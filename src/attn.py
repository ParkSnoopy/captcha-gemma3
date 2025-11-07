import torch
import torch.nn as nn
import torch.nn.functional as F

from util import RoPECache, apply_rope_single

from dataclasses import dataclass
from typing import Optional, Tuple
import math


@dataclass
class AttnConfig:
    d_model: int
    n_heads: int
    n_kv_heads: int
    rope_base: float
    max_seq_len: int
    attn_dropout: float = 0.0
    qk_norm: bool = True


class QKNorm(nn.Module):
    def __init__(self, head_dim: int, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.gamma_q = nn.Parameter(torch.ones(head_dim))
        self.gamma_k = nn.Parameter(torch.ones(head_dim))

    def forward(self, q, k):
        q = q / (q.norm(dim=-1, keepdim=True) + self.eps) * self.gamma_q
        k = k / (k.norm(dim=-1, keepdim=True) + self.eps) * self.gamma_k
        return q, k


class GQAAttention(nn.Module):
    def __init__(self, cfg: AttnConfig, is_local: bool, window: Optional[int]):
        super().__init__()
        self.cfg = cfg
        self.is_local = is_local
        self.window = window

        d = cfg.d_model
        h = cfg.n_heads
        hk = cfg.n_kv_heads
        hd = d // h
        self.h, self.hk, self.hd = h, hk, hd

        self.wq = nn.Linear(d, h * hd, bias=False)
        self.wk = nn.Linear(d, hk * hd, bias=False)
        self.wv = nn.Linear(d, hk * hd, bias=False)
        self.wo = nn.Linear(h * hd, d, bias=False)

        self.attn_drop = nn.Dropout(cfg.attn_dropout)
        self.rope = RoPECache(head_dim=hd, base=cfg.rope_base, max_len=cfg.max_seq_len)
        self.qk_norm = QKNorm(hd) if cfg.qk_norm else None

    def _sliding_window_mask(self, T: int, device):
        i = torch.arange(T, device=device)
        j = torch.arange(T, device=device)
        mask = i.unsqueeze(1) - j.unsqueeze(0)  # [T,T]
        causal = mask < 0
        sw = (
            torch.zeros_like(mask, dtype=torch.bool)
            if self.window is None
            else (mask > self.window)
        )
        return causal | sw

    @staticmethod
    def _rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        # Why: only document the reason. Keeps q slice separate from k full.
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)

    def forward(
        self, x: torch.Tensor, kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ):
        """
        x: [B, T, d]
        kv: optional (k,v) cache with shapes [B, hk, T_kv_prev, D]
        """
        B, T, _ = x.shape
        H, HK, D = self.h, self.hk, self.hd
        device = x.device

        q = self.wq(x).view(B, T, H, D).transpose(1, 2)  # [B,H,T,D]
        k_new = self.wk(x).view(B, T, HK, D).transpose(1, 2)  # [B,HK,T,D]
        v_new = self.wv(x).view(B, T, HK, D).transpose(1, 2)

        if kv is not None:
            k = torch.cat([kv[0], k_new], dim=2)  # [B,HK,T_kv,D]
            v = torch.cat([kv[1], v_new], dim=2)
        else:
            k, v = k_new, v_new

        T_q, T_kv = q.size(2), k.size(2)

        # ----- RoPE fix: same D, slice only time for q -----
        cos_k, sin_k = self.rope.get(T_kv)  # [1,1,T_kv,D]
        cos_k = cos_k.to(x.dtype).to(device)
        sin_k = sin_k.to(x.dtype).to(device)
        cos_q = cos_k[:, :, -T_q:, :]  # [1,1,T_q,D]
        sin_q = sin_k[:, :, -T_q:, :]

        q = apply_rope_single(q, cos_q, sin_q)
        k = apply_rope_single(k, cos_k, sin_k)

        # Optional QK norm
        if self.qk_norm is not None:
            q, k = self.qk_norm(q, k)

        # Keep HK version for returning cache
        k_cache, v_cache = k, v

        # GQA: expand K/V to H heads
        if H != HK:
            repeat = H // HK
            k = k.repeat_interleave(repeat, dim=1)  # [B,H,T_kv,D]
            v = v.repeat_interleave(repeat, dim=1)

        att = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(D)  # [B,H,T_q,T_kv]

        # Masks
        if self.is_local and self.window is not None and (kv is None):
            i = torch.arange(att.size(-1), device=device)
            j = torch.arange(att.size(-1), device=device)
            bad = (i.unsqueeze(1) - j.unsqueeze(0) < 0) | (
                i.unsqueeze(1) - j.unsqueeze(0) > self.window
            )
            att = att.masked_fill(bad.unsqueeze(0).unsqueeze(0), float("-inf"))
        else:
            i = torch.arange(att.size(-2), device=device)
            j = torch.arange(att.size(-1), device=device)
            att = att.masked_fill(
                (j > i.unsqueeze(-1)).unsqueeze(0).unsqueeze(0), float("-inf")
            )

        p = F.softmax(att, dim=-1)
        p = self.attn_drop(p)
        y = torch.matmul(p, v).transpose(1, 2).contiguous().view(B, T_q, H * D)
        y = self.wo(y)

        # Trim cache if sliding window
        if self.is_local and self.window is not None and k_cache.size(2) > self.window:
            k_cache = k_cache[:, :, -self.window :]
            v_cache = v_cache[:, :, -self.window :]

        return y, (k_cache, v_cache)


class SwiGLU(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w3 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))
