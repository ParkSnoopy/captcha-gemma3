import torch
import torch.nn as nn
import torch.nn.functional as F

from util import RoPECache, rotate_half  # <- rotate_half needed for local rope() helper

from dataclasses import dataclass
from typing import Optional, Tuple
import math


@dataclass
class AttnConfig:
    d_model: int
    n_heads: int
    n_kv_heads: int  # GQA: keys/values shared across groups of query heads
    rope_base: float
    max_seq_len: int
    attn_dropout: float = 0.0
    qk_norm: bool = True
    rope_scale_for_kv: bool = True  # apply rope to both q and k (standard)


class QKNorm(nn.Module):
    """L2-normalize q and k, then scale by learned gamma. Keeps logits well-behaved."""

    def __init__(self, head_dim: int, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.gamma_q = nn.Parameter(torch.ones(head_dim))
        self.gamma_k = nn.Parameter(torch.ones(head_dim))

    def forward(self, q, k):
        # q,k: [B, H, T, D]
        q = q / (q.norm(dim=-1, keepdim=True) + self.eps) * self.gamma_q
        k = k / (k.norm(dim=-1, keepdim=True) + self.eps) * self.gamma_k
        return q, k


class GQAAttention(nn.Module):
    """
    Grouped-Query Attention (shared K/V across head groups) with RoPE and optional sliding-window mask.
    """

    def __init__(self, cfg: AttnConfig, is_local: bool, window: Optional[int]):
        super().__init__()
        self.cfg = cfg
        self.is_local = is_local
        self.window = window  # e.g., 1024 for local; None for global

        d = cfg.d_model
        h = cfg.n_heads
        hk = cfg.n_kv_heads
        hd = d // h  # head dim
        self.h = h
        self.hk = hk
        self.hd = hd

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
        mask = i.unsqueeze(1) - j.unsqueeze(0)  # [T, T]
        causal = mask < 0
        if self.window is None:
            sw = torch.zeros_like(mask, dtype=torch.bool)
        else:
            sw = mask > self.window
        full = causal | sw
        return full  # True => -inf

    @staticmethod
    def _rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        # Why: we need separate cos/sin for q (tail slice) vs k (full).
        return (x * cos) + (rotate_half(x) * sin)

    def forward(
        self, x: torch.Tensor, kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ):
        """
        x: [B, T, d]
        kv (optional): external KV to use (past cache). If provided, expected shapes:
          k,v: [B, hk, T_kv, hd]
        Returns: y, new_k, new_v
        """
        B, T, _ = x.shape
        H, HK, D = self.h, self.hk, self.hd
        device = x.device

        q = self.wq(x).view(B, T, H, D).transpose(1, 2)  # [B, H, T, D]
        k_new = self.wk(x).view(B, T, HK, D).transpose(1, 2)  # [B, HK, T, D]
        v_new = self.wv(x).view(B, T, HK, D).transpose(1, 2)

        # Concatenate to cache if provided (inference)
        if kv is not None:
            k_prev, v_prev = kv
            k = torch.cat([k_prev, k_new], dim=2)  # time concat
            v = torch.cat([v_prev, v_new], dim=2)
        else:
            k, v = k_new, v_new

        # --- RoPE fix: apply correct positions to q (tail only) and k (full) ---
        T_q = q.size(2)
        T_kv = k.size(2)
        # full cos/sin for all keys
        cos_k, sin_k = self.rope.get(T_kv)
        cos_k = cos_k.to(x.dtype).to(device)
        sin_k = sin_k.to(x.dtype).to(device)
        # tail slice for queries (new positions only)
        cos_q = cos_k[..., -T_q:, :]
        sin_q = sin_k[..., -T_q:, :]

        # apply separately: avoids 1->T_kv broadcast on q during decoding
        q = self._rope(q, cos_q, sin_q)
        k = self._rope(k, cos_k, sin_k)

        # QK-norm
        if self.qk_norm is not None:
            q, k = self.qk_norm(q, k)

        # ---- keep HK version for the cache ----
        k_cache, v_cache = k, v

        # Expand K/V across query heads (GQA)
        if H != HK:
            repeat = H // HK
            k = k.repeat_interleave(repeat, dim=1)  # [B, H, T_kv, D]
            v = v.repeat_interleave(repeat, dim=1)

        # attn
        att = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(D)  # [B, H, T_q, T_kv]

        # Masks: causal + optional sliding window
        if self.is_local and self.window is not None and (kv is None):
            bad = self._sliding_window_mask(att.size(-1), device)  # [T_kv, T_kv]
            att = att.masked_fill(bad.unsqueeze(0).unsqueeze(0), float("-inf"))
        else:
            i = torch.arange(att.size(-2), device=device)  # T_q
            j = torch.arange(att.size(-1), device=device)  # T_kv
            causal = j > i.unsqueeze(-1)
            att = att.masked_fill(causal.unsqueeze(0).unsqueeze(0), float("-inf"))

        p = F.softmax(att, dim=-1)
        p = self.attn_drop(p)
        y = torch.matmul(p, v)  # [B, H, T_q, D]

        # reshape (robust to any T_q)
        y = y.transpose(1, 2).contiguous()
        T_out = y.size(1)
        # sanity (why: catch future mismatches early)
        assert T_out == T_q, f"T_out({T_out}) != T_q({T_q})"
        y = y.view(B, T_out, H * D)
        y = self.wo(y)

        # trim cache time dimension on the HK version
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
