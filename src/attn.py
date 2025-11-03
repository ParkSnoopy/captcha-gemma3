import torch
import torch.nn as nn

from util import RoPECache

from dataclasses import dataclass
from typing import Optional, Tuple



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
    """
    Simple, stable QK-norm: L2-normalize q and k, then scale by a learned gamma.
    This mirrors the 'QK-norm' idea used instead of soft-capping in Gemma 3.
    """
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
        hd = d // h                          # head dim
        self.h = h
        self.hk = hk
        self.hd = hd

        self.wq = nn.Linear(d, h * hd, bias=False)
        self.wk = nn.Linear(d, hk * hd, bias=False)
        self.wv = nn.Linear(d, hk * hd, bias=False)
        self.wo = nn.Linear(h * hd, d, bias=False)

        self.attn_drop = nn.Dropout(cfg.attn_dropout)

        self.rope = RoPECache(head_dim=hd,
                              base=cfg.rope_base,
                              max_len=cfg.max_seq_len)

        self.qk_norm = QKNorm(hd) if cfg.qk_norm else None

    def _sliding_window_mask(self, T: int, device):
        # causal + window: allow attending only to last 'window' tokens
        i = torch.arange(T, device=device)
        j = torch.arange(T, device=device)
        mask = i.unsqueeze(1) - j.unsqueeze(0)  # [T, T]
        # disallow j > i (future) OR (i-j) > window
        causal = mask < 0
        if self.window is None:
            sw = torch.zeros_like(mask, dtype=torch.bool)
        else:
            sw = mask > self.window
        full = causal | sw
        return full  # True => -inf

    def forward(self, x: torch.Tensor, kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None):
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

        # Apply RoPE (different base per-attn module via cfg.rope_base)
        cos, sin = self.rope.get(k.size(2))  # T_kv
        # broadcast to [1, *, T, D]
        cos = cos.to(x.dtype).to(device)
        sin = sin.to(x.dtype).to(device)

        # we only rotate first 2*floor(D/2) dims; our D is even by construction
        q, k = apply_rope(q, k, cos, sin)

        # QK-norm (Gemma3 uses QK-norm instead of soft-capping)
        if self.qk_norm is not None:
            q, k = self.qk_norm(q, k)

        # Expand K/V across query heads (GQA): map hk -> h by repeating groups
        # (each group of query heads shares one K/V head)
        if H != HK:
            repeat = H // HK
            k = k.repeat_interleave(repeat, dim=1)  # [B, H, T, D]
            v = v.repeat_interleave(repeat, dim=1)

        att = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(D)  # [B, H, T, T_kv]

        # Masks: causal + optional sliding window
        if self.is_local and self.window is not None:
            bad = self._sliding_window_mask(att.size(-1), device)  # [T_kv, T_kv] but we query full prefixes
            # We need mask aligned on [T (query), T_kv (key)]. For autoregressive prefill, T==T_kv.
            att = att.masked_fill(bad.unsqueeze(0).unsqueeze(0), float('-inf'))
        else:
            # causal mask (no future)
            i = torch.arange(att.size(-2), device=device)
            j = torch.arange(att.size(-1), device=device)
            causal = j > i.unsqueeze(-1)
            att = att.masked_fill(causal.unsqueeze(0).unsqueeze(0), float('-inf'))

        p = F.softmax(att, dim=-1)
        p = self.attn_drop(p)
        y = torch.matmul(p, v)  # [B, H, T, D]
        y = y.transpose(1, 2).contiguous().view(B, T, H * D)
        y = self.wo(y)

        # For local layers we can drop old cache beyond window to save memory
        if self.is_local and self.window is not None and k.size(2) > self.window:
            k = k[:, :, -self.window:]
            v = v[:, :, -self.window:]

        return y, (k, v)


class SwiGLU(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w3 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))
