import torch
import torch.nn as nn


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(q, k, cos, sin):
    # q,k: [B, n_heads, T, head_dim], cos/sin broadcastable to that
    q_ = (q * cos) + (rotate_half(q) * sin)
    k_ = (k * cos) + (rotate_half(k) * sin)
    return q_, k_


class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        # Root-mean-square layer norm (no bias)
        scale = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return self.weight * x * scale


class RoPECache:
    """
    Builds cos/sin caches for RoPE with a specified base.
    We keep separate caches for local/global (different bases).
    """

    def __init__(self, head_dim: int, base: float, max_len: int, device=None):
        assert head_dim % 2 == 0
        self.head_dim = head_dim
        self.base = base
        self.max_len = max_len
        self.device = device

        half = head_dim // 2
        # theta shape: [half]
        theta = 1.0 / (
            base ** (torch.arange(0, half, dtype=torch.float32, device=device) / half)
        )
        # positions: [max_len, 1]
        pos = torch.arange(max_len, device=device, dtype=torch.float32).unsqueeze(
            1
        )  # [T, 1]
        freqs = pos * theta  # [T, half]
        self.cos = torch.cos(freqs).repeat_interleave(2, dim=1)  # [T, head_dim]
        self.sin = torch.sin(freqs).repeat_interleave(2, dim=1)

    def get(self, t: int):
        # from:
        # cos = self.cos[:t].unsqueeze(1).unsqueeze(1)
        # sin = self.sin[:t].unsqueeze(1).unsqueeze(1)

        # to (shapes -> [1,1,T,D]):
        cos = self.cos[:t].unsqueeze(0).unsqueeze(0)
        sin = self.sin[:t].unsqueeze(0).unsqueeze(0)
        return cos, sin
