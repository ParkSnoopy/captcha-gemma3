import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Tuple, List

from config import BOS_TOKEN


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rope_single(x, cos, sin):
    """
    Apply RoPE to a single tensor.
    x:   [B, H, T, D]
    cos/sin: broadcastable to x (usually [1,1,T,D])
    """
    # why: lets q and k use different time indices during incremental decode
    return (x * cos) + (rotate_half(x) * sin)


def apply_rope(q, k, cos, sin):
    """
    Legacy pair-wise API (same cos/sin for both). Kept for compatibility.
    """
    q_ = apply_rope_single(q, cos, sin)
    k_ = apply_rope_single(k, cos, sin)
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
        pos = torch.arange(max_len, device=device, dtype=torch.float32).unsqueeze(1)
        freqs = pos * theta
        self.cos = torch.cos(freqs).repeat_interleave(2, dim=1)
        self.sin = torch.sin(freqs).repeat_interleave(2, dim=1)

    def get(self, t: int):
        cos = self.cos[:t].unsqueeze(0).unsqueeze(0)  # [1,1,T,D]
        sin = self.sin[:t].unsqueeze(0).unsqueeze(0)
        return cos, sin


# -----------------------
# Collate
# -----------------------
def collate_fn(batch, stoi: dict):
    xs, ys = zip(*batch)
    x = torch.stack(xs, dim=0)  # [B,1,H,W]
    bos_id = stoi[BOS_TOKEN]
    text_ids = []
    for lab in ys:
        seq = [bos_id] + [stoi[c] for c in lab]
        text_ids.append(torch.tensor(seq, dtype=torch.long))
    text = torch.stack(text_ids, dim=0)  # [B, 1+4]
    return x, text, ys


class Collate:
    """Windows-picklable collate wrapper (why: avoid lambda/closure in DataLoader)."""

    def __init__(self, stoi: dict):
        self.stoi = stoi

    def __call__(self, batch):
        return collate_fn(batch, self.stoi)


# -----------------------
# Loss helper
# -----------------------
def compute_autoregressive_loss(logits, text_ids, vision_len, ignore_index=-100):
    """
    logits: [B, V+T, vocab]
    text_ids: [B, T] (BOS + 4 chars)  -> we want to predict the last T-1 tokens (the 4 chars) conditioned on BOS
    Only compute loss on the text region; ignore all vision positions.
    """
    B, S, V = logits.shape
    T = text_ids.size(1)  # 1+4
    # shift as usual: logits[:-1] vs targets[1:]
    logits_shift = logits[:, :-1, :]  # [B, S-1, V]
    # Build "global tokens" stream indices: vision placeholders + text_ids
    # We only need targets for the text region; everything else = ignore_index
    targets = torch.full(
        (B, S - 1), ignore_index, dtype=torch.long, device=logits.device
    )
    # the first text position to supervise in logits_shift is index = vision_len
    # we have T-1 targets (exclude BOS)
    targets[:, vision_len : vision_len + (T - 1)] = text_ids[:, 1:]  # [B, 4]
    loss = F.cross_entropy(
        logits_shift.reshape(-1, V), targets.reshape(-1), ignore_index=ignore_index
    )
    return loss


# ---- metrics ----
def token_and_seq_accuracy(preds: List[str], targets: List[str]) -> Tuple[float, float]:
    correct_seq = sum(p == t for p, t in zip(preds, targets))
    seq_acc = correct_seq / max(1, len(preds))
    total_tok = sum(len(t) for t in targets)
    correct_tok = sum(
        sum(pi == ti for pi, ti in zip(p, t)) for p, t in zip(preds, targets)
    )
    tok_acc = correct_tok / max(1, total_tok)
    return tok_acc, seq_acc
