import torch
import torch.nn as nn

from util import RMSNorm
from attn import GQAAttention, AttnConfig, SwiGLU

from typing import Optional


class Gemma3Block(nn.Module):
    def __init__(
        self,
        d_model,
        n_heads,
        n_kv_heads,
        max_seq_len,
        is_local: bool,
        local_window: Optional[int],
        rope_base_local=10_000.0,
        rope_base_global=1_000_000.0,
        attn_dropout=0.0,
        mlp_ratio=4.0,
        qk_norm=True,
    ):
        super().__init__()
        rope_base = rope_base_local if is_local else rope_base_global
        self.is_local = is_local

        self.norm1 = RMSNorm(d_model)  # pre-norm
        self.attn = GQAAttention(
            AttnConfig(
                d_model=d_model,
                n_heads=n_heads,
                n_kv_heads=n_kv_heads,
                rope_base=rope_base,
                max_seq_len=max_seq_len,
                attn_dropout=attn_dropout,
                qk_norm=qk_norm,
            ),
            is_local=is_local,
            window=local_window if is_local else None,
        )
        self.norm2 = RMSNorm(d_model)
        self.mlp = SwiGLU(d_model, int(d_model * mlp_ratio))

        # (Optionally) a post-norm after residual could be added to mirror "pre+post" note;
        # in practice, Pre-LN with RMSNorm + final RMSNorm works well.

    def forward(self, x, kv=None):
        att_in = self.norm1(x)
        att_out, kv_new = self.attn(att_in, kv)
        x = x + att_out
        x = x + self.mlp(self.norm2(x))
        return x, kv_new


class Gemma3Model(nn.Module):
    """
    Decoder-only Transformer with 5:1 local:global interleaving.
    Optionally accepts 'vision_tokens' of shape [B, V=256, d_model] to prepend.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 3072,
        n_layers: int = 48,
        n_heads: int = 24,
        n_kv_heads: int = 8,
        max_seq_len: int = 128_000,
        local_window: int = 1024,
        l2g: int = 5,  # 5 locals then 1 global
        attn_dropout: float = 0.0,
        mlp_ratio: float = 4.0,
        qk_norm: bool = True,
        tie_embedding: bool = True,
        vision_enabled: bool = True,
        vision_tokens: int = 256,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.local_window = local_window
        self.l2g = l2g
        self.vision_enabled = vision_enabled
        self.vision_tokens = vision_tokens

        self.embed = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList()

        # Build interleaved stack: start with local
        for i in range(n_layers):
            is_local = (
                (i % (l2g + 1)) != l2g
            )  # e.g., for l2g=5: indices 0-4 local, 5 global, 6-10 local, 11 global, ...
            self.layers.append(
                Gemma3Block(
                    d_model=d_model,
                    n_heads=n_heads,
                    n_kv_heads=n_kv_heads,
                    max_seq_len=max_seq_len,
                    is_local=is_local,
                    local_window=local_window,
                    attn_dropout=attn_dropout,
                    mlp_ratio=mlp_ratio,
                    qk_norm=qk_norm,
                )
            )

        self.final_norm = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        if tie_embedding:
            self.lm_head.weight = self.embed.weight

        # Simple projector for vision embeddings (assuming SigLIP output already provided)
        if vision_enabled:
            self.vision_proj = nn.Linear(
                d_model, d_model, bias=False
            )  # identity-sized; adjust if your SigLIP dim differs

    def forward(
        self,
        input_ids: torch.Tensor,  # [B, T_text]
        vision_embeds: Optional[torch.Tensor] = None,  # [B, 256, d_v] if provided
        kv_cache: Optional[list] = None,  # list of (k,v) per layer for fast decode
    ):
        B, T_txt = input_ids.shape
        x = self.embed(input_ids)  # [B, T_text, d]

        # Prepend vision tokens if given
        if self.vision_enabled and vision_embeds is not None:
            # Project to LM dim if needed
            if vision_embeds.size(-1) != self.d_model:
                vision_tok = self.vision_proj(vision_embeds)
            else:
                vision_tok = vision_embeds
            x = torch.cat([vision_tok, x], dim=1)  # [B, V+T, d]

        T = x.size(1)
        assert T <= self.max_seq_len, "sequence exceeds max_seq_len"

        new_cache = []
        h = x
        for i, layer in enumerate(self.layers):
            kv = None if kv_cache is None else kv_cache[i]
            h, kv = layer(h, kv)
            new_cache.append(kv)

        h = self.final_norm(h)
        logits = self.lm_head(h)  # [B, T, vocab]
        return logits, new_cache


# Test
if __name__ == "__main__":
    g3 = Gemma3Model(vocab_size=100)
