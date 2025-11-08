import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision.transforms.functional import to_tensor

from PIL import Image

from pathlib import Path
from typing import Optional

from util import RMSNorm, build_vocab
from attn import GQAAttention, AttnConfig, SwiGLU


# -----------------------
# Vision Patch Embedder
# -----------------------
class PatchEmbed(nn.Module):
    """
    Simple non-pretrained patch embedder that converts a (B,1,H,W) grayscale image
    into a sequence of tokens of dimension d_model. For (H,W)=(100,250) and patch=10,
    tokens = (H/10)*(W/10) = 10*25 = 250.
    """

    def __init__(self, img_size=(100, 250), patch=10, in_chans=1, d_model=512):
        super().__init__()
        H, W = img_size
        assert H % patch == 0 and W % patch == 0, (
            "Image size must be divisible by patch size"
        )
        self.grid_h = H // patch
        self.grid_w = W // patch
        self.tokens = self.grid_h * self.grid_w
        self.proj = nn.Conv2d(
            in_chans, d_model, kernel_size=patch, stride=patch, bias=False
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        # x: [B,1,H,W]
        y = self.proj(x)  # [B, d_model, H', W']
        y = y.permute(0, 2, 3, 1)  # [B, H', W', d_model]
        y = y.reshape(y.size(0), -1, y.size(-1))  # [B, tokens, d_model]
        y = self.norm(y)
        return y


# -----------------------
# Dataset
# -----------------------
class CaptchaDataset(Dataset):
    def __init__(self, root: str, charset: str, img_size=(100, 250)):
        self.root = Path(root)
        self.img_size = img_size
        self.charset = charset
        self.itos, self.stoi = build_vocab(charset)

        self.items = []

        # parse label from filename stem (e.g., "AB1C.png" -> "AB1C")
        img_glob = list(
            self.root.glob("*.png")
        )  # only `png` for now # + list(self.root.glob("*.jpg")) + list(self.root.glob("*.jpeg"))
        for p in sorted(img_glob):
            label = p.stem.split(".")[
                0
            ]  # assume first segment separated by '.' is the label
            self.items.append((p, label))

        # keep only items with 4-char labels that exist in charset
        valid = []
        allowed = set(charset)
        for p, lab in self.items:
            if len(lab) == 4 and all(c in allowed for c in lab):
                valid.append((p, lab))
        self.items = valid

        if len(self.items) == 0:
            raise RuntimeError(
                "No valid samples found. Ensure images exist and labels are 4 chars inside charset."
            )

    def __len__(self):
        return len(self.items)

    def _load_img(self, path: Path):
        # grayscale, resize to (H,W) = (100,250)
        img = (
            Image.open(path)
            .convert("L")
            .resize(
                (self.img_size[1], self.img_size[0]), Image.BILINEAR
            )  # (W,H) for PIL
        )
        x = to_tensor(img)  # -> [1, H, W], float32 in [0,1]
        return x

    def __getitem__(self, idx):
        path, label = self.items[idx]
        x = self._load_img(path)  # [1,H,W]
        y = label
        return x, y


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
