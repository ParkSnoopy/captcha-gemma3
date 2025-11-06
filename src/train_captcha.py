import argparse
from pathlib import Path
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image

# Import your model code
from model import Gemma3Model

# -----------------------
# Charset / Tokenization
# -----------------------
# Default: 10 digits + 26 uppercase letters
DEFAULT_CHARSET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
PAD_TOKEN = "<pad>"
BOS_TOKEN = "<bos>"


def build_vocab(charset: str):
    itos = [PAD_TOKEN, BOS_TOKEN] + list(charset)
    stoi = {ch: i for i, ch in enumerate(itos)}
    return itos, stoi


def encode_text(text: str, stoi: dict) -> List[int]:
    return [stoi[c] for c in text]


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
        # grayscale, resize to (H,W) = (100,250) default
        img = (
            Image.open(path)
            .convert("L")
            .resize((self.img_size[1], self.img_size[0]), Image.BILINEAR)
        )
        # to tensor [0,1], shape [1,H,W]
        x = torch.from_numpy(
            (
                torch.ByteTensor(torch.ByteStorage.from_buffer(img.tobytes())).float()
                / 255.0
            ).numpy()
        )
        x = x.view(self.img_size[0], self.img_size[1])  # [H,W]
        x = x.unsqueeze(0)  # [1,H,W]
        return x

    def __getitem__(self, idx):
        path, label = self.items[idx]
        x = self._load_img(path)  # [1,H,W]
        y = label
        return x, y


# -----------------------
# Collate
# -----------------------
def collate_fn(batch, stoi: dict):
    xs, ys = zip(*batch)
    x = torch.stack(xs, dim=0)  # [B,1,H,W]
    # Build text ids: [BOS] + label_chars
    bos_id = stoi[BOS_TOKEN]
    text_ids = []
    for lab in ys:
        seq = [bos_id] + [stoi[c] for c in lab]
        text_ids.append(torch.tensor(seq, dtype=torch.long))
    text = torch.stack(text_ids, dim=0)  # [B, 1+4]
    return x, text, ys  # xs, tokenized text, and raw labels for debug


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


# -----------------------
# Training
# -----------------------
def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Data
    ds = CaptchaDataset(
        args.data,
        charset=DEFAULT_CHARSET,
        img_size=(args.height, args.width),
    )
    itos, stoi = ds.itos, ds.stoi
    print(f"Vocab size: {len(itos)}  (includes PAD and BOS)")
    train_loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=lambda b: collate_fn(b, stoi),
    )

    # Vision tokens via PatchEmbed
    patch = args.patch
    d_model = args.d_model
    patcher = PatchEmbed(
        img_size=(args.height, args.width), patch=patch, in_chans=1, d_model=d_model
    ).to(device)
    vision_tokens = patcher.tokens
    print(f"Vision tokens per image: {vision_tokens}")

    # Model
    model = Gemma3Model(
        vocab_size=len(itos),
        d_model=d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        n_kv_heads=args.n_kv_heads,
        max_seq_len=max(1024, vision_tokens + 16),
        local_window=args.local_window,
        l2g=args.l2g,
        attn_dropout=0.0,
        mlp_ratio=args.mlp_ratio,
        qk_norm=True,
        tie_embedding=True,
        vision_enabled=True,
        vision_tokens=vision_tokens,
    ).to(device)

    # Optimizer
    opt = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=0.1, betas=(0.9, 0.95)
    )
    scaler = torch.amp.GradScaler(str(device), enabled=args.amp)

    # Train
    model.train()
    global_step = 0
    for epoch in range(args.epochs):
        for imgs, text_ids, raw in train_loader:
            imgs = imgs.to(device=device, dtype=torch.float32)
            text_ids = text_ids.to(device)

            # Vision -> tokens
            with torch.amp.autocast(str(device), enabled=args.amp):
                vtok = patcher(imgs)  # [B, V, d_model]
                logits, _ = model(input_ids=text_ids, vision_embeds=vtok, kv_cache=None)
                loss = compute_autoregressive_loss(
                    logits, text_ids, vision_len=vision_tokens, ignore_index=-100
                )

            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)

            if global_step % args.log_every == 0:
                print(f"epoch {epoch} step {global_step}  loss {loss.item():.4f}")
            global_step += 1

        # (optional) save checkpoint per epoch
        if args.out:
            ckpt_path = Path(args.out) / f"captcha_gemma3_e{epoch + 1}.pt"
            ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model": model.state_dict(),
                    "patcher": patcher.state_dict(),
                    "stoi": stoi,
                    "itos": itos,
                    "args": vars(args),
                },
                ckpt_path,
            )
            print(f"Saved {ckpt_path}")

    print("Done.")


# -----------------------
# Inference helper
# -----------------------
@torch.no_grad()
def predict_batch(
    images: torch.Tensor,
    model: Gemma3Model,
    patcher: PatchEmbed,
    stoi: dict,
    itos: list,
    max_len=4,
    device="cpu",
):
    model.eval()
    bos_id = stoi[BOS_TOKEN]
    vtok = patcher(images.to(device=device, dtype=torch.float32))  # [B, V, d]
    B = images.size(0)

    # Autoregressive greedy decode for exactly 4 chars
    seq = torch.full((B, 1), bos_id, dtype=torch.long, device=device)  # [B,1]
    outputs = []
    kv_cache = None
    for t in range(max_len):
        logits, kv_cache = model(input_ids=seq, vision_embeds=vtok, kv_cache=kv_cache)
        next_logits = logits[:, -1, :]  # [B, vocab]
        next_id = next_logits[:, 2:].argmax(dim=-1) + 2  # skip PAD(0), BOS(1)
        seq = torch.cat([seq, next_id.unsqueeze(1)], dim=1)
        outputs.append(next_id)

    outputs = torch.stack(outputs, dim=1)  # [B, 4]
    preds = []
    for b in range(B):
        chars = [itos[idx.item()] for idx in outputs[b]]
        preds.append("".join(chars))
    return preds


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--data",
        type=str,
        required=True,
        help="Folder with images",
    )
    p.add_argument(
        "--out", type=str, default="checkpoints", help="Where to save checkpoints"
    )
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--amp", action="store_true", help="Enable mixed precision")
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument(
        "--no_csv",
        action="store_true",
        help="Ignore labels.csv and read labels from filenames",
    )

    # image + patch
    p.add_argument("--height", type=int, default=100)
    p.add_argument("--width", type=int, default=250)
    p.add_argument("--patch", type=int, default=25)

    # model size
    p.add_argument("--d_model", type=int, default=512)
    p.add_argument("--n_layers", type=int, default=12)
    p.add_argument("--n_heads", type=int, default=8)
    p.add_argument("--n_kv_heads", type=int, default=4)
    p.add_argument("--local_window", type=int, default=256)
    p.add_argument("--l2g", type=int, default=5)
    p.add_argument("--mlp_ratio", type=float, default=4.0)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
