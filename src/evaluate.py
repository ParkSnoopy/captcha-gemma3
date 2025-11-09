# src/evaluate.py
import argparse
from pathlib import Path
from typing import List

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image

from model import Gemma3Model, PatchEmbed
from util import build_vocab
from config import DEFAULT_CHARSET

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def _greedy_with_probe(
    images: torch.Tensor,
    model: Gemma3Model,
    patcher: PatchEmbed,
    itos: List[str],
    max_len: int = 4,
    topk: int = 5,
):
    model.eval()
    vtok = patcher(images.to(device=device, dtype=torch.float32))
    B = images.size(0)

    # Prefill from vision only (no BOS)
    empty = torch.empty((B, 0), dtype=torch.long, device=device)
    logits, kv = model(input_ids=empty, vision_embeds=vtok, kv_cache=None)

    def topk_tokens(log):  # [B, V]
        p = F.softmax(log, dim=-1)
        vals, idxs = torch.topk(p, k=min(topk, p.size(-1)), dim=-1)
        out = []
        for b in range(p.size(0)):
            out.append(
                [
                    (itos[idxs[b, j].item()], float(vals[b, j].item()))
                    for j in range(idxs.size(1))
                ]
            )
        return out

    step_log = []
    step0 = logits[:, -1, :]
    tk = topk_tokens(step0)
    step_log.append(tk)
    next_id = torch.argmax(step0, dim=-1)
    out_ids = [next_id]

    for _ in range(1, max_len):
        logits, kv = model(
            input_ids=next_id.unsqueeze(1), vision_embeds=None, kv_cache=kv
        )
        step = logits[:, -1, :]
        tk = topk_tokens(step)
        step_log.append(tk)
        next_id = torch.argmax(step, dim=-1)
        out_ids.append(next_id)

    outs = torch.stack(out_ids, dim=1)  # [B, T]
    preds = ["".join(itos[i.item()] for i in row) for row in outs]
    # [T][B][topk] -> [B][T][topk]
    Bsz = len(preds)
    T = len(step_log)
    probes = [[step_log[t][b] for t in range(T)] for b in range(Bsz)]
    return preds, probes


def _load_img(path: Path, H: int, W: int, n_channels: int) -> torch.Tensor:
    img = (
        Image.open(path)
        .convert("RGB" if n_channels == 3 else "L")
        .resize((W, H), Image.BILINEAR)
    )

    arr = np.array(img, dtype=np.float32) / 255.0  # [H,W] or [H,W,3]
    if n_channels == 1:
        if arr.ndim == 3:
            arr = arr[..., 0]
        x = torch.from_numpy(arr).view(H, W).unsqueeze(0)  # [1,H,W]
    else:
        if arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=-1)
        x = torch.from_numpy(arr).permute(2, 0, 1).contiguous()  # [3,H,W]
    return x


def evaluation(cli_args):
    ckpt_path = Path(cli_args.use_checkpoint)
    assert ckpt_path.exists(), f"Checkpoint not found: {ckpt_path}"

    ckpt = torch.load(ckpt_path, map_location="cpu")
    args = ckpt["args"]
    itos, _ = build_vocab(DEFAULT_CHARSET)

    model = (
        Gemma3Model(
            vocab_size=len(itos),
            d_model=args["d_model"],
            n_layers=args["n_layers"],
            n_heads=args["n_heads"],
            n_kv_heads=args["n_kv_heads"],
            max_seq_len=args["max_seq_len"],
            local_window=args["local_window"],
            l2g=args["l2g"],
            mlp_ratio=args["mlp_ratio"],
            vision_enabled=True,
            vision_tokens=(args["vision_tokens"]),
        )
        .to(device)
        .eval()
    )
    model.load_state_dict(ckpt["model"], strict=True)

    patcher = (
        PatchEmbed(
            img_size=(args["height"], args["width"]),
            patch=args["patch"],
            in_chans=args["n_channels"],
            d_model=args["d_model"],
        )
        .to(device)
        .eval()
    )
    patcher.load_state_dict(ckpt["patcher"])

    root = Path(cli_args.data)
    img_paths = sorted(root.glob("*.png"))

    for img_path in img_paths:
        x = (
            _load_img(
                path=img_path,
                H=args["height"],
                W=args["width"],
                n_channels=args["n_channels"],
            )
            .unsqueeze(0)
            .to(device)
        )  # [1,C,H,W]
        preds, probs = _greedy_with_probe(x, model, patcher, itos, max_len=4, topk=5)
        pred = preds[0]
        print(f"{img_path.name} -> {pred}")
        if cli_args.verbose:
            for t, top in enumerate(probs[0], 1):
                tops = ", ".join([f"{ch}:{prob:.2f}" for ch, prob in top])
                print(f"  t{t}: {tops}")


def parse_args(args=None):
    p = argparse.ArgumentParser()
    p.add_argument("--use-checkpoint", type=str, required=True)
    p.add_argument("--data", type=str, required=True)
    p.add_argument("--verbose", action="store_true")
    return p.parse_args() if args is None else p.parse_args(args)


if __name__ == "__main__":
    evaluation(parse_args())
