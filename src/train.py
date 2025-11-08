import torch
from torch.utils.data import DataLoader, random_split

import argparse
from tqdm.auto import tqdm, trange

import os
import random
from pathlib import Path
from typing import List

from config import DEFAULT_CHARSET
from util import Collate, compute_autoregressive_loss, token_and_seq_accuracy
from model import Gemma3Model, PatchEmbed, CaptchaDataset


# ---- data helpers ----
def make_dataloaders(args, device):
    g = torch.Generator().manual_seed(args.seed)
    ds = CaptchaDataset(
        args.data,
        charset=DEFAULT_CHARSET,
        img_size=(args.height, args.width),
        img_channels=args.n_channels,
    )
    itos, stoi = ds.itos, ds.stoi

    if args.no_val:
        train_set, val_set = ds, None
    else:
        n_val = max(1, int(len(ds) * args.val_split))
        n_train = len(ds) - n_val
        train_set, val_set = random_split(ds, [n_train, n_val], generator=g)

    # ---- Windows/CPU safe DataLoader settings ----
    is_windows = os.name == "nt"
    on_cuda = device.type == "cuda"
    # Workers: 0 on Windows or CPU; otherwise modest parallelism
    train_workers = 0 if (is_windows or not on_cuda) else 4
    val_workers = 0 if (is_windows or not on_cuda) else 2
    # Pin memory only helps on CUDA
    pin = on_cuda
    # persistent_workers must be False when num_workers == 0
    train_persistent = train_workers > 0
    val_persistent = val_workers > 0

    collate = Collate(stoi)

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=train_workers,
        pin_memory=pin,
        collate_fn=collate,  # <-- no lambda
        drop_last=True,
        persistent_workers=train_persistent,
    )
    val_loader = None
    if val_set is not None:
        val_loader = DataLoader(
            val_set,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=val_workers,
            pin_memory=pin,
            collate_fn=collate,  # <-- same wrapper
            drop_last=False,
            persistent_workers=val_persistent,
        )
    return train_loader, val_loader, itos, stoi


# ---- validation loop ----
@torch.no_grad()
def validate(model, patcher, loader, stoi, itos, device, max_len=4, no_tqdm=False):
    model.eval()
    losses = []
    preds_all, targs_all = [], []
    data_iter = (
        loader
        if no_tqdm
        else tqdm(loader, desc="Validate", leave=False, dynamic_ncols=True)
    )
    for imgs, text_ids, raw in data_iter:
        imgs = imgs.to(device=device, dtype=torch.float32)
        text_ids = text_ids.to(device)
        vtok = patcher(imgs)
        logits, _ = model(input_ids=text_ids, vision_embeds=vtok, kv_cache=None)
        loss = compute_autoregressive_loss(
            logits, text_ids, vision_len=patcher.tokens, ignore_index=-100
        )
        losses.append(loss.item())
        # decode
        batch_preds = predict_batch(
            imgs, model, patcher, stoi, itos, max_len=max_len, device=device
        )
        preds_all.extend(batch_preds)
        # recover targets as strings
        targs = ["".join(itos[j.item()] for j in row[:max_len]) for row in text_ids]
        targs_all.extend(targs)
    tok_acc, seq_acc = token_and_seq_accuracy(preds_all, targs_all)
    return float(sum(losses) / max(1, len(losses))), tok_acc, seq_acc


# -----------------------
# Training
# -----------------------
def train(args):
    # set seed
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Data
    train_loader, val_loader, itos, stoi = make_dataloaders(args, device)
    print(
        f"Dataset size: train={len(train_loader.dataset)} val={0 if val_loader is None else len(val_loader.dataset)}"
    )
    print(f"Vocab size: {len(itos)}")

    # Vision tokens via PatchEmbed
    patcher = PatchEmbed(
        patch=args.patch,
        img_size=(args.height, args.width),
        in_chans=args.n_channels,
        d_model=args.d_model,
    ).to(device)
    vision_tokens = patcher.tokens
    print(f"Vision tokens per image: {vision_tokens}")

    # Model
    model = Gemma3Model(
        vocab_size=len(itos),
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        n_kv_heads=args.n_kv_heads,
        max_seq_len=max(1024, vision_tokens + 16),
        local_window=args.local_window,
        l2g=args.l2g,
        attn_dropout=args.dropout,
        mlp_ratio=args.mlp_ratio,
        qk_norm=True,
        tie_embedding=True,
        vision_enabled=True,
        vision_tokens=vision_tokens,
    ).to(device)

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95),
    )
    scaler = torch.amp.GradScaler(str(device), enabled=args.amp)
    schedular = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=args.train_patience, factor=0.5
    )

    # Train
    model.train()
    global_step = 0
    best_seq_acc = -1.0
    epochs_no_improve = 0

    epoch_iter = (
        range(args.epochs)
        if args.no_tqdm
        else trange(args.epochs, desc="Train Progress", leave=True)
    )

    for epoch in epoch_iter:
        data_iter = (
            train_loader
            if args.no_tqdm
            else (
                tqdm(
                    train_loader,
                    total=len(train_loader),
                    leave=False,
                    desc=f"Epoch {epoch + 1}",
                    dynamic_ncols=True,
                )
            )
        )

        for i, (imgs, text_ids, raw) in enumerate(data_iter):
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
            total_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), args.grad_clip
            ).item()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

            if not args.no_tqdm:
                mem = (
                    torch.cuda.memory_reserved() / 1e6 if device.type == "cuda" else 0.0
                )
                data_iter.set_postfix(
                    loss=f"{loss.item():.4f}",
                    gnorm=f"{total_norm:.2f}",
                    lr=f"{optimizer.param_groups[0]['lr']:.2e}",
                    mem=f"{mem:.2f} MB",
                )
            else:
                if global_step % args.log_every == 0:
                    print(
                        f"epoch {epoch + 1} | step {global_step} | loss {loss.item():.4f} | gnorm {total_norm:.2f}"
                    )

            # if global_step % args.log_every == 0:
            #    print(f"epoch {epoch} step {global_step}  loss {loss.item():.4f}")

            global_step += 1

        # ---- end epoch: validate/save ----
        val_loss, tok_acc, seq_acc = (0.0, 0.0, 0.0)
        if val_loader is not None:
            val_loss, tok_acc, seq_acc = validate(
                model,
                patcher,
                val_loader,
                stoi,
                itos,
                device,
                max_len=4,
                no_tqdm=args.no_tqdm,
            )
            schedular.step(val_loss)  # why: reduce LR when val stalls

        # Save checkpoint
        ckpt_path = Path(args.out) / f"captcha_gemma3_e{epoch + 1}_last.pt"
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model": model.state_dict(),
                "patcher": patcher.state_dict(),
                "stoi": stoi,
                "itos": itos,
                "args": vars(args) | {"vision_tokens": vision_tokens},
                "epoch": epoch + 1,
                "val_loss": val_loss,
                "val_tok_acc": tok_acc,
                "val_seq_acc": seq_acc,
            },
            ckpt_path,
        )
        (print if args.no_tqdm else tqdm.write)(f"Saved curr to '{ckpt_path}'")

        # improve with `validation`
        improved = (seq_acc > best_seq_acc) if val_loader is not None else True
        if improved:
            best_ckpt_path = Path(args.out) / f"captcha_gemma3_e{epoch + 1}_best.pt"
            best_seq_acc = seq_acc
            epochs_no_improve = 0
            torch.save(
                torch.load(ckpt_path, map_location="cpu"), best_ckpt_path
            )  # copy last->best
            (print if args.no_tqdm else tqdm.write)(
                f"Saved best to '{best_ckpt_path}' (seq_acc={seq_acc:.3f}, tok_acc={tok_acc:.3f}, val_loss={val_loss:.4f})"
            )
        else:
            epochs_no_improve += 1
            (print if args.no_tqdm else tqdm.write)(
                f"No improvement: seq_acc={seq_acc:.3f}, tok_acc={tok_acc:.3f}, val_loss={val_loss:.4f}"
            )
            if epochs_no_improve >= args.val_patience and not args.no_val:
                print(f"Early stopping (patience={args.val_patience}).")
                break

    print("Done.")


# -----------------------
# Inference helper
# -----------------------
@torch.no_grad()
def predict_batch(
    images: torch.Tensor,
    model: Gemma3Model,
    patcher,
    stoi: dict,
    itos: List[str],
    max_len: int = 4,
    device: str = "cpu",
):
    model.eval()
    vtok = patcher(images.to(device=device, dtype=torch.float32))
    B = images.size(0)

    # prefill: vision only (no BOS)
    empty = torch.empty((B, 0), dtype=torch.long, device=device)
    logits, kv_cache = model(input_ids=empty, vision_embeds=vtok, kv_cache=None)

    def pick_next(log):
        # No special tokens to mask
        return log.argmax(dim=-1)

    next_id = pick_next(logits[:, -1, :])
    out_ids = [next_id]

    for _ in range(1, max_len):
        logits, kv_cache = model(
            input_ids=next_id.unsqueeze(1),
            vision_embeds=None,
            kv_cache=kv_cache,
        )
        next_id = pick_next(logits[:, -1, :])
        out_ids.append(next_id)

    outputs = torch.stack(out_ids, dim=1)  # [B, max_len]
    preds = ["".join(itos[i.item()] for i in row) for row in outputs]
    return preds


def parse_args(args=None):
    p = argparse.ArgumentParser()

    p.add_argument("--seed", type=int, default=random.randint(0, 999_999_999_999_999))

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
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--amp", action="store_true", help="Enable mixed precision")

    # logging
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--no-tqdm", action="store_true", help="Disable tqdm progress bars")

    # image + patch
    p.add_argument("--height", type=int, default=100)
    p.add_argument("--width", type=int, default=250)
    p.add_argument("--n-channels", type=int, default=1)
    p.add_argument("--patch", type=int, default=25)

    # model size
    p.add_argument("--d-model", type=int, default=512)
    p.add_argument("--n-layers", type=int, default=12)
    p.add_argument("--n-heads", type=int, default=8)
    p.add_argument("--n-kv-heads", type=int, default=4)
    p.add_argument("--local-window", type=int, default=256)
    p.add_argument("--l2g", type=int, default=5)
    p.add_argument("--mlp-ratio", type=float, default=4.0)

    p.add_argument("--train-patience", type=int, default=float("inf"))
    p.add_argument("--val-split", type=float, default=0.1)
    p.add_argument("--val-patience", type=int, default=3)

    args = p.parse_args() if args is None else p.parse_args(args)
    args.no_val = args.val_split <= 0.0
    args.train_patience = int(min(args.epochs, args.train_patience))

    return args


if __name__ == "__main__":
    args = parse_args(
        # args=list(filter(lambda x: x != "", args.split())),
    )
    train(args)
