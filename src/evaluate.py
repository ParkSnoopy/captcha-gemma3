import torch
from torchvision.transforms.functional import to_tensor

import argparse
from PIL import Image

import os
from pathlib import Path

from model import Gemma3Model, PatchEmbed
from train import predict_batch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluation(cli_args):
    # --- load checkpoint ---
    ckpt_path = cli_args.use_checkpoint  # change to your best .pt
    ckpt = torch.load(ckpt_path, map_location=device)

    args = ckpt["args"]  # hyperparams used for training
    itos, stoi = ckpt["itos"], ckpt["stoi"]

    # --- rebuild model & patcher and load weights ---
    model = (
        Gemma3Model(
            vocab_size=len(itos),
            d_model=args["d_model"],
            n_layers=args["n_layers"],
            n_heads=args["n_heads"],
            n_kv_heads=args["n_kv_heads"],
            max_seq_len=max(
                1024,
                (args["height"] // args["patch"]) * (args["width"] // args["patch"])
                + 16,
            ),
            local_window=args["local_window"],
            l2g=args["l2g"],
            mlp_ratio=args["mlp_ratio"],
            vision_enabled=True,
            vision_tokens=(args["height"] // args["patch"])
            * (args["width"] // args["patch"]),
        )
        .to(device)
        .eval()
    )
    model.load_state_dict(ckpt["model"], strict=True)

    patcher = (
        PatchEmbed(
            img_size=(args["height"], args["width"]),
            patch=args["patch"],
            d_model=args["d_model"],
        )
        .to(device)
        .eval()
    )
    patcher.load_state_dict(ckpt["patcher"])

    # --- prepare images ---
    def load_img(path, H=args["height"], W=args["width"]):
        img = Image.open(path).convert("L").resize((W, H))
        return to_tensor(img)  # [1,H,W], float32 in [0,1]

    image_paths = Path(cli_args.eval_target)
    image_paths = (
        list(image_paths.glob("*.png"))
        + list(image_paths.glob("*.jpg"))
        + list(image_paths.glob("*.jpeg"))
    )
    batch = torch.stack([load_img(p) for p in image_paths]).to(device)  # [B,1,H,W]

    # --- predict ---
    preds = predict_batch(batch, model, patcher, stoi, itos, max_len=4, device=device)
    for p, y in zip(image_paths, preds):
        print(os.path.basename(p), "->", y)


def parse_args(args=None):
    p = argparse.ArgumentParser()

    p.add_argument(
        "--use-checkpoint",
        type=str,
        required=True,
    )
    p.add_argument(
        "--eval-target",
        type=str,
        required=True,
    )

    return p.parse_args() if args is None else p.parse_args(args)


if __name__ == "__main__":
    args = """
        --use-checkpoint ./checkpoints/captcha_gemma3_e5.pt
        --eval-target ./data.eval/
    """
    args = parse_args(
        # args=list(filter(lambda x: x != "", args.split())),
    )
    evaluation(args)
