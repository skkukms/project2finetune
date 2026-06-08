"""FID evaluation across all refiner_1024 checkpoints.

Usage:
    python eval_fid.py \
        --g256-ckpt  /content/drive/MyDrive/project2claude/runs/finetune_256/final.pt \
        --r512-ckpt  /content/drive/MyDrive/project2claude/runs/refiner_512/ckpt_000050000.pt \
        --r1024-dir  /content/drive/MyDrive/project2claude/runs/refiner_1024 \
        --valid-zip  /content/drive/MyDrive/project2claude/valid_10k_1024.zip \
        --n-gen      10000 \
        --out        fid_results.txt
"""
from __future__ import annotations

import argparse
import io
import os
import shutil
import tempfile
import zipfile
from pathlib import Path

import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from src.model import build_baseline_256_generator
from src.refiner import Refiner512, Refiner512Config, Refiner1024, Refiner1024Config


# ---------------------------------------------------------------------------
# Model loading (reuse from export_onnx.py)
# ---------------------------------------------------------------------------

def load_g256(path: Path, device: str):
    G = build_baseline_256_generator().to(device).eval()
    state = torch.load(path, map_location=device, weights_only=False)
    G.load_state_dict(state["G_ema_state"])
    for p in G.parameters():
        p.requires_grad_(False)
    return G


def load_r512(path: Path, device: str):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg  = Refiner512Config(**ckpt["meta"]["refiner_config"])
    r    = Refiner512(cfg).to(device).eval()
    r.load_state_dict(ckpt["refiner_ema_state"])
    for p in r.parameters():
        p.requires_grad_(False)
    return r


def load_r1024(path: Path, device: str):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg  = Refiner1024Config(**ckpt["meta"]["refiner_config"])
    r    = Refiner1024(cfg).to(device).eval()
    r.load_state_dict(ckpt["refiner_ema_state"])
    for p in r.parameters():
        p.requires_grad_(False)
    return r


# ---------------------------------------------------------------------------
# Image generation
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate_images(G256, r512, r1024, n: int, batch: int, device: str, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    generated = 0
    while generated < n:
        b = min(batch, n - generated)
        z = torch.randn(b, 512, device=device)
        imgs = r1024(r512(G256(z)))           # [-1, 1]
        imgs = ((imgs + 1) / 2).clamp(0, 1)  # [0, 1]
        for i, img in enumerate(imgs):
            pil = TF.to_pil_image(img.cpu())
            pil.save(out_dir / f"{generated + i:06d}.png")
        generated += b
        if generated % 1000 == 0:
            print(f"  Generated {generated}/{n}", flush=True)


# ---------------------------------------------------------------------------
# Real image extraction
# ---------------------------------------------------------------------------

def extract_real_images(zip_path: Path, out_dir: Path, n: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        names = sorted(
            nm for nm in zf.namelist()
            if nm.lower().endswith((".png", ".jpg", ".jpeg"))
        )[:n]
        for i, nm in enumerate(names):
            with zf.open(nm) as f:
                img = Image.open(io.BytesIO(f.read())).convert("RGB")
            img.save(out_dir / f"{i:06d}.png")
            if (i + 1) % 1000 == 0:
                print(f"  Extracted {i+1}/{len(names)}", flush=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--g256-ckpt",  required=True, type=Path)
    parser.add_argument("--r512-ckpt",  required=True, type=Path)
    parser.add_argument("--r1024-dir",  required=True, type=Path,
                        help="Directory containing refiner_1024 checkpoints")
    parser.add_argument("--valid-zip",  required=True, type=Path)
    parser.add_argument("--n-gen",      type=int, default=10000,
                        help="Number of images to generate per checkpoint")
    parser.add_argument("--batch",      type=int, default=8)
    parser.add_argument("--out",        type=Path, default=Path("fid_results.txt"))
    parser.add_argument("--ckpts",      nargs="*", type=Path, default=None,
                        help="Specific checkpoint files to evaluate (default: all in r1024-dir)")
    args = parser.parse_args()

    try:
        import torch_fidelity
    except ImportError:
        raise SystemExit("Install torch-fidelity first:  pip install torch-fidelity")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Gather checkpoints
    if args.ckpts:
        ckpt_paths = sorted(args.ckpts)
    else:
        ckpt_paths = sorted(args.r1024_dir.glob("ckpt_*.pt"))
    if not ckpt_paths:
        raise SystemExit(f"No checkpoints found in {args.r1024_dir}")
    print(f"Evaluating {len(ckpt_paths)} checkpoints: {[p.name for p in ckpt_paths]}")

    # Load frozen models
    print("Loading G_256 and Refiner512 ...")
    G256 = load_g256(args.g256_ckpt, device)
    r512 = load_r512(args.r512_ckpt, device)

    # Extract real images once
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path  = Path(tmp)
        real_dir  = tmp_path / "real"
        fake_dir  = tmp_path / "fake"

        print(f"Extracting {args.n_gen} real images from {args.valid_zip.name} ...")
        extract_real_images(args.valid_zip, real_dir, args.n_gen)

        results: list[tuple[str, float]] = []

        for ckpt_path in ckpt_paths:
            print(f"\n[{ckpt_path.name}] Generating {args.n_gen} images ...")
            r1024 = load_r1024(ckpt_path, device)

            # Clean fake dir
            if fake_dir.exists():
                shutil.rmtree(fake_dir)

            generate_images(G256, r512, r1024, args.n_gen, args.batch, device, fake_dir)

            print(f"[{ckpt_path.name}] Computing FID ...")
            metrics = torch_fidelity.calculate_metrics(
                input1=str(real_dir),
                input2=str(fake_dir),
                cuda=(device == "cuda"),
                fid=True,
                verbose=False,
                cache=True,                       # cache real features across checkpoints
                cache_root=str(tmp_path / "cache"),
                input1_cache_name="real_valid",   # real features computed only once
            )
            fid = metrics["frechet_inception_distance"]
            print(f"  FID = {fid:.4f}")
            results.append((ckpt_path.name, fid))

            del r1024

        # Sort by FID (lower is better)
        results.sort(key=lambda x: x[1])

        print("\n========== FID Results (lower is better) ==========")
        for name, fid in results:
            print(f"  {name}: {fid:.4f}")
        print(f"\nBest: {results[0][0]}  (FID={results[0][1]:.4f})")

        # Save to file
        with open(args.out, "w") as f:
            f.write("checkpoint\tFID\n")
            for name, fid in results:
                f.write(f"{name}\t{fid:.4f}\n")
            f.write(f"\nBest: {results[0][0]}  FID={results[0][1]:.4f}\n")
        print(f"Results saved to {args.out}")


if __name__ == "__main__":
    main()
