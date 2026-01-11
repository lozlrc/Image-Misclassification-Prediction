"""
Evaluate base model on CIFAR-10 test set (final evaluation for base classifier).

Outputs:
- prints accuracy
- saves confusion matrix plot to ./artifacts/figures/confusion_matrix.png
- saves reliability diagram + ECE to ./artifacts/figures/reliability_diagram.png

Requires:
- ./artifacts/base_model.pt (run train_base first)

Usage:
  python -m src.eval_base --seed 42
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from src.utils_data import CIFAR10_CLASSES, SplitConfig, get_dataloaders, get_device, set_global_seed
from src.utils_model import build_model, load_base_checkpoint, require_file
from src.utils_calibration import (
    compute_confusion_matrix,
    compute_ece,
    plot_confusion_matrix,
    plot_reliability_diagram,
    softmax_np,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate base CIFAR-10 classifier.")
    p.add_argument("--data-dir", type=str, default="./data")
    p.add_argument("--artifacts-dir", type=str, default="./artifacts")
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"])
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-bins", type=int, default=15, help="Bins for reliability diagram / ECE.")
    return p.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    set_global_seed(args.seed)
    device = get_device(args.device)

    artifacts_dir = Path(args.artifacts_dir)
    fig_dir = artifacts_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = require_file(artifacts_dir / "base_model.pt", hint="Run: python -m src.train_base")
    payload = torch.load(ckpt_path, map_location="cpu")
    extra = payload.get("extra", {})
    model_name = extra.get("model", "resnet18")
    pretrained = bool(extra.get("pretrained", False))

    # Data: enforce standard split config
    split_cfg = SplitConfig(
        data_dir=args.data_dir,
        seed=args.seed,
        n_train_base=40_000,
        n_train_meta=10_000,
        assert_sizes=True,
    )
    _, _, test_loader = get_dataloaders(
        cfg=split_cfg,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=False,
    )

    model = build_model(model_name, num_classes=10, pretrained=pretrained).to(device)
    load_base_checkpoint(model, ckpt_path, map_location=device)
    model.eval()

    all_logits = []
    all_y = []
    for x, y in test_loader:
        x = x.to(device, non_blocking=True)
        logits = model(x).detach().cpu().float()
        all_logits.append(logits)
        all_y.append(y.detach().cpu())

    logits = torch.cat(all_logits, dim=0).numpy()
    y_true = torch.cat(all_y, dim=0).numpy().astype(np.int64)

    probs = softmax_np(logits)
    y_pred = probs.argmax(axis=1)
    acc = float((y_pred == y_true).mean())
    print(f"[base] test accuracy: {acc:.4f}")

    # Confusion matrix
    cm = compute_confusion_matrix(y_true, y_pred, num_classes=10)
    plot_confusion_matrix(
        cm,
        class_names=list(CIFAR10_CLASSES),
        out_path=fig_dir / "confusion_matrix.png",
        title="CIFAR-10 Confusion Matrix (Base Model)",
        normalize=True,
    )

    # Calibration / ECE
    calib = compute_ece(probs, y_true, n_bins=args.n_bins)
    print(f"[base] test ECE (n_bins={args.n_bins}): {calib.ece:.4f}")
    plot_reliability_diagram(
        calib,
        out_path=fig_dir / "reliability_diagram.png",
        title="CIFAR-10 Reliability Diagram (Base Model)",
    )

    # Save also a tiny json-ish text summary for report convenience
    summary_path = artifacts_dir / "base_eval_summary.txt"
    summary_path.write_text(
        "\n".join(
            [
                f"test_accuracy={acc:.6f}",
                f"ece_nbins_{args.n_bins}={calib.ece:.6f}",
                f"model={model_name}",
                f"pretrained={pretrained}",
            ]
        )
        + "\n"
    )
    print(f"[saved] {summary_path}")


if __name__ == "__main__":
    main()