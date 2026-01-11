"""
Train the base CIFAR-10 image classifier.

Requested changes:
- Default to 6 epochs
- Lower learning rate later using cosine schedule (CosineAnnealingLR)

Compatibility:
- Your src.utils_data.get_dataloaders(cfg) expects config fields like:
  cfg.n_train_base, cfg.n_train_meta, cfg.assert_sizes, etc.

Outputs:
- artifacts/base_model.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import src.utils_data as ud
from src.utils_model import build_model


def set_seed_fallback(seed: int) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def set_seed(seed: int) -> None:
    if hasattr(ud, "set_seed") and callable(getattr(ud, "set_seed")):
        ud.set_seed(seed)
    elif hasattr(ud, "seed_everything") and callable(getattr(ud, "seed_everything")):
        ud.seed_everything(seed)
    else:
        set_seed_fallback(seed)


class _Cfg:
    pass


def build_cfg(
    data_dir: str,
    batch_size: int,
    num_workers: int,
    n_train_base: int,
    n_train_meta: int,
    seed: int,
    quick: bool,
    device: torch.device,
) -> Any:
    """
    Build a cfg object that matches src/utils_data.py expectations.

    From your traceback, utils_data.py uses:
      - cfg.n_train_base
      - cfg.n_train_meta
      - cfg.assert_sizes
    And typically also uses:
      - cfg.data_dir
      - cfg.batch_size
      - cfg.num_workers
      - cfg.seed
      - cfg.quick
    """
    cfg = _Cfg()

    # Data location
    cfg.data_dir = data_dir
    cfg.root = data_dir

    # Loader params
    cfg.batch_size = int(batch_size)
    cfg.num_workers = int(num_workers)
    cfg.pin_memory = (device.type == "cuda")

    # Split sizes (names your utils_data.py uses)
    cfg.n_train_base = int(n_train_base)
    cfg.n_train_meta = int(n_train_meta)

    # Also set aliases in case other code expects them
    cfg.train_base_size = int(n_train_base)
    cfg.train_meta_size = int(n_train_meta)

    # Seed + quick flag
    cfg.seed = int(seed)
    cfg.quick = bool(quick)

    # Your utils_data.py checks this flag
    cfg.assert_sizes = True

    return cfg


def call_get_dataloaders(cfg: Any):
    if not hasattr(ud, "get_dataloaders"):
        raise ImportError("src.utils_data.get_dataloaders not found in your repo.")

    out = ud.get_dataloaders(cfg)
    if isinstance(out, (tuple, list)) and len(out) == 3:
        return out[0], out[1], out[2]

    raise TypeError(
        "utils_data.get_dataloaders(cfg) returned an unexpected value.\n"
        "Expected: (train_base_loader, train_meta_loader, test_loader)."
    )


def accuracy(logits: torch.Tensor, y: torch.Tensor) -> float:
    return float((logits.argmax(dim=1) == y).float().mean().item())


def train_one_epoch(model: nn.Module, loader, optimizer, device: torch.device) -> Tuple[float, float]:
    model.train()
    total_loss, total_acc, n = 0.0, 0.0, 0

    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(xb)
        loss = F.cross_entropy(logits, yb)
        loss.backward()
        optimizer.step()

        bs = xb.size(0)
        total_loss += float(loss.item()) * bs
        total_acc += accuracy(logits, yb) * bs
        n += bs

    return total_loss / max(n, 1), total_acc / max(n, 1)


@torch.no_grad()
def eval_on_loader(model: nn.Module, loader, device: torch.device) -> Tuple[float, float]:
    model.eval()
    total_loss, total_acc, n = 0.0, 0.0, 0

    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)

        logits = model(xb)
        loss = F.cross_entropy(logits, yb)

        bs = xb.size(0)
        total_loss += float(loss.item()) * bs
        total_acc += accuracy(logits, yb) * bs
        n += bs

    return total_loss / max(n, 1), total_acc / max(n, 1)


def main() -> None:
    p = argparse.ArgumentParser(description="Train base CIFAR-10 classifier")
    p.add_argument("--data-dir", type=str, default="data", help="CIFAR-10 download/cache dir")
    p.add_argument("--artifacts-dir", type=str, default="artifacts", help="Output artifacts dir")
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--model", type=str, default="resnet18")
    p.add_argument("--pretrained", action="store_true")
    p.add_argument("--epochs", type=int, default=6)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--weight-decay", type=float, default=5e-4)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--n-train-base", type=int, default=40000)
    p.add_argument("--n-train-meta", type=int, default=10000)
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()

    set_seed(args.seed)
    device = ud.get_device(args.device) if hasattr(ud, "get_device") else torch.device("cpu")
    print(f"[device] {device}")

    artifacts_dir = Path(args.artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    cfg = build_cfg(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        n_train_base=args.n_train_base,
        n_train_meta=args.n_train_meta,
        seed=args.seed,
        quick=args.quick,
        device=device,
    )

    train_base_loader, train_meta_loader, _test_loader = call_get_dataloaders(cfg)

    model = build_model(args.model, num_classes=10, pretrained=args.pretrained).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] {args.model} params={n_params:,}")

    optimizer = optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        nesterov=True,
    )

    epochs = args.epochs
    if args.quick:
        epochs = min(epochs, 3)
        print(f"[quick] enabled -> epochs={epochs}")

    # Cosine schedule: LR decreases later in training
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))

    for epoch in range(1, epochs + 1):
        lr_now = optimizer.param_groups[0]["lr"]
        print(f"\nEpoch {epoch}/{epochs}  lr={lr_now:.6f}")

        tr_loss, tr_acc = train_one_epoch(model, train_base_loader, optimizer, device)

        # sanity check only on Train_meta
        meta_loss, meta_acc = eval_on_loader(model, train_meta_loader, device)

        scheduler.step()

        print(f"  train_base: loss={tr_loss:.4f} acc={tr_acc:.4f}")
        print(f"  train_meta: loss={meta_loss:.4f} acc={meta_acc:.4f}  (sanity check only)")

    ckpt = {
        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "extra": {
            "model": args.model,
            "pretrained": bool(args.pretrained),
            "seed": args.seed,
            "epochs": epochs,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "momentum": args.momentum,
            "scheduler": "cosine",
            "n_train_base": int(args.n_train_base),
            "n_train_meta": int(args.n_train_meta),
        },
    }

    out_path = artifacts_dir / "base_model.pt"
    torch.save(ckpt, out_path)
    print(f"\n[saved] {out_path}")
    print("[next] Run: python3 -m src.make_meta_dataset --device mps")


if __name__ == "__main__":
    main()