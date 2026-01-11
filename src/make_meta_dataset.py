"""
Generate meta-datasets from the base model outputs (no leakage).

Leakage prevention (must):
- Base model is trained ONLY on Train_base
- Meta features for training the failure predictor come ONLY from Train_meta predictions
- CIFAR-10 test set is used ONLY for final evaluation

Outputs (saved to ./artifacts):
- meta_train.csv  (from Train_meta split)
- meta_test.csv   (from official CIFAR-10 test set)

Required meta-features:
- max_prob   = max softmax probability
- entropy    = -sum(p * log p)
- margin     = p_top1 - p_top2
- logit_gap  = z_top1 - z_top2 (logits)

Failure label:
- y_fail = 1 if y_pred != y_true else 0

Usage:
  python -m src.make_meta_dataset --seed 42
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset

from src.utils_data import SplitConfig, get_dataloaders, get_device, set_global_seed
from src.utils_model import build_model, load_base_checkpoint, require_file


class WithIndex(Dataset):
    """Wrap a dataset to also return a stable integer index for each item."""
    def __init__(self, ds: Dataset):
        self.ds = ds

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, i: int):
        x, y = self.ds[i]
        return x, y, i


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Create meta_train.csv and meta_test.csv from base model outputs.")
    p.add_argument("--data-dir", type=str, default="./data")
    p.add_argument("--artifacts-dir", type=str, default="./artifacts")
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"])
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def _compute_meta_features(logits: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Compute required meta-features from logits.
    logits: (N, C)
    Returns dict of feature arrays (N,)
    """
    # softmax probs (stable)
    z = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(z)
    probs = exp / (exp.sum(axis=1, keepdims=True) + 1e-12)

    # top1/top2 probs and logits
    top2_idx = np.argsort(-probs, axis=1)[:, :2]
    p_top1 = probs[np.arange(len(probs)), top2_idx[:, 0]]
    p_top2 = probs[np.arange(len(probs)), top2_idx[:, 1]]

    # logits top1/top2 aligned to prob ranking (equivalent to logits ranking)
    z_top1 = logits[np.arange(len(logits)), top2_idx[:, 0]]
    z_top2 = logits[np.arange(len(logits)), top2_idx[:, 1]]

    max_prob = p_top1
    margin = p_top1 - p_top2
    logit_gap = z_top1 - z_top2

    # entropy: -sum(p log p)
    entropy = -np.sum(probs * np.log(probs + 1e-12), axis=1)

    return {
        "max_prob": max_prob.astype(np.float32),
        "entropy": entropy.astype(np.float32),
        "margin": margin.astype(np.float32),
        "logit_gap": logit_gap.astype(np.float32),
        # include full probs optionally later; keep minimal now
    }


@torch.no_grad()
def _run_inference(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
      logits: (N,C)
      y_true: (N,)
      idx:    (N,) stable within-loader index (0..N-1)
    """
    model.eval()
    all_logits: List[torch.Tensor] = []
    all_y: List[torch.Tensor] = []
    all_idx: List[torch.Tensor] = []

    for x, y, idx in loader:
        x = x.to(device, non_blocking=True)
        logits = model(x).detach().cpu().float()
        all_logits.append(logits)
        all_y.append(y.detach().cpu())
        all_idx.append(idx.detach().cpu())

    logits_np = torch.cat(all_logits, dim=0).numpy().astype(np.float32)
    y_np = torch.cat(all_y, dim=0).numpy().astype(np.int64)
    idx_np = torch.cat(all_idx, dim=0).numpy().astype(np.int64)
    return logits_np, y_np, idx_np


def _build_dataframe(
    split_name: str,
    logits: np.ndarray,
    y_true: np.ndarray,
    idx: np.ndarray,
) -> pd.DataFrame:
    # predicted class from logits
    y_pred = logits.argmax(axis=1).astype(np.int64)
    y_fail = (y_pred != y_true).astype(np.int64)

    feats = _compute_meta_features(logits)

    df = pd.DataFrame(
        {
            "split": split_name,
            "idx": idx.astype(np.int64),        # stable index within this CSV split
            "y_true": y_true.astype(np.int64),
            "y_pred": y_pred.astype(np.int64),
            "y_fail": y_fail.astype(np.int64),
            **{k: v for k, v in feats.items()},
        }
    )
    return df


def main() -> None:
    args = parse_args()
    set_global_seed(args.seed)
    device = get_device(args.device)

    artifacts_dir = Path(args.artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = require_file(artifacts_dir / "base_model.pt", hint="Run: python -m src.train_base")
    payload = torch.load(ckpt_path, map_location="cpu")
    extra = payload.get("extra", {})
    model_name = extra.get("model", "resnet18")
    pretrained = bool(extra.get("pretrained", False))

    # Data loaders
    split_cfg = SplitConfig(
        data_dir=args.data_dir,
        seed=args.seed,
        n_train_base=40_000,
        n_train_meta=10_000,
        assert_sizes=True,
    )
    _, train_meta_loader, test_loader = get_dataloaders(
        cfg=split_cfg,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=False,
    )

    # Wrap datasets to include stable indices (0..N-1 in each split)
    train_meta_loader = DataLoader(
        WithIndex(train_meta_loader.dataset),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        WithIndex(test_loader.dataset),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    # Load base model
    model = build_model(model_name, num_classes=10, pretrained=pretrained).to(device)
    load_base_checkpoint(model, ckpt_path, map_location=device)

    # Inference -> meta features
    print("[meta] running base model on Train_meta ...")
    logits_tr, y_tr, idx_tr = _run_inference(model, train_meta_loader, device)
    df_tr = _build_dataframe("train_meta", logits_tr, y_tr, idx_tr)

    print("[meta] running base model on Test ...")
    logits_te, y_te, idx_te = _run_inference(model, test_loader, device)
    df_te = _build_dataframe("test", logits_te, y_te, idx_te)

    # Save
    out_train = artifacts_dir / "meta_train.csv"
    out_test = artifacts_dir / "meta_test.csv"
    df_tr.to_csv(out_train, index=False)
    df_te.to_csv(out_test, index=False)

    # Quick stats
    tr_fail_rate = float(df_tr["y_fail"].mean())
    te_fail_rate = float(df_te["y_fail"].mean())
    print(f"[saved] {out_train} (n={len(df_tr)}, fail_rate={tr_fail_rate:.3f})")
    print(f"[saved] {out_test}  (n={len(df_te)}, fail_rate={te_fail_rate:.3f})")
    print("[next] Train meta-model with: python -m src.train_meta")


if __name__ == "__main__":
    main()