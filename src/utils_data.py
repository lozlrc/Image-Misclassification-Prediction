"""
Data utilities for CIFAR-10 project.

Key requirements:
- Must ONLY use torchvision.datasets.CIFAR10(download=True).
- Auto-download into ./data/ when run.
- Deterministic split of CIFAR-10 train set into Train_base (e.g., 40k) and Train_meta (e.g., 10k).
- Provide DataLoaders and class names.

CIFAR-correct pipeline:
- Keep images at 32x32 (NO upscaling)
- Use CIFAR-10 normalization
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


CIFAR10_CLASSES: List[str] = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]

# Standard CIFAR-10 normalization
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)


@dataclass(frozen=True)
class SplitConfig:
    """Configuration for deterministic CIFAR-10 splits."""
    data_dir: str = "./data"
    seed: int = 42
    n_train_base: int = 40_000  # out of 50,000
    n_train_meta: int = 10_000  # out of 50,000
    assert_sizes: bool = True


def set_global_seed(seed: int) -> None:
    """Best-effort deterministic seeding across python, numpy, and torch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # safe even if no cuda
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(prefer: str = "auto") -> torch.device:
    """
    Select device.
    prefer: "auto" | "cpu" | "cuda" | "mps"
    """
    prefer = (prefer or "auto").lower()
    if prefer == "cpu":
        return torch.device("cpu")
    if prefer == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        raise RuntimeError("Requested device=cuda but CUDA is not available.")
    if prefer == "mps":
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return torch.device("mps")
        raise RuntimeError("Requested device=mps but MPS is not available.")
    # auto
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _cifar10_transforms(train: bool) -> transforms.Compose:
    """
    CIFAR-10 transforms for 32x32 inputs.
    """
    normalize = transforms.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD)

    if train:
        return transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]
        )
    return transforms.Compose(
        [
            transforms.ToTensor(),
            normalize,
        ]
    )


def get_cifar10_datasets(data_dir: str = "./data") -> Tuple[datasets.CIFAR10, datasets.CIFAR10]:
    """
    Returns:
      train_full: CIFAR-10 train split (50k)
      test: CIFAR-10 test split (10k)

    Both auto-download to data_dir.
    """
    os.makedirs(data_dir, exist_ok=True)

    train_full = datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=True,
        transform=_cifar10_transforms(train=True),
    )
    test = datasets.CIFAR10(
        root=data_dir,
        train=False,
        download=True,
        transform=_cifar10_transforms(train=False),
    )
    return train_full, test


def make_train_splits(train_full: datasets.CIFAR10, cfg: SplitConfig) -> Tuple[Subset, Subset]:
    """
    Deterministically split CIFAR-10 train set into:
      - Train_base (cfg.n_train_base)
      - Train_meta (cfg.n_train_meta)
    """
    n_total = len(train_full)
    n_base = int(cfg.n_train_base)
    n_meta = int(cfg.n_train_meta)

    if cfg.assert_sizes and (n_base + n_meta != n_total):
        raise ValueError(
            f"Split sizes must sum to {n_total}. Got n_train_base={n_base}, n_train_meta={n_meta}."
        )

    g = torch.Generator()
    g.manual_seed(cfg.seed)

    perm = torch.randperm(n_total, generator=g).tolist()
    base_idx = perm[:n_base]
    meta_idx = perm[n_base : n_base + n_meta]

    return Subset(train_full, base_idx), Subset(train_full, meta_idx)


def get_dataloaders(
    cfg: SplitConfig,
    batch_size: int = 128,
    num_workers: int = 2,
    pin_memory: bool = True,
    persistent_workers: bool = False,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Convenience:
    - Loads CIFAR-10 (auto-download)
    - Creates Train_base, Train_meta, and Test loaders
    """
    set_global_seed(cfg.seed)

    train_full, test_ds = get_cifar10_datasets(cfg.data_dir)
    train_base_ds, train_meta_ds = make_train_splits(train_full, cfg)

    pin = bool(pin_memory and torch.cuda.is_available())

    train_base_loader = DataLoader(
        train_base_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin,
        persistent_workers=persistent_workers and num_workers > 0,
    )
    train_meta_loader = DataLoader(
        train_meta_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin,
        persistent_workers=persistent_workers and num_workers > 0,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin,
        persistent_workers=persistent_workers and num_workers > 0,
    )

    return train_base_loader, train_meta_loader, test_loader


def get_class_names() -> List[str]:
    return list(CIFAR10_CLASSES)


def get_raw_test_dataset(data_dir: str = "./data") -> datasets.CIFAR10:
    """
    Raw test dataset for visualization: NO normalization, NO resize (32x32).
    """
    os.makedirs(data_dir, exist_ok=True)
    return datasets.CIFAR10(
        root=data_dir,
        train=False,
        download=True,
        transform=transforms.ToTensor(),
    )