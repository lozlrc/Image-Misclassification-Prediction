"""
Model utilities for misclass-risk.

CIFAR-correct ResNet18:
- Keep images at 32x32
- Modify ResNet18 stem:
  * conv1: 7x7/stride2 -> 3x3/stride1
  * maxpool -> Identity

Includes:
- Model builders (resnet18, small CNN fallback)
- Training / evaluation loops
- Checkpoint save/load helpers
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


def ensure_dir(p: str | Path) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


def default_artifacts_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "artifacts"


def default_base_model_path() -> Path:
    return default_artifacts_dir() / "base_model.pt"


def require_file(path: str | Path, hint: str) -> Path:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}\nHint: {hint}")
    return path


def build_resnet18_cifar(num_classes: int = 10, pretrained: bool = False) -> nn.Module:
    """
    ResNet18 adapted for CIFAR-10 (32x32).
    If pretrained=True, we load ImageNet weights for layers *except* the stem is replaced,
    so conv1 weights won't match; we simply re-init conv1. This is okay, but not guaranteed to help.
    """
    # Build base resnet18
    try:
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)
    except Exception:
        model = models.resnet18(pretrained=pretrained)

    # Replace stem for CIFAR
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    nn.init.kaiming_normal_(model.conv1.weight, mode="fan_out", nonlinearity="relu")
    model.maxpool = nn.Identity()

    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


class SmallCNN(nn.Module):
    """Small CNN fallback for CIFAR-10 (expects 32x32)."""
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),  # 16x16
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),  # 8x8
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.flatten(1)
        return self.classifier(x)


def build_model(name: str = "resnet18", num_classes: int = 10, pretrained: bool = False) -> nn.Module:
    name = (name or "resnet18").lower()
    if name == "resnet18":
        return build_resnet18_cifar(num_classes=num_classes, pretrained=pretrained)
    if name in {"cnn", "smallcnn"}:
        return SmallCNN(num_classes=num_classes)
    raise ValueError(f"Unknown model '{name}'. Use 'resnet18' or 'cnn'.")


@dataclass
class TrainConfig:
    epochs: int = 20
    lr: float = 3e-4
    weight_decay: float = 1e-4
    device: str = "auto"
    seed: int = 42
    log_every: int = 100
    quick: bool = False


@torch.no_grad()
def accuracy_from_logits(logits: torch.Tensor, y: torch.Tensor) -> float:
    pred = logits.argmax(dim=1)
    return (pred == y).float().mean().item()


def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    log_every: int = 100,
) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_n = 0

    for step, (x, y) in enumerate(loader):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        logits = model(x)
        loss = F.cross_entropy(logits, y)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        bs = y.size(0)
        total_loss += loss.item() * bs
        total_correct += (logits.argmax(1) == y).sum().item()
        total_n += bs

        if log_every > 0 and (step + 1) % log_every == 0:
            avg_loss = total_loss / max(total_n, 1)
            avg_acc = total_correct / max(total_n, 1)
            print(f"  step {step+1:04d}/{len(loader)}  loss={avg_loss:.4f}  acc={avg_acc:.4f}")

    return {"loss": total_loss / max(total_n, 1), "acc": total_correct / max(total_n, 1)}


@torch.no_grad()
def eval_one_epoch(model: nn.Module, loader: torch.utils.data.DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_n = 0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        logits = model(x)
        loss = F.cross_entropy(logits, y)

        bs = y.size(0)
        total_loss += loss.item() * bs
        total_correct += (logits.argmax(1) == y).sum().item()
        total_n += bs

    return {"loss": total_loss / max(total_n, 1), "acc": total_correct / max(total_n, 1)}


@torch.no_grad()
def predict_logits(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_logits = []
    all_y = []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        logits = model(x).detach().cpu().float()
        all_logits.append(logits)
        all_y.append(y.detach().cpu())
    logits_np = torch.cat(all_logits, dim=0).numpy().astype(np.float32)
    y_np = torch.cat(all_y, dim=0).numpy().astype(np.int64)
    return logits_np, y_np


def save_base_checkpoint(model: nn.Module, path: str | Path, extra: Optional[Dict] = None) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    payload = {"state_dict": model.state_dict(), "extra": extra or {}}
    torch.save(payload, path)
    print(f"[saved] base model -> {path}")


def load_base_checkpoint(
    model: nn.Module, path: str | Path, map_location: Optional[str | torch.device] = "cpu"
) -> Dict:
    path = require_file(path, hint="Run: python -m src.train_base")
    payload = torch.load(path, map_location=map_location)
    if "state_dict" not in payload:
        raise RuntimeError(f"Checkpoint at {path} missing 'state_dict'.")
    model.load_state_dict(payload["state_dict"])
    return payload.get("extra", {})


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def pretty_device(d: torch.device) -> str:
    return str(d).replace("cuda", "cuda:0")