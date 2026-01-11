"""
Calibration utilities:
- ECE (Expected Calibration Error)
- Reliability diagram (calibration curve) plotting

Allowed deps: numpy, matplotlib
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import matplotlib.pyplot as plt


@dataclass(frozen=True)
class CalibrationResult:
    ece: float
    bin_edges: np.ndarray
    bin_acc: np.ndarray
    bin_conf: np.ndarray
    bin_count: np.ndarray


def softmax_np(logits: np.ndarray) -> np.ndarray:
    """Numerically stable softmax."""
    z = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(z)
    return exp / (exp.sum(axis=1, keepdims=True) + 1e-12)


def compute_ece(
    probs: np.ndarray,
    y_true: np.ndarray,
    n_bins: int = 15,
) -> CalibrationResult:
    """
    Compute ECE using max-prob confidence and correctness.

    Args:
      probs: (N, C) probabilities
      y_true: (N,) true labels
      n_bins: number of bins in [0,1]

    Returns:
      CalibrationResult with per-bin accuracy/confidence and ECE.
    """
    if probs.ndim != 2:
        raise ValueError(f"probs must be 2D (N,C). Got shape={probs.shape}")
    if y_true.ndim != 1:
        raise ValueError(f"y_true must be 1D (N,). Got shape={y_true.shape}")
    if probs.shape[0] != y_true.shape[0]:
        raise ValueError("probs and y_true must have same N.")

    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == y_true).astype(np.float32)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_acc = np.zeros(n_bins, dtype=np.float32)
    bin_conf = np.zeros(n_bins, dtype=np.float32)
    bin_count = np.zeros(n_bins, dtype=np.int64)

    ece = 0.0
    N = len(y_true)

    # bins: (bin_edges[i], bin_edges[i+1]] except first includes 0.0
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        if i == 0:
            mask = (conf >= lo) & (conf <= hi)
        else:
            mask = (conf > lo) & (conf <= hi)

        cnt = int(mask.sum())
        bin_count[i] = cnt
        if cnt == 0:
            continue

        acc_i = float(correct[mask].mean())
        conf_i = float(conf[mask].mean())
        bin_acc[i] = acc_i
        bin_conf[i] = conf_i

        ece += (cnt / N) * abs(acc_i - conf_i)

    return CalibrationResult(
        ece=float(ece),
        bin_edges=bin_edges,
        bin_acc=bin_acc,
        bin_conf=bin_conf,
        bin_count=bin_count,
    )


def plot_reliability_diagram(
    calib: CalibrationResult,
    out_path: str | Path,
    title: str = "Reliability diagram",
) -> None:
    """
    Save a reliability diagram plot.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_bins = len(calib.bin_acc)
    # Use bin centers for plotting points/bars
    edges = calib.bin_edges
    centers = (edges[:-1] + edges[1:]) / 2.0
    width = (edges[1] - edges[0]) * 0.9

    plt.figure()
    # Perfect calibration line
    plt.plot([0, 1], [0, 1])
    # Bar: accuracy per bin (y) vs confidence (x centers)
    plt.bar(centers, calib.bin_acc, width=width, alpha=0.8, edgecolor="black", linewidth=0.5, label="Accuracy")
    # Overlay: average confidence
    plt.plot(centers, calib.bin_conf, marker="o", linestyle="--", label="Avg confidence")

    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel("Confidence (max softmax probability)")
    plt.ylabel("Accuracy")
    plt.title(f"{title}\nECE={calib.ece:.4f}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"[saved] {out_path}")


def compute_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    """
    Compute confusion matrix counts: rows=true, cols=pred.
    """
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true.astype(int), y_pred.astype(int)):
        cm[t, p] += 1
    return cm


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: list[str],
    out_path: str | Path,
    title: str = "Confusion matrix",
    normalize: bool = True,
) -> None:
    """
    Save confusion matrix plot. If normalize=True, normalize rows to proportions.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    plot_cm = cm.astype(np.float32)
    if normalize:
        row_sums = plot_cm.sum(axis=1, keepdims=True) + 1e-12
        plot_cm = plot_cm / row_sums

    plt.figure(figsize=(8, 7))
    plt.imshow(plot_cm, interpolation="nearest")
    plt.title(title + (" (normalized)" if normalize else ""))
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45, ha="right")
    plt.yticks(tick_marks, class_names)

    # Light annotation (avoid huge text)
    if len(class_names) <= 15:
        for i in range(plot_cm.shape[0]):
            for j in range(plot_cm.shape[1]):
                val = plot_cm[i, j]
                txt = f"{val:.2f}" if normalize else str(int(val))
                plt.text(j, i, txt, ha="center", va="center", fontsize=7)

    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"[saved] {out_path}")