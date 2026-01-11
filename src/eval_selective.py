"""
Evaluate selective classification using the failure predictor.

Inputs (required):
- ./artifacts/meta_test.csv
- ./artifacts/meta_model.pkl

Outputs:
- ./artifacts/figures/risk_coverage_curve.png
- ./artifacts/figures/selective_accuracy_curve.png
- ./artifacts/selective_eval_summary.txt

Metrics:
- Failure predictor AUROC on test
- Risk–coverage curve (risk = error rate among non-abstained)
- Selective accuracy vs coverage

Usage:
  python -m src.eval_selective
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate selective prediction (abstain based on failure risk).")
    p.add_argument("--artifacts-dir", type=str, default="./artifacts")
    p.add_argument("--n-points", type=int, default=101, help="Number of thresholds for curves.")
    return p.parse_args()


def _require_file(path: Path, hint: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}\nHint: {hint}")
    return path


def risk_coverage_curves(
    y_fail: np.ndarray,
    y_correct: np.ndarray,
    p_fail: np.ndarray,
    n_points: int = 101,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute risk (error rate on selected) and selective accuracy vs coverage by thresholding p_fail.

    We abstain when p_fail > threshold; predict otherwise.

    Returns:
      coverage: fraction selected (non-abstained)
      risk: error rate among selected = mean(y_fail | selected)
      sel_acc: accuracy among selected = mean(y_correct | selected)
    """
    thresholds = np.linspace(0.0, 1.0, n_points)
    coverage = np.zeros_like(thresholds, dtype=np.float32)
    risk = np.zeros_like(thresholds, dtype=np.float32)
    sel_acc = np.zeros_like(thresholds, dtype=np.float32)

    for i, t in enumerate(thresholds):
        selected = p_fail <= t
        cov = float(selected.mean())
        coverage[i] = cov
        if selected.sum() == 0:
            risk[i] = np.nan
            sel_acc[i] = np.nan
        else:
            risk[i] = float(y_fail[selected].mean())
            sel_acc[i] = float(y_correct[selected].mean())
    return coverage, risk, sel_acc


def save_curve_plot(x: np.ndarray, y: np.ndarray, xlabel: str, ylabel: str, title: str, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure()
    plt.plot(x, y)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"[saved] {out_path}")


def main() -> None:
    args = parse_args()
    artifacts_dir = Path(args.artifacts_dir)
    fig_dir = artifacts_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    meta_test_path = _require_file(artifacts_dir / "meta_test.csv", hint="Run: python -m src.make_meta_dataset")
    meta_model_path = _require_file(artifacts_dir / "meta_model.pkl", hint="Run: python -m src.train_meta")

    df = pd.read_csv(meta_test_path)
    bundle = joblib.load(meta_model_path)
    model = bundle["model"]
    features = bundle["features"]

    if any(f not in df.columns for f in features):
        missing = [f for f in features if f not in df.columns]
        raise ValueError(f"meta_test.csv missing features required by meta_model: {missing}")

    X = df[features].values.astype(np.float32)
    y_fail = df["y_fail"].values.astype(np.int64)

    # correctness label derived from y_fail
    y_correct = (1 - y_fail).astype(np.int64)

    p_fail = model.predict_proba(X)[:, 1]
    auroc = float(roc_auc_score(y_fail, p_fail))
    print(f"[selective] failure predictor AUROC (test): {auroc:.4f}")

    coverage, risk, sel_acc = risk_coverage_curves(y_fail=y_fail, y_correct=y_correct, p_fail=p_fail, n_points=args.n_points)

    # Save plots
    save_curve_plot(
        coverage,
        risk,
        xlabel="Coverage (fraction predicted)",
        ylabel="Risk (error rate among predicted)",
        title="Risk–Coverage Curve (Selective Classification)",
        out_path=fig_dir / "risk_coverage_curve.png",
    )
    save_curve_plot(
        coverage,
        sel_acc,
        xlabel="Coverage (fraction predicted)",
        ylabel="Selective Accuracy (accuracy among predicted)",
        title="Selective Accuracy vs Coverage",
        out_path=fig_dir / "selective_accuracy_curve.png",
    )

    # Summary at a few coverages
    def interp_at(c: float) -> Tuple[float, float]:
        # nearest neighbor on coverage grid
        j = int(np.nanargmin(np.abs(coverage - c)))
        return float(risk[j]), float(sel_acc[j])

    r50, a50 = interp_at(0.50)
    r80, a80 = interp_at(0.80)
    r95, a95 = interp_at(0.95)

    summary = "\n".join(
        [
            f"failure_auroc_test={auroc:.6f}",
            f"risk_at_coverage_0.50={r50:.6f}",
            f"sel_acc_at_coverage_0.50={a50:.6f}",
            f"risk_at_coverage_0.80={r80:.6f}",
            f"sel_acc_at_coverage_0.80={a80:.6f}",
            f"risk_at_coverage_0.95={r95:.6f}",
            f"sel_acc_at_coverage_0.95={a95:.6f}",
        ]
    ) + "\n"

    out_summary = artifacts_dir / "selective_eval_summary.txt"
    out_summary.write_text(summary)
    print(f"[saved] {out_summary}")


if __name__ == "__main__":
    main()