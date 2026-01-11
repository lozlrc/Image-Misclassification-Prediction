"""
Train failure predictor (meta-model) to predict whether base classifier will be wrong.

Default meta-model:
- LogisticRegression (scikit-learn)

Inputs (required):
- ./artifacts/meta_train.csv
- ./artifacts/meta_test.csv

Output:
- ./artifacts/meta_model.pkl
- ./artifacts/meta_eval_summary.txt (AUROC on test)

Usage:
  python -m src.train_meta
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


REQUIRED_FEATURES = ["max_prob", "entropy", "margin", "logit_gap"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train failure predictor (LogReg) on meta features.")
    p.add_argument("--artifacts-dir", type=str, default="./artifacts")
    p.add_argument("--C", type=float, default=1.0, help="Inverse regularization for LogisticRegression.")
    p.add_argument("--max-iter", type=int, default=2000)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def _require_file(path: Path, hint: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}\nHint: {hint}")
    return path


def _load_meta_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [c for c in (REQUIRED_FEATURES + ["y_fail"]) if c not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")
    return df


def main() -> None:
    args = parse_args()
    artifacts_dir = Path(args.artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    train_path = _require_file(artifacts_dir / "meta_train.csv", hint="Run: python -m src.make_meta_dataset")
    test_path = _require_file(artifacts_dir / "meta_test.csv", hint="Run: python -m src.make_meta_dataset")

    df_tr = _load_meta_csv(train_path)
    df_te = _load_meta_csv(test_path)

    X_tr = df_tr[REQUIRED_FEATURES].values.astype(np.float32)
    y_tr = df_tr["y_fail"].values.astype(np.int64)

    X_te = df_te[REQUIRED_FEATURES].values.astype(np.float32)
    y_te = df_te["y_fail"].values.astype(np.int64)

    # Standardize features; LogisticRegression tends to work better.
    model = Pipeline(
        steps=[
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            (
                "clf",
                LogisticRegression(
                    C=args.C,
                    max_iter=args.max_iter,
                    random_state=args.seed,
                    solver="lbfgs",
                    class_weight="balanced",
                ),
            ),
        ]
    )

    model.fit(X_tr, y_tr)

    # Prob of failure
    p_fail = model.predict_proba(X_te)[:, 1]
    auroc = float(roc_auc_score(y_te, p_fail))

    out_model = artifacts_dir / "meta_model.pkl"
    joblib.dump(
        {
            "model": model,
            "features": REQUIRED_FEATURES,
            "meta": {
                "C": args.C,
                "max_iter": args.max_iter,
                "seed": args.seed,
            },
        },
        out_model,
    )
    print(f"[saved] {out_model}")

    summary_path = artifacts_dir / "meta_eval_summary.txt"
    summary_path.write_text("\n".join([f"failure_auroc_test={auroc:.6f}"]) + "\n")
    print(f"[meta] test AUROC (failure predictor): {auroc:.4f}")
    print(f"[saved] {summary_path}")


if __name__ == "__main__":
    main()