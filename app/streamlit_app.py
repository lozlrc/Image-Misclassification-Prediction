"""
Streamlit demo: Image Misclassification Prediction (CIFAR-10)

Loads:
- ./artifacts/base_model.pt
- ./artifacts/meta_model.pkl

Shows:
- CIFAR-10 test image by index (32x32)
- base prediction + top-3 probabilities
- predicted failure risk (meta-model)
- ABSTAIN/PREDICT based on threshold control

UI:
- Wide layout with clean columns
- Softmax formula shown on UI
- Threshold can be set with slider or manual entry (synced)
- Meta-feature explanations focus on entropy and margin only
- "How good is abstaining at catching mistakes?" uses a slider (0..100 step 1)
  and reports abstention precision:
    (# wrong AND abstained) / (# abstained)
  plus the total number abstained on the test set at that threshold.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms

from src.utils_data import CIFAR10_CLASSES, CIFAR10_MEAN, CIFAR10_STD, get_device

REQUIRED_META_FEATURES = ["max_prob", "entropy", "margin", "logit_gap"]

FEATURE_EXPLAIN_SIMPLE = {
    "entropy": "Entropy measures uncertainty. Higher entropy means the model is less sure.",
    "margin": "Margin is the gap between the top two probabilities. Smaller margin means it was a close call.",
}


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def artifacts_dir() -> Path:
    return repo_root() / "artifacts"


def require_file(path: Path, hint: str) -> Path:
    if not path.exists():
        st.error(f"Missing required file: {path}\n\nHint: {hint}")
        st.stop()
    return path


def cifar10_test_raw(data_dir: Path) -> datasets.CIFAR10:
    """For display: no normalization, 32x32."""
    data_dir.mkdir(parents=True, exist_ok=True)
    return datasets.CIFAR10(
        root=str(data_dir),
        train=False,
        download=True,
        transform=transforms.ToTensor(),
    )


def cifar10_test_model(data_dir: Path) -> datasets.CIFAR10:
    """For inference: CIFAR normalization, 32x32."""
    data_dir.mkdir(parents=True, exist_ok=True)
    normalize = transforms.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD)
    return datasets.CIFAR10(
        root=str(data_dir),
        train=False,
        download=True,
        transform=transforms.Compose([transforms.ToTensor(), normalize]),
    )


def compute_meta_features_from_logits(logits: np.ndarray) -> np.ndarray:
    """Return [max_prob, entropy, margin, logit_gap] from a single (C,) logits vector."""
    z = logits - np.max(logits)
    exp = np.exp(z)
    probs = exp / (np.sum(exp) + 1e-12)

    top2 = np.argsort(-probs)[:2]
    p1, p2 = probs[top2[0]], probs[top2[1]]
    z1, z2 = logits[top2[0]], logits[top2[1]]

    max_prob = float(p1)
    entropy = float(-np.sum(probs * np.log(probs + 1e-12)))
    margin = float(p1 - p2)
    logit_gap = float(z1 - z2)

    return np.array([max_prob, entropy, margin, logit_gap], dtype=np.float32)


def render_cifar_image_crisp(x_3chw: torch.Tensor, size: int = 256) -> np.ndarray:
    """Upscale with nearest neighbor so CIFAR looks crisp instead of blurry."""
    x = x_3chw.unsqueeze(0)  # (1,3,32,32)
    x_big = F.interpolate(x, size=(size, size), mode="nearest").squeeze(0)  # (3,H,W)
    return x_big.permute(1, 2, 0).cpu().numpy()


@st.cache_resource
def load_models(device_choice: str = "auto"):
    device = get_device(device_choice)

    a_dir = artifacts_dir()
    base_path = require_file(a_dir / "base_model.pt", hint="Run: python -m src.train_base")
    meta_path = require_file(
        a_dir / "meta_model.pkl",
        hint="Run: python -m src.make_meta_dataset && python -m src.train_meta",
    )

    payload = torch.load(base_path, map_location="cpu")
    extra = payload.get("extra", {})
    model_name = extra.get("model", "resnet18")
    pretrained = bool(extra.get("pretrained", False))

    from src.utils_model import build_model, load_base_checkpoint  # local import

    base_model = build_model(model_name, num_classes=10, pretrained=pretrained).to(device)
    load_base_checkpoint(base_model, base_path, map_location=device)
    base_model.eval()

    meta_bundle = joblib.load(meta_path)
    meta_model = meta_bundle["model"]
    features = meta_bundle.get("features", REQUIRED_META_FEATURES)

    return device, base_model, meta_model, features


@st.cache_data
def precompute_test_pfail(meta_test_path: str, feature_names: list[str], _meta_model) -> dict:
    """
    Precompute p_fail for the whole test set once (cached).
    NOTE: _meta_model has a leading underscore so Streamlit won't try to hash it.
    """
    df = pd.read_csv(meta_test_path)
    X = df[feature_names].values.astype(np.float32)
    p_fail = _meta_model.predict_proba(X)[:, 1]
    y_fail = df["y_fail"].values.astype(int)
    return {"p_fail": p_fail, "y_fail": y_fail}


def abstention_precision_pct(p_fail: np.ndarray, y_fail: np.ndarray, threshold: float) -> float:
    """
    Percent of abstained examples that are actually base-model errors.

    abstained = p_fail > threshold
    precision = (# wrong AND abstained) / (# abstained)
    """
    abstained = (p_fail > threshold)
    denom = int(abstained.sum())
    if denom == 0:
        return float("nan")
    correct_abstains = int(((y_fail == 1) & abstained).sum())
    return 100.0 * correct_abstains / denom


def abstained_count(p_fail: np.ndarray, threshold: float) -> int:
    """Number of examples with p_fail > threshold."""
    return int((p_fail > threshold).sum())


def main() -> None:
    st.set_page_config(page_title="Image Misclassification Prediction", layout="wide")

    st.title("Image Misclassification Prediction")
    st.caption("Predict when a base image classifier will be wrong and abstain when risk is high.")

    # --- Sidebar ---
    st.sidebar.header("Controls")
    device_choice = st.sidebar.selectbox("Device", ["auto", "cpu", "cuda", "mps"], index=0)
    idx = st.sidebar.number_input("Test image index", min_value=0, max_value=9999, value=0, step=1)

    # Threshold controls: slider + manual, synced
    if "threshold" not in st.session_state:
        st.session_state["threshold"] = 0.50

    def _sync_from_slider():
        st.session_state["threshold"] = float(st.session_state["threshold_slider"])

    def _sync_from_number():
        v = float(st.session_state["threshold_number"])
        st.session_state["threshold"] = max(0.0, min(1.0, v))

    st.sidebar.markdown("### Threshold")
    st.sidebar.slider(
        "Scroll to set threshold",
        min_value=0.0,
        max_value=1.0,
        value=float(st.session_state["threshold"]),
        step=0.01,
        key="threshold_slider",
        on_change=_sync_from_slider,
    )
    st.sidebar.number_input(
        "Or type it in",
        min_value=0.0,
        max_value=1.0,
        value=float(st.session_state["threshold"]),
        step=0.01,
        format="%.2f",
        key="threshold_number",
        on_change=_sync_from_number,
    )
    threshold = float(st.session_state["threshold"])

    st.sidebar.markdown("---")
    st.sidebar.markdown("**What is failure risk?**")
    st.sidebar.write("The meta-model outputs p_fail. Higher means more likely the base model is wrong.")

    # --- Softmax formula shown on UI ---
    with st.expander("How probabilities are computed", expanded=True):
        st.write("The base model outputs logits. We convert logits to probabilities using softmax.")
        st.latex(r"p_i = \frac{e^{z_i}}{\sum_{j=1}^{K} e^{z_j}}")
        st.write("Here, z is the logit score for a class, and p is the probability for that class.")

    # --- Load models + data ---
    device, base_model, meta_model, feature_names = load_models(device_choice=device_choice)

    data_dir = repo_root() / "data"
    ds_raw = cifar10_test_raw(data_dir)
    ds_model = cifar10_test_model(data_dir)

    left, right = st.columns([1.05, 1.0], gap="large")

    # -------- Left: image + base prediction --------
    with left:
        x_raw, y_true = ds_raw[int(idx)]
        st.subheader("Input image")
        st.image(
            render_cifar_image_crisp(x_raw, size=256),
            caption=f"True label: {CIFAR10_CLASSES[int(y_true)]}  |  index: {int(idx)}",
        )

        x_infer, _ = ds_model[int(idx)]
        x_infer = x_infer.unsqueeze(0).to(device)

        with torch.no_grad():
            logits_t = base_model(x_infer).detach().cpu().float().squeeze(0)
            probs_t = F.softmax(logits_t, dim=0)

        logits = logits_t.numpy()
        probs = probs_t.numpy()

        top3 = np.argsort(-probs)[:3]
        pred = int(top3[0])
        correct = pred == int(y_true)

        st.subheader("Base model output")
        a, b, c = st.columns(3)
        a.metric("Predicted class", CIFAR10_CLASSES[pred])
        b.metric("Top-1 probability", f"{float(probs[pred]):.3f}")
        c.metric("Correct", "Yes" if correct else "No")

        df_top = pd.DataFrame(
            {
                "Rank": [1, 2, 3],
                "Class": [CIFAR10_CLASSES[int(k)] for k in top3],
                "Probability": [float(probs[int(k)]) for k in top3],
            }
        )
        df_top["Probability"] = df_top["Probability"].map(lambda v: f"{v:.3f}")
        st.markdown("**Top-3 probabilities**")
        st.dataframe(df_top, use_container_width=True, hide_index=True)

    # -------- Right: decision + abstention precision + meta features --------
    with right:
        st.subheader("Selective decision")

        feats = compute_meta_features_from_logits(logits).reshape(1, -1)
        if list(feature_names) != REQUIRED_META_FEATURES:
            st.warning(
                f"Meta-model expects features={feature_names}, but app provides {REQUIRED_META_FEATURES}. "
                "Re-train meta-model with the default feature set."
            )
            st.stop()

        p_fail = float(meta_model.predict_proba(feats)[:, 1][0])
        decision = "ABSTAIN" if p_fail > threshold else "PREDICT"

        m1, m2 = st.columns(2)
        m1.metric("Failure risk p_fail", f"{p_fail:.3f}")
        m2.metric("Decision", decision)

        st.progress(min(max(p_fail, 0.0), 1.0))

        if decision == "ABSTAIN":
            st.error(f"Abstain because p_fail {p_fail:.3f} is above threshold {threshold:.2f}.")
        else:
            st.success(f"Predict because p_fail {p_fail:.3f} is not above threshold {threshold:.2f}.")

        st.markdown("---")
        st.subheader("How good is abstaining at catching mistakes?")

        meta_test_path = artifacts_dir() / "meta_test.csv"
        if meta_test_path.exists():
            # Precompute p_fail for all test samples (cached)
            pre = precompute_test_pfail(str(meta_test_path), feature_names, meta_model)

            # Slider + abstained count shown next to it
            s_col, n_col = st.columns([2, 1])
            with s_col:
                eval_thresh_pct = st.slider(
                    "Evaluation threshold",
                    min_value=0,
                    max_value=100,
                    value=50,
                    step=1,
                    help="Threshold in percent. We abstain when p_fail is above this value.",
                )
            eval_thresh = eval_thresh_pct / 100.0

            with n_col:
                n_abst = abstained_count(pre["p_fail"], eval_thresh)
                st.metric("Abstained", f"{n_abst}")

            st.caption(f"Using threshold: {eval_thresh_pct}%  meaning p_fail > {eval_thresh:.2f}")

            pct = abstention_precision_pct(pre["p_fail"], pre["y_fail"], eval_thresh)
            st.metric("Abstention precision", f"{pct:.1f}%")
            st.caption("Among images above the threshold, this is the percent where the base model is actually wrong.")
        else:
            st.info("meta_test.csv not found. Run: python -m src.make_meta_dataset")

        st.markdown("---")
        st.subheader("Meta-features")
        st.write("The failure predictor uses several features. Below we explain only entropy and margin.")

        entropy_val = float(feats[0, 1])
        margin_val = float(feats[0, 2])

        df_feat = pd.DataFrame(
            [
                ("entropy", f"{entropy_val:.4f}", FEATURE_EXPLAIN_SIMPLE["entropy"]),
                ("margin", f"{margin_val:.4f}", FEATURE_EXPLAIN_SIMPLE["margin"]),
            ],
            columns=["Feature", "Value", "Meaning"],
        )
        st.dataframe(df_feat, use_container_width=True, hide_index=True)

        with st.expander("Formulas for entropy and margin"):
            st.latex(r"\text{entropy} = -\sum_{i=1}^{K} p_i \log(p_i)")
            st.latex(r"\text{margin} = p_{\text{top1}} - p_{\text{top2}}")

        st.markdown("**Quick interpretation**")
        bullets = []
        bullets.append("Entropy is high. The model is uncertain." if entropy_val > 1.5 else "Entropy is lower. The model is more confident.")
        bullets.append("Margin is small. The top two classes are very close." if margin_val < 0.05 else "Margin is larger. The top class stands out more.")
        for b in bullets:
            st.write(f"- {b}")

        with st.expander("What does abstaining mean?"):
            st.write(
                "Abstaining means the system chooses not to answer on this image. "
                "In real use, abstained cases can go to a stronger model or a human."
            )


if __name__ == "__main__":
    main()