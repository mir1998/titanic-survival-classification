"""Streamlit app for the Titanic survival classifier.

Two screens, matching the two requirements in the assignment:

**Validation Results** - the metrics and plots for the held-out validation
split. Reads ``artifacts/val_predictions.csv`` rather than re-running the
model, so this screen needs neither Kaggle access nor the original data. That
also makes the threshold slider instant: changing it only recomputes metrics
from probabilities that are already on disk.

**Inference** - the user gives a path to a CSV, the trained model and
preprocessor are loaded from disk, and predictions are produced. If the file
happens to contain a ``Survived`` column, the same evaluation is shown for it;
if not, predictions are shown on their own rather than the app failing.

Run with::

    streamlit run ds_app.py

Requires ``python train.py`` to have been run first.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from src.evaluate import (
    compute_metrics,
    plot_confusion_matrix,
    plot_precision_recall_curve,
    plot_probability_distribution,
    plot_roc_curve,
)
from src.model import load_model, predict_proba
from src.preprocess import add_engineered_features, load_preprocessor

PROJECT_ROOT = Path(__file__).resolve().parent
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
SAMPLE_CSV = PROJECT_ROOT / "data" / "sample_titanic.csv"

TARGET = "Survived"

#: Raw columns the preprocessing pipeline needs. PassengerId and Ticket are
#: not here: they are dropped, so a file without them still works. Survived is
#: also absent by design - it is the label, and inference must work without it.
REQUIRED_COLUMNS = [
    "Pclass", "Name", "Sex", "Age", "SibSp", "Parch", "Fare", "Cabin", "Embarked",
]


# --- Artifact loading --------------------------------------------------------

@st.cache_resource
def load_artifacts():
    """Load model, preprocessor and metadata from ``artifacts/``.

    Decorated with ``st.cache_resource`` because these are heavy, unhashable
    objects that should be created once per session rather than on every
    rerun. Streamlit re-executes this entire script top to bottom on every
    widget interaction, so without caching the model would be rebuilt from
    disk each time the user nudged a slider.

    Returns
    -------
    tuple
        ``(model, preprocessor, metadata)``.
    """
    metadata_path = ARTIFACTS_DIR / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"No artifacts found in {ARTIFACTS_DIR}. "
            "Run `python train.py` first."
        )

    metadata = json.loads(metadata_path.read_text())
    preprocessor = load_preprocessor(ARTIFACTS_DIR / "preprocessor.joblib")
    model = load_model(
        ARTIFACTS_DIR / "model.pt",
        input_dim=metadata["input_dim"],
        hidden_dims=tuple(metadata["hidden_dims"]),
        dropout=metadata["dropout"],
    )
    return model, preprocessor, metadata


@st.cache_data
def load_val_predictions() -> pd.DataFrame:
    """Load the saved validation predictions written by ``train.py``.

    ``st.cache_data`` (not ``cache_resource``) because this returns a
    dataframe: Streamlit caches a copy, so a mutation on one rerun cannot
    corrupt what later reruns see.
    """
    path = ARTIFACTS_DIR / "val_predictions.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python train.py` first."
        )
    return pd.read_csv(path)


# --- Input handling ----------------------------------------------------------

def read_csv_from_path(path_str: str) -> pd.DataFrame:
    """Read a user-supplied CSV path, with errors aimed at the user.

    Raises
    ------
    FileNotFoundError, ValueError
        With messages meant to be shown directly in the UI.
    """
    if not path_str or not path_str.strip():
        raise ValueError("Please enter a path to a CSV file.")

    path = Path(path_str.strip().strip('"'))  # tolerate pasted quoted paths
    if not path.exists():
        raise FileNotFoundError(f"No file at: {path}")
    if path.is_dir():
        raise ValueError(f"That is a directory, not a file: {path}")

    try:
        df = pd.read_csv(path)
    except Exception as exc:
        raise ValueError(f"Could not parse '{path.name}' as CSV: {exc}") from exc

    if df.empty:
        raise ValueError(f"'{path.name}' contains no rows.")
    return df


def find_missing_columns(df: pd.DataFrame) -> list[str]:
    """Return the required columns absent from ``df``."""
    return [c for c in REQUIRED_COLUMNS if c not in df.columns]


def run_inference(
    df: pd.DataFrame,
    model,
    preprocessor,
    threshold: float,
) -> pd.DataFrame:
    """Predict survival for every row.

    Returns
    -------
    pandas.DataFrame
        The input columns plus ``survival_probability`` and ``prediction``.
        ``PassengerId`` is carried through when present so results stay
        traceable to the source rows.
    """
    features = add_engineered_features(df)
    X = preprocessor.transform(features)
    prob = predict_proba(model, X)

    out = df.copy()
    out["survival_probability"] = prob
    out["prediction"] = (prob >= threshold).astype(int)
    return out


# --- Screens -----------------------------------------------------------------

def render_validation_page() -> None:
    """Screen 1: results on the held-out validation split."""
    st.header("Validation Results")

    try:
        _, _, metadata = load_artifacts()
        val_predictions = load_val_predictions()
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.stop()

    y_true = val_predictions["y_true"].to_numpy()
    y_prob = val_predictions["y_prob"].to_numpy()

    with st.expander("Training configuration"):
        st.write({
            "seed": metadata["seed"],
            "validation_size": metadata["val_size"],
            "best_epoch": metadata["best_epoch"],
            "input_dim": metadata["input_dim"],
            "learning_rate": metadata["learning_rate"],
            "batch_size": metadata["batch_size"],
        })

    threshold = st.slider(
        "Decision threshold",
        min_value=0.0,
        max_value=1.0,
        value=float(metadata["threshold"]),
        step=0.01,
    )

    metrics = compute_metrics(
        y_true,
        y_prob,
        threshold=threshold,
    )

    majority_baseline = max(
        y_true.mean(),
        1 - y_true.mean(),
    )

    st.caption(
        f"Majority-class accuracy baseline: {majority_baseline:.3f}"
    )

    cols = st.columns(5)

    cols[0].metric("Accuracy", f"{metrics['accuracy']:.3f}")
    cols[1].metric("Precision", f"{metrics['precision']:.3f}")
    cols[2].metric("Recall", f"{metrics['recall']:.3f}")
    cols[3].metric("F1", f"{metrics['f1']:.3f}")
    cols[4].metric("ROC-AUC", f"{metrics['roc_auc']:.3f}")

    col1, col2 = st.columns(2)

    with col1:
        st.pyplot(
            plot_confusion_matrix(
                y_true,
                y_prob,
                threshold=threshold,
            )
        )

    with col2:
        st.pyplot(
            plot_roc_curve(
                y_true,
                y_prob,
            )
        )

    col3, col4 = st.columns(2)

    with col3:
        st.pyplot(
            plot_precision_recall_curve(
                y_true,
                y_prob,
            )
        )

    with col4:
        st.pyplot(
            plot_probability_distribution(
                y_true,
                y_prob,
                threshold=threshold,
            )
        )


def render_inference_page() -> None:
    """Screen 2: run the saved model against a user-supplied CSV."""
    st.header("Inference")

    path_str = st.text_input(
        "Path to CSV file",
        value=str(SAMPLE_CSV),
    )

    threshold = st.slider(
        "Decision threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.01,
        key="inference_threshold",
    )

    if not st.button("Run inference"):
        return

    try:
        model, preprocessor, _ = load_artifacts()
        df = read_csv_from_path(path_str)
    except (FileNotFoundError, ValueError) as exc:
        st.error(str(exc))
        return

    missing_columns = find_missing_columns(df)

    if missing_columns:
        st.error(
            "Missing required columns: "
            + ", ".join(missing_columns)
        )
        return

    try:
        results = run_inference(
            df,
            model,
            preprocessor,
            threshold=threshold,
        )
    except Exception as exc:
        st.error(f"Inference failed: {exc}")
        return

    st.subheader("Predictions")
    st.dataframe(results, use_container_width=True)

    csv_bytes = results.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Download predictions as CSV",
        data=csv_bytes,
        file_name="titanic_predictions.csv",
        mime="text/csv",
    )

    if TARGET not in df.columns:
        st.info(
            "No 'Survived' column was found, so predictions are shown "
            "without evaluation metrics."
        )
        return

    y_true = df[TARGET].to_numpy()

    if not set(pd.Series(y_true).dropna().unique()).issubset({0, 1}):
        st.error(
            "'Survived' must contain only binary values 0 and 1 "
            "to compute evaluation metrics."
        )
        return

    y_prob = results["survival_probability"].to_numpy()

    metrics = compute_metrics(
        y_true,
        y_prob,
        threshold=threshold,
    )

    st.subheader("Evaluation")

    cols = st.columns(5)

    cols[0].metric("Accuracy", f"{metrics['accuracy']:.3f}")
    cols[1].metric("Precision", f"{metrics['precision']:.3f}")
    cols[2].metric("Recall", f"{metrics['recall']:.3f}")
    cols[3].metric("F1", f"{metrics['f1']:.3f}")
    cols[4].metric("ROC-AUC", f"{metrics['roc_auc']:.3f}")

    col1, col2 = st.columns(2)

    with col1:
        st.pyplot(
            plot_confusion_matrix(
                y_true,
                y_prob,
                threshold=threshold,
            )
        )

    with col2:
        st.pyplot(
            plot_roc_curve(
                y_true,
                y_prob,
            )
        )

    col3, col4 = st.columns(2)

    with col3:
        st.pyplot(
            plot_precision_recall_curve(
                y_true,
                y_prob,
            )
        )

    with col4:
        st.pyplot(
            plot_probability_distribution(
                y_true,
                y_prob,
                threshold=threshold,
            )
        )


# --- Entry point -------------------------------------------------------------

def main() -> None:
    st.set_page_config(page_title="Titanic Survival Classifier", layout="wide")
    st.title("Titanic Survival Classifier")

    page = st.sidebar.radio("Screen", ["Validation Results", "Inference"])
    st.sidebar.markdown("---")
    st.sidebar.caption(
        "Artifacts are produced by `python train.py` and read from "
        "`artifacts/`."
    )

    if page == "Validation Results":
        render_validation_page()
    else:
        render_inference_page()


if __name__ == "__main__":
    main()
