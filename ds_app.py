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

    # TODO (Miriam):
    #  1. Wrap load_artifacts() / load_val_predictions() in try/except
    #     FileNotFoundError and st.error(...) + st.stop() with the
    #     "run python train.py first" message. The app must not traceback
    #     when someone clones the repo and opens it before training.
    #  2. Pull metadata out for display: seed, val_size, best_epoch,
    #     input_dim, learning_rate, batch_size. st.caption or an expander
    #     is a tidy place for this - it documents reproducibility in the UI.
    #  3. Threshold slider:
    #       threshold = st.slider("Decision threshold", 0.0, 1.0,
    #                             value=metadata["threshold"], step=0.01)
    #     Then recompute compute_metrics(y_true, y_prob, threshold) live.
    #     This is why train.py saved probabilities rather than hard labels.
    #  4. Metrics row: st.columns(5) with st.metric() for accuracy,
    #     precision, recall, f1, roc_auc.
    #     Worth adding: the majority-class baseline (1 - y_true.mean()) as a
    #     reference point, so a reviewer sees 82.7% against 61.7% rather
    #     than a bare number.
    #  5. Plots: st.columns(2) and st.pyplot(fig) for each of the four
    #     evaluate.py figures. Note roc/pr take (y_true, y_prob) only, while
    #     confusion matrix and probability distribution also take threshold.
    raise NotImplementedError


def render_inference_page() -> None:
    """Screen 2: run the saved model against a user-supplied CSV."""
    st.header("Inference")

    # TODO (Miriam):
    #  1. Input. The assignment explicitly asks for a *path*, so
    #     st.text_input is the primary control - keep it. A st.file_uploader
    #     alongside it is a nice extra, but must not replace the path box.
    #     Pre-fill the default with str(SAMPLE_CSV) so the app is usable
    #     immediately by a reviewer with no data of their own.
    #  2. A "Run inference" button (st.button) so it does not fire on every
    #     keystroke while the path is being typed.
    #  3. Load: read_csv_from_path -> catch (FileNotFoundError, ValueError)
    #     -> st.error(str(exc)) and return. Never let a traceback reach the UI.
    #  4. Validate: find_missing_columns(df). If non-empty, st.error listing
    #     exactly which columns are missing, then return.
    #  5. Predict: run_inference(...) and show the table with st.dataframe.
    #     Add st.download_button for the results as CSV - cheap and it makes
    #     the screen genuinely useful rather than just a demo.
    #  6. THE IMPORTANT BRANCH:
    #       if TARGET in df.columns:
    #           -> full evaluation: metrics + the same four plots
    #       else:
    #           -> st.info("No 'Survived' column, so predictions are shown
    #              without evaluation") and display predictions only.
    #     A file without labels is the normal inference case, not an error.
    #     This branch is the "error handling and robustness" criterion.
    raise NotImplementedError


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
