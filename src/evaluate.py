"""Metrics and plots, shared by ``train.py`` and ``ds_app.py``.

Every plotting function **returns** a ``matplotlib.figure.Figure`` and never
calls ``plt.show()`` or writes to disk. That is what lets the same code serve
two very different consumers: ``train.py`` can save the figure to a file,
and Streamlit can hand it straight to ``st.pyplot()``. A function that showed
or saved internally would only work for one of them.

All functions take ``y_prob`` (predicted probabilities), not hard labels, so
the decision threshold stays an explicit argument rather than being baked in.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # headless backend: no display needed on a server

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

DEFAULT_THRESHOLD = 0.5


def compute_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = DEFAULT_THRESHOLD,
) -> dict[str, float]:
    """Compute the headline classification metrics."""

    y_pred = (y_prob >= threshold).astype(int)

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(
            precision_score(y_true, y_pred, zero_division=0)
        ),
        "recall": float(
            recall_score(y_true, y_pred, zero_division=0)
        ),
        "f1": float(
            f1_score(y_true, y_pred, zero_division=0)
        ),
        "roc_auc": float(
            roc_auc_score(y_true, y_prob)
        ),
    }

    return metrics


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = DEFAULT_THRESHOLD,
) -> plt.Figure:
    """Confusion matrix as an annotated heatmap."""

    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(5, 4))

    im = ax.imshow(cm, cmap="Blues")

    labels = ["Did not survive", "Survived"]

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)

    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(f"Confusion Matrix (threshold = {threshold:.2f})")

    # Annotate each cell with its count
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
            )

    fig.colorbar(im, ax=ax)
    fig.tight_layout()

    return fig


def plot_roc_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
) -> plt.Figure:
    """ROC curve with the AUC in the legend, plus the chance diagonal."""

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc_value = roc_auc_score(y_true, y_prob)

    fig, ax = plt.subplots(figsize=(6, 5))

    ax.plot(
        fpr,
        tpr,
        label=f"ROC curve (AUC = {auc_value:.3f})",
    )

    # Chance-level classifier
    ax.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Chance",
    )

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)

    fig.tight_layout()

    return fig


def plot_precision_recall_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
) -> plt.Figure:
    """Precision-recall curve with average precision in the legend."""

    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    average_precision = average_precision_score(y_true, y_prob)

    fig, ax = plt.subplots(figsize=(6, 5))

    ax.plot(
        recall,
        precision,
        label=f"PR curve (AP = {average_precision:.3f})",
    )

    # Baseline equals the positive-class prevalence
    baseline = y_true.mean()
    ax.axhline(
        baseline,
        linestyle="--",
        label=f"Baseline = {baseline:.3f}",
    )

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    ax.legend(loc="lower left")
    ax.grid(alpha=0.3)

    fig.tight_layout()

    return fig


def plot_probability_distribution(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = DEFAULT_THRESHOLD,
) -> plt.Figure:
    """Overlaid histograms of predicted probability, split by true class."""

    fig, ax = plt.subplots(figsize=(6, 5))

    ax.hist(
        y_prob[y_true == 0],
        bins=15,
        alpha=0.6,
        label="Did not survive",
    )

    ax.hist(
        y_prob[y_true == 1],
        bins=15,
        alpha=0.6,
        label="Survived",
    )

    ax.axvline(
        threshold,
        linestyle="--",
        label=f"Threshold = {threshold:.2f}",
    )

    ax.set_xlabel("Predicted survival probability")
    ax.set_ylabel("Count")
    ax.set_title("Predicted Probability Distribution")
    ax.legend()
    ax.grid(alpha=0.3)

    fig.tight_layout()

    return fig


# --- Smoke test --------------------------------------------------------------

if __name__ == "__main__":
    # Synthetic data: a deliberately mediocre classifier, so the curves have
    # a realistic shape rather than being perfect.
    rng = np.random.default_rng(0)
    n = 300
    y_true = rng.binomial(1, 0.38, size=n)
    y_prob = np.clip(0.3 + 0.4 * y_true + rng.normal(0, 0.2, size=n), 0, 1)

    metrics = compute_metrics(y_true, y_prob)
    print("metrics:")
    for k, v in metrics.items():
        print(f"  {k:<12} {v:.4f}")
    assert all(isinstance(v, float) for v in metrics.values()), "must be plain floats"

    import json

    json.dumps(metrics)  # must not raise
    print("JSON-serialisable: True")

    figs = {
        "confusion_matrix": plot_confusion_matrix(y_true, y_prob),
        "roc_curve": plot_roc_curve(y_true, y_prob),
        "pr_curve": plot_precision_recall_curve(y_true, y_prob),
        "prob_distribution": plot_probability_distribution(y_true, y_prob),
    }
    for name, fig in figs.items():
        assert isinstance(fig, plt.Figure), f"{name} must return a Figure"
        print(f"  {name}: Figure OK")
    print("all plot functions return figures")
