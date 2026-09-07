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
    auc,
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
    """Compute the headline classification metrics.

    Parameters
    ----------
    y_true:
        Ground-truth labels, 0/1.
    y_prob:
        Predicted probabilities in ``[0, 1]``.
    threshold:
        Probability above which a passenger is predicted to have survived.

    Returns
    -------
    dict
        ``accuracy``, ``precision``, ``recall``, ``f1``, ``roc_auc``.
        Plain Python floats, so the dict is JSON-serialisable for
        ``metadata.json``.

    Notes
    -----
    ``roc_auc`` is computed from ``y_prob`` directly and is therefore
    threshold-independent; the other four all depend on ``threshold``.
    That is the point of reporting both kinds together.
    """
    # TODO (Miriam):
    #  1. y_pred = (y_prob >= threshold).astype(int)
    #  2. accuracy_score / precision_score / recall_score / f1_score on y_pred
    #     Use zero_division=0 on precision - if the model predicts no
    #     survivors at all at some threshold, precision is 0/0 and sklearn
    #     warns loudly. Relevant once the app has a threshold slider.
    #  3. roc_auc_score on y_PROB, not y_pred. Passing hard labels here is a
    #     classic mistake: it silently returns a much lower, meaningless number.
    #  4. Wrap each value in float() so json.dumps() accepts them
    #     (numpy scalars are not JSON-serialisable).
    raise NotImplementedError


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = DEFAULT_THRESHOLD,
) -> plt.Figure:
    """Confusion matrix as an annotated heatmap.

    Returns
    -------
    matplotlib.figure.Figure
    """
    # TODO (Miriam):
    #  1. y_pred from threshold, then confusion_matrix(y_true, y_pred)
    #  2. fig, ax = plt.subplots(); ax.imshow(cm, cmap="Blues")
    #  3. Annotate each cell with its count (nested loop + ax.text)
    #  4. Tick labels: ["Did not survive", "Survived"] on both axes -
    #     the same wording you used in the notebook.
    #  5. Return fig. No plt.show().
    raise NotImplementedError


def plot_roc_curve(y_true: np.ndarray, y_prob: np.ndarray) -> plt.Figure:
    """ROC curve with the AUC in the legend, plus the chance diagonal."""
    # TODO (Miriam):
    #  fpr, tpr, _ = roc_curve(y_true, y_prob); auc_value = roc_auc_score(...)
    #  Plot the curve, add the y=x dashed diagonal (a random classifier),
    #  label the axes, put AUC in the legend, return fig.
    raise NotImplementedError


def plot_precision_recall_curve(y_true: np.ndarray, y_prob: np.ndarray) -> plt.Figure:
    """Precision-recall curve with average precision in the legend.

    More informative than ROC on imbalanced data, which is why both appear.
    """
    # TODO (Miriam):
    #  precision, recall, _ = precision_recall_curve(y_true, y_prob)
    #  auc(recall, precision) gives the area. Add a horizontal baseline at
    #  y_true.mean() - that is what a random classifier achieves here, and it
    #  is exactly the 38.3% survival rate from the EDA.
    raise NotImplementedError


def plot_probability_distribution(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = DEFAULT_THRESHOLD,
) -> plt.Figure:
    """Overlaid histograms of predicted probability, split by true class.

    Shows how well-separated the two classes are, and where the threshold
    falls relative to that separation. Good separation looks like two humps
    pushed to opposite ends.
    """
    # TODO (Miriam):
    #  Two ax.hist() calls (y_prob[y_true == 0] and y_prob[y_true == 1]) with
    #  alpha for overlap, plus ax.axvline(threshold, linestyle="--") to mark
    #  the cut. Use the same red/green you used in the notebook for consistency.
    raise NotImplementedError


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
