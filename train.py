"""Standalone training script for the Titanic survival classifier.

Runs the whole pipeline end to end and writes everything the Streamlit app
needs to ``artifacts/``:

===========================  ==================================================
File                         Contents
===========================  ==================================================
``model.pt``                 Network weights (state dict)
``preprocessor.joblib``      Fitted ColumnTransformer
``metadata.json``            Architecture, hyperparameters, feature names,
                             decision threshold, validation metrics
``val_predictions.csv``      Per-row ``y_true`` / ``y_prob`` on the held-out
                             validation split
===========================  ==================================================

``val_predictions.csv`` exists so the app never has to reconstruct the split
or re-run the model just to display validation results: it loads the
predictions and renders metrics from them. That also makes an interactive
threshold slider cheap, since nothing needs recomputing.

Usage
-----
    python train.py
    python train.py --epochs 300 --lr 5e-4 --seed 7
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.data import (
    DEFAULT_SEED,
    DEFAULT_VAL_SIZE,
    TARGET,
    download_titanic,
    load_train_csv,
    split_train_val,
)
from src.model import (
    DEFAULT_DROPOUT,
    DEFAULT_HIDDEN_DIMS,
    TitanicNet,
    predict_proba,
    save_model,
)
from src.preprocess import (
    add_engineered_features,
    build_preprocessor,
    get_feature_names,
    save_preprocessor,
)

PROJECT_ROOT = Path(__file__).resolve().parent
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

DEFAULT_EPOCHS = 200
DEFAULT_LR = 1e-3
DEFAULT_BATCH_SIZE = 32
DEFAULT_PATIENCE = 20
DEFAULT_THRESHOLD = 0.5


# --- Reproducibility ---------------------------------------------------------

def set_seed(seed: int) -> None:
    """Seed every source of randomness the run touches.

    Covers Python's ``random`` (used by some sklearn internals), NumPy (the
    split and any array shuffling) and PyTorch (weight initialisation and
    DataLoader shuffling). Without all three, two runs of this script produce
    different weights and the reported metrics are not reproducible.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# --- Data preparation --------------------------------------------------------

def build_tensors(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, object, list[str]]:
    """Engineer features, fit the preprocessor, and return tensors.

    Returns
    -------
    tuple
        ``(X_train, y_train, X_val, y_val, preprocessor, feature_names)``
        where the X tensors are float32 of shape ``(n, input_dim)`` and the
        y tensors are float32 of shape ``(n, 1)``.
    """
    # TODO (Miriam):
    #  1. add_engineered_features on BOTH frames.
    #  2. pre = build_preprocessor()
    #  3. X_train = pre.fit_transform(train_feat)   <- fit ONLY on train
    #     X_val   = pre.transform(val_feat)         <- transform only
    #     This is the single most important line ordering in the project.
    #  4. Convert to tensors:
    #       torch.tensor(np.asarray(X, dtype=np.float32))
    #  5. Targets: df[TARGET].to_numpy(dtype=np.float32), then
    #       .reshape(-1, 1)  <- shape MUST match the model's (batch, 1) output.
    #       BCEWithLogitsLoss will broadcast a (batch,) target against a
    #       (batch, 1) prediction into a (batch, batch) loss without
    #       complaining. The loss still decreases, the model still "trains",
    #       and the result is silently wrong. Keep both at (n, 1).
    #  6. Return everything plus get_feature_names(pre).
    raise NotImplementedError


# --- Training ----------------------------------------------------------------

def train_one_epoch(
    model: TitanicNet,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
) -> float:
    """Run one training epoch and return the mean batch loss."""
    # TODO (Miriam): the core loop. For each (xb, yb) in loader:
    #  1. optimizer.zero_grad()
    #     PyTorch ACCUMULATES gradients across .backward() calls by design
    #     (it is what makes gradient accumulation and RNNs work). Skip this
    #     and every batch's gradients pile onto the previous ones. No error
    #     is raised - the model just trains badly, which is hard to spot.
    #  2. logits = model(xb)
    #  3. loss = criterion(logits, yb)
    #  4. loss.backward()      <- computes gradients
    #  5. optimizer.step()     <- applies them to the weights
    #     This order matters: step() uses whatever backward() just produced.
    #  6. Accumulate loss.item() and return the mean over batches.
    #
    #  Remember model.train() before the loop - that is the caller's job in
    #  main(), but double-check it is happening.
    raise NotImplementedError


@torch.no_grad()
def evaluate_loss(
    model: TitanicNet,
    X: torch.Tensor,
    y: torch.Tensor,
    criterion: nn.Module,
) -> float:
    """Compute loss on a full tensor without updating anything."""
    # TODO (Miriam):
    #  1. model.eval()  <- turns Dropout off. With dropout still active the
    #     validation loss is noisy and systematically worse than reality,
    #     which corrupts early stopping.
    #  2. logits = model(X); return criterion(logits, y).item()
    #  The decorator already handles no_grad.
    raise NotImplementedError


def fit(
    model: TitanicNet,
    train_loader: DataLoader,
    X_val: torch.Tensor,
    y_val: torch.Tensor,
    epochs: int,
    lr: float,
    patience: int,
) -> dict:
    """Train with early stopping on validation loss.

    Returns
    -------
    dict
        History with ``train_loss`` and ``val_loss`` lists, plus
        ``best_epoch``. The model is left holding the **best** weights, not
        the last ones.
    """
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # TODO (Miriam):
    #  1. Track best_val_loss (start at inf), best_state (None), epochs_without_improvement.
    #  2. For each epoch:
    #       model.train()
    #       train_loss = train_one_epoch(...)
    #       val_loss   = evaluate_loss(...)
    #       append both to history
    #       if val_loss < best_val_loss:
    #           record best_val_loss, best_epoch
    #           best_state = copy.deepcopy(model.state_dict())
    #               <- deepcopy is REQUIRED. state_dict() returns references
    #                  to the live tensors, so without a copy your "best"
    #                  snapshot mutates as training continues and you end up
    #                  restoring the final weights instead.
    #           reset the counter
    #       else:
    #           increment counter; break if it reaches patience
    #       print progress every ~10 epochs so the run is not silent
    #  3. After the loop: model.load_state_dict(best_state) to restore the best.
    #  4. Return the history dict.
    raise NotImplementedError


# --- Orchestration -----------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--patience", type=int, default=DEFAULT_PATIENCE)
    parser.add_argument("--dropout", type=float, default=DEFAULT_DROPOUT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--val-size", type=float, default=DEFAULT_VAL_SIZE)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--out-dir", type=Path, default=ARTIFACTS_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    print("Loading data...")
    full_df = load_train_csv(download_titanic())
    train_df, val_df = split_train_val(full_df, val_size=args.val_size, seed=args.seed)
    print(f"  train: {train_df.shape}   val: {val_df.shape}")

    print("Preprocessing (fitting on the training split only)...")
    X_train, y_train, X_val, y_val, preprocessor, feature_names = build_tensors(
        train_df, val_df
    )
    input_dim = X_train.shape[1]
    print(f"  input_dim: {input_dim}")

    # A seeded generator makes DataLoader shuffling reproducible too.
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
    )

    print(f"Training (max {args.epochs} epochs, patience {args.patience})...")
    model = TitanicNet(
        input_dim=input_dim,
        hidden_dims=DEFAULT_HIDDEN_DIMS,
        dropout=args.dropout,
    )
    history = fit(
        model,
        train_loader,
        X_val,
        y_val,
        epochs=args.epochs,
        lr=args.lr,
        patience=args.patience,
    )
    print(f"  best epoch: {history['best_epoch']}")

    # --- Validation predictions, computed once and saved ---------------------
    val_prob = predict_proba(model, X_val)
    val_true = y_val.numpy().ravel().astype(int)

    from src.evaluate import compute_metrics  # noqa: PLC0415 - Block 5

    metrics = compute_metrics(val_true, val_prob, threshold=args.threshold)
    print("\nValidation metrics:")
    for name, value in metrics.items():
        print(f"  {name:<12} {value:.4f}")

    # --- Persist everything --------------------------------------------------
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    save_model(model, out_dir / "model.pt")
    save_preprocessor(preprocessor, out_dir / "preprocessor.joblib")

    pd.DataFrame({"y_true": val_true, "y_prob": val_prob}).to_csv(
        out_dir / "val_predictions.csv", index=False
    )

    metadata = {
        "input_dim": input_dim,
        "hidden_dims": list(DEFAULT_HIDDEN_DIMS),
        "dropout": args.dropout,
        "feature_names": feature_names,
        "threshold": args.threshold,
        "seed": args.seed,
        "val_size": args.val_size,
        "epochs_requested": args.epochs,
        "best_epoch": history["best_epoch"],
        "learning_rate": args.lr,
        "batch_size": args.batch_size,
        "val_metrics": metrics,
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

    print(f"\nArtifacts written to {out_dir}")
    for name in ("model.pt", "preprocessor.joblib", "metadata.json", "val_predictions.csv"):
        print(f"  {name}")


if __name__ == "__main__":
    main()
