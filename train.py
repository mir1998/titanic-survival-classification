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
import copy
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
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    object,
    list[str],
]:
    """Engineer features, fit the preprocessor, and return tensors."""

    # Add row-local engineered features
    train_feat = add_engineered_features(train_df)
    val_feat = add_engineered_features(val_df)

    # Fit preprocessing only on the training split
    preprocessor = build_preprocessor()

    X_train_np = preprocessor.fit_transform(train_feat)
    X_val_np = preprocessor.transform(val_feat)

    # Convert feature matrices to float32 tensors
    X_train = torch.tensor(
        np.asarray(X_train_np, dtype=np.float32)
    )
    X_val = torch.tensor(
        np.asarray(X_val_np, dtype=np.float32)
    )

    # Targets must match the model output shape: (n_samples, 1)
    y_train = torch.tensor(
        train_df[TARGET].to_numpy(dtype=np.float32).reshape(-1, 1)
    )
    y_val = torch.tensor(
        val_df[TARGET].to_numpy(dtype=np.float32).reshape(-1, 1)
    )

    feature_names = get_feature_names(preprocessor)

    return (
        X_train,
        y_train,
        X_val,
        y_val,
        preprocessor,
        feature_names,
    )


# --- Training ----------------------------------------------------------------

def train_one_epoch(
    model: TitanicNet,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
) -> float:
    """Run one training epoch and return the mean batch loss."""

    model.train()

    total_loss = 0.0
    n_batches = 0

    for xb, yb in loader:
        optimizer.zero_grad()

        logits = model(xb)
        loss = criterion(logits, yb)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / n_batches


@torch.no_grad()
def evaluate_loss(
    model: TitanicNet,
    X: torch.Tensor,
    y: torch.Tensor,
    criterion: nn.Module,
) -> float:
    """Compute loss on a full tensor without updating anything."""

    model.eval()

    logits = model(X)
    loss = criterion(logits, y)

    return loss.item()


def fit(
    model: TitanicNet,
    train_loader: DataLoader,
    X_val: torch.Tensor,
    y_val: torch.Tensor,
    epochs: int,
    lr: float,
    patience: int,
) -> dict:
    """Train with early stopping on validation loss."""

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history = {
        "train_loss": [],
        "val_loss": [],
        "best_epoch": None,
    }

    best_val_loss = float("inf")
    best_state = None
    epochs_without_improvement = 0

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
        )

        val_loss = evaluate_loss(
            model,
            X_val,
            y_val,
            criterion,
        )

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            history["best_epoch"] = epoch
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epoch == 1 or epoch % 10 == 0:
            print(
                f"Epoch {epoch:03d} | "
                f"train_loss={train_loss:.4f} | "
                f"val_loss={val_loss:.4f}"
            )

        if epochs_without_improvement >= patience:
            print(
                f"Early stopping at epoch {epoch} "
                f"(best epoch: {history['best_epoch']})"
            )
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return history


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
