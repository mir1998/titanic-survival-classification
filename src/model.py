"""The PyTorch classifier for Titanic survival prediction.

A small feed-forward network (MLP). With ~700 training rows and 17 input
features, capacity is deliberately modest: a large network would memorise the
training split long before it learned anything general.

The network outputs a single **logit**, not a probability. Applying the
sigmoid is deferred to ``BCEWithLogitsLoss`` during training and to
:func:`predict_proba` at inference. See the note in :meth:`TitanicNet.forward`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn

#: Default architecture, sized for a dataset of this scale.
DEFAULT_HIDDEN_DIMS = (64, 32)
DEFAULT_DROPOUT = 0.3


class TitanicNet(nn.Module):
    """A small MLP binary classifier.

    Parameters
    ----------
    input_dim:
        Number of input features, i.e. the width of the matrix produced by
        the fitted preprocessor. Must be passed explicitly rather than
        hard-coded, because it changes whenever the feature set changes
        (one-hot encoding makes it non-obvious).
    hidden_dims:
        Width of each hidden layer.
    dropout:
        Dropout probability applied after each hidden layer.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: tuple[int, ...] = DEFAULT_HIDDEN_DIMS,
        dropout: float = DEFAULT_DROPOUT,
        ) -> None:
        super().__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = hidden_dim

        # Final layer outputs a single raw logit
        layers.append(nn.Linear(prev_dim, 1))

        self.network = nn.Sequential(*layers)
  

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a forward pass."""
        return self.network(x)


@torch.no_grad()
def predict_proba(
    model: TitanicNet,
    X: np.ndarray | torch.Tensor
) -> np.ndarray:
    """Return survival probabilities for a feature matrix."""

    # Disable dropout and use inference behavior
    model.eval()

    # Convert input to float32 tensor if needed
    if isinstance(X, torch.Tensor):
        x_tensor = X.float()
    else:
        x_tensor = torch.tensor(X, dtype=torch.float32)

    # Convert raw logits to probabilities
    logits = model(x_tensor)
    probabilities = torch.sigmoid(logits)

    return probabilities.squeeze(1).cpu().numpy()


def save_model(model: TitanicNet, path: Path | str) -> Path:
    """Persist model weights to disk."""

    path = Path(path)

    # Ensure the destination directory exists
    path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(model.state_dict(), path)

    return path


def load_model(
    path: Path | str,
    input_dim: int,
    hidden_dims: tuple[int, ...] = DEFAULT_HIDDEN_DIMS,
    dropout: float = DEFAULT_DROPOUT,
) -> TitanicNet:
    """Rebuild the network and load saved weights into it."""

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Model weights not found at '{path}'. "
            "Run `python train.py` first to generate the training artifacts."
        )

    model = TitanicNet(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        dropout=dropout,
    )

    state_dict = torch.load(path, map_location="cpu")
    model.load_state_dict(state_dict)

    model.eval()

    return model


# --- Smoke test --------------------------------------------------------------

if __name__ == "__main__":
    # Quick shape check with random data, no real dataset needed.
    import tempfile

    torch.manual_seed(0)
    net = TitanicNet(input_dim=17)
    print(net)
    print(f"parameters: {sum(p.numel() for p in net.parameters()):,}")

    dummy = torch.randn(8, 17)
    out = net(dummy)
    print(f"forward output shape: {tuple(out.shape)}  <- expected (8, 1)")

    probs = predict_proba(net, dummy)
    print(f"probs shape: {probs.shape}, range: [{probs.min():.3f}, {probs.max():.3f}]")
    assert probs.shape == (8,), "predict_proba should return a 1-D array"
    assert ((probs >= 0) & (probs <= 1)).all(), "probabilities must lie in [0, 1]"

    # eval() must make repeated calls deterministic
    again = predict_proba(net, dummy)
    print(f"deterministic in eval mode: {np.allclose(probs, again)}  <- must be True")

    tmp = Path(tempfile.gettempdir()) / "model_smoketest.pt"
    save_model(net, tmp)
    reloaded = load_model(tmp, input_dim=17)
    print(f"save/load preserves output: "
          f"{np.allclose(probs, predict_proba(reloaded, dummy))}  <- must be True")
    tmp.unlink()
