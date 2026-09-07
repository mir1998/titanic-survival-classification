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
        # TODO (Miriam):
        #  Build the stack. For each hidden dim, in order:
        #      Linear(prev_dim, h) -> ReLU -> Dropout(dropout)
        #  then a final Linear(last_hidden, 1).
        #
        #  Two ways to do it, both fine:
        #    (a) collect the layers in a list and wrap in nn.Sequential(*layers)
        #    (b) write them out explicitly as self.fc1, self.fc2, ...
        #  (a) generalises to any hidden_dims without editing forward().
        #
        #  NOTE: no Sigmoid at the end. The final layer outputs a raw logit.
        #  See forward() for why.
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a forward pass.

        Parameters
        ----------
        x:
            Float tensor of shape ``(batch_size, input_dim)``.

        Returns
        -------
        torch.Tensor
            Raw logits of shape ``(batch_size, 1)``. **Not** probabilities.

        Notes
        -----
        Returning logits rather than probabilities is deliberate.
        ``nn.BCEWithLogitsLoss`` combines the sigmoid and the binary
        cross-entropy into one operation that is numerically stable via the
        log-sum-exp trick. Doing ``Sigmoid()`` followed by ``nn.BCELoss()``
        is mathematically identical but can overflow to ``inf``/``NaN`` once
        a logit grows large, because the sigmoid saturates to exactly 0.0 or
        1.0 in float32 and the loss then takes ``log(0)``.
        """
        # TODO (Miriam): pass x through the stack and return the result.
        raise NotImplementedError


@torch.no_grad()
def predict_proba(model: TitanicNet, X: np.ndarray | torch.Tensor) -> np.ndarray:
    """Return survival probabilities for a feature matrix.

    This is the single place where the sigmoid is applied, so training and
    inference cannot drift apart.

    Parameters
    ----------
    model:
        The network. Set to eval mode internally.
    X:
        Preprocessed feature matrix of shape ``(n_samples, input_dim)``.

    Returns
    -------
    numpy.ndarray
        1-D array of probabilities in ``[0, 1]``, length ``n_samples``.
    """
    # TODO (Miriam):
    #  1. model.eval()   <- switches Dropout OFF. Without this the same input
    #     returns a different answer on every call, because dropout keeps
    #     randomly zeroing units. Silent and very confusing to debug.
    #  2. Convert X to a float32 tensor if it is not one already.
    #  3. logits = model(X)
    #  4. torch.sigmoid(logits) -> squeeze to 1-D -> .numpy()
    #
    #  The @torch.no_grad() decorator above already disables gradient
    #  tracking for the whole function, so no need to wrap anything.
    raise NotImplementedError


def save_model(model: TitanicNet, path: Path | str) -> Path:
    """Persist model weights to disk.

    Saves ``state_dict()`` (the tensors only) rather than the model object.
    Pickling the object itself embeds a reference to this module's import
    path, so it breaks if the file is later moved or renamed. A state dict
    is just weights and reloads into any matching architecture.
    """
    # TODO (Miriam): mkdir parent, torch.save(model.state_dict(), path), return Path.
    raise NotImplementedError


def load_model(
    path: Path | str,
    input_dim: int,
    hidden_dims: tuple[int, ...] = DEFAULT_HIDDEN_DIMS,
    dropout: float = DEFAULT_DROPOUT,
) -> TitanicNet:
    """Rebuild the network and load saved weights into it.

    Because a state dict holds no architecture information, the caller must
    supply ``input_dim`` (and the hidden sizes, if non-default). ``train.py``
    writes these into ``metadata.json`` for exactly this reason - the
    Streamlit app reads them back rather than guessing.

    Raises
    ------
    FileNotFoundError
        Pointing the user at ``python train.py``.
    """
    # TODO (Miriam):
    #  1. Path(path); if not exists -> FileNotFoundError mentioning train.py
    #  2. model = TitanicNet(input_dim, hidden_dims, dropout)
    #  3. model.load_state_dict(torch.load(path, map_location="cpu"))
    #     map_location="cpu" keeps it working on a machine without a GPU.
    #  4. model.eval()  <- load for inference, so leave it in eval mode
    #  5. return model
    raise NotImplementedError


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
