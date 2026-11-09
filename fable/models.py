"""The network this project trains: a small convolutional digit classifier."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import functional as F
from .nn import Conv2d, Dropout, Linear, Module
from .tensor import Tensor

__all__ = ["FableNet", "save_checkpoint", "load_checkpoint"]


class FableNet(Module):
    """LeNet-style CNN for 28x28 greyscale digits.

        input            1 x 28 x 28
        conv 5x5 + ReLU  c1 x 24 x 24
        max-pool 2x2     c1 x 12 x 12
        conv 5x5 + ReLU  c2 x  8 x  8
        max-pool 2x2     c2 x  4 x  4
        flatten          c2 * 16
        linear + ReLU    hidden
        dropout
        linear           10 logits
    """

    def __init__(self, c1: int = 16, c2: int = 32, hidden: int = 128, dropout: float = 0.3):
        self.config = {"c1": c1, "c2": c2, "hidden": hidden, "dropout": dropout}
        self.conv1 = Conv2d(1, c1, kernel_size=5)
        self.conv2 = Conv2d(c1, c2, kernel_size=5)
        self.fc1 = Linear(c2 * 4 * 4, hidden)
        self.drop = Dropout(dropout)
        self.fc2 = Linear(hidden, 10)

    def forward(self, x: Tensor) -> Tensor:
        x = F.max_pool2d(self.conv1(x).relu(), 2)
        x = F.max_pool2d(self.conv2(x).relu(), 2)
        x = x.flatten(1)
        x = self.drop(self.fc1(x).relu())
        return self.fc2(x)


def save_checkpoint(model: FableNet, path, extra: dict | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = {"config": model.config, **(extra or {})}
    np.savez(path, __meta__=np.array(json.dumps(meta)), **model.state_dict())


def load_checkpoint(path):
    """Returns (model in eval mode, metadata dict)."""
    with np.load(Path(path), allow_pickle=False) as archive:
        meta = json.loads(str(archive["__meta__"]))
        state = {name: archive[name] for name in archive.files if name != "__meta__"}
    model = FableNet(**meta["config"])
    model.load_state_dict(state)
    return model.eval(), meta
