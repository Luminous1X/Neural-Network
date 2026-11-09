"""Layers and the Module container that tracks their parameters."""
from __future__ import annotations

import math

import numpy as np

from . import functional as F
from .tensor import Tensor, get_default_dtype, get_rng

__all__ = [
    "Parameter",
    "Module",
    "Linear",
    "Conv2d",
    "MaxPool2d",
    "ReLU",
    "Tanh",
    "Flatten",
    "Dropout",
    "Sequential",
]


class Parameter(Tensor):
    """A tensor that a Module should train."""

    __slots__ = ()

    def __init__(self, data):
        super().__init__(data, requires_grad=True)


def _kaiming_uniform(shape: tuple, fan_in: int) -> np.ndarray:
    """He initialisation: keeps activation variance stable through ReLU layers."""
    bound = math.sqrt(6.0 / fan_in)
    return get_rng().uniform(-bound, bound, size=shape).astype(get_default_dtype())


class Module:
    """Base class. Parameters and sub-modules are found by inspecting attributes."""

    training = True

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def named_children(self):
        for name, value in vars(self).items():
            if isinstance(value, Module):
                yield name, value
            elif isinstance(value, (list, tuple)):
                for i, item in enumerate(value):
                    if isinstance(item, Module):
                        yield f"{name}.{i}", item

    def named_parameters(self, prefix: str = ""):
        for name, value in vars(self).items():
            if isinstance(value, Parameter):
                yield prefix + name, value
        for name, child in self.named_children():
            yield from child.named_parameters(prefix + name + ".")

    def parameters(self) -> list:
        return [p for _, p in self.named_parameters()]

    def num_parameters(self) -> int:
        return sum(p.size for p in self.parameters())

    def zero_grad(self) -> None:
        for p in self.parameters():
            p.grad = None

    def train(self, mode: bool = True) -> "Module":
        self.training = mode
        for _, child in self.named_children():
            child.train(mode)
        return self

    def eval(self) -> "Module":
        return self.train(False)

    def state_dict(self) -> dict:
        return {name: p.data.copy() for name, p in self.named_parameters()}

    def load_state_dict(self, state: dict) -> None:
        own = dict(self.named_parameters())
        missing = sorted(set(own) - set(state))
        unexpected = sorted(set(state) - set(own))
        if missing or unexpected:
            raise KeyError(f"state dict mismatch; missing={missing}, unexpected={unexpected}")
        for name, p in own.items():
            value = np.asarray(state[name])
            if value.shape != p.shape:
                raise ValueError(f"shape mismatch for {name}: {value.shape} vs {p.shape}")
            p.data = value.astype(p.dtype, copy=True)


class Linear(Module):
    """y = x @ W.T + b"""

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        self.weight = Parameter(_kaiming_uniform((out_features, in_features), in_features))
        self.bias = Parameter(np.zeros(out_features, dtype=get_default_dtype())) if bias else None

    def forward(self, x: Tensor) -> Tensor:
        out = x @ self.weight.T
        return out if self.bias is None else out + self.bias


class Conv2d(Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        bias: bool = True,
    ):
        shape = (out_channels, in_channels, kernel_size, kernel_size)
        self.weight = Parameter(_kaiming_uniform(shape, in_channels * kernel_size * kernel_size))
        self.bias = Parameter(np.zeros(out_channels, dtype=get_default_dtype())) if bias else None
        self.stride = stride
        self.padding = padding

    def forward(self, x: Tensor) -> Tensor:
        return F.conv2d(x, self.weight, self.bias, stride=self.stride, padding=self.padding)


class MaxPool2d(Module):
    def __init__(self, kernel_size: int = 2):
        self.kernel_size = kernel_size

    def forward(self, x: Tensor) -> Tensor:
        return F.max_pool2d(x, self.kernel_size)


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        return x.relu()


class Tanh(Module):
    def forward(self, x: Tensor) -> Tensor:
        return x.tanh()


class Flatten(Module):
    def forward(self, x: Tensor) -> Tensor:
        return x.flatten(1)


class Dropout(Module):
    def __init__(self, p: float = 0.5):
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        return F.dropout(x, self.p, self.training)


class Sequential(Module):
    def __init__(self, *layers: Module):
        self.layers = list(layers)

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers:
            x = layer(x)
        return x
