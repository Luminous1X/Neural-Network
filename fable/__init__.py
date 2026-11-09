"""fable: a tiny deep-learning framework written from scratch on NumPy."""
from . import data, functional, nn, optim
from .tensor import (
    Tensor,
    default_dtype,
    get_default_dtype,
    get_rng,
    is_grad_enabled,
    manual_seed,
    no_grad,
    set_default_dtype,
)

__all__ = [
    "Tensor",
    "no_grad",
    "is_grad_enabled",
    "manual_seed",
    "get_rng",
    "get_default_dtype",
    "set_default_dtype",
    "default_dtype",
    "nn",
    "optim",
    "functional",
    "data",
]
