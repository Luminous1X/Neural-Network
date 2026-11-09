"""Check analytic gradients against finite differences."""
from __future__ import annotations

import numpy as np

from .tensor import Tensor

__all__ = ["numerical_grad", "gradcheck"]


def numerical_grad(fn, x: Tensor, eps: float = 1e-6) -> np.ndarray:
    """Central-difference estimate of d fn() / d x, perturbing x in place.

    `fn` takes no arguments and returns a scalar Tensor. Use float64 data.
    """
    grad = np.zeros_like(x.data)
    flat = x.data.reshape(-1)  # a view: writes go through to x.data
    out = grad.reshape(-1)
    for i in range(flat.size):
        original = flat[i]
        flat[i] = original + eps
        plus = float(fn().data)
        flat[i] = original - eps
        minus = float(fn().data)
        flat[i] = original
        out[i] = (plus - minus) / (2 * eps)
    return grad


def gradcheck(fn, inputs, eps: float = 1e-6, tol: float = 1e-6) -> float:
    """Compare backward() against finite differences for every tensor in `inputs`.

    Returns the worst relative error and raises AssertionError if it exceeds `tol`.
    """
    for x in inputs:
        if x.data.dtype != np.float64:
            raise TypeError("gradcheck needs float64 inputs")
        if not x.data.flags.c_contiguous:
            raise ValueError("gradcheck needs C-contiguous inputs")
        x.grad = None
    fn().backward()

    worst = 0.0
    for i, x in enumerate(inputs):
        analytic = np.zeros_like(x.data) if x.grad is None else x.grad
        numeric = numerical_grad(fn, x, eps)
        scale = max(1.0, float(np.abs(analytic).max()), float(np.abs(numeric).max()))
        error = float(np.abs(analytic - numeric).max()) / scale
        worst = max(worst, error)
        if error > tol:
            raise AssertionError(f"gradient mismatch for input {i}: relative error {error:.3e} > {tol:.1e}")
    return worst
