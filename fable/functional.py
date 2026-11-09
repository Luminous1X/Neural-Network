"""Neural-network operations that are implemented as single graph nodes.

These could be composed from the primitive ops in `tensor.py`, but writing the
forward and backward passes by hand is far faster (convolution, pooling) or
numerically safer (log-softmax, cross-entropy).
"""
from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from .tensor import Tensor, _op, get_rng

__all__ = ["conv2d", "max_pool2d", "log_softmax", "softmax", "cross_entropy", "dropout"]


def conv2d(x: Tensor, weight: Tensor, bias: Tensor | None = None, stride: int = 1, padding: int = 0) -> Tensor:
    """2-D cross-correlation.

    x:      (N, C, H, W)
    weight: (F, C, KH, KW)
    bias:   (F,) or None
    returns (N, F, OH, OW)

    Uses the im2col trick: every receptive field is unrolled into one row of a
    matrix, which turns the whole convolution into a single matrix product.
    """
    n, c, h, w = x.shape
    f, wc, kh, kw = weight.shape
    if wc != c:
        raise ValueError(f"input has {c} channels but weight expects {wc}")

    xp = x.data
    if padding:
        xp = np.pad(xp, ((0, 0), (0, 0), (padding, padding), (padding, padding)))
    hp, wp = xp.shape[2], xp.shape[3]
    oh = (hp - kh) // stride + 1
    ow = (wp - kw) // stride + 1

    # (N, C, OH, OW, KH, KW) view of every window, no copy yet.
    windows = sliding_window_view(xp, (kh, kw), axis=(2, 3))[:, :, ::stride, ::stride]
    # One row per output position, one column per (channel, ky, kx).
    cols = windows.transpose(0, 2, 3, 1, 4, 5).reshape(n * oh * ow, c * kh * kw)
    wmat = weight.data.reshape(f, c * kh * kw)

    out = cols @ wmat.T  # (N*OH*OW, F)
    if bias is not None:
        out += bias.data
    out = np.ascontiguousarray(out.reshape(n, oh, ow, f).transpose(0, 3, 1, 2))

    def backward(g):
        gmat = g.transpose(0, 2, 3, 1).reshape(n * oh * ow, f)
        # Written as (cols.T @ gmat).T rather than gmat.T @ cols: same result, but
        # OpenBLAS runs this operand layout ~100x faster for the shapes used here.
        grad_w = (cols.T @ gmat).T.reshape(weight.shape) if weight.requires_grad else None
        grad_b = gmat.sum(axis=0) if bias is not None and bias.requires_grad else None
        grad_x = None
        if x.requires_grad:
            # Gradient of every window, then scatter the windows back (col2im).
            gcols = (gmat @ wmat).reshape(n, oh, ow, c, kh, kw)
            gxp = np.zeros_like(xp)
            for i in range(kh):
                for j in range(kw):
                    gxp[:, :, i : i + stride * oh : stride, j : j + stride * ow : stride] += gcols[
                        :, :, :, :, i, j
                    ].transpose(0, 3, 1, 2)
            grad_x = gxp[:, :, padding : padding + h, padding : padding + w] if padding else gxp
        return (grad_x, grad_w) if bias is None else (grad_x, grad_w, grad_b)

    parents = (x, weight) if bias is None else (x, weight, bias)
    return _op(out, parents, backward)


def max_pool2d(x: Tensor, kernel_size: int = 2) -> Tensor:
    """Non-overlapping max pooling (stride equals kernel size)."""
    k = kernel_size
    n, c, h, w = x.shape
    if h % k or w % k:
        raise ValueError(f"spatial size {(h, w)} is not divisible by kernel size {k}")
    oh, ow = h // k, w // k

    # (N, C, OH, OW, K*K): each window flattened along the last axis.
    windows = x.data.reshape(n, c, oh, k, ow, k).transpose(0, 1, 2, 4, 3, 5).reshape(n, c, oh, ow, k * k)
    winner = windows.argmax(axis=-1)[..., None]
    out = np.take_along_axis(windows, winner, axis=-1)[..., 0]

    def backward(g):
        # Only the winning element of each window receives gradient.
        gw = np.zeros((n, c, oh, ow, k * k), dtype=g.dtype)
        np.put_along_axis(gw, winner, g[..., None], axis=-1)
        return (gw.reshape(n, c, oh, ow, k, k).transpose(0, 1, 2, 4, 3, 5).reshape(n, c, h, w),)

    return _op(out, (x,), backward)


def _log_softmax(z: np.ndarray, axis: int) -> np.ndarray:
    shifted = z - z.max(axis=axis, keepdims=True)  # subtract the max so exp() cannot overflow
    return shifted - np.log(np.exp(shifted).sum(axis=axis, keepdims=True))


def log_softmax(x: Tensor, axis: int = -1) -> Tensor:
    out = _log_softmax(x.data, axis)

    def backward(g):
        return (g - np.exp(out) * g.sum(axis=axis, keepdims=True),)

    return _op(out, (x,), backward)


def softmax(x: Tensor, axis: int = -1) -> Tensor:
    return log_softmax(x, axis).exp()


def cross_entropy(logits: Tensor, targets, label_smoothing: float = 0.0) -> Tensor:
    """Mean cross-entropy between `logits` (N, K) and integer class `targets` (N,).

    With label smoothing e, the target distribution puts 1 - e on the true
    class and spreads e evenly over all K classes.
    """
    targets = np.asarray(targets)
    n, k = logits.shape
    logp = _log_softmax(logits.data, axis=1)

    q = np.full((n, k), label_smoothing / k, dtype=logp.dtype)
    q[np.arange(n), targets] += 1.0 - label_smoothing
    loss = -(q * logp).sum() / n

    def backward(g):
        # d/dlogits of -sum(q * log_softmax(logits)) is simply softmax - q.
        return (g * (np.exp(logp) - q) / n,)

    return _op(loss, (logits,), backward)


def dropout(x: Tensor, p: float = 0.5, training: bool = True) -> Tensor:
    """Inverted dropout: zero each element with probability p, rescale the rest."""
    if not training or p == 0.0:
        return x
    if not 0.0 <= p < 1.0:
        raise ValueError("dropout probability must be in [0, 1)")
    keep = 1.0 - p
    mask = (get_rng().random(x.shape) < keep).astype(x.dtype) / keep
    return x * mask
