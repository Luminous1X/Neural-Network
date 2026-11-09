"""A small reverse-mode automatic differentiation engine on top of NumPy.

Every `Tensor` wraps an ndarray. Operations on tensors build a graph: each
result remembers its parents and a function that maps the gradient flowing
into the result onto gradients for those parents. `Tensor.backward()` walks
that graph in reverse topological order and applies the chain rule.

NumPy does the arithmetic. Everything about differentiation lives here.
"""
from __future__ import annotations

import contextlib

import numpy as np

__all__ = [
    "Tensor",
    "no_grad",
    "is_grad_enabled",
    "get_default_dtype",
    "set_default_dtype",
    "default_dtype",
    "manual_seed",
    "get_rng",
]


class _State:
    grad_enabled = True
    dtype = np.float32
    rng = np.random.default_rng(0)


def is_grad_enabled() -> bool:
    return _State.grad_enabled


@contextlib.contextmanager
def no_grad():
    """Inside this block operations do not record a graph (used for inference)."""
    previous = _State.grad_enabled
    _State.grad_enabled = False
    try:
        yield
    finally:
        _State.grad_enabled = previous


def get_default_dtype():
    return _State.dtype


def set_default_dtype(dtype) -> None:
    _State.dtype = np.dtype(dtype).type


@contextlib.contextmanager
def default_dtype(dtype):
    """Temporarily switch the dtype used for new parameters and Python data."""
    previous = _State.dtype
    set_default_dtype(dtype)
    try:
        yield
    finally:
        _State.dtype = previous


def manual_seed(seed: int) -> None:
    """Seed the generator used for weight initialisation and dropout."""
    _State.rng = np.random.default_rng(seed)


def get_rng() -> np.random.Generator:
    return _State.rng


def _unbroadcast(grad: np.ndarray, shape: tuple) -> np.ndarray:
    """Sum `grad` back down to `shape`, undoing NumPy broadcasting.

    If an operand was broadcast up to produce the output, each of its elements
    contributed to several output elements, so its gradient is the sum of the
    gradients of all of them.
    """
    if grad.shape == shape:
        return grad
    extra = grad.ndim - len(shape)
    if extra > 0:
        grad = grad.sum(axis=tuple(range(extra)))
    stretched = tuple(i for i, n in enumerate(shape) if n == 1 and grad.shape[i] != 1)
    if stretched:
        grad = grad.sum(axis=stretched, keepdims=True)
    return grad


class Tensor:
    """An n-dimensional array that can track gradients."""

    __slots__ = ("data", "grad", "requires_grad", "_parents", "_backward")

    # Make `ndarray <op> Tensor` defer to Tensor's reflected operators.
    __array_ufunc__ = None

    def __init__(self, data, requires_grad: bool = False):
        if isinstance(data, Tensor):
            data = data.data
        if isinstance(data, (np.ndarray, np.generic)) and data.dtype.kind == "f":
            data = np.asarray(data)
        else:
            data = np.asarray(data, dtype=_State.dtype)
        self.data = data
        self.grad = None
        self.requires_grad = bool(requires_grad)
        self._parents = ()
        self._backward = None

    # ------------------------------------------------------------------ basics

    @property
    def shape(self) -> tuple:
        return self.data.shape

    @property
    def ndim(self) -> int:
        return self.data.ndim

    @property
    def size(self) -> int:
        return self.data.size

    @property
    def dtype(self):
        return self.data.dtype

    def __len__(self) -> int:
        return len(self.data)

    def __repr__(self) -> str:
        suffix = ", requires_grad=True" if self.requires_grad else ""
        return f"Tensor({np.array2string(self.data, precision=4, threshold=20)}{suffix})"

    def item(self) -> float:
        return float(self.data)

    def numpy(self) -> np.ndarray:
        return self.data

    def detach(self) -> "Tensor":
        """A tensor sharing this data but cut off from the graph."""
        return Tensor(self.data)

    def zero_grad(self) -> None:
        self.grad = None

    # ---------------------------------------------------------------- backward

    def backward(self, grad=None) -> None:
        """Accumulate d(self)/d(leaf) into `.grad` of every leaf that requires it."""
        if not self.requires_grad:
            raise RuntimeError("backward() called on a tensor that does not require grad")
        if grad is None:
            if self.data.size != 1:
                raise RuntimeError("backward() without an explicit grad needs a scalar output")
            grad = np.ones_like(self.data)
        else:
            grad = np.asarray(grad, dtype=self.data.dtype)
            if grad.shape != self.data.shape:
                raise ValueError(f"grad shape {grad.shape} does not match tensor shape {self.data.shape}")

        # Post-order depth-first walk: every node lands after all of its inputs.
        order = []
        seen = set()
        stack = [(self, False)]
        while stack:
            node, expanded = stack.pop()
            if expanded:
                order.append(node)
                continue
            if id(node) in seen:
                continue
            seen.add(id(node))
            stack.append((node, True))
            for parent in node._parents:
                if parent.requires_grad and id(parent) not in seen:
                    stack.append((parent, False))

        # Reverse it, so a node is handled only once every consumer has reported.
        pending = {id(self): grad}
        for node in reversed(order):
            g = pending.pop(id(node), None)
            if g is None:
                continue
            if node._backward is None:  # a leaf: this is where gradients end up
                node.grad = np.array(g) if node.grad is None else node.grad + g
                continue
            for parent, parent_grad in zip(node._parents, node._backward(g)):
                if parent_grad is None or not parent.requires_grad:
                    continue
                key = id(parent)
                if key in pending:
                    pending[key] = pending[key] + parent_grad
                else:
                    pending[key] = parent_grad

    # -------------------------------------------------------------- arithmetic

    def __add__(self, other):
        a, b = self, _wrap(other, self)

        def backward(g):
            return (
                _unbroadcast(g, a.shape) if a.requires_grad else None,
                _unbroadcast(g, b.shape) if b.requires_grad else None,
            )

        return _op(a.data + b.data, (a, b), backward)

    __radd__ = __add__

    def __sub__(self, other):
        a, b = self, _wrap(other, self)

        def backward(g):
            return (
                _unbroadcast(g, a.shape) if a.requires_grad else None,
                _unbroadcast(-g, b.shape) if b.requires_grad else None,
            )

        return _op(a.data - b.data, (a, b), backward)

    def __rsub__(self, other):
        return _wrap(other, self) - self

    def __mul__(self, other):
        a, b = self, _wrap(other, self)

        def backward(g):
            return (
                _unbroadcast(g * b.data, a.shape) if a.requires_grad else None,
                _unbroadcast(g * a.data, b.shape) if b.requires_grad else None,
            )

        return _op(a.data * b.data, (a, b), backward)

    __rmul__ = __mul__

    def __truediv__(self, other):
        a, b = self, _wrap(other, self)

        def backward(g):
            return (
                _unbroadcast(g / b.data, a.shape) if a.requires_grad else None,
                _unbroadcast(-g * a.data / (b.data * b.data), b.shape) if b.requires_grad else None,
            )

        return _op(a.data / b.data, (a, b), backward)

    def __rtruediv__(self, other):
        return _wrap(other, self) / self

    def __neg__(self):
        return _op(-self.data, (self,), lambda g: (-g,))

    def __pow__(self, exponent):
        if not isinstance(exponent, (int, float)):
            raise TypeError("only constant Python exponents are supported")
        a = self

        def backward(g):
            return (g * exponent * a.data ** (exponent - 1),)

        return _op(a.data ** exponent, (a,), backward)

    def __matmul__(self, other):
        a, b = self, _wrap(other, self)
        if a.ndim < 2 or b.ndim < 2:
            raise ValueError("matmul needs operands with at least 2 dimensions; reshape vectors first")

        def backward(g):
            return (
                _unbroadcast(g @ np.swapaxes(b.data, -1, -2), a.shape) if a.requires_grad else None,
                _unbroadcast(np.swapaxes(a.data, -1, -2) @ g, b.shape) if b.requires_grad else None,
            )

        return _op(a.data @ b.data, (a, b), backward)

    def __rmatmul__(self, other):
        return _wrap(other, self) @ self

    # ------------------------------------------------------- elementwise maths

    def exp(self):
        out = np.exp(self.data)
        return _op(out, (self,), lambda g: (g * out,))

    def log(self):
        a = self
        return _op(np.log(a.data), (a,), lambda g: (g / a.data,))

    def sqrt(self):
        out = np.sqrt(self.data)
        return _op(out, (self,), lambda g: (g / (2 * out),))

    def tanh(self):
        out = np.tanh(self.data)
        return _op(out, (self,), lambda g: (g * (1 - out * out),))

    def sigmoid(self):
        out = 1 / (1 + np.exp(-self.data))
        return _op(out, (self,), lambda g: (g * out * (1 - out),))

    def relu(self):
        mask = self.data > 0
        return _op(self.data * mask, (self,), lambda g: (g * mask,))

    # -------------------------------------------------------------- reductions

    def sum(self, axis=None, keepdims: bool = False):
        a = self

        def backward(g):
            if axis is not None and not keepdims:
                g = np.expand_dims(g, axis)
            return (np.broadcast_to(g, a.shape),)

        return _op(a.data.sum(axis=axis, keepdims=keepdims), (a,), backward)

    def mean(self, axis=None, keepdims: bool = False):
        total = self.sum(axis=axis, keepdims=keepdims)
        count = self.data.size // max(total.data.size, 1)
        return total * (1.0 / count)

    # ------------------------------------------------------------------ shapes

    def reshape(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        a = self
        return _op(a.data.reshape(shape), (a,), lambda g: (g.reshape(a.shape),))

    def flatten(self, start_dim: int = 1):
        return self.reshape(self.shape[:start_dim] + (-1,))

    def transpose(self, *axes):
        if len(axes) == 1 and isinstance(axes[0], (tuple, list)):
            axes = tuple(axes[0])
        if not axes:
            axes = tuple(reversed(range(self.ndim)))
        inverse = tuple(np.argsort(axes))
        return _op(self.data.transpose(axes), (self,), lambda g: (g.transpose(inverse),))

    @property
    def T(self):
        return self.transpose()

    def __getitem__(self, index):
        a = self

        def backward(g):
            out = np.zeros_like(a.data)
            np.add.at(out, index, g)  # add.at so repeated indices accumulate
            return (out,)

        return _op(a.data[index], (a,), backward)


def _wrap(value, like: Tensor) -> Tensor:
    """Turn a Python number or ndarray into a constant Tensor matching `like`'s dtype."""
    if isinstance(value, Tensor):
        return value
    return Tensor(np.asarray(value, dtype=like.data.dtype))


def _op(data, parents: tuple, backward) -> Tensor:
    """Build the result of an operation.

    `backward(g)` receives dLoss/d(result) and returns one gradient per parent
    (or None for a parent that does not need one). The graph is recorded only
    when gradients are enabled and at least one parent requires them.
    """
    out = Tensor(data)
    if _State.grad_enabled and any(p.requires_grad for p in parents):
        out.requires_grad = True
        out._parents = parents
        out._backward = backward
    return out
