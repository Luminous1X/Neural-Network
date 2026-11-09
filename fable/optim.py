"""Optimisers and a learning-rate schedule."""
from __future__ import annotations

import math

import numpy as np

__all__ = ["SGD", "Adam", "WarmupCosine"]


class Optimizer:
    def __init__(self, params, lr: float):
        self.params = list(params)
        self.lr = lr

    def zero_grad(self) -> None:
        for p in self.params:
            p.grad = None

    def step(self) -> None:
        raise NotImplementedError


class SGD(Optimizer):
    """Stochastic gradient descent with optional (Nesterov) momentum and L2 decay."""

    def __init__(self, params, lr: float = 0.01, momentum: float = 0.0, nesterov: bool = False, weight_decay: float = 0.0):
        super().__init__(params, lr)
        self.momentum = momentum
        self.nesterov = nesterov
        self.weight_decay = weight_decay
        self.velocity = [np.zeros_like(p.data) for p in self.params]

    def step(self) -> None:
        for p, v in zip(self.params, self.velocity):
            if p.grad is None:
                continue
            g = p.grad
            if self.weight_decay and p.ndim > 1:
                g = g + self.weight_decay * p.data
            if self.momentum:
                v *= self.momentum
                v += g
                g = g + self.momentum * v if self.nesterov else v
            p.data -= self.lr * g


class Adam(Optimizer):
    """Adam with decoupled weight decay (AdamW).

    Weight decay is applied to weight matrices and kernels only (ndim > 1),
    never to biases.
    """

    def __init__(self, params, lr: float = 1e-3, betas=(0.9, 0.999), eps: float = 1e-8, weight_decay: float = 0.0):
        super().__init__(params, lr)
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0
        self.m = [np.zeros_like(p.data) for p in self.params]
        self.v = [np.zeros_like(p.data) for p in self.params]

    def step(self) -> None:
        self.t += 1
        correction1 = 1.0 - self.beta1**self.t
        correction2 = 1.0 - self.beta2**self.t
        for p, m, v in zip(self.params, self.m, self.v):
            if p.grad is None:
                continue
            g = p.grad
            m *= self.beta1
            m += (1.0 - self.beta1) * g
            v *= self.beta2
            v += (1.0 - self.beta2) * g * g
            if self.weight_decay and p.ndim > 1:
                p.data *= 1.0 - self.lr * self.weight_decay
            p.data -= self.lr * (m / correction1) / (np.sqrt(v / correction2) + self.eps)


class WarmupCosine:
    """Linear warm-up to `peak_lr`, then cosine decay to `final_lr`. Call step() once per batch."""

    def __init__(self, optimizer: Optimizer, peak_lr: float, total_steps: int, warmup_steps: int = 0, final_lr: float = 0.0):
        self.optimizer = optimizer
        self.peak_lr = peak_lr
        self.total_steps = max(total_steps, 1)
        self.warmup_steps = warmup_steps
        self.final_lr = final_lr
        self.step_count = 0
        optimizer.lr = self.lr_at(0)

    def lr_at(self, step: int) -> float:
        if step < self.warmup_steps:
            return self.peak_lr * (step + 1) / self.warmup_steps
        span = max(self.total_steps - self.warmup_steps, 1)
        progress = min((step - self.warmup_steps) / span, 1.0)
        return self.final_lr + 0.5 * (self.peak_lr - self.final_lr) * (1.0 + math.cos(math.pi * progress))

    def step(self) -> float:
        self.step_count += 1
        self.optimizer.lr = self.lr_at(self.step_count)
        return self.optimizer.lr
