"""Gradient checks for the core autograd engine."""
import unittest

import numpy as np

from fable import Tensor, no_grad
from fable.gradcheck import gradcheck


def leaf(rng, *shape, positive=False):
    data = rng.standard_normal(shape)
    if positive:
        data = np.abs(data) + 0.5
    return Tensor(data, requires_grad=True)


class ArithmeticGradients(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(0)

    def test_add_sub_mul_div(self):
        a, b = leaf(self.rng, 3, 4), leaf(self.rng, 3, 4, positive=True)
        gradcheck(lambda: (a + b).sum(), [a, b])
        gradcheck(lambda: (a - b).sum(), [a, b])
        gradcheck(lambda: (a * b).sum(), [a, b])
        gradcheck(lambda: (a / b).sum(), [a, b])

    def test_broadcasting(self):
        a = leaf(self.rng, 2, 3, 4)
        row = leaf(self.rng, 4)
        col = leaf(self.rng, 3, 1, positive=True)
        scalar = leaf(self.rng)
        gradcheck(lambda: ((a + row) * col / (scalar * scalar + 1.0)).sum(), [a, row, col, scalar])
        gradcheck(lambda: (row - a).sum(), [a, row])
        gradcheck(lambda: (col / a.exp()).sum(), [a, col])

    def test_python_scalars_and_reflected_ops(self):
        a = leaf(self.rng, 5, positive=True)
        gradcheck(lambda: (2.0 * a + 1.0 - a / 3.0).sum(), [a])
        gradcheck(lambda: (1.0 - a).sum() + (2.0 / a).sum(), [a])
        gradcheck(lambda: (-a).sum(), [a])

    def test_ndarray_on_the_left(self):
        a = leaf(self.rng, 3)
        k = np.array([1.0, 2.0, 3.0])
        out = k * a
        self.assertIsInstance(out, Tensor)
        gradcheck(lambda: (k * a + k - a).sum(), [a])

    def test_pow_and_sqrt(self):
        a = leaf(self.rng, 4, positive=True)
        gradcheck(lambda: (a**3).sum(), [a])
        gradcheck(lambda: (a**-1.5).sum(), [a])
        gradcheck(lambda: a.sqrt().sum(), [a])

    def test_matmul(self):
        a, b = leaf(self.rng, 3, 4), leaf(self.rng, 4, 5)
        gradcheck(lambda: (a @ b).sum(), [a, b])
        weights = Tensor(self.rng.standard_normal((3, 5)))
        gradcheck(lambda: ((a @ b) * weights).sum(), [a, b])

    def test_batched_matmul(self):
        a, b = leaf(self.rng, 2, 3, 4), leaf(self.rng, 4, 5)
        weights = Tensor(self.rng.standard_normal((2, 3, 5)))
        gradcheck(lambda: ((a @ b) * weights).sum(), [a, b])

    def test_matmul_rejects_vectors(self):
        with self.assertRaises(ValueError):
            leaf(self.rng, 3) @ leaf(self.rng, 3)


class ElementwiseGradients(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(1)

    def test_exp_log_tanh_sigmoid(self):
        a = leaf(self.rng, 3, 4)
        p = leaf(self.rng, 3, 4, positive=True)
        gradcheck(lambda: a.exp().sum(), [a])
        gradcheck(lambda: p.log().sum(), [p])
        gradcheck(lambda: a.tanh().sum(), [a])
        gradcheck(lambda: a.sigmoid().sum(), [a])

    def test_relu(self):
        a = leaf(self.rng, 6, 5)
        a.data[np.abs(a.data) < 1e-3] = 0.5  # stay clear of the kink at zero
        gradcheck(lambda: (a.relu() * a).sum(), [a])
        np.testing.assert_array_equal(Tensor([-1.0, 0.0, 2.0]).relu().data, [0.0, 0.0, 2.0])


class ReductionAndShapeGradients(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(2)

    def test_sum_variants(self):
        a = leaf(self.rng, 2, 3, 4)
        w = Tensor(self.rng.standard_normal((2, 4)))
        gradcheck(lambda: (a.sum(axis=1) * w).sum(), [a])
        gradcheck(lambda: (a.sum(axis=(0, 2), keepdims=True) ** 2).sum(), [a])
        gradcheck(lambda: (a.sum(axis=-1) ** 2).sum(), [a])
        gradcheck(lambda: a.sum() * a.sum(), [a])

    def test_mean(self):
        a = leaf(self.rng, 3, 5)
        np.testing.assert_allclose(a.mean().data, a.data.mean())
        np.testing.assert_allclose(a.mean(axis=0).data, a.data.mean(axis=0))
        gradcheck(lambda: (a.mean(axis=0) ** 2).sum() + a.mean(), [a])

    def test_reshape_transpose_flatten(self):
        a = leaf(self.rng, 2, 3, 4)
        w = Tensor(self.rng.standard_normal((4, 6)))
        gradcheck(lambda: (a.reshape(6, 4).T * w).sum(), [a])
        gradcheck(lambda: (a.transpose(2, 0, 1).reshape(4, 6) * w).sum(), [a])
        gradcheck(lambda: (a.flatten(1) ** 2).sum(), [a])

    def test_getitem(self):
        a = leaf(self.rng, 4, 5)
        gradcheck(lambda: (a[1:3, ::2] ** 2).sum(), [a])
        rows, cols = np.array([0, 1, 1, 3]), np.array([4, 2, 2, 0])  # (1, 2) is picked twice
        gradcheck(lambda: (a[rows, cols] ** 2).sum(), [a])


class GraphBehaviour(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(3)

    def test_node_used_many_times(self):
        a = leaf(self.rng, 3)
        gradcheck(lambda: (a * a + a * a * a + a).sum(), [a])

    def test_diamond_graph(self):
        a = leaf(self.rng, 3)

        def fn():
            shared = a.tanh()
            left, right = shared * 2.0, shared.exp()
            return (left * right + shared).sum()

        gradcheck(fn, [a])

    def test_deep_chain_does_not_recurse(self):
        a = Tensor(np.array(1.0), requires_grad=True)
        out = a
        for _ in range(5000):  # far deeper than Python's recursion limit
            out = out * 1.0
        out.backward()
        self.assertEqual(float(a.grad), 1.0)

    def test_known_value(self):
        x = Tensor(np.array(3.0), requires_grad=True)
        y = Tensor(np.array(-2.0), requires_grad=True)
        (x * x * y + y).backward()
        self.assertAlmostEqual(float(x.grad), 2 * 3.0 * -2.0)
        self.assertAlmostEqual(float(y.grad), 3.0 * 3.0 + 1.0)

    def test_gradients_accumulate_until_zeroed(self):
        a = Tensor(np.array([1.0, 2.0]), requires_grad=True)
        (a * 3.0).sum().backward()
        (a * 3.0).sum().backward()
        np.testing.assert_allclose(a.grad, [6.0, 6.0])
        a.zero_grad()
        self.assertIsNone(a.grad)

    def test_leaf_gradients_do_not_alias(self):
        a = Tensor(np.ones(3), requires_grad=True)
        b = Tensor(np.ones(3), requires_grad=True)
        (a + b).sum().backward()
        a.grad *= 5.0
        np.testing.assert_allclose(b.grad, [1.0, 1.0, 1.0])

    def test_no_grad_records_nothing(self):
        a = leaf(self.rng, 3)
        with no_grad():
            out = (a * 2.0).sum()
        self.assertFalse(out.requires_grad)
        self.assertEqual(out._parents, ())
        self.assertTrue((a * 2.0).requires_grad)

    def test_constants_get_no_gradient(self):
        a = leaf(self.rng, 3)
        c = Tensor(self.rng.standard_normal(3))
        (a * c).sum().backward()
        self.assertIsNone(c.grad)
        np.testing.assert_allclose(a.grad, c.data)

    def test_backward_requires_scalar(self):
        a = leaf(self.rng, 3)
        with self.assertRaises(RuntimeError):
            (a * 2.0).backward()
        (a * 2.0).backward(np.ones(3))
        np.testing.assert_allclose(a.grad, [2.0, 2.0, 2.0])

    def test_dtype_is_preserved(self):
        a = Tensor(np.ones((2, 2), dtype=np.float32), requires_grad=True)
        out = ((a * 2.0 + 1.0) / 3.0 - 0.5).exp().mean()
        self.assertEqual(out.dtype, np.float32)
        out.backward()
        self.assertEqual(a.grad.dtype, np.float32)


if __name__ == "__main__":
    unittest.main()
