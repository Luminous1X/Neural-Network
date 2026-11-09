"""Convolution, pooling, softmax, loss and dropout: values and gradients."""
import unittest

import numpy as np

import fable
from fable import Tensor
from fable import functional as F
from fable.gradcheck import gradcheck


def naive_conv2d(x, w, b, stride, padding):
    """The definition of cross-correlation, written as plain loops."""
    n, c, h, wd = x.shape
    f, _, kh, kw = w.shape
    xp = np.pad(x, ((0, 0), (0, 0), (padding, padding), (padding, padding)))
    oh = (h + 2 * padding - kh) // stride + 1
    ow = (wd + 2 * padding - kw) // stride + 1
    out = np.zeros((n, f, oh, ow))
    for i in range(n):
        for j in range(f):
            for y in range(oh):
                for z in range(ow):
                    patch = xp[i, :, y * stride : y * stride + kh, z * stride : z * stride + kw]
                    out[i, j, y, z] = (patch * w[j]).sum() + (0.0 if b is None else b[j])
    return out


class Conv2d(unittest.TestCase):
    CASES = [  # (stride, padding)
        (1, 0),
        (1, 1),
        (2, 0),
        (2, 2),
    ]

    def setUp(self):
        self.rng = np.random.default_rng(0)

    def test_forward_matches_definition(self):
        x = self.rng.standard_normal((2, 3, 7, 6))
        w = self.rng.standard_normal((4, 3, 3, 2))
        b = self.rng.standard_normal(4)
        for stride, padding in self.CASES:
            with self.subTest(stride=stride, padding=padding):
                got = F.conv2d(Tensor(x), Tensor(w), Tensor(b), stride=stride, padding=padding).data
                np.testing.assert_allclose(got, naive_conv2d(x, w, b, stride, padding), atol=1e-12)
        got = F.conv2d(Tensor(x), Tensor(w)).data
        np.testing.assert_allclose(got, naive_conv2d(x, w, None, 1, 0), atol=1e-12)

    def test_gradients(self):
        for stride, padding in self.CASES:
            with self.subTest(stride=stride, padding=padding):
                x = Tensor(self.rng.standard_normal((2, 3, 6, 5)), requires_grad=True)
                w = Tensor(self.rng.standard_normal((4, 3, 3, 2)), requires_grad=True)
                b = Tensor(self.rng.standard_normal(4), requires_grad=True)
                probe = Tensor(self.rng.standard_normal(F.conv2d(x, w, b, stride, padding).shape))
                gradcheck(lambda: (F.conv2d(x, w, b, stride, padding) * probe).sum(), [x, w, b])

    def test_gradients_without_bias(self):
        x = Tensor(self.rng.standard_normal((1, 2, 5, 5)), requires_grad=True)
        w = Tensor(self.rng.standard_normal((3, 2, 3, 3)), requires_grad=True)
        gradcheck(lambda: (F.conv2d(x, w) ** 2).sum(), [x, w])

    def test_channel_mismatch(self):
        with self.assertRaises(ValueError):
            F.conv2d(Tensor(np.zeros((1, 2, 5, 5))), Tensor(np.zeros((3, 4, 3, 3))))


class MaxPool(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(1)

    def test_forward(self):
        x = np.arange(16.0).reshape(1, 1, 4, 4)
        np.testing.assert_array_equal(F.max_pool2d(Tensor(x), 2).data, [[[[5.0, 7.0], [13.0, 15.0]]]])

    def test_forward_matches_definition(self):
        x = self.rng.standard_normal((2, 3, 6, 9))
        got = F.max_pool2d(Tensor(x), 3).data
        for y in range(2):
            for z in range(3):
                expected = x[:, :, y * 3 : y * 3 + 3, z * 3 : z * 3 + 3].max(axis=(2, 3))
                np.testing.assert_array_equal(got[:, :, y, z], expected)

    def test_gradients(self):
        x = Tensor(self.rng.standard_normal((2, 3, 4, 6)), requires_grad=True)
        probe = Tensor(self.rng.standard_normal((2, 3, 2, 3)))
        gradcheck(lambda: (F.max_pool2d(x, 2) * probe).sum(), [x])

    def test_ties_send_gradient_to_one_element(self):
        x = Tensor(np.zeros((1, 1, 2, 2)), requires_grad=True)
        F.max_pool2d(x, 2).sum().backward()
        self.assertEqual(x.grad.sum(), 1.0)
        self.assertEqual(np.count_nonzero(x.grad), 1)

    def test_indivisible_size(self):
        with self.assertRaises(ValueError):
            F.max_pool2d(Tensor(np.zeros((1, 1, 5, 4))), 2)


class SoftmaxAndLoss(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(2)

    def test_softmax_values(self):
        z = self.rng.standard_normal((4, 6))
        expected = np.exp(z) / np.exp(z).sum(axis=1, keepdims=True)
        np.testing.assert_allclose(F.softmax(Tensor(z)).data, expected, atol=1e-12)
        np.testing.assert_allclose(F.log_softmax(Tensor(z)).data, np.log(expected), atol=1e-12)

    def test_log_softmax_is_stable(self):
        z = Tensor(np.array([[1000.0, 0.0, -1000.0]]))
        out = F.log_softmax(z).data
        self.assertTrue(np.all(np.isfinite(out)))
        np.testing.assert_allclose(out[0, 0], 0.0, atol=1e-12)

    def test_log_softmax_gradients(self):
        z = Tensor(self.rng.standard_normal((3, 5)), requires_grad=True)
        probe = Tensor(self.rng.standard_normal((3, 5)))
        gradcheck(lambda: (F.log_softmax(z) * probe).sum(), [z])
        gradcheck(lambda: (F.log_softmax(z, axis=0) * probe).sum(), [z])
        gradcheck(lambda: (F.softmax(z) * probe).sum(), [z])

    def test_cross_entropy_matches_composition(self):
        z = Tensor(self.rng.standard_normal((6, 5)), requires_grad=True)
        y = self.rng.integers(0, 5, size=6)
        composed = -F.log_softmax(z)[np.arange(6), y].mean()
        np.testing.assert_allclose(F.cross_entropy(z, y).data, composed.data, atol=1e-12)

        composed.backward()
        from_composition = z.grad
        z.grad = None
        F.cross_entropy(z, y).backward()
        np.testing.assert_allclose(z.grad, from_composition, atol=1e-12)

    def test_cross_entropy_gradients(self):
        z = Tensor(self.rng.standard_normal((6, 5)), requires_grad=True)
        y = self.rng.integers(0, 5, size=6)
        gradcheck(lambda: F.cross_entropy(z, y), [z])
        gradcheck(lambda: F.cross_entropy(z, y, label_smoothing=0.1), [z])

    def test_cross_entropy_known_values(self):
        uniform = Tensor(np.zeros((3, 10)))
        np.testing.assert_allclose(F.cross_entropy(uniform, [0, 5, 9]).data, np.log(10.0))
        confident = Tensor(np.array([[50.0, 0.0, 0.0]]))
        self.assertLess(float(F.cross_entropy(confident, [0]).data), 1e-12)
        self.assertGreater(float(F.cross_entropy(confident, [1]).data), 49.0)

    def test_label_smoothing_value(self):
        z = self.rng.standard_normal((4, 5))
        y = self.rng.integers(0, 5, size=4)
        logp = F.log_softmax(Tensor(z)).data
        q = np.full((4, 5), 0.2 / 5)
        q[np.arange(4), y] += 0.8
        np.testing.assert_allclose(F.cross_entropy(Tensor(z), y, label_smoothing=0.2).data, -(q * logp).sum() / 4)


class Dropout(unittest.TestCase):
    def test_identity_when_not_training(self):
        x = Tensor(np.ones((4, 4)))
        self.assertIs(F.dropout(x, 0.5, training=False), x)
        self.assertIs(F.dropout(x, 0.0, training=True), x)

    def test_scales_survivors_and_keeps_expectation(self):
        fable.manual_seed(0)
        x = Tensor(np.ones((200, 200)))
        out = F.dropout(x, 0.25, training=True).data
        np.testing.assert_allclose(np.unique(out), [0.0, 1 / 0.75])
        self.assertAlmostEqual(float(out.mean()), 1.0, delta=0.02)
        self.assertAlmostEqual(float((out == 0).mean()), 0.25, delta=0.02)

    def test_gradient_follows_mask(self):
        fable.manual_seed(1)
        x = Tensor(np.ones((10, 10)), requires_grad=True)
        out = F.dropout(x, 0.5, training=True)
        out.sum().backward()
        np.testing.assert_array_equal(x.grad, out.data)

    def test_invalid_probability(self):
        with self.assertRaises(ValueError):
            F.dropout(Tensor(np.ones(3)), 1.0, training=True)


if __name__ == "__main__":
    unittest.main()
