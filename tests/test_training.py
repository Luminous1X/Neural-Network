"""Modules, optimisers, data utilities, and end-to-end learning."""
import tempfile
import unittest
from pathlib import Path

import numpy as np

import fable
from fable import Tensor, nn, optim
from fable import functional as F
from fable.data import iterate_batches, normalize, random_affine
from fable.gradcheck import gradcheck
from fable.models import FableNet, load_checkpoint, save_checkpoint


class Modules(unittest.TestCase):
    def setUp(self):
        fable.manual_seed(0)

    def test_parameter_discovery(self):
        model = nn.Sequential(nn.Linear(4, 3), nn.ReLU(), nn.Linear(3, 2, bias=False))
        names = [name for name, _ in model.named_parameters()]
        self.assertEqual(names, ["layers.0.weight", "layers.0.bias", "layers.2.weight"])
        self.assertEqual(model.num_parameters(), 4 * 3 + 3 + 3 * 2)

    def test_fablenet_shapes(self):
        model = FableNet()
        shapes = {name: p.shape for name, p in model.named_parameters()}
        self.assertEqual(
            shapes,
            {
                "conv1.weight": (16, 1, 5, 5),
                "conv1.bias": (16,),
                "conv2.weight": (32, 16, 5, 5),
                "conv2.bias": (32,),
                "fc1.weight": (128, 512),
                "fc1.bias": (128,),
                "fc2.weight": (10, 128),
                "fc2.bias": (10,),
            },
        )
        out = model(Tensor(np.zeros((3, 1, 28, 28), dtype=np.float32)))
        self.assertEqual(out.shape, (3, 10))
        self.assertEqual(out.dtype, np.float32)

    def test_train_eval_propagates(self):
        model = FableNet()
        self.assertTrue(model.drop.training)
        model.eval()
        self.assertFalse(model.drop.training)
        x = Tensor(np.random.default_rng(0).standard_normal((2, 1, 28, 28)).astype(np.float32))
        np.testing.assert_array_equal(model(x).data, model(x).data)  # deterministic in eval mode
        model.train()
        self.assertTrue(model.drop.training)

    def test_state_dict_round_trip(self):
        a, b = FableNet(), FableNet()
        x = Tensor(np.random.default_rng(0).standard_normal((2, 1, 28, 28)).astype(np.float32))
        a.eval(), b.eval()
        self.assertFalse(np.allclose(a(x).data, b(x).data))
        b.load_state_dict(a.state_dict())
        np.testing.assert_array_equal(a(x).data, b(x).data)
        with self.assertRaises(KeyError):
            b.load_state_dict({"nope": np.zeros(1)})

    def test_checkpoint_round_trip(self):
        model = FableNet(c1=4, c2=6, hidden=10).eval()
        x = Tensor(np.random.default_rng(0).standard_normal((2, 1, 28, 28)).astype(np.float32))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.npz"
            save_checkpoint(model, path, {"test_accuracy": 0.5})
            loaded, meta = load_checkpoint(path)
        self.assertEqual(meta["config"], model.config)
        self.assertEqual(meta["test_accuracy"], 0.5)
        np.testing.assert_array_equal(model(x).data, loaded(x).data)

    def test_whole_network_gradients(self):
        """Finite-difference check through conv, pool, flatten, linear and the loss at once."""
        with fable.default_dtype(np.float64):
            model = FableNet(c1=2, c2=3, hidden=5, dropout=0.0)
        rng = np.random.default_rng(0)
        x = Tensor(rng.standard_normal((2, 1, 28, 28)))
        y = np.array([3, 7])
        worst = gradcheck(lambda: F.cross_entropy(model(x), y), model.parameters(), tol=1e-5)
        self.assertLess(worst, 1e-5)


class Optimisers(unittest.TestCase):
    @staticmethod
    def minimise(make_optimizer, steps):
        target = np.array([3.0, -2.0, 0.5])
        p = nn.Parameter(np.zeros(3))
        opt = make_optimizer([p])
        for _ in range(steps):
            opt.zero_grad()
            ((p - target) ** 2).sum().backward()
            opt.step()
        return float(np.abs(p.data - target).max())

    def test_sgd(self):
        self.assertLess(self.minimise(lambda ps: optim.SGD(ps, lr=0.1), 200), 1e-6)

    def test_sgd_momentum(self):
        self.assertLess(self.minimise(lambda ps: optim.SGD(ps, lr=0.05, momentum=0.9), 300), 1e-6)
        self.assertLess(self.minimise(lambda ps: optim.SGD(ps, lr=0.05, momentum=0.9, nesterov=True), 300), 1e-6)

    def test_adam(self):
        self.assertLess(self.minimise(lambda ps: optim.Adam(ps, lr=0.1), 500), 1e-3)

    def test_adam_first_step_is_lr_sized(self):
        p = nn.Parameter(np.array([1.0, -1.0]))
        opt = optim.Adam([p], lr=0.01)
        (p * np.array([5.0, -0.001])).sum().backward()
        opt.step()
        # Bias correction makes the first step ~lr in the direction of -sign(grad).
        np.testing.assert_allclose(p.data, [0.99, -0.99], atol=1e-6)

    def test_weight_decay_skips_biases(self):
        w, b = nn.Parameter(np.ones((2, 2))), nn.Parameter(np.ones(2))
        opt = optim.Adam([w, b], lr=0.1, weight_decay=0.5)
        (w.sum() * 0.0 + b.sum() * 0.0).backward()  # zero gradients: only decay can move things
        opt.step()
        np.testing.assert_allclose(w.data, 0.95)
        np.testing.assert_allclose(b.data, 1.0)

    def test_warmup_cosine(self):
        opt = optim.SGD([nn.Parameter(np.zeros(1))], lr=123.0)
        schedule = optim.WarmupCosine(opt, peak_lr=1.0, total_steps=110, warmup_steps=10)
        lrs = [opt.lr] + [schedule.step() for _ in range(110)]
        self.assertAlmostEqual(lrs[0], 0.1)
        self.assertAlmostEqual(lrs[9], 1.0)
        self.assertAlmostEqual(max(lrs), 1.0)
        self.assertAlmostEqual(lrs[60], 0.5)  # halfway through the cosine
        self.assertAlmostEqual(lrs[-1], 0.0)
        self.assertTrue(all(a >= b for a, b in zip(lrs[9:], lrs[10:])))


class Data(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(0)
        self.images = self.rng.integers(0, 256, size=(10, 28, 28), dtype=np.uint8)

    def test_normalize(self):
        x = normalize(self.images)
        self.assertEqual(x.shape, (10, 1, 28, 28))
        self.assertEqual(x.dtype, np.float32)
        np.testing.assert_allclose(x[0, 0, 0, 0], (self.images[0, 0, 0] / 255 - 0.1307) / 0.3081, rtol=1e-5)

    def test_identity_affine_changes_nothing(self):
        out = random_affine(self.images, self.rng, max_rotate=0, max_scale=0, max_shift=0)
        np.testing.assert_allclose(out, self.images, atol=1e-3)

    def test_affine_shift_moves_pixels(self):
        image = np.zeros((1, 28, 28), dtype=np.uint8)
        image[0, 10, 10] = 255

        class FixedShift:
            """Stands in for the rng: no rotation, no zoom, shift of exactly +2 px."""

            def uniform(self, low, high, n):
                return np.full(n, 0.0 if high in (0, 0.0) else (1.0 if low == 1.0 else 2.0))

        out = random_affine(image, FixedShift(), max_rotate=0, max_scale=0, max_shift=2)
        self.assertAlmostEqual(float(out[0, 12, 12]), 255.0, places=3)
        self.assertAlmostEqual(float(out.sum()), 255.0, places=2)

    def test_affine_stays_in_range_and_roughly_preserves_mass(self):
        image = np.zeros((50, 28, 28), dtype=np.uint8)
        image[:, 9:19, 11:17] = 200
        out = random_affine(image, self.rng)
        self.assertGreaterEqual(float(out.min()), 0.0)
        self.assertLessEqual(float(out.max()), 200.0 + 1e-3)
        ratio = out.sum(axis=(1, 2)) / image.sum(axis=(1, 2))
        self.assertTrue(np.all((ratio > 0.7) & (ratio < 1.35)), ratio)

    def test_batches_cover_everything_once(self):
        labels = np.arange(10, dtype=np.uint8)
        seen = []
        for x, y in iterate_batches(self.images, labels, batch_size=4, rng=self.rng):
            self.assertEqual(x.shape[1:], (1, 28, 28))
            self.assertEqual(len(x), len(y))
            seen.extend(y.tolist())
        self.assertEqual(sorted(seen), list(range(10)))
        self.assertNotEqual(seen, list(range(10)))  # shuffled

        ordered = [y for _, ys in iterate_batches(self.images, labels, batch_size=4) for y in ys.tolist()]
        self.assertEqual(ordered, list(range(10)))


class Learning(unittest.TestCase):
    def test_mlp_learns_xor(self):
        fable.manual_seed(0)
        x = Tensor(np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32))
        y = np.array([0, 1, 1, 0])
        model = nn.Sequential(nn.Linear(2, 16), nn.Tanh(), nn.Linear(16, 2))
        opt = optim.Adam(model.parameters(), lr=0.05)
        for _ in range(300):
            opt.zero_grad()
            loss = F.cross_entropy(model(x), y)
            loss.backward()
            opt.step()
        self.assertLess(loss.item(), 0.01)
        np.testing.assert_array_equal(model(x).data.argmax(axis=1), y)

    def test_cnn_memorises_a_small_batch(self):
        fable.manual_seed(0)
        rng = np.random.default_rng(0)
        x = Tensor(rng.standard_normal((16, 1, 28, 28)).astype(np.float32))
        y = rng.integers(0, 10, size=16)
        model = FableNet(c1=4, c2=8, hidden=32, dropout=0.0)
        opt = optim.Adam(model.parameters(), lr=3e-3)
        first = None
        for _ in range(60):
            opt.zero_grad()
            loss = F.cross_entropy(model(x), y)
            loss.backward()
            opt.step()
            first = loss.item() if first is None else first
        self.assertGreater(first, 1.5)
        self.assertLess(loss.item(), 0.05)
        np.testing.assert_array_equal(model(x).data.argmax(axis=1), y)


if __name__ == "__main__":
    unittest.main()
