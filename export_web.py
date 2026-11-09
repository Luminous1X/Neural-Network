"""Export a trained checkpoint for the browser demo.

Writes:
  demo/weights.js          the weights, plus a few test digits to try
  tests/js_fixture.json    test images and the logits NumPy produces for them,
                           so tests/js_parity.js can check the JavaScript port

    python export_web.py
"""
from __future__ import annotations

import argparse
import base64
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

import fable  # noqa: E402
from fable import Tensor  # noqa: E402
from fable.data import MNIST_MEAN, MNIST_STD, load_mnist, normalize  # noqa: E402
from fable.models import load_checkpoint  # noqa: E402


def b64(array: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(array).tobytes()).decode("ascii")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", default=str(ROOT / "checkpoints" / "fablenet.npz"))
    parser.add_argument("--data-dir", default=str(ROOT / "data" / "mnist"))
    parser.add_argument("--fixture-size", type=int, default=300)
    args = parser.parse_args()

    model, meta = load_checkpoint(args.checkpoint)
    _, _, test_x, test_y = load_mnist(args.data_dir)

    # Samples for the demo's "show me a test digit" button: three of each digit.
    sample_idx = np.concatenate([np.flatnonzero(test_y == d)[:3] for d in range(10)])

    weights = {
        "meta": {
            "config": meta["config"],
            "mean": MNIST_MEAN,
            "std": MNIST_STD,
            "test_accuracy": meta["test_accuracy"],
            "parameters": meta["parameters"],
            "epochs": meta["epochs"],
        },
        "tensors": {
            name: {"shape": list(p.shape), "data": b64(p.data.astype("<f4"))} for name, p in model.named_parameters()
        },
        "samples": {"labels": test_y[sample_idx].tolist(), "pixels": b64(test_x[sample_idx])},
    }
    weights_path = ROOT / "demo" / "weights.js"
    weights_path.write_text("globalThis.FABLE_WEIGHTS = " + json.dumps(weights) + ";\n", encoding="ascii")

    n = args.fixture_size
    with fable.no_grad():
        logits = model(Tensor(normalize(test_x[:n]))).data
    fixture = {
        "count": n,
        "labels": test_y[:n].tolist(),
        "pixels": b64(test_x[:n]),
        "logits": b64(logits.astype("<f4")),
    }
    fixture_path = ROOT / "tests" / "js_fixture.json"
    fixture_path.write_text(json.dumps(fixture), encoding="ascii")

    print(f"wrote {weights_path} ({weights_path.stat().st_size / 1024:.0f} KB)")
    print(f"wrote {fixture_path} ({n} images, NumPy accuracy on them "
          f"{(logits.argmax(axis=1) == test_y[:n]).mean():.2%})")


if __name__ == "__main__":
    main()
