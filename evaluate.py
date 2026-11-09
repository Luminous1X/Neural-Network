"""Evaluate a trained checkpoint on the MNIST test set.

Prints overall and per-digit accuracy, the confusion matrix, and the test
images the network is most confidently wrong about.

    python evaluate.py
    python evaluate.py --show 10
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

import fable  # noqa: E402
from fable import Tensor  # noqa: E402
from fable import functional as F  # noqa: E402
from fable.data import load_mnist, normalize  # noqa: E402
from fable.models import load_checkpoint  # noqa: E402

SHADES = " .:-=+*#%@"


def ascii_digit(image: np.ndarray) -> list:
    """Render a 28x28 uint8 image as 28 lines of text."""
    levels = (image.astype(np.int32) * (len(SHADES) - 1) + 127) // 255
    return ["".join(SHADES[v] * 2 for v in row) for row in levels]


def predict(model, images: np.ndarray, batch_size: int = 500) -> np.ndarray:
    """Class probabilities, shape (N, 10)."""
    out = []
    with fable.no_grad():
        for start in range(0, len(images), batch_size):
            logits = model(Tensor(normalize(images[start : start + batch_size])))
            out.append(F.softmax(logits).data)
    return np.concatenate(out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", default=str(ROOT / "checkpoints" / "fablenet.npz"))
    parser.add_argument("--data-dir", default=str(ROOT / "data" / "mnist"))
    parser.add_argument("--show", type=int, default=4, help="how many mistakes to draw")
    args = parser.parse_args()

    model, meta = load_checkpoint(args.checkpoint)
    _, _, test_x, test_y = load_mnist(args.data_dir)
    probs = predict(model, test_x)
    pred = probs.argmax(axis=1)
    wrong = np.flatnonzero(pred != test_y)

    print(f"checkpoint: {args.checkpoint}")
    print(f"config:     {meta['config']}  ({model.num_parameters():,} parameters)")
    print(f"test accuracy: {(pred == test_y).mean():.2%}  ({len(wrong)} errors out of {len(test_y):,})")

    print("\nper-digit accuracy")
    for digit in range(10):
        mask = test_y == digit
        print(f"  {digit}: {(pred[mask] == digit).mean():7.2%}  ({int((pred[mask] != digit).sum())} of {int(mask.sum())} wrong)")

    confusion = np.zeros((10, 10), dtype=np.int64)
    np.add.at(confusion, (test_y, pred), 1)
    print("\nconfusion matrix (rows: true digit, columns: predicted)")
    print("      " + "".join(f"{d:>6}" for d in range(10)))
    for digit in range(10):
        print(f"  {digit} | " + "".join(f"{v:>6}" if v else f"{'.':>6}" for v in confusion[digit]))

    if args.show and len(wrong):
        confidence = probs[wrong, pred[wrong]]
        worst = wrong[np.argsort(-confidence)][: args.show]
        print(f"\nmost confident mistakes ({len(worst)} of {len(wrong)})")
        for i in worst:
            print(f"\n  test image #{i}: labelled {test_y[i]}, predicted {pred[i]} "
                  f"({probs[i, pred[i]]:.1%} confident; gave the label {probs[i, test_y[i]]:.1%})")
            for line in ascii_digit(test_x[i]):
                if line.strip():
                    print("    " + line)


if __name__ == "__main__":
    main()
