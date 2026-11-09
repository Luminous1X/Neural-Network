"""Train FableNet on MNIST.

    python train.py                  # full run
    python train.py --limit 2000 --epochs 1   # quick smoke test
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# The matrices here are small, so a large BLAS thread pool costs more in
# wake-ups than it gains. Must be set before NumPy is imported.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

import fable  # noqa: E402
from fable import Tensor, optim  # noqa: E402
from fable import functional as F  # noqa: E402
from fable.data import iterate_batches, load_mnist  # noqa: E402
from fable.models import FableNet, save_checkpoint  # noqa: E402


def evaluate(model, images, labels, batch_size: int = 500):
    """Returns (mean loss, accuracy) over a dataset, without augmentation or dropout."""
    model.eval()
    loss_sum, correct = 0.0, 0
    with fable.no_grad():
        for x, y in iterate_batches(images, labels, batch_size):
            logits = model(Tensor(x))
            loss_sum += F.cross_entropy(logits, y).item() * len(y)
            correct += int((logits.data.argmax(axis=1) == y).sum())
    return loss_sum / len(images), correct / len(images)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-3, help="peak learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--c1", type=int, default=16, help="channels in the first conv layer")
    parser.add_argument("--c2", type=int, default=32, help="channels in the second conv layer")
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--no-augment", action="store_true", help="disable random affine augmentation")
    parser.add_argument("--limit", type=int, default=None, help="use only the first N training images")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--data-dir", default=str(ROOT / "data" / "mnist"))
    parser.add_argument("--out", default=str(ROOT / "checkpoints" / "fablenet.npz"))
    args = parser.parse_args()

    train_x, train_y, test_x, test_y = load_mnist(args.data_dir)
    if args.limit:
        train_x, train_y = train_x[: args.limit], train_y[: args.limit]

    fable.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    model = FableNet(c1=args.c1, c2=args.c2, hidden=args.hidden, dropout=args.dropout)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    steps_per_epoch = -(-len(train_x) // args.batch_size)
    schedule = optim.WarmupCosine(
        optimizer,
        peak_lr=args.lr,
        total_steps=args.epochs * steps_per_epoch,
        warmup_steps=min(steps_per_epoch, 500),
    )

    print(f"FableNet: {model.num_parameters():,} parameters", flush=True)
    print(f"training on {len(train_x):,} images, testing on {len(test_x):,}; "
          f"{args.epochs} epochs of {steps_per_epoch} steps", flush=True)
    print(f"{'epoch':>5} {'lr':>9} {'train loss':>10} {'train acc':>9} {'test loss':>9} {'test acc':>8} {'time':>6}", flush=True)

    history = []
    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        epoch_started = time.perf_counter()
        model.train()
        loss_sum, correct, seen = 0.0, 0, 0
        for x, y in iterate_batches(train_x, train_y, args.batch_size, rng=rng, augment=not args.no_augment):
            logits = model(Tensor(x))
            loss = F.cross_entropy(logits, y, label_smoothing=args.label_smoothing)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            schedule.step()

            loss_sum += loss.item() * len(y)
            correct += int((logits.data.argmax(axis=1) == y).sum())
            seen += len(y)

        test_loss, test_acc = evaluate(model, test_x, test_y)
        record = {
            "epoch": epoch,
            "lr": optimizer.lr,
            "train_loss": loss_sum / seen,
            "train_acc": correct / seen,
            "test_loss": test_loss,
            "test_acc": test_acc,
            "seconds": time.perf_counter() - epoch_started,
        }
        history.append(record)
        print(f"{epoch:>5} {record['lr']:>9.2e} {record['train_loss']:>10.4f} {record['train_acc']:>9.2%} "
              f"{test_loss:>9.4f} {test_acc:>8.2%} {record['seconds']:>5.1f}s", flush=True)

    total = time.perf_counter() - started
    final = history[-1]
    print(f"done in {total:.0f}s; final test accuracy {final['test_acc']:.2%} "
          f"({round((1 - final['test_acc']) * len(test_x))} errors out of {len(test_x):,})", flush=True)

    out = Path(args.out)
    save_checkpoint(
        model,
        out,
        {
            "test_accuracy": final["test_acc"],
            "test_loss": final["test_loss"],
            "epochs": args.epochs,
            "train_images": len(train_x),
            "parameters": model.num_parameters(),
            "args": {k: v for k, v in vars(args).items() if k not in ("data_dir", "out")},
        },
    )
    history_path = out.with_name(out.stem + "_history.json")
    history_path.write_text(json.dumps(history, indent=2))
    print(f"saved {out} and {history_path.name}", flush=True)


if __name__ == "__main__":
    main()
