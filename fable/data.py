"""MNIST download and parsing, mini-batching, and data augmentation."""
from __future__ import annotations

import gzip
import hashlib
import struct
import urllib.request
from pathlib import Path

import numpy as np

__all__ = ["load_mnist", "normalize", "iterate_batches", "random_affine", "MNIST_MEAN", "MNIST_STD"]

MNIST_MEAN = 0.1307
MNIST_STD = 0.3081

_MIRRORS = (
    "https://ossci-datasets.s3.amazonaws.com/mnist/",
    "https://storage.googleapis.com/cvdf-datasets/mnist/",
)

# file name -> MD5 of the gzip archive
_FILES = {
    "train-images-idx3-ubyte.gz": "f68b3c2dcbeaaa9fbdd348bbdeb94873",
    "train-labels-idx1-ubyte.gz": "d53e105ee54ea40749a09fcbcd1e9432",
    "t10k-images-idx3-ubyte.gz": "9fb629c4189551a2d022fa330f9573f3",
    "t10k-labels-idx1-ubyte.gz": "ec29112dd5afa0611ce80d1b7f02629c",
}


def _md5(path: Path) -> str:
    digest = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fetch(name: str, directory: Path) -> Path:
    """Return the path to a verified copy of `name`, downloading it if needed."""
    target = directory / name
    expected = _FILES[name]
    if target.exists() and _md5(target) == expected:
        return target

    directory.mkdir(parents=True, exist_ok=True)
    errors = []
    for mirror in _MIRRORS:
        partial = directory / (name + ".part")
        try:
            print(f"downloading {mirror}{name}", flush=True)
            with urllib.request.urlopen(mirror + name, timeout=60) as response, open(partial, "wb") as fh:
                while chunk := response.read(1 << 16):
                    fh.write(chunk)
            actual = _md5(partial)
            if actual != expected:
                raise IOError(f"checksum mismatch (got {actual}, expected {expected})")
            partial.replace(target)
            return target
        except Exception as exc:  # try the next mirror
            errors.append(f"{mirror}: {exc}")
            partial.unlink(missing_ok=True)
    raise RuntimeError(f"could not download {name}:\n  " + "\n  ".join(errors))


def _read_idx(path: Path) -> np.ndarray:
    """Parse the IDX format: a big-endian header followed by raw unsigned bytes."""
    with gzip.open(path, "rb") as fh:
        raw = fh.read()
    zero, dtype_code, ndim = struct.unpack(">HBB", raw[:4])
    if zero != 0 or dtype_code != 0x08:
        raise ValueError(f"{path.name} is not an unsigned-byte IDX file")
    shape = struct.unpack(">" + "I" * ndim, raw[4 : 4 + 4 * ndim])
    data = np.frombuffer(raw, dtype=np.uint8, offset=4 + 4 * ndim)
    if data.size != int(np.prod(shape)):
        raise ValueError(f"{path.name} is truncated")
    return data.reshape(shape)


def load_mnist(directory="data/mnist"):
    """Returns (train_images, train_labels, test_images, test_labels).

    Images are uint8 arrays of shape (N, 28, 28); labels are uint8 of shape (N,).
    Files are downloaded on first use and verified against known checksums.
    """
    directory = Path(directory)
    arrays = [_read_idx(_fetch(name, directory)) for name in _FILES]
    train_x, train_y, test_x, test_y = arrays
    if train_x.shape != (60000, 28, 28) or test_x.shape != (10000, 28, 28):
        raise ValueError("unexpected MNIST image shapes")
    if train_y.shape != (60000,) or test_y.shape != (10000,):
        raise ValueError("unexpected MNIST label shapes")
    return train_x, train_y, test_x, test_y


def normalize(images: np.ndarray) -> np.ndarray:
    """(N, 28, 28) pixels in [0, 255] -> standardised float32 of shape (N, 1, 28, 28)."""
    x = images.astype(np.float32) / 255.0
    return ((x - MNIST_MEAN) / MNIST_STD)[:, None, :, :]


def random_affine(
    images: np.ndarray,
    rng: np.random.Generator,
    max_rotate: float = 10.0,
    max_scale: float = 0.10,
    max_shift: float = 2.0,
) -> np.ndarray:
    """Randomly rotate, scale and shift each image, with bilinear resampling.

    images: (N, H, W), any real dtype. Returns float32 in the same value range.
    Each image gets its own rotation (degrees), zoom factor and pixel shift,
    drawn uniformly from the given ranges. Pixels sampled from outside the
    source image are zero.
    """
    n, h, w = images.shape
    theta = np.deg2rad(rng.uniform(-max_rotate, max_rotate, n))
    scale = rng.uniform(1.0 - max_scale, 1.0 + max_scale, n)
    shift_x = rng.uniform(-max_shift, max_shift, n)
    shift_y = rng.uniform(-max_shift, max_shift, n)

    # For every output pixel, find where in the source image it comes from
    # (inverse mapping), rotating and scaling about the image centre.
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    ys, xs = ys - cy, xs - cx
    cos = (np.cos(theta) / scale).astype(np.float32)[:, None, None]
    sin = (np.sin(theta) / scale).astype(np.float32)[:, None, None]
    src_x = cos * xs - sin * ys + cx - shift_x.astype(np.float32)[:, None, None]
    src_y = sin * xs + cos * ys + cy - shift_y.astype(np.float32)[:, None, None]

    x0 = np.floor(src_x).astype(np.intp)
    y0 = np.floor(src_y).astype(np.intp)
    fx = src_x - x0
    fy = src_y - y0
    batch = np.arange(n)[:, None, None]

    def sample(yy, xx):
        inside = (yy >= 0) & (yy < h) & (xx >= 0) & (xx < w)
        values = images[batch, np.clip(yy, 0, h - 1), np.clip(xx, 0, w - 1)]
        return values * inside

    out = (
        (1 - fy) * (1 - fx) * sample(y0, x0)
        + (1 - fy) * fx * sample(y0, x0 + 1)
        + fy * (1 - fx) * sample(y0 + 1, x0)
        + fy * fx * sample(y0 + 1, x0 + 1)
    )
    return out.astype(np.float32)


def iterate_batches(images, labels, batch_size: int, rng=None, augment: bool = False):
    """Yield (x, y) mini-batches with x normalised to (B, 1, 28, 28) float32.

    Passing `rng` shuffles the data; `augment` additionally applies a random
    affine distortion to every image.
    """
    n = len(images)
    order = rng.permutation(n) if rng is not None else np.arange(n)
    if augment and rng is None:
        raise ValueError("augmentation needs an rng")
    for start in range(0, n, batch_size):
        idx = order[start : start + batch_size]
        batch = images[idx]
        if augment:
            batch = random_affine(batch, rng)
        yield normalize(batch), labels[idx].astype(np.intp)
