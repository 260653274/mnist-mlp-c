"""Load MNIST IDX binary files and return numpy arrays."""

import struct
import numpy as np
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"


def load_idx_images(path: Path) -> np.ndarray:
    with open(path, "rb") as f:
        magic, n, rows, cols = struct.unpack(">4I", f.read(16))
        assert magic == 2051, f"bad magic {magic}"
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data.reshape(n, rows * cols).astype(np.float64) / 255.0


def load_idx_labels(path: Path) -> np.ndarray:
    with open(path, "rb") as f:
        magic, n = struct.unpack(">2I", f.read(8))
        assert magic == 2049, f"bad magic {magic}"
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    return labels.astype(np.int32)


def to_one_hot(labels: np.ndarray, num_classes: int = 10) -> np.ndarray:
    oh = np.zeros((len(labels), num_classes), dtype=np.float64)
    oh[np.arange(len(labels)), labels] = 1.0
    return oh


def load_mnist():
    X_train = load_idx_images(DATA_DIR / "train-images-idx3-ubyte")
    y_train = load_idx_labels(DATA_DIR / "train-labels-idx1-ubyte")
    X_test  = load_idx_images(DATA_DIR / "t10k-images-idx3-ubyte")
    y_test  = load_idx_labels(DATA_DIR / "t10k-labels-idx1-ubyte")

    Y_train = to_one_hot(y_train)
    Y_test  = to_one_hot(y_test)

    return X_train, Y_train, y_train, X_test, Y_test, y_test


if __name__ == "__main__":
    X_tr, Y_tr, y_tr, X_te, Y_te, y_te = load_mnist()
    print(f"Train: X={X_tr.shape}, Y={Y_tr.shape}, range=[{X_tr.min():.3f}, {X_tr.max():.3f}]")
    print(f"Test:  X={X_te.shape}, Y={Y_te.shape}")
    print(f"Label sample: {y_tr[:10]}")
