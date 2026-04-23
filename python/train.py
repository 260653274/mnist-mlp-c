"""
784 -> 512 -> 10 MLP, mini-batch SGD, He init, ReLU + Softmax.
Reference config: batch=50, lr=0.09, epochs=50 -> ~98.26% test accuracy.
"""

import csv
import time
import numpy as np
from pathlib import Path
from prepare_data import load_mnist

# ── hyper-parameters ──────────────────────────────────────────────────────────
HIDDEN     = 512
LR         = 0.09
BATCH_SIZE = 50
EPOCHS     = 50
SEED       = 42
EPS        = 1e-8   # cross-entropy log clip

LOG_DIR    = Path(__file__).parent.parent / "logs"
LOG_DIR.mkdir(exist_ok=True)


# ── activations ───────────────────────────────────────────────────────────────
def relu(z):
    return np.maximum(0.0, z)

def relu_grad(z):
    return (z > 0).astype(np.float64)

def softmax(z):
    # numerically stable: subtract row-wise max
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)

def cross_entropy(a2, y):
    return -np.mean(np.sum(y * np.log(a2 + EPS), axis=1))


# ── parameter init ────────────────────────────────────────────────────────────
def init_params(rng, n_in, n_hid, n_out):
    W1 = rng.randn(n_in,  n_hid) * np.sqrt(2.0 / n_in)
    b1 = np.zeros((1, n_hid))
    W2 = rng.randn(n_hid, n_out) * np.sqrt(2.0 / n_hid)
    b2 = np.zeros((1, n_out))
    return W1, b1, W2, b2


# ── forward / backward ────────────────────────────────────────────────────────
def forward(X, W1, b1, W2, b2):
    Z1 = X  @ W1 + b1
    A1 = relu(Z1)
    Z2 = A1 @ W2 + b2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2

def backward(X, Y, Z1, A1, A2, W2, batch_size):
    dZ2 = (A2 - Y) / batch_size          # Softmax + CE combined gradient
    dW2 = A1.T @ dZ2
    db2 = dZ2.sum(axis=0, keepdims=True)
    dA1 = dZ2 @ W2.T
    dZ1 = dA1 * relu_grad(Z1)
    dW1 = X.T  @ dZ1
    db1 = dZ1.sum(axis=0, keepdims=True)
    return dW1, db1, dW2, db2


# ── accuracy helper ───────────────────────────────────────────────────────────
def accuracy(X, y_int, W1, b1, W2, b2, chunk=1000):
    correct = 0
    for i in range(0, len(X), chunk):
        _, _, _, A2 = forward(X[i:i+chunk], W1, b1, W2, b2)
        correct += (A2.argmax(axis=1) == y_int[i:i+chunk]).sum()
    return correct / len(X)


# ── training loop ─────────────────────────────────────────────────────────────
def train():
    X_train, Y_train, y_train, X_test, Y_test, y_test = load_mnist()
    n = len(X_train)

    rng = np.random.RandomState(SEED)
    W1, b1, W2, b2 = init_params(rng, 784, HIDDEN, 10)

    log_rows = []
    t0 = time.time()

    for epoch in range(1, EPOCHS + 1):
        # shuffle
        idx = rng.permutation(n)
        X_shuf, Y_shuf = X_train[idx], Y_train[idx]

        epoch_loss = 0.0
        n_batches  = 0

        for start in range(0, n, BATCH_SIZE):
            Xb = X_shuf[start:start + BATCH_SIZE]
            Yb = Y_shuf[start:start + BATCH_SIZE]
            bs = len(Xb)

            Z1, A1, Z2, A2 = forward(Xb, W1, b1, W2, b2)
            loss = cross_entropy(A2, Yb)
            epoch_loss += loss
            n_batches  += 1

            dW1, db1_, dW2, db2_ = backward(Xb, Yb, Z1, A1, A2, W2, bs)

            W1 -= LR * dW1
            b1 -= LR * db1_
            W2 -= LR * dW2
            b2 -= LR * db2_

        avg_loss = epoch_loss / n_batches
        test_acc = accuracy(X_test, y_test, W1, b1, W2, b2)
        elapsed  = time.time() - t0

        print(f"epoch {epoch:3d}/{EPOCHS}  loss={avg_loss:.6f}  test_acc={test_acc*100:.2f}%  t={elapsed:.1f}s")
        log_rows.append([epoch, avg_loss, test_acc])

    # save training log
    log_path = LOG_DIR / "train_log.csv"
    with open(log_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "loss", "test_acc"])
        w.writerows(log_rows)
    print(f"\nLog saved to {log_path}")

    return W1, b1, W2, b2


if __name__ == "__main__":
    train()
