"""
Train the MLP and export weights + first-batch intermediates for C alignment.

Outputs (all in weights/):
  W1.npy, b1.npy, W2.npy, b2.npy    -- trained parameters
  init_W1.npy, init_b1.npy, ...      -- initial parameters (He init, same seed)

Outputs (logs/):
  batch0_Z1.npy, batch0_A1.npy, batch0_Z2.npy, batch0_A2.npy
  batch0_dW1.npy, batch0_db1.npy, batch0_dW2.npy, batch0_db2.npy
  batch0_X.npy, batch0_Y.npy        -- the actual input batch used
"""

import numpy as np
from pathlib import Path
from prepare_data import load_mnist
from train import (
    init_params, forward, backward, HIDDEN, LR, BATCH_SIZE, SEED
)

WEIGHTS_DIR = Path(__file__).parent.parent / "weights"
LOGS_DIR    = Path(__file__).parent.parent / "logs"
WEIGHTS_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)


def main():
    X_train, Y_train, y_train, X_test, Y_test, y_test = load_mnist()

    rng = np.random.RandomState(SEED)
    W1, b1, W2, b2 = init_params(rng, 784, HIDDEN, 10)

    # save initial weights (for C to load the same starting point)
    np.save(WEIGHTS_DIR / "init_W1.npy", W1)
    np.save(WEIGHTS_DIR / "init_b1.npy", b1)
    np.save(WEIGHTS_DIR / "init_W2.npy", W2)
    np.save(WEIGHTS_DIR / "init_b2.npy", b2)
    print("Saved initial weights.")

    # capture first batch intermediates (before any weight update)
    idx  = rng.permutation(len(X_train))   # same rng state as train.py epoch-1
    Xb0  = X_train[idx[:BATCH_SIZE]]
    Yb0  = Y_train[idx[:BATCH_SIZE]]

    Z1, A1, Z2, A2 = forward(Xb0, W1, b1, W2, b2)
    dW1, db1_, dW2, db2_ = backward(Xb0, Yb0, Z1, A1, A2, W2, BATCH_SIZE)

    for name, arr in [
        ("batch0_X",   Xb0),  ("batch0_Y",   Yb0),
        ("batch0_Z1",  Z1),   ("batch0_A1",  A1),
        ("batch0_Z2",  Z2),   ("batch0_A2",  A2),
        ("batch0_dW1", dW1),  ("batch0_db1", db1_),
        ("batch0_dW2", dW2),  ("batch0_db2", db2_),
    ]:
        np.save(LOGS_DIR / f"{name}.npy", arr)
    print("Saved batch-0 intermediates.")

    # run full training
    from train import train, EPOCHS
    print("\n--- Starting full training ---")
    W1, b1, W2, b2 = train()

    # save final trained weights
    np.save(WEIGHTS_DIR / "W1.npy", W1)
    np.save(WEIGHTS_DIR / "b1.npy", b1)
    np.save(WEIGHTS_DIR / "W2.npy", W2)
    np.save(WEIGHTS_DIR / "b2.npy", b2)
    print("Saved trained weights.")


if __name__ == "__main__":
    main()
