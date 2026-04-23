"""
Convert .npy files (float64) to raw little-endian double binary files (.bin)
so that C can load them with a simple fread().

Run this once after export_weights.py to prepare files for test_alignment.
"""

import numpy as np
from pathlib import Path

ROOT        = Path(__file__).parent.parent
WEIGHTS_DIR = ROOT / "weights"
LOGS_DIR    = ROOT / "logs"

conversions = [
    # (src .npy, dst .bin, expected shape)
    (WEIGHTS_DIR / "init_W1.npy", WEIGHTS_DIR / "init_W1.bin", (784, 512)),
    (WEIGHTS_DIR / "init_b1.npy", WEIGHTS_DIR / "init_b1.bin", (1, 512)),
    (WEIGHTS_DIR / "init_W2.npy", WEIGHTS_DIR / "init_W2.bin", (512, 10)),
    (WEIGHTS_DIR / "init_b2.npy", WEIGHTS_DIR / "init_b2.bin", (1, 10)),
    (WEIGHTS_DIR / "W1.npy",      WEIGHTS_DIR / "W1.bin",      (784, 512)),
    (WEIGHTS_DIR / "b1.npy",      WEIGHTS_DIR / "b1.bin",      (1, 512)),
    (WEIGHTS_DIR / "W2.npy",      WEIGHTS_DIR / "W2.bin",      (512, 10)),
    (WEIGHTS_DIR / "b2.npy",      WEIGHTS_DIR / "b2.bin",      (1, 10)),
    (LOGS_DIR / "batch0_X.npy",   LOGS_DIR / "batch0_X.bin",   (50, 784)),
    (LOGS_DIR / "batch0_Y.npy",   LOGS_DIR / "batch0_Y.bin",   (50, 10)),
    (LOGS_DIR / "batch0_Z1.npy",  LOGS_DIR / "batch0_Z1.bin",  (50, 512)),
    (LOGS_DIR / "batch0_A1.npy",  LOGS_DIR / "batch0_A1.bin",  (50, 512)),
    (LOGS_DIR / "batch0_Z2.npy",  LOGS_DIR / "batch0_Z2.bin",  (50, 10)),
    (LOGS_DIR / "batch0_A2.npy",  LOGS_DIR / "batch0_A2.bin",  (50, 10)),
    (LOGS_DIR / "batch0_dW1.npy", LOGS_DIR / "batch0_dW1.bin", (784, 512)),
    (LOGS_DIR / "batch0_db1.npy", LOGS_DIR / "batch0_db1.bin", (1, 512)),
    (LOGS_DIR / "batch0_dW2.npy", LOGS_DIR / "batch0_dW2.bin", (512, 10)),
    (LOGS_DIR / "batch0_db2.npy", LOGS_DIR / "batch0_db2.bin", (1, 10)),
]

for src, dst, expected_shape in conversions:
    arr = np.load(src)
    assert arr.shape == expected_shape, \
        f"{src.name}: expected {expected_shape}, got {arr.shape}"
    arr = np.ascontiguousarray(arr, dtype=np.float64)
    arr.tofile(dst)
    print(f"  {src.name:25s} -> {dst.name}  shape={arr.shape}")

print("\nDone. All .bin files written.")
