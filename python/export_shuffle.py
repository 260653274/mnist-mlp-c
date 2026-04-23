"""
Export Python's shuffle indices (first 5 epochs) and He-init weights
as raw binary int32 / float64 so C can replicate the exact training trajectory.

Outputs (logs/):
  shuffle_ep{1..5}.bin  -- int32[60000] Fisher-Yates permutation indices
"""

import numpy as np
from pathlib import Path

LOGS_DIR = Path(__file__).parent.parent / "logs"
LOGS_DIR.mkdir(exist_ok=True)

N_EPOCHS_EXPORT = 5
SEED            = 42

rng = np.random.RandomState(SEED)

# Consume the same RNG state as train.py: init_params uses rng for W1, W2
# He init: rng.randn(784, 512)  then rng.randn(512, 10)
_ = rng.randn(784, 512)
_ = rng.randn(512, 10)

# Now export shuffle indices for epochs 1..N
for ep in range(1, N_EPOCHS_EXPORT + 1):
    idx = rng.permutation(60000).astype(np.int32)
    out = LOGS_DIR / f"shuffle_ep{ep}.bin"
    idx.tofile(out)
    print(f"  epoch {ep}: shuffle_ep{ep}.bin  first5={idx[:5].tolist()}")

print(f"\nExported {N_EPOCHS_EXPORT} epoch shuffle indices to {LOGS_DIR}")
