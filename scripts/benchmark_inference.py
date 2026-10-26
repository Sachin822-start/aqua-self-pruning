import time
import sys
from pathlib import Path

import numpy as np

# Add the project root to Python's import path.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from nn.layers import Linear



def measure(function, repeats=3):
    # Warm up once.
    function()

    timings = []
    for _ in range(repeats):
        start = time.perf_counter()
        for _ in range(3):
            function()
        timings.append((time.perf_counter() - start) / 3)

    return float(np.median(timings))



def main():
    rng = np.random.default_rng(42)

    batch_size = 256
    in_features = 256
    out_features = 256

    x = rng.standard_normal((batch_size, in_features))
    layer = Linear(in_features, out_features, seed=42)

    print("Inference benchmark")
    print(f"Input shape: {x.shape}")
    print(f"Weight shape: {layer.weight.data.shape}")
    print()

    for sparsity in (0.0, 0.5, 0.75, 0.9):
        # Retain the requested approximate fraction of connections.
        mask = rng.random(layer.weight.data.shape) >= sparsity
        layer.set_mask(mask)

        dense_time = measure(
    lambda: x @ (layer.weight.data * layer.mask) + layer.bias.data
)
        sparse_time = measure(lambda: layer.forward_sparse(x))

        total = layer.mask.size
        active = int(np.count_nonzero(layer.mask))
        actual_sparsity = 1.0 - active / total

        print(
            f"Target sparsity: {sparsity:.0%} | "
            f"Actual sparsity: {actual_sparsity:.2%} | "
            f"Active weights: {active}/{total}"
        )
        print(f"  Dense forward:        {dense_time * 1000:.3f} ms")
        print(f"  Sparse-aware forward: {sparse_time * 1000:.3f} ms")
        print(f"  Sparse/dense ratio:   {sparse_time / dense_time:.2f}x")
        print()


if __name__ == "__main__":
    main()
