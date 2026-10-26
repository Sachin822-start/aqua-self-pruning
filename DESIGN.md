# Design Notes — AQUA Self-Pruning Neural Network

## 1. Architecture

The project is organized into five main components:

- **Autodiff engine:** The `Tensor` class tracks operations and calculates gradients through reverse-mode automatic differentiation.
- **Neural-network layers:** `Linear` implements a fully connected layer with a weight matrix, bias vector, and Boolean pruning mask.
- **Optimizer:** The custom Adam optimizer updates trainable parameters and handles optimizer state for changed mask entries.
- **Pruner:** The pruning implementation supports magnitude and gradient-based saliency scores.
- **Experiments:** The training and experiment scripts evaluate accuracy across pruning targets and random seeds.

## 2. Pruning methods

### Magnitude pruning

The importance score for each eligible weight is its absolute value:

`score = abs(weight)`

Weights with lower scores are candidates for pruning.

### Gradient-based saliency pruning

The implementation uses:

`score = abs(weight * gradient)`

This estimates how strongly a weight and its current gradient contribute to a local change in the loss. It is a heuristic, not a guarantee that removing a weight will have the smallest possible effect on final accuracy.

The experiment compares both methods at matched target sparsity levels.

## 3. Masking and optimizer state

Each layer maintains a Boolean mask with the same shape as its weight matrix. Active connections have a true mask entry; pruned connections have a false entry.

The forward computation multiplies the weights by the mask, preventing pruned connections from contributing to the output. The layer also sets pruned weight values to zero.

The optimizer resets state for entries whose masks change. This prevents stale Adam momentum from being carried into a connection when its mask status changes.

## 4. Experimental design

The pruning experiment evaluates two methods at five target sparsities:

- 0%
- 50%
- 75%
- 90%
- 95%

Five random seeds are used for each method and target. The resulting 50 runs are saved to a CSV file.

The validator checks the result file and summarizes mean test accuracy and standard deviation across seeds.

## 5. Observed accuracy results

In the current five-seed experiment, mean test accuracy was:

| Target sparsity | Magnitude | Saliency |
|---:|---:|---:|
| 0% | 99.80% | 99.80% |
| 50% | 99.80% | 99.80% |
| 75% | 99.40% | 99.80% |
| 90% | 98.40% | 98.60% |
| 95% | 66.80% | 56.40% |

These results show that the model retains high accuracy up to 90% target sparsity on this synthetic dataset. At 95% sparsity, accuracy decreases substantially and varies considerably between seeds.

Saliency performs slightly better at 75% and 90% sparsity in this experiment, but it performs worse at 95%. The evidence therefore does not support claiming that saliency always outperforms magnitude pruning.

## 6. Computational cost and limitations

The layer's standard forward path uses dense matrix multiplication after applying the mask. Zero-valued weights do not automatically reduce the amount of dense matrix computation.

The separate `forward_sparse` method skips inactive connections using Python loops. This demonstrates how inactive connections can be excluded from the calculation, but Python-level loops introduce substantial overhead. The method may be slower than dense NumPy matrix multiplication.

Benchmark results should compare equivalent inference operations, use repeated measurements, and record the execution environment. Parameter sparsity and measured latency must be reported separately.

Potential improvements include using an optimized sparse-matrix representation, profiling memory usage, benchmarking on larger models, and evaluating on a real-world dataset.

## 7. Further validation

Before using this implementation in a production setting, additional work would be needed:

- Broader numerical gradient checks.
- More datasets and model architectures.
- Repeated benchmarks across hardware and batch sizes.
- Memory and inference-latency profiling.
- Testing with an optimized sparse inference backend.
