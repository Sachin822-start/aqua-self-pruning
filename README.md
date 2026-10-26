<<<<<<< HEAD
# AQUA Self-Pruning Neural Network

## Overview

This project implements a small neural-network training and pruning system using Python and NumPy. It includes a custom reverse-mode automatic differentiation engine, an MLP classifier, an optimizer, and experiments comparing magnitude-based and gradient-based saliency pruning.

## Features

- Custom Tensor engine with reverse-mode  autodiff.
- Arithmetic operations, matrix multiplication, reductions, activation functions, and softmax cross-entropy.
- MLP classifier trained with a custom Adam optimizer.
- Magnitude pruning and gradient-based saliency pruning.
- Weight masks that prevent pruned connections from contributing to the forward pass.
- Tests for gradients, layers, pruning, and masked weights.
- Multi-seed pruning experiments with raw CSV results and a sparsity–accuracy plot.
- Experimental sparse-aware inference benchmark.

## Requirements

- Python 3.10 or newer
- NumPy
- Matplotlib
- pytest

## Setup

Create and activate a virtual environment, then install the dependencies:

```bash
python -m venv .venv
```

Windows Command Prompt:

```cmd
.venv\Scripts\activate
python -m pip install numpy matplotlib pytest
```

## Run the tests

```bash
python -m pytest tests -v
```

## Train the baseline MLP

```bash
python -m train.train_mlp
```

## Run the pruning experiments

```bash
python -m train.pruning_experiment
```

This generates the experiment CSV and plot under `results/`.

## Validate experiment results

```bash
python scripts/validate_results.py
```

## Benchmark inference

```bash
python scripts/benchmark_inference.py
```

The benchmark compares dense masked inference with the project's sparse-aware inference implementation. Results depend on the hardware, Python and NumPy versions, and implementation details. Sparse-aware inference is not guaranteed to be faster.

## Results

The experiment evaluates magnitude and saliency pruning at target sparsities of 0%, 50%, 75%, 90%, and 95%, using five random seeds.

The checked-in CSV contains the measured results. Use the generated plot and CSV when reporting accuracy and sparsity. Do not claim one pruning method is universally better unless the results support that conclusion.

## Limitations

- Experiments use a synthetic classification dataset; results do not establish performance on real-world datasets.
- The sparse-aware inference method uses Python loops and may be slower than dense matrix multiplication.
- Weight sparsity alone does not guarantee lower latency or memory use.
- The implementation is an educational prototype, not a production inference engine.

## Project structure

- `engine/` — autodiff Tensor implementation.
- `nn/` — layers, model, and optimizer.
- `train/` — baseline training and pruning experiments.
- `prune/` — pruning logic.
- `tests/` — automated tests.
- `scripts/` — validation and benchmark scripts.
- `results/` — experiment CSV and plot.
=======
# aqua-self-pruning
AQUA self-pruning neural network with custom autodiff, saliency pruning, and multi-seed experiments.
>>>>>>> origin/main
