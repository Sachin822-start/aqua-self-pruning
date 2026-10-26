
import csv
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from engine.tensor import Tensor, softmax_cross_entropy
from nn.model import MLP
from nn.optim import Adam
from train.train_mlp import make_dataset, evaluate
from prune.pruner import prune_model


TARGET_SPARSITIES = [0.0, 0.50, 0.75, 0.90, 0.95]
METHODS = ["magnitude", "saliency"]
SEEDS = [42, 43, 44, 45, 46]
EPOCHS = 40
BATCH_SIZE = 32


def train_pruned_model(target_sparsity, method, seed):
    """Train a fresh model with gradual pruning toward the target."""
    x_train, y_train, x_test, y_test = make_dataset(seed=seed)

    model = MLP(2, 16, 2, seed=seed)

    optimizer = Adam(
        model.parameters(),
        lr=0.01,
        masks=[
            model.layer1.mask,
            None,
            model.layer2.mask,
            None,
        ],
    )

    rng = np.random.default_rng(seed)

    for epoch in range(EPOCHS):
        order = rng.permutation(len(y_train))

        for start in range(0, len(order), BATCH_SIZE):
            indices = order[start:start + BATCH_SIZE]
            xb = x_train[indices]
            yb = y_train[indices]

            optimizer.zero_grad()

            logits = model.forward(Tensor(xb, requires_grad=False))
            loss = softmax_cross_entropy(logits, yb)

            loss.backward()
            optimizer.step()

        # Gradually increase sparsity using a cubic schedule.
        progress = (epoch + 1) / EPOCHS
        scheduled_target = target_sparsity * (
            1.0 - (1.0 - progress) ** 3
        )

        if scheduled_target > 0:
            # Use the latest training batch's freshly computed gradients.
            result = prune_model(
                model,
                optimizer,
                target_sparsity=scheduled_target,
                method=method,
            )

    test_loss, test_accuracy = evaluate(model, x_test, y_test)

    total_weights = sum(
        layer.weight.data.size for layer in model.layers()
    )
    active_weights = sum(
        int(np.count_nonzero(layer.mask)) for layer in model.layers()
    )

    return {
        "method": method,
        "seed": seed,
        "target_sparsity": target_sparsity,
        "achieved_sparsity": 1.0 - active_weights / total_weights,
        "total_weights": total_weights,
        "active_weights": active_weights,
        "test_loss": test_loss,
        "test_accuracy": test_accuracy,
    }


def main():
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    rows = []

    for method in METHODS:
        for target in TARGET_SPARSITIES:
            for seed in SEEDS:
                print(
                    f"Running method={method}, "
                    f"target={target:.0%}, seed={seed}"
                )

                row = train_pruned_model(target, method, seed)
                rows.append(row)

                print(
                    f"  achieved={row['achieved_sparsity']:.1%}, "
                    f"test_accuracy={row['test_accuracy']:.2%}, "
                    f"active_weights={row['active_weights']}"
                )

    csv_path = results_dir / "pruning_results.csv"

    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


    # Plot mean accuracy and standard deviation across random seeds.
    plt.figure(figsize=(8, 5))

    for method in METHODS:
        means = []
        stds = []
        sparsities = []

        for target in TARGET_SPARSITIES:
            group = [
                row for row in rows
                if row["method"] == method
                and row["target_sparsity"] == target
            ]

            accuracies = np.array(
                [row["test_accuracy"] for row in group]
            )
            achieved = np.mean(
                [row["achieved_sparsity"] for row in group]
            )

            sparsities.append(achieved * 100)
            means.append(np.mean(accuracies) * 100)
            stds.append(np.std(accuracies, ddof=1) * 100)

        plt.errorbar(
            sparsities,
            means,
            yerr=stds,
            marker="o",
            capsize=4,
            label=method,
        )

    plt.xlabel("Mean achieved sparsity (%)")
    plt.ylabel("Mean test accuracy (%)")
    plt.title("Sparsity vs Accuracy (5 random seeds)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plot_path = results_dir / "sparsity_accuracy.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()

    print("\nExperiment completed.")
    print(f"Raw results: {csv_path}")
    print(f"Plot: {plot_path}")
    print(f"Runs: {len(rows)}")


    for method in METHODS:
        method_rows = [
            row for row in rows if row["method"] == method
        ]
        method_rows.sort(key=lambda row: row["achieved_sparsity"])

        plt.plot(
            [row["achieved_sparsity"] * 100 for row in method_rows],
            [row["test_accuracy"] * 100 for row in method_rows],
            marker="o",
            label=method,
        )

    plt.xlabel("Achieved sparsity (%)")
    plt.ylabel("Test accuracy (%)")
    plt.title("AQUA: Sparsity vs Test Accuracy")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plot_path = results_dir / "sparsity_accuracy.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()

    print("\nExperiment completed.")
    print(f"Raw results: {csv_path}")
    print(f"Plot: {plot_path}")
    print(
        "Note: results currently use one seed. "
        "Run multiple seeds before drawing strong conclusions."
    )


if __name__ == "__main__":
    main()
