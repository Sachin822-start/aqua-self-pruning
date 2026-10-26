
import csv
import math
from collections import defaultdict
from pathlib import Path

RESULTS_FILE = Path("results/pruning_results.csv")

METHODS = {"magnitude", "saliency"}
TARGETS = {0.0, 0.50, 0.75, 0.90, 0.95}
SEEDS = {42, 43, 44, 45, 46}

REQUIRED_COLUMNS = {
    "method",
    "seed",
    "target_sparsity",
    "achieved_sparsity",
    "total_weights",
    "active_weights",
    "test_loss",
    "test_accuracy",
}


def main():
    if not RESULTS_FILE.exists():
        raise FileNotFoundError(f"Missing file: {RESULTS_FILE}")

    with RESULTS_FILE.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        rows = list(reader)
        columns = set(reader.fieldnames or [])

    assert REQUIRED_COLUMNS <= columns, (
        f"Missing columns: {REQUIRED_COLUMNS - columns}"
    )

    expected_count = len(METHODS) * len(TARGETS) * len(SEEDS)
    assert len(rows) == expected_count, (
        f"Expected {expected_count} rows, found {len(rows)}"
    )

    seen = set()
    grouped_accuracy = defaultdict(list)

    for row in rows:
        method = row["method"]
        seed = int(row["seed"])
        target = float(row["target_sparsity"])
        achieved = float(row["achieved_sparsity"])
        total = int(row["total_weights"])
        active = int(row["active_weights"])
        loss = float(row["test_loss"])
        accuracy = float(row["test_accuracy"])

        assert method in METHODS
        assert seed in SEEDS
        assert target in TARGETS

        key = (method, seed, target)
        assert key not in seen, f"Duplicate run: {key}"
        seen.add(key)

        assert total > 0
        assert 0 <= active <= total
        assert math.isclose(
            achieved, 1.0 - active / total, abs_tol=1e-9
        )
        assert math.isfinite(loss) and loss >= 0
        assert math.isfinite(accuracy) and 0 <= accuracy <= 1

        # Allow at most one connection of sparsity rounding error.
        assert achieved + 1e-9 >= target - 1.0 / total, (
            f"Target not reached within rounding tolerance: {key}, "
            f"achieved={achieved:.4f}"
        )

        grouped_accuracy[(method, target)].append(accuracy)

    assert len(seen) == expected_count

    print("CSV validation PASSED")
    print(f"Runs validated: {len(rows)}")
    print(f"Seeds: {sorted(SEEDS)}")
    print("\nMean test accuracy across seeds:")

    for method in sorted(METHODS):
        for target in sorted(TARGETS):
            values = grouped_accuracy[(method, target)]
            mean = sum(values) / len(values)
            variance = sum((v - mean) ** 2 for v in values) / (
                len(values) - 1
            )
            std = math.sqrt(variance)

            print(
                f"{method:9s} target={target:.0%} "
                f"accuracy={mean:.2%} "
                f"std={std:.2%} "
                f"n={len(values)}"
            )


if __name__ == "__main__":
    main()
