
import numpy as np

from engine.tensor import Tensor, softmax_cross_entropy
from nn.model import MLP
from nn.optim import Adam


def make_dataset(seed=42, samples_per_class=250):
    """Create a reproducible two-class synthetic dataset."""
    rng = np.random.default_rng(seed)

    class_0 = rng.normal(
        loc=[-1.5, -1.5],
        scale=0.8,
        size=(samples_per_class, 2),
    )
    class_1 = rng.normal(
        loc=[1.5, 1.5],
        scale=0.8,
        size=(samples_per_class, 2),
    )

    x = np.vstack([class_0, class_1])
    y = np.concatenate([
        np.zeros(samples_per_class, dtype=np.int64),
        np.ones(samples_per_class, dtype=np.int64),
    ])

    permutation = rng.permutation(len(y))
    x, y = x[permutation], y[permutation]

    split = int(0.8 * len(y))
    return x[:split], y[:split], x[split:], y[split:]


def evaluate(model, x, y):
    logits = model.forward(Tensor(x, requires_grad=False))
    loss = softmax_cross_entropy(logits, y)

    predictions = np.argmax(logits.data, axis=1)
    accuracy = np.mean(predictions == y)

    return float(loss.data), float(accuracy)


def main():
    np.random.seed(42)

    x_train, y_train, x_test, y_test = make_dataset()

    model = MLP(
        input_size=2,
        hidden_size=16,
        num_classes=2,
        seed=42,
    )

    # Each weight matrix has a mask. Bias vectors are unmasked.
    masks = [
        model.layer1.mask,
        None,
        model.layer2.mask,
        None,
    ]

    optimizer = Adam(
        model.parameters(),
        lr=0.01,
        masks=masks,
    )

    epochs = 40
    batch_size = 32
    rng = np.random.default_rng(42)

    for epoch in range(epochs):
        order = rng.permutation(len(y_train))

        for start in range(0, len(order), batch_size):
            indices = order[start:start + batch_size]
            xb = x_train[indices]
            yb = y_train[indices]

            optimizer.zero_grad()

            logits = model.forward(
                Tensor(xb, requires_grad=False)
            )
            loss = softmax_cross_entropy(logits, yb)

            loss.backward()
            optimizer.step()

        train_loss, train_acc = evaluate(model, x_train, y_train)
        test_loss, test_acc = evaluate(model, x_test, y_test)

        print(
            f"Epoch {epoch + 1:02d}/{epochs} | "
            f"train_loss={train_loss:.4f} | "
            f"train_acc={train_acc:.3f} | "
            f"test_loss={test_loss:.4f} | "
            f"test_acc={test_acc:.3f}"
        )

        if not np.isfinite(train_loss) or not np.isfinite(test_loss):
            raise RuntimeError("Non-finite loss detected.")

    print("\nTraining completed.")
    print(f"Final test accuracy: {test_acc:.2%}")


if __name__ == "__main__":
    main()
