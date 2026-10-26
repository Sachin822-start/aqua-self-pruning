
import numpy as np
from engine.tensor import Tensor


class Linear:
    """Fully connected neural-network layer: output = input @ weight + bias."""

    def __init__(self, in_features, out_features, seed=None):
        if in_features <= 0 or out_features <= 0:
            raise ValueError("Feature dimensions must be positive.")

        rng = np.random.default_rng(seed)

        # He initialization for layers followed by ReLU.
        scale = np.sqrt(2.0 / in_features)
        initial_weights = rng.standard_normal(
            (in_features, out_features)
        ) * scale

        self.weight = Tensor(initial_weights)
        self.bias = Tensor(np.zeros(out_features))

        # True means active; False means pruned.
        self.mask = np.ones(
            (in_features, out_features), dtype=bool
        )

    def forward(self, x):
        if not isinstance(x, Tensor):
            x = Tensor(x, requires_grad=False)

        if x.data.ndim != 2:
            raise ValueError("Input must have shape (batch, features).")

        if x.data.shape[1] != self.weight.data.shape[0]:
            raise ValueError("Input feature count does not match layer.")

        # Apply the mask in the actual forward computation.
        effective_weight = self.weight * Tensor(
            self.mask.astype(float), requires_grad=False
        )

        return x @ effective_weight + self.bias

    def parameters(self):
        return [self.weight, self.bias]

    def set_mask(self, new_mask):
        new_mask = np.asarray(new_mask, dtype=bool)

        if new_mask.shape != self.weight.data.shape:
            raise ValueError("Mask shape must match the weight matrix.")

        self.mask = new_mask.copy()
        self.weight.data[~self.mask] = 0.0

    
    def forward_sparse(self, x):
        """Run inference while skipping pruned connections.

        This method uses NumPy arrays directly and does not build an
        autodiff graph. It is intended for inference, not training.
        """
        x = np.asarray(x, dtype=float)

        if x.ndim != 2:
            raise ValueError("Input must have shape (batch, features).")

        if x.shape[1] != self.weight.data.shape[0]:
            raise ValueError("Input feature count does not match layer.")

        batch_size = x.shape[0]
        out_features = self.weight.data.shape[1]
        output = np.zeros((batch_size, out_features), dtype=float)

        # Process only connections whose masks are active.
        for input_index in range(self.weight.data.shape[0]):
            active_outputs = np.flatnonzero(self.mask[input_index])

            if active_outputs.size == 0:
                continue

            output[:, active_outputs] += (
                x[:, input_index:input_index + 1]
                * self.weight.data[input_index, active_outputs]
            )

        # Biases are not pruned in this project.
        output += self.bias.data

        return output

