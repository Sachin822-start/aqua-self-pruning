
import numpy as np


class Adam:
    """Adam optimizer with explicit support for pruned parameters."""

    def __init__(
        self,
        parameters,
        lr=0.001,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
        masks=None,
    ):
        self.parameters = list(parameters)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0

        if masks is None:
            self.masks = [None] * len(self.parameters)
        else:
            if len(masks) != len(self.parameters):
                raise ValueError("Provide one mask per parameter.")
            self.masks = [
                None if mask is None else np.asarray(mask, dtype=bool).copy()
                for mask in masks
            ]

        self.m = [np.zeros_like(p.data) for p in self.parameters]
        self.v = [np.zeros_like(p.data) for p in self.parameters]

        for p, mask in zip(self.parameters, self.masks):
            if mask is not None and mask.shape != p.data.shape:
                raise ValueError("Mask shape must match parameter shape.")
            if mask is not None:
                p.data[~mask] = 0.0

   
    def zero_grad(self):
        for p in self.parameters:
            if p.grad is not None:
                p.grad.fill(0.0)
            p._grad_computed = False


    def set_mask(self, index, new_mask):
        """Update a parameter's mask and reset state for changed entries."""
        p = self.parameters[index]
        new_mask = np.asarray(new_mask, dtype=bool)

        if new_mask.shape != p.data.shape:
            raise ValueError("Mask shape must match parameter shape.")

        old_mask = self.masks[index]
        if old_mask is None:
            old_mask = np.ones_like(new_mask, dtype=bool)

        changed = old_mask != new_mask
        self.m[index][changed] = 0.0
        self.v[index][changed] = 0.0

        self.masks[index] = new_mask.copy()
        p.data[~new_mask] = 0.0

        if p.grad is not None:
            p.grad[~new_mask] = 0.0

    def step(self):
        self.t += 1

        for i, p in enumerate(self.parameters):
            if p.grad is None:
                continue

            mask = self.masks[i]
            grad = p.grad.copy()

            if mask is not None:
                # Pruned connections receive no optimizer update.
                grad[~mask] = 0.0
                self.m[i][~mask] = 0.0
                self.v[i][~mask] = 0.0

            self.m[i] = (
                self.beta1 * self.m[i] + (1 - self.beta1) * grad
            )
            self.v[i] = (
                self.beta2 * self.v[i] + (1 - self.beta2) * grad**2
            )

            m_hat = self.m[i] / (1 - self.beta1**self.t)
            v_hat = self.v[i] / (1 - self.beta2**self.t)

            p.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

            if mask is not None:
                p.data[~mask] = 0.0
                self.m[i][~mask] = 0.0
                self.v[i][~mask] = 0.0
