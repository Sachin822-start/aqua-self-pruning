
import numpy as np


def unbroadcast(grad, shape):
    """Reduce a broadcast gradient to the original input shape."""
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)

    for axis, size in enumerate(shape):
        if size == 1 and grad.shape[axis] != 1:
            grad = grad.sum(axis=axis, keepdims=True)

    return grad.reshape(shape)


class Tensor:
    """A NumPy tensor supporting reverse-mode automatic differentiation."""

    def __init__(self, data, requires_grad=True, _children=(), _op=""):
        self.data = np.asarray(data, dtype=np.float64)
        self.requires_grad = requires_grad
        self.grad = np.zeros_like(self.data) if requires_grad else None

        self._prev = set(_children)
        self._op = _op
        self._backward = lambda: None

        # True only after backward() has calculated this tensor's gradient.
        self._grad_computed = False

    def __repr__(self):
        return f"Tensor(data={self.data}, grad={self.grad})"

    def _make_result(self, data, children, op, backward_fn):
        result = Tensor(
            data,
            requires_grad=any(x.requires_grad for x in children),
            _children=children,
            _op=op,
        )
        result._backward = backward_fn
        return result

    def __add__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other, requires_grad=False)

        out = self._make_result(
            self.data + other.data, (self, other), "add", lambda: None
        )

        def backward():
            if self.requires_grad:
                self.grad += unbroadcast(out.grad, self.data.shape)
            if other.requires_grad:
                other.grad += unbroadcast(out.grad, other.data.shape)

        out._backward = backward
        return out

    __radd__ = __add__

    def __neg__(self):
        return self * -1.0

    def __sub__(self, other):
        return self + (-other if isinstance(other, Tensor) else -other)

    def __rsub__(self, other):
        return (-self) + other

    def __mul__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other, requires_grad=False)

        out = self._make_result(
            self.data * other.data, (self, other), "mul", lambda: None
        )

        def backward():
            if self.requires_grad:
                self.grad += unbroadcast(
                    other.data * out.grad, self.data.shape
                )
            if other.requires_grad:
                other.grad += unbroadcast(
                    self.data * out.grad, other.data.shape
                )

        out._backward = backward
        return out

    __rmul__ = __mul__

    def __truediv__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other, requires_grad=False)

        out = self._make_result(
            self.data / other.data, (self, other), "div", lambda: None
        )

        def backward():
            if self.requires_grad:
                self.grad += unbroadcast(
                    out.grad / other.data, self.data.shape
                )
            if other.requires_grad:
                other.grad += unbroadcast(
                    -out.grad * self.data / (other.data ** 2),
                    other.data.shape,
                )

        out._backward = backward
        return out

    def __matmul__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other, requires_grad=False)

        out = self._make_result(
            self.data @ other.data, (self, other), "matmul", lambda: None
        )

        def backward():
            if self.requires_grad:
                self.grad += out.grad @ np.swapaxes(other.data, -1, -2)
            if other.requires_grad:
                other.grad += np.swapaxes(self.data, -1, -2) @ out.grad

        out._backward = backward
        return out

    def sum(self, axis=None, keepdims=False):
        out = self._make_result(
            self.data.sum(axis=axis, keepdims=keepdims),
            (self,),
            "sum",
            lambda: None,
        )

        def backward():
            if not self.requires_grad:
                return

            grad = out.grad

            if axis is not None and not keepdims:
                axes = (axis,) if isinstance(axis, int) else tuple(axis)
                axes = sorted(
                    a if a >= 0 else a + self.data.ndim for a in axes
                )
                for ax in axes:
                    grad = np.expand_dims(grad, axis=ax)

            self.grad += np.broadcast_to(grad, self.data.shape)

        out._backward = backward
        return out

    def mean(self, axis=None, keepdims=False):
        if axis is None:
            count = self.data.size
        else:
            axes = (axis,) if isinstance(axis, int) else tuple(axis)
            count = int(
                np.prod([
                    self.data.shape[a if a >= 0 else a + self.data.ndim]
                    for a in axes
                ])
            )

        return self.sum(axis=axis, keepdims=keepdims) / count

    def relu(self):
        out = self._make_result(
            np.maximum(0, self.data), (self,), "relu", lambda: None
        )

        def backward():
            if self.requires_grad:
                self.grad += (self.data > 0) * out.grad

        out._backward = backward
        return out

    def tanh(self):
        value = np.tanh(self.data)
        out = self._make_result(value, (self,), "tanh", lambda: None)

        def backward():
            if self.requires_grad:
                self.grad += (1.0 - value**2) * out.grad

        out._backward = backward
        return out

    def sigmoid(self):
        value = np.empty_like(self.data)
        positive = self.data >= 0

        value[positive] = 1.0 / (1.0 + np.exp(-self.data[positive]))

        exp_x = np.exp(self.data[~positive])
        value[~positive] = exp_x / (1.0 + exp_x)

        out = self._make_result(value, (self,), "sigmoid", lambda: None)

        def backward():
            if self.requires_grad:
                self.grad += value * (1.0 - value) * out.grad

        out._backward = backward
        return out

    def backward(self):
        if self.data.size != 1:
            raise ValueError(
                "backward() requires a scalar output. "
                "Use sum() or mean() first."
            )

        topo = []
        visited = set()

        def build(node):
            if node not in visited:
                visited.add(node)
                for parent in node._prev:
                    build(parent)
                topo.append(node)

        build(self)

        # Clear gradients and computed flags for this graph.
        for node in topo:
            node._grad_computed = False
            if node.requires_grad:
                node.grad.fill(0.0)

        if self.requires_grad:
            self.grad[...] = 1.0

        for node in reversed(topo):
            node._backward()

        # Mark gradients as computed, including valid zero gradients.
        for node in topo:
            if node.requires_grad:
                node._grad_computed = True


def softmax_cross_entropy(logits, labels):
    """Numerically stable mean cross-entropy for (batch, classes) logits."""
    if logits.data.ndim != 2:
        raise ValueError("Logits must have shape (batch, classes).")

    labels = np.asarray(labels, dtype=np.int64)
    batch_size, num_classes = logits.data.shape

    if labels.shape != (batch_size,):
        raise ValueError("Labels must have shape (batch,).")

    if np.any(labels < 0) or np.any(labels >= num_classes):
        raise ValueError("Labels contain an invalid class index.")

    shifted = logits.data - np.max(logits.data, axis=1, keepdims=True)
    exp_values = np.exp(shifted)
    probabilities = exp_values / exp_values.sum(axis=1, keepdims=True)

    chosen = probabilities[np.arange(batch_size), labels]
    loss_value = -np.log(np.clip(chosen, 1e-15, 1.0)).mean()

    loss = Tensor(
        loss_value,
        requires_grad=logits.requires_grad,
        _children=(logits,),
        _op="softmax_cross_entropy",
    )

    def backward():
        if logits.requires_grad:
            grad = probabilities.copy()
            grad[np.arange(batch_size), labels] -= 1.0
            logits.grad += grad * loss.grad / batch_size

    loss._backward = backward
    return loss
