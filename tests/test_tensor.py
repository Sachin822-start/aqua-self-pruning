
import numpy as np
from engine.tensor import Tensor, softmax_cross_entropy


def numerical_gradient(function, values, eps=1e-5):
    """Estimate gradients using central finite differences."""
    gradients = np.zeros_like(values, dtype=float)

    for index in np.ndindex(values.shape):
        original = values[index]

        values[index] = original + eps
        positive = function(values)

        values[index] = original - eps
        negative = function(values)

        values[index] = original
        gradients[index] = (positive - negative) / (2 * eps)

    return gradients


def check_gradient(function, values, atol=1e-5):
    values = np.array(values, dtype=float)

    x = Tensor(values.copy())
    loss = function(x)
    loss.backward()

    numerical = numerical_gradient(
        lambda v: float(function(Tensor(v.copy())).data),
        values.copy(),
    )

    np.testing.assert_allclose(
        x.grad, numerical, atol=atol, rtol=1e-4
    )


def test_add_gradient():
    check_gradient(
        lambda x: (x + 3.0).mean(),
        np.array([1.0, 2.0, 4.0]),
    )


def test_multiply_gradient():
    check_gradient(
        lambda x: (x * x).sum(),
        np.array([1.0, 2.0, 3.0]),
    )


def test_division_gradient():
    check_gradient(
        lambda x: (x / 2.0).sum(),
        np.array([1.0, 2.0, 4.0]),
    )


def test_matmul_gradient():
    values = np.array([[1.0, 2.0], [3.0, 4.0]])
    other = np.array([[2.0, 1.0], [1.0, 3.0]])

    def function(x):
        return (x @ Tensor(other, requires_grad=False)).sum()

    check_gradient(function, values)


def test_sum_gradient():
    check_gradient(
        lambda x: x.sum(),
        np.array([1.0, 2.0, 3.0]),
    )


def test_mean_gradient():
    check_gradient(
        lambda x: x.mean(),
        np.array([1.0, 2.0, 3.0]),
    )


def test_relu_gradient():
    check_gradient(
        lambda x: x.relu().sum(),
        np.array([-2.0, -0.5, 1.0, 3.0]),
    )


def test_tanh_gradient():
    check_gradient(
        lambda x: x.tanh().sum(),
        np.array([-1.0, 0.2, 1.5]),
    )


def test_sigmoid_gradient():
    check_gradient(
        lambda x: x.sigmoid().sum(),
        np.array([-2.0, 0.2, 1.5]),
    )


def test_broadcasting_gradient():
    x = Tensor(np.ones((2, 3)))
    bias = Tensor(np.array([1.0, 2.0, 3.0]))

    loss = (x + bias).sum()
    loss.backward()

    np.testing.assert_allclose(x.grad, np.ones((2, 3)))
    np.testing.assert_allclose(bias.grad, np.array([2.0, 2.0, 2.0]))


def test_softmax_cross_entropy_gradient():
    logits = np.array([
        [1.0, 2.0, -0.5],
        [0.2, -0.1, 1.2],
    ])
    labels = np.array([1, 2])

    def function(x):
        return softmax_cross_entropy(x, labels)

    check_gradient(function, logits, atol=2e-5)
