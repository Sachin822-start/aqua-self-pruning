
import numpy as np

from engine.tensor import Tensor
from nn.layers import Linear


def test_linear_output_shape():
    layer = Linear(3, 2, seed=42)
    x = Tensor(np.ones((4, 3)), requires_grad=False)

    output = layer.forward(x)

    assert output.data.shape == (4, 2)


def test_linear_backward():
    layer = Linear(3, 2, seed=42)
    x = Tensor(np.ones((4, 3)), requires_grad=False)

    output = layer.forward(x)
    loss = output.mean()
    loss.backward()

    assert layer.weight.grad.shape == (3, 2)
    assert layer.bias.grad.shape == (2,)
    assert np.all(np.isfinite(layer.weight.grad))
    assert np.all(np.isfinite(layer.bias.grad))


def test_masked_connections_have_zero_effect_and_gradient():
    layer = Linear(2, 1, seed=42)

    layer.weight.data[:] = [[2.0], [3.0]]
    layer.bias.data[:] = [0.0]

    layer.set_mask(np.array([[True], [False]]))

    x = Tensor([[1.0, 100.0]], requires_grad=False)
    output = layer.forward(x)

    # The second input's connection is pruned.
    np.testing.assert_allclose(output.data, [[2.0]])

    output.sum().backward()

    # The pruned weight receives zero gradient.
    assert layer.weight.grad[1, 0] == 0.0


def test_invalid_mask_shape():
    layer = Linear(2, 3, seed=42)

    try:
        layer.set_mask(np.ones((2, 2), dtype=bool))
    except ValueError:
        pass
    else:
        raise AssertionError("Invalid mask shape should raise ValueError.")
