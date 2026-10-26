
import numpy as np
import pytest

from engine.tensor import Tensor, softmax_cross_entropy
from nn.model import MLP
from nn.optim import Adam
from prune.pruner import prune_model


def make_model_and_optimizer():
    model = MLP(2, 4, 2, seed=42)
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
    return model, optimizer


@pytest.mark.parametrize("method", ["magnitude", "saliency"])
def test_pruning_reaches_target_and_keeps_weights_zero(method):
    model, optimizer = make_model_and_optimizer()

    x = Tensor([[1.0, 2.0], [-1.0, 0.5]])
    labels = np.array([1, 0])

    loss = softmax_cross_entropy(model.forward(x), labels)
    loss.backward()

    result = prune_model(
        model, optimizer, target_sparsity=0.5, method=method
    )

    assert result["achieved_sparsity"] == 0.5
    assert result["active_weights"] == 8

    for layer in model.layers():
        assert np.all(layer.weight.data[~layer.mask] == 0.0)


def test_invalid_pruning_method():
    model, optimizer = make_model_and_optimizer()

    with pytest.raises(ValueError):
        prune_model(model, optimizer, 0.5, method="unknown")


def test_saliency_requires_gradients():
    model, optimizer = make_model_and_optimizer()

    with pytest.raises(ValueError):
        prune_model(model, optimizer, 0.5, method="saliency")
