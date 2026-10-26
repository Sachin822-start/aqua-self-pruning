
import numpy as np

from engine.tensor import Tensor
from nn.optim import Adam


def test_pruned_weight_stays_zero_and_state_is_reset():
    weight = Tensor(np.array([2.0, 3.0]))
    mask = np.array([True, True])

    optimizer = Adam([weight], lr=0.1, masks=[mask])

    # First, train both connections so Adam has nonzero state.
    weight.grad[:] = [1.0, 1.0]
    optimizer.step()

    assert np.any(optimizer.m[0] != 0.0)

    # Prune the second connection.
    mask = np.array([True, False])
    optimizer.set_mask(0, mask)

    assert weight.data[1] == 0.0
    assert optimizer.m[0][1] == 0.0
    assert optimizer.v[0][1] == 0.0

    # Simulate a gradient arriving at the pruned parameter.
    weight.grad[:] = [1.0, 100.0]
    optimizer.step()

    assert weight.data[1] == 0.0
    assert optimizer.m[0][1] == 0.0
    assert optimizer.v[0][1] == 0.0


def test_revived_weight_starts_with_fresh_optimizer_state():
    weight = Tensor(np.array([2.0]))
    optimizer = Adam([weight], lr=0.1, masks=[np.array([True])])

    weight.grad[:] = 2.0
    optimizer.step()

    # Prune the connection, then revive it.
    optimizer.set_mask(0, np.array([False]))
    optimizer.set_mask(0, np.array([True]))

    assert optimizer.m[0][0] == 0.0
    assert optimizer.v[0][0] == 0.0

    weight.grad[:] = 1.0
    optimizer.step()

    assert weight.data[0] != 0.0
    assert np.isfinite(weight.data[0])
