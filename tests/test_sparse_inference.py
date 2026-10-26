
import numpy as np

from nn.layers import Linear


def test_sparse_inference_matches_dense_forward():
    layer = Linear(4, 3, seed=42)

    layer.set_mask(
        np.array([
            [True, False, True],
            [False, True, False],
            [True, True, False],
            [False, False, True],
        ])
    )

    x = np.array([
        [1.0, 2.0, 3.0, 4.0],
        [0.5, -1.0, 2.0, 0.0],
    ])

    dense_output = layer.forward(x).data
    sparse_output = layer.forward_sparse(x)

    np.testing.assert_allclose(
        sparse_output,
        dense_output,
        rtol=1e-10,
        atol=1e-10,
    )


def test_sparse_inference_with_all_weights_pruned():
    layer = Linear(3, 2, seed=42)
    layer.set_mask(np.zeros((3, 2), dtype=bool))
    layer.bias.data[:] = [1.5, -2.0]

    x = np.array([
        [1.0, 2.0, 3.0],
        [-1.0, 0.0, 4.0],
    ])

    output = layer.forward_sparse(x)

    expected = np.array([
        [1.5, -2.0],
        [1.5, -2.0],
    ])

    np.testing.assert_allclose(output, expected)
