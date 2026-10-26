
import numpy as np


def prune_model(model, optimizer, target_sparsity, method="saliency"):
    """
    Globally prune active weights to a target sparsity.

    magnitude: score = abs(weight)
    saliency:  score = abs(weight * gradient)

    Pruned connections remain pruned. Newly pruned connections have their
    optimizer state reset by optimizer.set_mask().
    """
    if not 0.0 <= target_sparsity < 1.0:
        raise ValueError("target_sparsity must be >= 0 and < 1.")

    if method not in ("magnitude", "saliency"):
        raise ValueError("method must be 'magnitude' or 'saliency'.")

    layers = model.layers()
    total_weights = sum(layer.weight.data.size for layer in layers)

    target_pruned = int(np.ceil(target_sparsity * total_weights))
    currently_pruned = sum(
        int(np.count_nonzero(~layer.mask)) for layer in layers
    )
    to_prune = max(0, target_pruned - currently_pruned)

    candidates = []

    for layer_index, layer in enumerate(layers):
        weight = layer.weight.data

        if method == "magnitude":
            scores = np.abs(weight)
        else:
            if (
                not layer.weight.requires_grad
                or layer.weight.grad is None
                or not layer.weight._grad_computed
            ):
                raise ValueError(
                    "Saliency pruning requires a fresh backward pass. "
                    "Compute loss.backward() before pruning."
                )

            scores = np.abs(weight * layer.weight.grad)

        for index in np.ndindex(weight.shape):
            if layer.mask[index]:
                candidates.append(
                    (float(scores[index]), layer_index, index)
                )

    to_prune = min(to_prune, len(candidates))
    candidates.sort(key=lambda item: item[0])

    new_masks = [layer.mask.copy() for layer in layers]

    for _, layer_index, index in candidates[:to_prune]:
        new_masks[layer_index][index] = False

    # Optimizer order: weight, bias for each layer.
    for layer_index, layer in enumerate(layers):
        optimizer.set_mask(2 * layer_index, new_masks[layer_index])
        layer.set_mask(new_masks[layer_index])

    active_weights = sum(
        int(np.count_nonzero(layer.mask)) for layer in layers
    )

    return {
        "method": method,
        "target_sparsity": target_sparsity,
        "achieved_sparsity": 1.0 - active_weights / total_weights,
        "total_weights": total_weights,
        "active_weights": active_weights,
        "pruned_weights": total_weights - active_weights,
    }
