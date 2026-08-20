"""Neuron importance — identify which hidden-layer neurons matter most per class.

For each (class, layer) pair, we measure how much each neuron's activation
responds to inputs of that class versus all other classes. The attacker uses
this to surgically perturb only the hidden neurons that encode the target
class's representation, leaving the output layer untouched.

Importance score per neuron:
    score = mean_activation(class_c) / (mean_activation(all_classes) + eps)

A score >> 1 means the neuron fires disproportionately for class c.
We return the top-k indices per (class, layer) sorted by score descending.
"""

import logging
from collections import defaultdict

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def compute_neuron_importance(
    model: nn.Module,
    data_loader,
    n_classes: int = 10,
    top_k: int = 5,
    device: str = "cpu",
) -> dict[int, dict[str, list[int]]]:
    """Compute per-class neuron importance for all HIDDEN layers.

    Args:
        model: the global model (used in eval mode).
        data_loader: DataLoader yielding (inputs, labels).
        n_classes: number of classes in the task.
        top_k: how many top neurons to return per (class, layer).
        device: torch device.

    Returns:
        {class_id: {layer_name: [top-k neuron indices]}}
        Only includes hidden layers (excludes the output/classifier layer).
    """
    model = model.to(device)
    model.eval()

    # Identify all Linear layers and their names in the state_dict convention.
    linear_layers = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            # state_dict key is e.g. "net.2" for nn.Sequential
            linear_layers[name] = module

    if not linear_layers:
        logger.warning("No Linear layers found in model")
        return {}

    # The OUTPUT layer is the last Linear layer (the one with out_features == n_classes).
    # We exclude it — the attack must not touch the output layer.
    layer_names = list(linear_layers.keys())
    output_layer_name = None
    for name in reversed(layer_names):
        if linear_layers[name].out_features == n_classes:
            output_layer_name = name
            break
    if output_layer_name is None:
        # Fallback: just treat the last Linear as the output layer
        output_layer_name = layer_names[-1]

    hidden_layers = {n: m for n, m in linear_layers.items() if n != output_layer_name}
    if not hidden_layers:
        logger.warning("No hidden Linear layers found (only output layer exists)")
        return {}

    logger.info(f"Neuron importance: hidden_layers={list(hidden_layers.keys())}, "
                f"output_layer={output_layer_name} (excluded)")

    # Collect activations per class per layer.
    # activation_sums[layer_name][class_id] = sum of neuron activations (vector)
    # activation_counts[class_id] = count of samples
    activation_sums: dict[str, dict[int, torch.Tensor]] = {
        name: defaultdict(lambda dev=device, m=mod: torch.zeros(m.out_features, device=dev))
        for name, mod in hidden_layers.items()
    }
    class_counts: dict[int, int] = defaultdict(int)

    # Register forward hooks to capture activations.
    hooks = []
    captured: dict[str, torch.Tensor] = {}

    def _make_hook(layer_name):
        def hook_fn(module, inp, out):
            # out shape: [batch, out_features] for Linear
            captured[layer_name] = out.detach()
        return hook_fn

    for name, mod in hidden_layers.items():
        hooks.append(mod.register_forward_hook(_make_hook(name)))

    try:
        with torch.no_grad():
            for batch_x, batch_y in data_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                _ = model(batch_x)  # triggers hooks

                for c in range(n_classes):
                    mask = (batch_y == c)
                    if mask.sum() == 0:
                        continue
                    class_counts[c] += mask.sum().item()
                    for name in hidden_layers:
                        act = captured[name]  # [batch, neurons]
                        # Mean activation magnitude for class c samples
                        class_act = act[mask].abs().sum(dim=0)  # [neurons]
                        activation_sums[name][c] = activation_sums[name][c] + class_act
    finally:
        for h in hooks:
            h.remove()

    # Compute importance: class-conditional mean / global mean
    eps = 1e-8
    result: dict[int, dict[str, list[int]]] = {}

    for c in range(n_classes):
        if class_counts.get(c, 0) == 0:
            continue
        result[c] = {}
        for name in hidden_layers:
            class_mean = activation_sums[name][c] / (class_counts[c] + eps)
            # Global mean across ALL classes
            global_sum = torch.zeros_like(class_mean)
            global_count = 0
            for cc in range(n_classes):
                if class_counts.get(cc, 0) > 0:
                    global_sum += activation_sums[name][cc]
                    global_count += class_counts[cc]
            global_mean = global_sum / (global_count + eps)

            # Importance = class-specific / global (how much MORE this neuron fires for class c)
            importance = class_mean / (global_mean + eps)

            # Top-k indices
            k = min(top_k, importance.numel())
            _, top_indices = importance.topk(k)
            # Convert state_dict key: "net.2" → "net.2.weight"
            weight_key = f"{name}.weight"
            result[c][weight_key] = top_indices.cpu().tolist()

    logger.info(f"Neuron importance computed for {len(result)} classes, "
                f"top_k={top_k} per layer")
    return result
