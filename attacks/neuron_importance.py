"""Neuron importance computation for targeted class poisoning.

This module computes the importance of individual neurons across all hidden layers
for a specific target class. It uses both Average Activation and Gradient Magnitude
to find the most critical neurons for the class.
"""

import logging
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

def compute_neuron_importance(model: nn.Module, dataloader, target_class: int, device: str = "cpu", top_k: int = 5):
    """
    Computes neuron importance across both Linear and Conv2d layers for a target class.
    Uses both average activation and gradient magnitude.
    
    Returns:
        dict: mapping layer names (e.g., 'net.2', 'net.4') to lists of the most 
              important neuron indices (up to top_k).
    """
    model.eval()
    activations = {}
    gradients = {}
    
    # Register hooks
    def get_activation(name):
        def hook(model, input, output):
            if name not in activations:
                activations[name] = []
            activations[name].append(output.detach())
        return hook

    def get_gradient(name):
        def hook(model, grad_input, grad_output):
            if name not in gradients:
                gradients[name] = []
            gradients[name].append(grad_output[0].detach())
        return hook

    handles = []
    layer_names = []
    
    # Find all linear and conv layers to hook
    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            layer_names.append(name)
            handles.append(module.register_forward_hook(get_activation(name)))
            # Use register_full_backward_hook for gradients
            handles.append(module.register_full_backward_hook(get_gradient(name)))

    criterion = nn.CrossEntropyLoss()
    
    # Process only samples of the target class
    samples_processed = 0
    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        samples_processed += inputs.size(0)

        model.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()

    # Remove hooks to clean up
    for h in handles:
        h.remove()
        
    if samples_processed == 0:
        return {} # No samples of target class found in the provided dataloader

    importance_scores = {}
    
    for name in layer_names:
        if name not in activations or name not in gradients:
            continue
            
        # Average activation: (N, out_features) or (N, C, H, W) -> mean over N (and H, W)
        act = torch.cat(activations[name], dim=0)
        if act.dim() > 2:
            avg_act = act.abs().mean(dim=list(range(2, act.dim()))).mean(dim=0)
        else:
            avg_act = act.abs().mean(dim=0)
        
        # Gradient magnitude: (N, out_features) or (N, C, H, W) -> mean over batches
        grad = torch.cat(gradients[name], dim=0)
        if grad.dim() > 2:
            avg_grad = grad.abs().mean(dim=list(range(2, grad.dim()))).mean(dim=0)
        else:
            avg_grad = grad.abs().mean(dim=0)
        
        # Normalize both to [0, 1] so they can be blended equally
        if avg_act.max() > 0:
            avg_act = avg_act / avg_act.max()
        if avg_grad.max() > 0:
            avg_grad = avg_grad / avg_grad.max()
            
        blended_score = avg_act + avg_grad
        
        # Get top-k indices
        k = min(top_k, blended_score.size(0))
        top_indices = torch.topk(blended_score, k).indices.tolist()
        importance_scores[name] = top_indices
        
        logger.info(f"[Targeted Attack] Class {target_class} | Layer: {name} | Selected Indices: {top_indices}")
        for idx in top_indices:
            logger.info(f"  -> Idx {idx:3d}: Blended={blended_score[idx]:.4f} (AvgAct={avg_act[idx]:.4f}, AvgGrad={avg_grad[idx]:.4f})")
        
    return importance_scores
