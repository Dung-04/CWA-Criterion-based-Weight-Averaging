"""Weight-averaging primitives used by CWA.

Two operations, both experiment-critical — changing either changes the reported
numbers:

``average_weights``
    Uniform element-wise mean of the learnable parameters across checkpoints.
    BatchNorm running statistics are population statistics, not learned
    parameters, so they are NOT averaged; they are carried over from the first
    (best) checkpoint and then re-estimated.

``update_bn``
    Re-estimates those BatchNorm statistics under the averaged weights by
    running forward-only passes over training data — the same idea as
    ``torch.optim.swa_utils.update_bn``.
"""
import copy

import torch

from utils.amp import autocast_context


def update_bn(model, train_loader, device, num_batches=100):
    """
    Update BatchNorm running statistics after loading averaged weights

    IMPORTANT: For frozen backbone models, we should NOT update the backbone BN layers
    because they already have good statistics from ImageNet pretraining.
    We only need to update BN layers in the classifier (if any).

    However, since our custom classifiers don't use BatchNorm,
    we can skip this step entirely for most cases.

    For safety, we only update BN layers that are in trainable (unfrozen) parts.

    Args:
        model: Model with averaged weights
        train_loader: Training data loader
        device: Device to run on
        num_batches: Number of batches to use for BN update (default 100)
    """
    # First, identify which BN layers are in trainable parts
    trainable_bn_layers = []

    for name, module in model.named_modules():
        if isinstance(module, (torch.nn.BatchNorm2d, torch.nn.BatchNorm1d)):
            # Check if this BN layer has trainable parameters
            has_trainable = False
            for param in module.parameters():
                if param.requires_grad:
                    has_trainable = True
                    break

            if has_trainable:
                trainable_bn_layers.append((name, module))

    # If no trainable BN layers, skip update entirely
    if not trainable_bn_layers:
        print(f"      (No trainable BN layers found, skipping BN update)")
        return

    print(f"      (Found {len(trainable_bn_layers)} trainable BN layers to update)")

    # Set model to eval mode first
    model.eval()

    # Only set trainable BN layers to train mode and reset their statistics
    for name, module in trainable_bn_layers:
        module.train()
        module.momentum = None  # Use cumulative moving average
        module.reset_running_stats()

    # Forward pass to accumulate BN statistics (no gradient computation)
    with torch.no_grad():
        for batch_idx, (images, _) in enumerate(train_loader):
            if batch_idx >= num_batches:
                break
            images = images.to(device, non_blocking=True)
            with autocast_context(device):
                _ = model(images)

    # Set everything back to eval mode
    model.eval()


def average_weights(checkpoint_paths, device):
    """
    Average model weights from multiple checkpoints

    IMPORTANT FOR FROZEN BACKBONE MODELS:
    - Frozen backbone weights are IDENTICAL across all checkpoints (they don't change during training)
    - Only the classifier/head weights differ between checkpoints
    - BatchNorm running statistics (running_mean, running_var) should NOT be averaged
      because they track population statistics, not learned parameters

    This function averages ALL learnable weights (including frozen ones, which are identical anyway)
    and keeps the BatchNorm running statistics from the FIRST checkpoint.

    Args:
        checkpoint_paths: List of checkpoint file paths
        device: Device to load checkpoints on

    Returns:
        averaged_state_dict: Averaged state dictionary
    """

    if not checkpoint_paths:
        return None

    if len(checkpoint_paths) == 1:
        # Only one checkpoint, no need to average
        checkpoint = torch.load(checkpoint_paths[0], map_location=device)
        return checkpoint['model_state_dict']

    # Load first checkpoint as base
    first_checkpoint = torch.load(checkpoint_paths[0], map_location=device)
    averaged_state_dict = copy.deepcopy(first_checkpoint['model_state_dict'])

    # Identify keys to average vs keys to keep from first checkpoint
    keys_to_average = []
    keys_to_keep = []

    for key in averaged_state_dict.keys():
        # Skip BatchNorm running statistics - these should NOT be averaged
        # They are population statistics, not learned parameters
        if 'running_mean' in key or 'running_var' in key or 'num_batches_tracked' in key:
            keys_to_keep.append(key)
        else:
            keys_to_average.append(key)

    # Sum weights from remaining checkpoints (only for keys_to_average)
    for checkpoint_path in checkpoint_paths[1:]:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        state_dict = checkpoint['model_state_dict']

        for key in keys_to_average:
            averaged_state_dict[key] = averaged_state_dict[key] + state_dict[key]

    # Compute average
    num_checkpoints = len(checkpoint_paths)
    for key in keys_to_average:
        averaged_state_dict[key] = averaged_state_dict[key] / num_checkpoints

    # keys_to_keep already have values from first checkpoint (no changes needed)

    return averaged_state_dict
