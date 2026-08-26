"""Optimizer and LR-schedule construction.

EXPERIMENT-CRITICAL. Two things here differ between datasets and must not be
unified away:

* ``Config.OPTIMIZER_PARAM_GROUPS`` decides whether weight decay skips biases
  and 1D/norm parameters (timm grouping, used by CIFAR-100 / Tiny ImageNet) or
  applies to every trainable parameter (used by the agricultural datasets).
  With WEIGHT_DECAY=0.1 the two are very different.
* ``Config.WARMUP_START_FACTOR`` is 0.01 for CIFAR-100 / Tiny ImageNet and 0.1
  for the agricultural runs.
"""
import inspect

import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from configs import Config

OPTIMIZER_CLASSES = {
    "adamw": optim.AdamW,
    "adam": optim.Adam,
    "sgd": optim.SGD,
    "rmsprop": optim.RMSprop,
}


def _parameter_groups(model):
    """Build the weight-decay parameter groups selected by config."""
    if Config.OPTIMIZER_PARAM_GROUPS == "all_trainable":
        # Single group: decay reaches biases and norm affine parameters too.
        return [
            {
                "params": [p for p in model.parameters() if p.requires_grad],
                "weight_decay": Config.WEIGHT_DECAY,
            }
        ]

    from timm.optim import param_groups_weight_decay

    return param_groups_weight_decay(model, weight_decay=Config.WEIGHT_DECAY)


def build_optimizer(model, device):
    """Build the optimizer selected by ``Config.OPTIMIZER``.

    Note that AdamW decouples weight decay while Adam/SGD/RMSprop apply it as
    classic L2 on the gradients, so the same WEIGHT_DECAY value is not
    equivalent across optimizers.

    Args:
        model: Model whose parameters are optimized
        device: Device used for training (fused kernels need CUDA)

    Returns:
        torch.optim.Optimizer
    """
    name = Config.OPTIMIZER.lower()
    optimizer_cls = OPTIMIZER_CLASSES.get(name)
    if optimizer_cls is None:
        raise ValueError(
            f"Unsupported OPTIMIZER '{Config.OPTIMIZER}'. "
            f"Choose one of: {', '.join(sorted(OPTIMIZER_CLASSES))}"
        )

    parameter_groups = _parameter_groups(model)

    optimizer_kwargs = {"lr": Config.LEARNING_RATE}
    if name in ("adamw", "adam"):
        optimizer_kwargs["betas"] = Config.OPTIMIZER_BETAS
        optimizer_kwargs["eps"] = Config.OPTIMIZER_EPS
        extra = f"betas={Config.OPTIMIZER_BETAS}, eps={Config.OPTIMIZER_EPS}"
    elif name == "sgd":
        optimizer_kwargs["momentum"] = Config.SGD_MOMENTUM
        optimizer_kwargs["nesterov"] = Config.SGD_NESTEROV
        extra = f"momentum={Config.SGD_MOMENTUM}, nesterov={Config.SGD_NESTEROV}"
    else:  # rmsprop
        optimizer_kwargs["momentum"] = Config.SGD_MOMENTUM
        optimizer_kwargs["alpha"] = Config.RMSPROP_ALPHA
        optimizer_kwargs["eps"] = Config.OPTIMIZER_EPS
        extra = f"momentum={Config.SGD_MOMENTUM}, alpha={Config.RMSPROP_ALPHA}"

    # Fused kernels only exist for some optimizers and only on CUDA.
    fused_supported = "fused" in inspect.signature(optimizer_cls).parameters
    fused_enabled = (
        Config.USE_FUSED_OPTIMIZER and device.type == "cuda" and fused_supported
    )
    if fused_enabled:
        optimizer_kwargs["fused"] = True
    elif Config.USE_FUSED_OPTIMIZER and device.type == "cuda" and not fused_supported:
        print(f"  Note: {optimizer_cls.__name__} has no fused implementation, using the default one")

    optimizer = optimizer_cls(parameter_groups, **optimizer_kwargs)
    print(
        f"  Optimizer: {optimizer_cls.__name__}(lr={Config.LEARNING_RATE}, "
        f"weight_decay={Config.WEIGHT_DECAY}, {extra}, fused={fused_enabled}, "
        f"groups={Config.OPTIMIZER_PARAM_GROUPS})"
    )
    if name != "adamw":
        print("    Weight decay is coupled (classic L2) for this optimizer, not decoupled as in AdamW")
    return optimizer


def build_scheduler(optimizer):
    """Linear warmup followed by cosine annealing."""
    warmup = LinearLR(
        optimizer,
        start_factor=Config.WARMUP_START_FACTOR,
        end_factor=1.0,
        total_iters=Config.WARMUP_EPOCHS,
    )
    cosine = CosineAnnealingLR(
        optimizer,
        T_max=Config.NUM_EPOCHS - Config.WARMUP_EPOCHS,
        eta_min=Config.ETA_MIN,
    )
    return SequentialLR(
        optimizer,
        schedulers=[warmup, cosine],
        milestones=[Config.WARMUP_EPOCHS],
    )
