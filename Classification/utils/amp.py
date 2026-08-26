"""Mixed-precision helper shared by training, evaluation and BN recalibration.

All three must run under the same autocast settings, otherwise averaged
weights would be evaluated at a different precision than they were trained at.
"""
import torch

from configs import Config


def autocast_context(device):
    """BF16 autocast on CUDA when enabled; a no-op context otherwise."""
    return torch.autocast(
        device_type=device.type,
        dtype=torch.bfloat16,
        enabled=Config.USE_AMP and device.type == "cuda",
    )
