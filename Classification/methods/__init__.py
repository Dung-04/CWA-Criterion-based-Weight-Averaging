"""Checkpoint-selection methods evaluated in the paper.

``baseline``
    The single checkpoint with the lowest validation loss. This is the
    conventional early-stopping result and the comparison point for CWA.

``cwa`` (proposed)
    Checkpoint Weight Averaging: uniform element-wise average of the Top-K
    checkpoints ranked by validation loss, followed by BatchNorm
    recalibration. Reported for every K in ``Config.TOP_K_VALUES``.

Both are selected purely on validation loss — the test set is never consulted
when picking or ranking checkpoints, and no code path chooses a "best K" from
test results.
"""
import os

import torch

from configs import Config
from evaluation.metrics import evaluate_model, print_eval_results
from models import get_model
from .averaging import average_weights, update_bn

BASELINE_NAME = "Baseline"


def cwa_name(k):
    """Result label for CWA at a given K."""
    return f"CWA (K={k})"


def run_baseline(model_name, checkpoint_manager, test_loader, num_classes, device,
                 class_names=None, save_dir=None):
    """Evaluate the checkpoint with the lowest validation loss.

    Returns:
        result: Dictionary with 'metrics', 'per_class', 'confusion_matrix'
    """
    print(f"\n  {BASELINE_NAME}: best checkpoint (lowest val_loss)")

    epoch, val_loss, checkpoint_path = checkpoint_manager.get_best_checkpoint()
    print(f"    Best checkpoint: Epoch {epoch}, Val Loss: {val_loss:.4f}")

    model = get_model(model_name, num_classes, freeze_backbone=False)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)

    result = evaluate_model(model, test_loader, device, num_classes, class_names)

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, 'baseline_best.pth')
        torch.save({
            'model_state_dict': model.state_dict(),
            'method': BASELINE_NAME,
            'epoch': epoch,
            'val_loss': val_loss
        }, save_path)
        print(f"    ✓ Checkpoint saved: {save_path}")

    print_eval_results(result['metrics'], result['per_class'],
                       prefix="    ", header=f"TEST RESULTS - {BASELINE_NAME}")

    return result


def run_cwa(model_name, checkpoint_manager, test_loader, train_loader, num_classes,
            device, class_names=None, save_dir=None):
    """Average the Top-K validation-ranked checkpoints, then recalibrate BatchNorm.

    CRITICAL: BatchNorm statistics must be updated after loading averaged
    weights, otherwise the running statistics belong to the best checkpoint
    while the weights no longer do.

    Returns:
        results: Dictionary with k as key and result dict as value
    """
    print(f"\n  CWA: Top-K checkpoint averaging")

    results = {}

    for k in Config.TOP_K_VALUES:
        print(f"    K={k}:")

        top_k = checkpoint_manager.get_top_k_checkpoints(k)

        if len(top_k) < k:
            print(f"      Warning: Only {len(top_k)} checkpoints available")

        checkpoint_paths = [path for _, _, path in top_k]

        averaged_weights = average_weights(checkpoint_paths, device)

        model = get_model(model_name, num_classes, freeze_backbone=False)
        # strict=True: every key is handled explicitly by average_weights
        model.load_state_dict(averaged_weights, strict=True)
        model = model.to(device)

        print(f"      Updating BatchNorm statistics...")
        update_bn(model, train_loader, device, num_batches=100)

        result = evaluate_model(model, test_loader, device, num_classes, class_names)
        results[k] = result

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f'cwa_K{k}.pth')
            torch.save({
                'model_state_dict': model.state_dict(),
                'method': cwa_name(k),
                'k': k,
                'checkpoint_epochs': [ep for ep, _, _ in top_k]
            }, save_path)
            print(f"      ✓ Checkpoint saved: {save_path}")

        print_eval_results(result['metrics'], result['per_class'],
                           prefix="      ", header=f"TEST RESULTS - {cwa_name(k)}")

    return results


def evaluate_all_methods(model_name, checkpoint_manager, test_loader, train_loader,
                         num_classes, device, class_names=None, save_dir=None):
    """Run every reported method for one trained model.

    Args:
        model_name: Model name
        checkpoint_manager: Checkpoint manager holding this run's checkpoints
        test_loader: Test data loader
        train_loader: Training data loader (needed for BatchNorm recalibration)
        num_classes: Number of classes
        device: Device
        class_names: List of class names
        save_dir: Directory to save per-method checkpoints (None = don't save)

    Returns:
        all_results: {method_name: {'metrics', 'per_class', 'confusion_matrix'}}
    """
    print(f"\n{'='*70}")
    print(f" EVALUATING {model_name.upper()}")
    print(f"{'='*70}")

    all_results = {
        BASELINE_NAME: run_baseline(
            model_name, checkpoint_manager, test_loader, num_classes, device,
            class_names=class_names, save_dir=save_dir,
        )
    }

    for k, result in run_cwa(
        model_name, checkpoint_manager, test_loader, train_loader, num_classes,
        device, class_names=class_names, save_dir=save_dir,
    ).items():
        all_results[cwa_name(k)] = result

    return all_results


__all__ = [
    "BASELINE_NAME",
    "average_weights",
    "cwa_name",
    "evaluate_all_methods",
    "run_baseline",
    "run_cwa",
    "update_bn",
]
