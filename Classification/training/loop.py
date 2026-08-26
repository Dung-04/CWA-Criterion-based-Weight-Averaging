"""Training loop.

EXPERIMENT-CRITICAL: epoch loop, loss selection, Mixup/CutMix handling, the
warmup+cosine schedule, early stopping and per-epoch checkpointing all live
here. Behaviour that differs per dataset is gated on config flags
(``USE_AMP``, ``USE_MIXUP_CUTMIX``, ``GRAD_CLIP_NORM``, ``USE_TORCH_COMPILE``,
``PROFILE_BATCHES``), never forked into a second loop.
"""
import csv
import os
import time

import torch
import torch.nn as nn
from tqdm import tqdm

from configs import Config
from utils.amp import autocast_context
from models import get_model
from utils.plots import plot_patience_period, plot_training_history
from .checkpoints import CheckpointManager, EarlyStopping
from .losses import PolyFocalLoss, compute_class_weights
from .optim import build_optimizer, build_scheduler


def train_one_epoch(model, train_loader, criterion, optimizer, device,
                    freeze_backbone=True, mixup_fn=None):
    """Train for one epoch."""
    model.train()

    # CRITICAL FIX: Set frozen backbone modules to eval mode to prevent BatchNorm stats update
    if freeze_backbone:
        for name, module in model.named_modules():
            # Identify backbone modules (not the classifier/head/fc)
            if any(backbone_name in name for backbone_name in
                   ['features', 'layer1', 'layer2', 'layer3', 'layer4',  # VGG, ResNet
                    'blocks', 'stages',  # EfficientNet, ConvNeXt
                    'patch_embed', 'layers', 'pos_drop',  # ViT, Swin
                    'conv_stem', 'bn1']):
                if hasattr(module, 'parameters'):
                    params = list(module.parameters())
                    if params and all(not p.requires_grad for p in params):
                        module.eval()

    running_loss = 0.0
    correct = 0
    total = 0
    profile_batches = getattr(Config, 'PROFILE_BATCHES', 0)
    profile_data_time = 0.0
    profile_compute_time = 0.0
    profile_count = 0
    end_time = time.time()

    pbar = tqdm(train_loader, desc='Training', leave=False)
    for batch_idx, (images, labels) in enumerate(pbar, 1):
        data_loaded_time = time.time()
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # DeiT-style Mixup/CutMix is a batch-level augmentation. It converts
        # integer labels into soft class distributions.
        targets = labels
        if mixup_fn is not None:
            images, targets = mixup_fn(images, labels)

        # Forward pass
        optimizer.zero_grad(set_to_none=True)
        with autocast_context(device):
            outputs = model(images)
            loss = criterion(outputs, targets)

        # Backward pass
        loss.backward()
        if Config.GRAD_CLIP_NORM is not None:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=Config.GRAD_CLIP_NORM,
            )
        optimizer.step()

        if profile_batches and batch_idx <= profile_batches:
            if device.type == 'cuda':
                torch.cuda.synchronize()
            batch_end_time = time.time()
            profile_data_time += data_loaded_time - end_time
            profile_compute_time += batch_end_time - data_loaded_time
            profile_count += 1
            end_time = batch_end_time

        # Statistics
        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        if targets.ndim == 2:
            # Expected correctness under the soft target distribution. This is
            # more meaningful than comparing mixed images to one hard label.
            correct += targets.gather(1, predicted.unsqueeze(1)).sum().item()
        else:
            correct += (predicted == targets).sum().item()

        if not profile_batches or batch_idx > profile_batches:
            end_time = time.time()

        pbar.set_postfix({'loss': loss.item(), 'acc': 100. * correct / total})

    if profile_count:
        print(
            f"  Profile first {profile_count} train batches: "
            f"data={profile_data_time / profile_count:.3f}s/batch, "
            f"compute={profile_compute_time / profile_count:.3f}s/batch"
        )

    return running_loss / total, 100. * correct / total


def validate(model, val_loader, criterion, device):
    """Validate model."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        pbar = tqdm(val_loader, desc='Validation', leave=False)
        for images, labels in pbar:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with autocast_context(device):
                outputs = model(images)
                loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            pbar.set_postfix({'loss': loss.item(), 'acc': 100. * correct / total})

    return running_loss / total, 100. * correct / total


def _build_criterion(train_labels, class_names):
    """Build the configured loss function."""
    if Config.LOSS_FUNCTION == 'poly_focal':
        if train_labels is not None:
            class_weights = compute_class_weights(
                train_labels, method=Config.CLASS_WEIGHT_METHOD
            )
            print(f"  Class weights ({Config.CLASS_WEIGHT_METHOD}):")
            if class_names:
                for i, name in enumerate(class_names):
                    print(f"    {name}: {class_weights[i]:.4f}")
            else:
                print(f"    {class_weights.tolist()}")
        else:
            class_weights = None
            print("  Warning: No train_labels provided, using equal class weights")

        criterion = PolyFocalLoss(
            gamma=Config.FOCAL_GAMMA,
            epsilon=Config.POLY_EPSILON,
            alpha=class_weights,
        )
        print(f"  Loss: PolyFocalLoss(gamma={Config.FOCAL_GAMMA}, epsilon={Config.POLY_EPSILON})")
        return criterion

    return nn.CrossEntropyLoss(label_smoothing=Config.LABEL_SMOOTHING)


def _build_mixup(num_classes):
    """Build the Mixup/CutMix batch augmenter, when enabled."""
    if not Config.USE_MIXUP_CUTMIX:
        return None

    from timm.data import Mixup

    mixup_fn = Mixup(
        mixup_alpha=Config.MIXUP_ALPHA,
        cutmix_alpha=Config.CUTMIX_ALPHA,
        prob=Config.MIXUP_PROB,
        switch_prob=Config.MIXUP_SWITCH_PROB,
        mode=Config.MIXUP_MODE,
        label_smoothing=Config.LABEL_SMOOTHING,
        num_classes=num_classes,
    )
    print(
        "  Batch augmentation: "
        f"Mixup(alpha={Config.MIXUP_ALPHA}) / "
        f"CutMix(alpha={Config.CUTMIX_ALPHA}), "
        f"prob={Config.MIXUP_PROB}, "
        f"switch_prob={Config.MIXUP_SWITCH_PROB}"
    )
    return mixup_fn


def train_model(model_name, train_loader, val_loader, num_classes, device,
                class_names=None, test_loader=None, train_labels=None,
                save_dir=None, checkpoints_dir=None):
    """
    Train a single model

    Args:
        model_name: Name of the model
        train_loader: Training dataloader
        val_loader: Validation dataloader
        num_classes: Number of classes
        device: Device to train on
        class_names: List of class names (optional, for visualization)
        test_loader: Test dataloader (optional, for final test evaluation)
        train_labels: List of training labels (optional, for computing class weights)
        save_dir: Directory to save training curves (per-run folder). If None, uses Config.RESULTS_DIR
        checkpoints_dir: Base directory for CheckpointManager. If None, uses Config.CHECKPOINTS_DIR.
                         Pass a fold-specific path in CV mode to isolate checkpoints per fold.

    Returns:
        checkpoint_manager: CheckpointManager object
        history: Training history dictionary
    """
    torch.set_float32_matmul_precision(Config.FLOAT32_MATMUL_PRECISION)

    print(f"\n{'='*70}")
    print(f"Training {model_name}")
    print(f"{'='*70}")

    model = get_model(model_name, num_classes, freeze_backbone=False)
    model = model.to(device)

    compile_enabled = Config.USE_TORCH_COMPILE and device.type == "cuda"
    if compile_enabled:
        if hasattr(model, "compile"):
            model.compile(mode=Config.TORCH_COMPILE_MODE)
        else:
            model = torch.compile(model, mode=Config.TORCH_COMPILE_MODE)
        print(f"  torch.compile: ENABLED ({Config.TORCH_COMPILE_MODE})")
    else:
        print("  torch.compile: DISABLED")

    criterion = _build_criterion(train_labels, class_names)
    mixup_fn = _build_mixup(num_classes)

    optimizer = build_optimizer(model, device)
    print(f"  Precision: {'BF16 AMP' if Config.USE_AMP and device.type == 'cuda' else 'FP32'}")

    scheduler = build_scheduler(optimizer)

    early_stopping = EarlyStopping(patience=Config.EARLY_STOPPING_PATIENCE)
    checkpoint_manager = CheckpointManager(
        checkpoints_dir if checkpoints_dir is not None else Config.CHECKPOINTS_DIR,
        model_name,
        keep_last_n=Config.KEEP_LAST_N_CHECKPOINTS,
        keep_top_k=Config.KEEP_TOP_K_CHECKPOINTS,
    )

    best_val_loss = float('inf')

    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'learning_rate': []
    }

    for epoch in range(1, Config.NUM_EPOCHS + 1):
        epoch_start_time = time.time()

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
            freeze_backbone=False, mixup_fn=mixup_fn,
        )

        val_loss, val_acc = validate(model, val_loader, criterion, device)

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['learning_rate'].append(optimizer.param_groups[0]['lr'])

        epoch_time = time.time() - epoch_start_time
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch [{epoch}/{Config.NUM_EPOCHS}] ({epoch_time:.2f}s) - LR: {current_lr:.6f}")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

        scheduler.step()

        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            print(f"  ✓ New best validation loss!")

        checkpoint_manager.save_checkpoint(model, epoch, val_loss, is_best)

        early_stopping(val_loss)
        if early_stopping.early_stop:
            print(f"\n✓ Early stopping triggered at epoch {epoch}")
            break

    checkpoint_manager.save_checkpoint_info()

    print(f"\n✓ Training completed for {model_name}")
    print(f"  Best Val Loss: {best_val_loss:.4f}")
    print(f"  Total checkpoints saved: {len(checkpoint_manager.checkpoints)}")

    # ================= FINAL TEST EVALUATION =================
    if test_loader is not None:
        print(f"\n{'='*70}")
        print(f"Final Test Evaluation on Best Checkpoint")
        print(f"{'='*70}")

        best_epoch, best_val_loss_cp, best_checkpoint_path = checkpoint_manager.get_best_checkpoint()
        checkpoint = torch.load(best_checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])

        test_loss, test_acc = validate(model, test_loader, criterion, device)

        print(f"  Best Checkpoint: Epoch {best_epoch}, Val Loss: {best_val_loss_cp:.4f}")
        print(f"  Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")

    if save_dir is not None:
        curves_dir = os.path.join(save_dir, model_name, "training_curves")
    else:
        curves_dir = os.path.join(Config.RESULTS_DIR, "training_curves")
    os.makedirs(curves_dir, exist_ok=True)

    history_csv_path = os.path.join(curves_dir, f"{model_name}_training_history.csv")
    with open(history_csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'train_loss', 'train_acc', 'val_loss', 'val_acc', 'learning_rate'])
        for i in range(len(history['train_loss'])):
            writer.writerow([
                i + 1,
                history['train_loss'][i],
                history['train_acc'][i],
                history['val_loss'][i],
                history['val_acc'][i],
                history['learning_rate'][i]
            ])
    print(f"\n✓ Training history CSV saved to: {history_csv_path}")

    plot_training_history(
        history, model_name,
        save_path=os.path.join(curves_dir, f"{model_name}_training_history.png"),
    )
    plot_patience_period(
        history, Config.EARLY_STOPPING_PATIENCE, model_name,
        save_path=os.path.join(curves_dir, f"{model_name}_patience_period.png"),
    )

    return checkpoint_manager, history
