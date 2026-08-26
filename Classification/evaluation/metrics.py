"""Test-set metric computation.

Metric definitions are experiment-critical: macro-averaged accuracy,
precision, recall, F1 and one-vs-rest AUC, plus a per-class breakdown derived
from the confusion matrix.
"""
import numpy as np
import torch
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix)
from tqdm import tqdm

from utils.amp import autocast_context


def evaluate_model(model, test_loader, device, num_classes, class_names=None):
    """
    Evaluate model and compute metrics including test loss, per-class metrics, and confusion matrix

    Args:
        model: Model to evaluate
        test_loader: Test data loader
        device: Device to run on
        num_classes: Number of classes
        class_names: List of class names (optional, defaults to Class 0, Class 1, ...)

    Returns:
        result: Dictionary with keys:
            - 'metrics': Overall macro-averaged metrics
            - 'per_class': Per-class metrics dict {class_name: {metric: value}}
            - 'confusion_matrix': Confusion matrix as numpy array
    """

    if class_names is None:
        class_names = [f'Class {i}' for i in range(num_classes)]

    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    running_loss = 0.0
    total = 0

    # Create criterion for test loss calculation
    criterion = torch.nn.CrossEntropyLoss()

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc='Evaluating', leave=False):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with autocast_context(device):
                outputs = model(images)
                loss = criterion(outputs, labels)

            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)

            running_loss += loss.item() * images.size(0)
            total += labels.size(0)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.float().cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    test_loss = running_loss / total

    # Macro-averaged metrics
    accuracy = accuracy_score(all_labels, all_preds) * 100
    precision = precision_score(all_labels, all_preds, average='macro', zero_division=0) * 100
    recall = recall_score(all_labels, all_preds, average='macro', zero_division=0) * 100
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0) * 100
    # AUC (one-vs-rest)
    try:
        auc = roc_auc_score(all_labels, all_probs, multi_class='ovr', average='macro') * 100
    except Exception:
        auc = 0.0

    metrics = {
        'Test Loss': test_loss,
        'Accuracy (%)': accuracy,
        'Precision (%)': precision,
        'Recall (%)': recall,
        'F1-Score (%)': f1,
        'AUC (%)': auc
    }

    # ========== Per-class metrics ==========
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(num_classes)))

    pc_precision = precision_score(all_labels, all_preds, average=None, labels=list(range(num_classes)), zero_division=0) * 100
    pc_recall = recall_score(all_labels, all_preds, average=None, labels=list(range(num_classes)), zero_division=0) * 100
    pc_f1 = f1_score(all_labels, all_preds, average=None, labels=list(range(num_classes)), zero_division=0) * 100

    per_class = {}
    for i in range(num_classes):
        cls_name = class_names[i]

        # Derive TP, FP, FN, TN from confusion matrix
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = cm.sum() - tp - fn - fp

        # Specificity = TN / (TN + FP)
        specificity = (tn / (tn + fp) * 100) if (tn + fp) > 0 else 0.0

        # Support = number of true samples for this class
        support = int(cm[i, :].sum())

        # Per-class AUC (one-vs-rest)
        try:
            binary_labels = (all_labels == i).astype(int)
            class_auc = roc_auc_score(binary_labels, all_probs[:, i]) * 100
        except Exception:
            class_auc = 0.0

        # Accuracy = (TP + TN) / Total
        cm_total = cm.sum()
        class_accuracy = ((tp + tn) / cm_total * 100) if cm_total > 0 else 0.0

        per_class[cls_name] = {
            'Accuracy (%)': class_accuracy,
            'Precision (%)': pc_precision[i],
            'Recall (%)': pc_recall[i],
            'F1-Score (%)': pc_f1[i],
            'Specificity (%)': specificity,
            'AUC (%)': class_auc,
            'Support': support
        }

    return {
        'metrics': metrics,
        'per_class': per_class,
        'confusion_matrix': cm
    }


def print_eval_results(metrics, per_class, prefix="    ", header="TEST RESULTS"):
    """Print macro and per-class evaluation results."""
    print(f"{prefix}{'='*60}")
    print(f"{prefix}📊 {header}:")
    print(f"{prefix}{'='*60}")
    print(f"{prefix}Test Loss : {metrics['Test Loss']:>6.4f}")
    print(f"{prefix}Accuracy  : {metrics['Accuracy (%)']:>6.2f}%")
    print(f"{prefix}Precision : {metrics['Precision (%)']:>6.2f}%")
    print(f"{prefix}Recall    : {metrics['Recall (%)']:>6.2f}%")
    print(f"{prefix}F1-Score  : {metrics['F1-Score (%)']:>6.2f}%")
    print(f"{prefix}AUC       : {metrics['AUC (%)']:>6.2f}%")

    print(f"{prefix}{'-'*60}")
    print(f"{prefix}Per-Class Breakdown:")
    print(f"{prefix}{'-'*60}")
    print(f"{prefix}  {'Class':<35} {'Acc':>6} {'Prec':>6} {'Rec':>6} {'F1':>6} {'Spec':>6} {'AUC':>6} {'Sup':>5}")
    print(f"{prefix}  {'-'*35} {'-'*6} {'-'*6} {'-'*6} {'-'*6} {'-'*6} {'-'*6} {'-'*5}")
    for cls_name, cls_metrics in per_class.items():
        print(f"{prefix}  {cls_name:<35} "
              f"{cls_metrics['Accuracy (%)']:>5.1f}% "
              f"{cls_metrics['Precision (%)']:>5.1f}% "
              f"{cls_metrics['Recall (%)']:>5.1f}% "
              f"{cls_metrics['F1-Score (%)']:>5.1f}% "
              f"{cls_metrics['Specificity (%)']:>5.1f}% "
              f"{cls_metrics['AUC (%)']:>5.1f}% "
              f"{cls_metrics['Support']:>5d}")
    print(f"{prefix}{'='*60}")
