"""Self-contained checks for the classification pipeline.

These verify plumbing and behaviour-preservation, not model quality: they run
on a tiny synthetic ImageFolder dataset with a stub backbone, so no dataset
download and no pretrained weights are needed.

    python tests/test_pipeline.py

Exit code 0 means every check passed.
"""
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

import configs

FAILURES = []
CHECKS = 0


def check(condition, label):
    global CHECKS
    CHECKS += 1
    if condition:
        print(f"  PASS  {label}")
    else:
        print(f"  FAIL  {label}")
        FAILURES.append(label)


def section(title):
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def make_dataset(root, n_classes=3, per_class=12, size=64):
    """Write a tiny ImageFolder tree of random images."""
    rng = np.random.default_rng(0)
    for idx in range(n_classes):
        cls_dir = os.path.join(root, f"class{idx}")
        os.makedirs(cls_dir, exist_ok=True)
        for i in range(per_class):
            arr = (rng.random((size, size, 3)) * 255).astype("uint8")
            Image.fromarray(arr).save(os.path.join(cls_dir, f"{i:03d}.jpg"))
    return root


class StubNet(nn.Module):
    """Tiny backbone WITH BatchNorm so BN recalibration is actually exercised."""

    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 8, 3, stride=8), nn.BatchNorm2d(8), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
        )
        self.classifier = nn.Linear(8, num_classes)

    def forward(self, x):
        return self.classifier(self.features(x))


def install_stub_model():
    """Point every get_model call site at StubNet."""
    import models
    import methods
    import training.loop as loop

    def stub(name, num_classes, freeze_backbone=False):
        return StubNet(num_classes)

    models.get_model = methods.get_model = loop.get_model = stub


# ----------------------------------------------------------------------
# 1. Config registry
# ----------------------------------------------------------------------
def test_configs():
    section("1. Config registry - per-dataset values are preserved")
    from configs import Config

    expected = {
        "cifar100":     dict(BATCH_SIZE=128,  LEARNING_RATE=1e-4, WEIGHT_DECAY=1e-4,
                             USE_AMP=True,  USE_MIXUP_CUTMIX=True,  AUGMENTATION="pretrain_224",
                             OPTIMIZER_PARAM_GROUPS="timm_no_decay_1d", WARMUP_START_FACTOR=0.01,
                             GRAD_CLIP_NORM=1.0, TRAIN_DROP_LAST=True, USE_WEIGHTED_SAMPLER=False),
        "tinyimagenet": dict(BATCH_SIZE=1024, LEARNING_RATE=2e-4, WEIGHT_DECAY=1e-4,
                             USE_AMP=True,  USE_MIXUP_CUTMIX=True,  AUGMENTATION="pretrain_224",
                             OPTIMIZER_PARAM_GROUPS="timm_no_decay_1d", WARMUP_START_FACTOR=0.01,
                             GRAD_CLIP_NORM=1.0, TRAIN_DROP_LAST=True, USE_WEIGHTED_SAMPLER=False),
        "burmese":      dict(BATCH_SIZE=32,   LEARNING_RATE=9e-7, WEIGHT_DECAY=0.1,
                             USE_AMP=False, USE_MIXUP_CUTMIX=False, AUGMENTATION="agri",
                             OPTIMIZER_PARAM_GROUPS="all_trainable",  WARMUP_START_FACTOR=0.1,
                             GRAD_CLIP_NORM=None, TRAIN_DROP_LAST=False, USE_WEIGHTED_SAMPLER=True),
    }

    for dataset, values in expected.items():
        configs.select_dataset(dataset)
        wrong = {k: (getattr(Config, k), v) for k, v in values.items()
                 if getattr(Config, k) != v}
        check(not wrong, f"{dataset}: {len(values)} settings match the original"
                         + (f" (differs: {wrong})" if wrong else ""))

    # Switching datasets must fully swap the active config.
    configs.select_dataset("cifar100")
    before = Config.BATCH_SIZE
    configs.select_dataset("burmese")
    check(Config.BATCH_SIZE == 32 and before == 128, "switching datasets swaps every value")

    # CLI-style overrides must stick.
    configs.select_dataset("cifar100")
    Config.BATCH_SIZE = 7
    check(Config.BATCH_SIZE == 7, "overrides written through the proxy persist")
    Config.BATCH_SIZE = 128

    check(set(configs.DATASET_CHOICES) ==
          {"burmese", "potato", "tomato", "cifar100", "tinyimagenet"},
          f"registered datasets: {configs.DATASET_CHOICES}")


# ----------------------------------------------------------------------
# 2. Transforms
# ----------------------------------------------------------------------
def test_transforms():
    section("2. Transforms - pipelines match the pre-refactor originals")
    from data.transforms import get_transforms
    from configs import Config

    configs.select_dataset("cifar100")
    train_ops = [type(t).__name__ for t in get_transforms("train").transforms]
    eval_ops = [type(t).__name__ for t in get_transforms("test").transforms]
    check(train_ops == ["Resize", "RandomHorizontalFlip", "ToTensor",
                        "Normalize", "RandomErasing"],
          f"cifar100 train pipeline: {train_ops}")
    check(eval_ops == ["Resize", "ToTensor", "Normalize"],
          f"cifar100 eval pipeline: {eval_ops}")

    configs.select_dataset("tinyimagenet")
    tin_ops = [type(t).__name__ for t in get_transforms("train").transforms]
    check(tin_ops == train_ops, "tinyimagenet shares cifar100's augmentation")

    configs.select_dataset("burmese")
    agri_ops = [type(t).__name__ for t in get_transforms("train").transforms]
    check(agri_ops == ["Resize", "RandomHorizontalFlip", "RandomVerticalFlip",
                       "RandomRotation", "ColorJitter", "ToTensor", "Normalize"],
          f"burmese keeps its own augmentation: {agri_ops}")
    check("RandomErasing" not in agri_ops, "burmese has no RandomErasing")
    check(Config.IMAGE_MEAN == (0.485, 0.456, 0.406), "burmese uses ImageNet normalization")

    configs.select_dataset("cifar100")
    check(Config.IMAGE_MEAN == (0.5, 0.5, 0.5), "cifar100 uses 0.5 normalization")


# ----------------------------------------------------------------------
# 3. Optimizer parameter grouping and LR schedule
# ----------------------------------------------------------------------
def test_optimizer():
    section("3. Optimizer - weight-decay grouping differs per dataset, as reported")
    from training.optim import build_optimizer, build_scheduler
    from configs import Config

    device = torch.device("cpu")

    configs.select_dataset("burmese")
    opt = build_optimizer(StubNet(3), device)
    undecayed = sum(len(g["params"]) for g in opt.param_groups if g["weight_decay"] == 0)
    check(len(opt.param_groups) == 1 and undecayed == 0,
          "burmese decays every trainable parameter (plain Adam)")

    configs.select_dataset("cifar100")
    opt = build_optimizer(StubNet(3), device)
    undecayed = sum(len(g["params"]) for g in opt.param_groups if g["weight_decay"] == 0)
    check(len(opt.param_groups) == 2 and undecayed > 0,
          "cifar100 leaves biases and 1D/norm parameters undecayed (timm grouping)")

    # The agricultural LR schedule really does ramp up - preserved deliberately.
    configs.select_dataset("burmese")
    Config.NUM_EPOCHS, Config.WARMUP_EPOCHS = 50, 5
    opt = build_optimizer(StubNet(3), device)
    sched = build_scheduler(opt)
    lrs = []
    for _ in range(Config.NUM_EPOCHS):
        lrs.append(opt.param_groups[0]["lr"])
        sched.step()
    check(abs(lrs[0] - 9e-8) < 1e-12, f"burmese warmup starts at lr*0.1 ({lrs[0]:.2e})")
    check(lrs[-1] > lrs[Config.WARMUP_EPOCHS],
          "burmese cosine phase raises LR toward ETA_MIN (known, preserved)")

    configs.select_dataset("cifar100")
    opt = build_optimizer(StubNet(3), device)
    sched = build_scheduler(opt)
    lrs = []
    for _ in range(Config.NUM_EPOCHS):
        lrs.append(opt.param_groups[0]["lr"])
        sched.step()
    check(lrs[-1] < lrs[Config.WARMUP_EPOCHS], "cifar100 cosine phase anneals LR down")


# ----------------------------------------------------------------------
# 4. Model specs
# ----------------------------------------------------------------------
def test_models():
    section("4. Models - every spec wires head/in_features/freezing correctly")
    import models as M
    from configs import Config

    configs.select_dataset("cifar100")
    Config.PRETRAINED = False
    Config.CLASSIFIER_CONFIG = [64]

    import timm
    from torchvision import models as tv

    builders = {
        'vgg16': lambda: tv.vgg16(weights=None),
        'resnet18': lambda: tv.resnet18(weights=None),
        'resnet101': lambda: tv.resnet101(weights=None),
        'mobilenet_v2': lambda: tv.mobilenet_v2(weights=None),
        'densenet121': lambda: tv.densenet121(weights=None),
        'efficientnet_b0': lambda: timm.create_model('efficientnet_b0', pretrained=False),
        'convnext_tiny': lambda: timm.create_model('convnext_tiny', pretrained=False),
        'vit_base_patch16_224': lambda: timm.create_model('vit_base_patch16_224', pretrained=False),
        'swin_tiny_patch4_window7_224': lambda: timm.create_model('swin_tiny_patch4_window7_224', pretrained=False),
        'convit_tiny': lambda: timm.create_model('convit_tiny', pretrained=False),
    }
    check(set(builders) == set(M.SUPPORTED_MODELS), "every supported model is covered")

    for name, build in builders.items():
        spec = M._SPECS[name]
        saved, spec.build = spec.build, build
        try:
            ok = True
            for freeze in (False, True):
                model = M.get_model(name, 7, freeze_backbone=freeze)
                head = M._resolve(model, spec.head_path)
                ok &= isinstance(head, spec.head_cls)
                ok &= head.classifier[-1].out_features == 7
                if spec.backbone:
                    grads = {p.requires_grad
                             for p in M._resolve(model, spec.backbone).parameters()}
                else:
                    prefix = spec.head_path.split(".")[0]
                    grads = {p.requires_grad for n, p in model.named_parameters()
                             if not n.startswith(prefix)}
                ok &= grads == {not freeze}
            check(ok, f"{name}: head at '{spec.head_path}', freeze/unfreeze correct")
        finally:
            spec.build = saved

    check(M.parse_model_list(["vit_base"]) == ["vit_base_patch16_224"], "model alias resolves")
    check(M.parse_model_list(["resnet18,resnet101"]) == ["resnet18", "resnet101"],
          "comma-separated model list parses")
    try:
        M.parse_model_list(["not_a_model"])
        check(False, "unknown model is rejected")
    except ValueError:
        check(True, "unknown model is rejected")


# ----------------------------------------------------------------------
# 5. Checkpoint selection
# ----------------------------------------------------------------------
def test_checkpoint_selection(tmp):
    section("5. Checkpoint selection - Baseline and CWA rank on val loss only")
    from training.checkpoints import CheckpointManager, EarlyStopping

    manager = CheckpointManager(tmp, "sel", keep_last_n=3, keep_top_k=3)
    losses = [0.9, 0.5, 0.7, 0.2, 0.8, 0.3, 0.6, 0.4, 0.95, 0.85, 0.75, 0.65]
    model = StubNet(3)
    for epoch, loss in enumerate(losses, 1):
        manager.save_checkpoint(model, epoch, loss, is_best=(loss == min(losses[:epoch])))

    best = manager.get_best_checkpoint()
    check(best[1] == 0.2 and best[0] == 4, "best checkpoint is the global lowest val_loss")

    top3 = [round(l, 3) for _, l, _ in manager.get_top_k_checkpoints(3)]
    check(top3 == [0.2, 0.3, 0.4], f"top-3 is the global 3 lowest losses: {top3}")

    top5 = [round(l, 3) for _, l, _ in manager.get_top_k_checkpoints(5)]
    check(top5[:3] == [0.2, 0.3, 0.4] and top5 == sorted(top5),
          "top-K is sorted ascending by val_loss")

    check(all(os.path.exists(p) for _, _, p in manager.checkpoints),
          f"every retained checkpoint exists on disk ({len(manager.checkpoints)} files)")

    stopper = EarlyStopping(patience=3)
    for loss in [1.0, 0.9, 0.95, 0.96, 0.97]:
        stopper(loss)
    check(stopper.early_stop, "early stopping fires after patience epochs without improvement")

    stopper = EarlyStopping(patience=3)
    for loss in [1.0, 0.9, 0.8, 0.7, 0.6]:
        stopper(loss)
    check(not stopper.early_stop, "early stopping does not fire while loss improves")


# ----------------------------------------------------------------------
# 6. Weight averaging
# ----------------------------------------------------------------------
def test_averaging(tmp):
    section("6. CWA averaging - uniform mean, BN running stats excluded")
    from methods.averaging import average_weights

    device = torch.device("cpu")
    paths = []
    for i in range(4):
        model = StubNet(3)
        with torch.no_grad():
            for p in model.parameters():
                p.fill_(float(i + 1))
            for m in model.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.running_mean.fill_(float(i + 1) * 10)
                    m.running_var.fill_(float(i + 1) * 100)
        path = os.path.join(tmp, f"avg_{i}.pth")
        torch.save({"model_state_dict": model.state_dict(), "epoch": i, "val_loss": 0.1 * i}, path)
        paths.append(path)

    averaged = average_weights(paths, device)

    # Learnable params filled with 1,2,3,4 -> uniform mean is 2.5
    weight = averaged["classifier.weight"]
    check(torch.allclose(weight, torch.full_like(weight, 2.5)),
          f"learnable params are the uniform mean (got {weight.flatten()[0].item()})")

    # BN population stats must be carried from the FIRST checkpoint, not averaged.
    check(torch.allclose(averaged["features.1.running_mean"],
                         torch.full_like(averaged["features.1.running_mean"], 10.0)),
          "BN running_mean taken from the first checkpoint, not averaged")
    check(torch.allclose(averaged["features.1.running_var"],
                         torch.full_like(averaged["features.1.running_var"], 100.0)),
          "BN running_var taken from the first checkpoint, not averaged")

    single = average_weights(paths[:1], device)
    check(torch.allclose(single["classifier.weight"],
                         torch.full_like(single["classifier.weight"], 1.0)),
          "K=1 returns the checkpoint unchanged")

    # Averaging must not depend on the order the checkpoints are listed in.
    reversed_avg = average_weights(list(reversed(paths)), device)
    check(torch.allclose(reversed_avg["classifier.weight"], weight),
          "averaging is order-independent for learnable params")

    check(set(averaged) == set(torch.load(paths[0])["model_state_dict"]),
          "averaged state dict has exactly the original keys")


# ----------------------------------------------------------------------
# 7. End-to-end
# ----------------------------------------------------------------------
def test_end_to_end(tmp, data_root):
    section("7. End-to-end - train, evaluate Baseline + CWA, export results")
    from configs import Config

    configs.select_dataset("burmese")
    Config.DATASET_PATH = data_root
    Config.NUM_EPOCHS, Config.WARMUP_EPOCHS, Config.EARLY_STOPPING_PATIENCE = 6, 1, 5
    Config.BATCH_SIZE, Config.NUM_WORKERS = 8, 0
    Config.TOP_K_VALUES, Config.KEEP_TOP_K_CHECKPOINTS = [2, 3], 3
    Config.RESULTS_DIR = tmp

    install_stub_model()

    import methods
    from data import create_dataloaders, load_dataset
    from evaluation import export_results_to_excel, save_confusion_matrices
    from training import train_model
    from utils import set_seed

    set_seed(Config.RANDOM_SEED)
    Config.validate_config()

    splits = load_dataset(random_seed=Config.RANDOM_SEED)
    train_data, train_labels, val_data, _, test_data, _, class_names = splits
    check(len(train_data) + len(val_data) + len(test_data) == 36,
          "70/15/15 split covers every image exactly once")

    train_loader, val_loader, test_loader = create_dataloaders(
        train_data, train_labels, val_data, test_data, Config.BATCH_SIZE, 0)

    out = os.path.join(tmp, "e2e")
    os.makedirs(out, exist_ok=True)
    manager, history = train_model(
        "stub", train_loader, val_loader, len(class_names), torch.device("cpu"),
        class_names=class_names, train_labels=train_labels, save_dir=out,
        checkpoints_dir=os.path.join(out, "ckpt"),
    )
    check(len(history["train_loss"]) > 0, "training produced a history")
    check(len(history["train_loss"]) == len(history["val_loss"]) == len(history["learning_rate"]),
          "history series have matching lengths")

    results = methods.evaluate_all_methods(
        "stub", manager, test_loader, train_loader, len(class_names),
        torch.device("cpu"), class_names=class_names,
    )
    check(list(results) == ["Baseline", "CWA (K=2)", "CWA (K=3)"],
          f"methods produced: {list(results)}")

    metric_keys = {"Test Loss", "Accuracy (%)", "Precision (%)",
                   "Recall (%)", "F1-Score (%)", "AUC (%)"}
    check(all(metric_keys <= set(r["metrics"]) for r in results.values()),
          "every method reports the full metric set")
    check(all(len(r["per_class"]) == len(class_names) for r in results.values()),
          "every method reports per-class metrics for all classes")
    check(all(r["confusion_matrix"].shape == (len(class_names), len(class_names))
              for r in results.values()), "confusion matrices are square and correctly sized")

    excel = os.path.join(out, "results.xlsx")
    df = export_results_to_excel({"stub": results}, excel)
    check(os.path.exists(excel), "Excel workbook written")
    check(list(df["Method"]) == ["Baseline", "CWA (K=2)", "CWA (K=3)"],
          "Excel 'Method' column uses the paper's method names")
    check("Strategy" not in df.columns, "no legacy 'Strategy' column remains")

    save_confusion_matrices({"stub": results}, out, class_names=class_names)
    check(len(os.listdir(os.path.join(out, "confusion_matrices"))) == 3,
          "one confusion-matrix image per method")

    curves = os.path.join(out, "stub", "training_curves")
    check(os.path.exists(os.path.join(curves, "stub_training_history.csv")),
          "training-history CSV written")


# ----------------------------------------------------------------------
# 8. Dataset verification
# ----------------------------------------------------------------------
def test_verify(tmp, data_root):
    section("8. Dataset verification - --check-dataset detects unreadable images")
    from configs import Config
    from data import verify_dataset

    configs.select_dataset("burmese")
    Config.DATASET_PATH = data_root
    total, corrupted = verify_dataset()
    check(total == 36 and not corrupted, f"clean dataset: {total} images, 0 unreadable")

    broken_root = os.path.join(tmp, "broken")
    shutil.copytree(data_root, broken_root)
    with open(os.path.join(broken_root, "class0", "000.jpg"), "w") as f:
        f.write("this is not a jpeg")
    Config.DATASET_PATH = broken_root
    total, corrupted = verify_dataset()
    check(len(corrupted) == 1, f"corrupted image detected ({len(corrupted)} found)")
    Config.DATASET_PATH = data_root


def main():
    tmp = tempfile.mkdtemp(prefix="cwa_tests_")
    try:
        data_root = make_dataset(os.path.join(tmp, "data"))
        test_configs()
        test_transforms()
        test_optimizer()
        test_models()
        test_checkpoint_selection(tmp)
        test_averaging(tmp)
        test_end_to_end(tmp, data_root)
        test_verify(tmp, data_root)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print(f"\n{'=' * 70}")
    if FAILURES:
        print(f" {len(FAILURES)}/{CHECKS} CHECKS FAILED")
        for failure in FAILURES:
            print(f"   - {failure}")
        print("=" * 70)
        return 1
    print(f" ALL {CHECKS} CHECKS PASSED")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
