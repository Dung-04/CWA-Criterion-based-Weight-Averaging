"""Agricultural leaf-disease datasets (Burmese grape, Potato, Tomato).

All three are ImageFolder-style trees (``<root>/<class_name>/*.jpg``) split
70/15/15 per class with a fixed seed, and they share one set of
hyperparameters. They differ only in the dataset root, so each one is a
two-line subclass.

The values below are the ones the reported agricultural runs used, and several
of them differ deliberately from the CIFAR-100 / Tiny ImageNet defaults:
no AMP, no Mixup/CutMix, no gradient clipping, weight decay applied to every
trainable parameter, and the "agri" augmentation policy.
"""
import os

from .base import BaseConfig


class AgriculturalConfig(BaseConfig):
    """Shared settings for the ImageFolder-based agricultural datasets."""

    LOADER = "image_folder"

    DATASET_PATH = None            # set per dataset below, or with --data-root
    TRAIN_RATIO = 0.7
    VAL_RATIO = 0.15
    TEST_RATIO = 0.15

    # ===================== Training =====================
    BATCH_SIZE = 32
    NUM_EPOCHS = 50
    LEARNING_RATE = 9e-7
    WEIGHT_DECAY = 0.1
    WARMUP_EPOCHS = 5              # int(NUM_EPOCHS * 0.1)
    WARMUP_START_FACTOR = 0.1
    # NOTE: ETA_MIN (1e-5) is ABOVE LEARNING_RATE (9e-7). This is what the
    # reported agricultural runs used: after warmup the cosine phase ramps the
    # LR *up* toward 1e-5 rather than annealing down. Kept as-is deliberately —
    # changing it would change the results.
    ETA_MIN = 1e-5
    ALLOW_ETA_MIN_ABOVE_LR = True
    NUM_WORKERS = 16
    EARLY_STOPPING_PATIENCE = 10

    # Plain Adam over every trainable parameter — biases and norm affine
    # parameters are decayed too. Do not swap this for the timm grouping
    # without re-running: WEIGHT_DECAY=0.1 makes the two clearly different.
    OPTIMIZER = "adam"
    OPTIMIZER_PARAM_GROUPS = "all_trainable"
    USE_FUSED_OPTIMIZER = False
    GRAD_CLIP_NORM = None

    # Full FP32; these runs predate the AMP/compile path.
    USE_AMP = False
    FLOAT32_MATMUL_PRECISION = "highest"
    USE_TORCH_COMPILE = False

    # ===================== DataLoader =====================
    TRAIN_DROP_LAST = False
    WORKER_SEED_MODE = "config_seed"

    # ===================== Class imbalance =====================
    USE_WEIGHTED_SAMPLER = True

    # ===================== Loss =====================
    LOSS_FUNCTION = "cross_entropy"
    LABEL_SMOOTHING = 0.15

    # ===================== Model =====================
    MODELS = ["vit_base_patch16_224"]
    VIT_PRETRAINED_MODEL_ID = "vit_base_patch16_224"
    CLASSIFIER_CONFIG = [512]
    DROPOUT_RATE = 0.4
    MODEL_DROP_RATE = 0.1
    MODEL_ATTN_DROP_RATE = 0.1
    MODEL_DROP_PATH_RATE = 0.2

    # ===================== Image / augmentation =====================
    IMAGE_SIZE = 224
    RESIZE_INTERPOLATION = "bilinear"   # torchvision Resize default
    IMAGE_MEAN = (0.485, 0.456, 0.406)
    IMAGE_STD = (0.229, 0.224, 0.225)
    AUGMENTATION = "agri"
    USE_MIXUP_CUTMIX = False
    HORIZONTAL_FLIP_PROB = 0.3
    VERTICAL_FLIP_PROB = 0.3
    ROTATION_DEGREES = 90
    COLOR_JITTER_BRIGHTNESS = (0.8, 1.2)
    COLOR_JITTER_CONTRAST = (0.8, 1.2)

    # ===================== Output =====================
    AUTO_DELETE_CHECKPOINTS = False

    # ===================== Reproducibility =====================
    SEEDS = [42]
    RANDOM_SEED = 42

    @classmethod
    def get_num_classes(cls):
        """Count class sub-directories under the dataset root."""
        if cls.DATASET_PATH and os.path.exists(cls.DATASET_PATH):
            return len(
                [
                    name
                    for name in os.listdir(cls.DATASET_PATH)
                    if os.path.isdir(os.path.join(cls.DATASET_PATH, name))
                ]
            )
        return 0

    @classmethod
    def validate_dataset(cls):
        if not cls.DATASET_PATH:
            raise ValueError(
                f"DATASET_PATH is not set for '{cls.DATASET}'. Set it in "
                "configs/agricultural.py or pass --data-root."
            )

        if not os.path.exists(cls.DATASET_PATH):
            raise ValueError(f"Dataset path does not exist: {cls.DATASET_PATH}")

        if cls.TRAIN_RATIO + cls.VAL_RATIO + cls.TEST_RATIO != 1.0:
            raise ValueError("Train/Val/Test ratios must sum to 1.0")

    @classmethod
    def describe_source(cls):
        return [
            ("dataset_path", cls.DATASET_PATH),
            ("dataset_source", "ImageFolder tree, per-class 70/15/15 split"),
            ("train_ratio", cls.TRAIN_RATIO),
            ("val_ratio", cls.VAL_RATIO),
            ("test_ratio", cls.TEST_RATIO),
        ]


class BurmeseConfig(AgriculturalConfig):
    DATASET = "burmese"
    DATASET_PATH = r"/home/student/BurmeseGrapeDataset"


class PotatoConfig(AgriculturalConfig):
    DATASET = "potato"
    DATASET_PATH = r"/home/student/PotatoDataset"


class TomatoConfig(AgriculturalConfig):
    DATASET = "tomato"
    DATASET_PATH = r"/home/student/TomatoDataset"
