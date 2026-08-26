"""Shared defaults for every classification dataset.

Per-dataset modules subclass :class:`BaseConfig` and override only what is
genuinely different for that dataset. Nothing here is dataset-specific, so a
new dataset only has to declare its own loader plus the handful of values that
differ from these defaults.
"""
import os


class BaseConfig:
    """Values shared by all datasets. Subclasses override, never delete."""

    # Set by configs.select_dataset(); identifies the active dataset.
    DATASET = None

    # ===================== Training Configuration =====================
    BATCH_SIZE = 128
    NUM_EPOCHS = 60
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-4
    WARMUP_EPOCHS = 6
    WARMUP_START_FACTOR = 0.01
    ETA_MIN = 1e-6
    SCHEDULER = "linear_warmup_cosine"

    # Optimizer.
    # Only AdamW uses decoupled weight decay; adam/sgd/rmsprop apply classic
    # L2, so the same WEIGHT_DECAY is NOT equivalent across optimizers.
    SUPPORTED_OPTIMIZERS = ("adamw", "adam", "sgd", "rmsprop")
    OPTIMIZER = "adam"
    OPTIMIZER_BETAS = (0.9, 0.999)   # adam / adamw
    OPTIMIZER_EPS = 1e-8             # adam / adamw / rmsprop
    SGD_MOMENTUM = 0.9               # sgd / rmsprop
    SGD_NESTEROV = True              # sgd
    RMSPROP_ALPHA = 0.99             # rmsprop
    USE_FUSED_OPTIMIZER = True       # ignored when the optimizer has no fused kernel
    GRAD_CLIP_NORM = 1.0

    # How weight decay is distributed over parameters:
    #   "timm_no_decay_1d" — timm param_groups_weight_decay: biases and 1D/norm
    #                        parameters are left undecayed.
    #   "all_trainable"    — a single group of every requires_grad parameter, so
    #                        decay also hits biases and norm affine parameters.
    # These are NOT equivalent. Each dataset keeps whichever one its reported
    # runs actually used.
    OPTIMIZER_PARAM_GROUPS = "timm_no_decay_1d"

    # ===================== Execution =====================
    USE_AMP = True
    AMP_DTYPE = "bfloat16"
    FLOAT32_MATMUL_PRECISION = "high"
    USE_TORCH_COMPILE = False
    TORCH_COMPILE_MODE = "reduce-overhead"

    # ===================== DataLoader =====================
    NUM_WORKERS = 16
    PREFETCH_FACTOR = 2
    PERSISTENT_WORKERS = True
    PIN_MEMORY = True
    TRAIN_DROP_LAST = True
    PROFILE_BATCHES = 0
    PRINT_DATASET_STATS = False
    # How DataLoader workers seed Python's RNG:
    #   "torch_initial_seed" — derive from torch.initial_seed() (per-epoch fresh)
    #   "config_seed"        — Config.RANDOM_SEED + worker_id (fixed every epoch)
    WORKER_SEED_MODE = "torch_initial_seed"

    # ===================== Early stopping / LR decay =====================
    EARLY_STOPPING_PATIENCE = 12
    LR_DECAY_PATIENCE = 5
    LR_DECAY_FACTOR = 0.5

    # ===================== Sampler =====================
    USE_WEIGHTED_SAMPLER = False

    # ===================== Cross-Validation (optional) =====================
    USE_CROSS_VALIDATION = False
    CV_N_SPLITS = 5

    # ===================== Loss Function =====================
    LOSS_FUNCTION = "cross_entropy"
    LABEL_SMOOTHING = 0.05
    FOCAL_GAMMA = 2.0
    POLY_EPSILON = 1.0
    CLASS_WEIGHT_METHOD = "inverse_freq"

    # ===================== Model =====================
    MODELS = ["vit_base_patch16_224"]
    PRETRAINED = True
    VIT_PRETRAINED_MODEL_ID = "vit_base_patch16_224.augreg2_in21k_ft_in1k"

    CLASSIFIER_CONFIG = [512, 256]
    DROPOUT_RATE = 0.3
    MODEL_DROP_RATE = 0.0
    MODEL_ATTN_DROP_RATE = 0.0
    MODEL_DROP_PATH_RATE = 0.1

    # ===================== Image / augmentation =====================
    IMAGE_SIZE = 224
    RESIZE_INTERPOLATION = "bicubic"
    IMAGE_MEAN = (0.5, 0.5, 0.5)
    IMAGE_STD = (0.5, 0.5, 0.5)

    # Which train-time augmentation policy data/transforms.py builds.
    #   "pretrain_224" — resize + hflip + RandomErasing (+ Mixup/CutMix in the
    #                    training loop). Used by CIFAR-100 and Tiny ImageNet.
    #   "agri"         — resize + h/v flip + rotation + colour jitter, no
    #                    erasing, no Mixup. Used by the agricultural datasets.
    AUGMENTATION = "pretrain_224"

    USE_MIXUP_CUTMIX = True
    MIXUP_ALPHA = 0.8
    CUTMIX_ALPHA = 1.0
    MIXUP_PROB = 1.0
    MIXUP_SWITCH_PROB = 0.5
    MIXUP_MODE = "batch"
    HORIZONTAL_FLIP_PROB = 0.5
    RANDOM_ERASING_PROB = 0.25
    RANDOM_ERASING_SCALE = (0.02, 0.33)
    RANDOM_ERASING_RATIO = (0.3, 3.3)
    RANDOM_ERASING_VALUE = "random"

    # "agri" policy only.
    VERTICAL_FLIP_PROB = 0.3
    ROTATION_DEGREES = 90
    COLOR_JITTER_BRIGHTNESS = (0.8, 1.2)
    COLOR_JITTER_CONTRAST = (0.8, 1.2)

    # ===================== Evaluation =====================
    TOP_K_VALUES = [2, 3, 4, 5]      # K values reported for CWA
    KEEP_TOP_K_CHECKPOINTS = 5       # must be >= max(TOP_K_VALUES)
    # Disk-retention only: how many recent epochs stay on disk alongside the
    # top-K. Does not affect which checkpoints CWA averages (the top-K set is
    # always retained), only peak checkpoint disk usage during training.
    KEEP_LAST_N_CHECKPOINTS = 10

    # ===================== Output =====================
    if os.path.exists("/kaggle"):
        CHECKPOINTS_DIR = "/kaggle/working/checkpoints"
        RESULTS_DIR = "/kaggle/working/results"
    else:
        CHECKPOINTS_DIR = "checkpoints"
        RESULTS_DIR = "results"

    AUTO_DELETE_CHECKPOINTS = True
    SAVE_METHOD_CHECKPOINTS = False
    KEEP_RESULTS = True

    # ===================== Reproducibility =====================
    SEEDS = [1, 10, 42, 100, 500]
    RANDOM_SEED = 1

    # Normally ETA_MIN < LEARNING_RATE. Set True only for a dataset whose
    # reported runs really did use ETA_MIN >= LEARNING_RATE, so validation
    # warns instead of refusing to reproduce them.
    ALLOW_ETA_MIN_ABOVE_LR = False

    # ------------------------------------------------------------------
    # Hooks a dataset config may override
    # ------------------------------------------------------------------
    @classmethod
    def get_num_classes(cls):
        raise NotImplementedError(f"{cls.__name__} must implement get_num_classes()")

    @classmethod
    def validate_dataset(cls):
        """Dataset-specific validation. Override when there is something to check."""

    @classmethod
    def describe_source(cls):
        """Rows describing where the data came from, for the run-config export."""
        return []

    @classmethod
    def validate_config(cls):
        """Validate the active configuration."""
        cls.validate_dataset()

        for name in ("MIXUP_PROB", "MIXUP_SWITCH_PROB", "RANDOM_ERASING_PROB"):
            value = getattr(cls, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1")

        if cls.USE_MIXUP_CUTMIX and cls.LOSS_FUNCTION != "cross_entropy":
            raise ValueError(
                "Mixup/CutMix currently requires LOSS_FUNCTION='cross_entropy' "
                "because PolyFocalLoss only accepts hard labels"
            )

        if cls.EARLY_STOPPING_PATIENCE >= cls.NUM_EPOCHS:
            raise ValueError("Early stopping patience should be less than num_epochs")

        if cls.WARMUP_EPOCHS >= cls.NUM_EPOCHS:
            raise ValueError("Warmup epochs should be less than num_epochs")

        if cls.ETA_MIN >= cls.LEARNING_RATE:
            if not cls.ALLOW_ETA_MIN_ABOVE_LR:
                raise ValueError("ETA_MIN must be smaller than LEARNING_RATE")
            print(
                f"  NOTE: ETA_MIN ({cls.ETA_MIN}) >= LEARNING_RATE "
                f"({cls.LEARNING_RATE}) — the cosine phase raises the LR toward "
                "ETA_MIN instead of decaying it. This reproduces the reported runs."
            )

        if cls.BATCH_SIZE <= 0:
            raise ValueError("BATCH_SIZE must be positive")

        if cls.NUM_WORKERS < 0:
            raise ValueError("NUM_WORKERS must be non-negative")

        if cls.PREFETCH_FACTOR <= 0:
            raise ValueError("PREFETCH_FACTOR must be positive")

        if cls.USE_AMP and cls.AMP_DTYPE != "bfloat16":
            raise ValueError("This pipeline currently supports AMP_DTYPE='bfloat16'")

        if cls.OPTIMIZER.lower() not in cls.SUPPORTED_OPTIMIZERS:
            raise ValueError(
                f"Unsupported OPTIMIZER '{cls.OPTIMIZER}'. "
                f"Choose one of: {', '.join(cls.SUPPORTED_OPTIMIZERS)}"
            )

        if cls.OPTIMIZER_PARAM_GROUPS not in ("timm_no_decay_1d", "all_trainable"):
            raise ValueError(
                "OPTIMIZER_PARAM_GROUPS must be 'timm_no_decay_1d' or 'all_trainable'"
            )

        if cls.OPTIMIZER.lower() == "sgd" and cls.SGD_NESTEROV and cls.SGD_MOMENTUM <= 0:
            raise ValueError("SGD_NESTEROV=True requires SGD_MOMENTUM > 0")

        if not cls.TOP_K_VALUES or any(int(k) <= 1 for k in cls.TOP_K_VALUES):
            raise ValueError("TOP_K_VALUES must be a non-empty list of ints > 1")

        if int(cls.KEEP_TOP_K_CHECKPOINTS) < max(cls.TOP_K_VALUES):
            raise ValueError(
                "KEEP_TOP_K_CHECKPOINTS must be >= max(TOP_K_VALUES) so every "
                "reported K has enough checkpoints to average"
            )

        if cls.NUM_EPOCHS < max(cls.TOP_K_VALUES):
            raise ValueError(
                f"NUM_EPOCHS={cls.NUM_EPOCHS} < max(TOP_K_VALUES)="
                f"{max(cls.TOP_K_VALUES)} — not enough checkpoints to average"
            )

        if cls.AUGMENTATION not in ("pretrain_224", "agri"):
            raise ValueError("AUGMENTATION must be 'pretrain_224' or 'agri'")

        print("[OK] Config validated successfully")
        print(f"  Dataset: {cls.DATASET}")
        print(f"  Number of classes: {cls.get_num_classes()}")
        print(f"  Optimizer: {cls.OPTIMIZER} ({cls.OPTIMIZER_PARAM_GROUPS})")
        print(f"  Augmentation: {cls.AUGMENTATION}")
        print(f"  Models to train: {len(cls.MODELS)}")
