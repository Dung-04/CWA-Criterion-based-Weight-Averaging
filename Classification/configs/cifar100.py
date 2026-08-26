"""CIFAR-100 configuration (torchvision, read from disk)."""
import os

from .base import BaseConfig


class CIFAR100Config(BaseConfig):
    """CIFAR-100.

    The official test split (10,000 images) is kept untouched as the final test
    set; validation is a stratified holdout from the official train split
    (45,000 / 5,000).
    """

    DATASET = "cifar100"
    LOADER = "cifar100"

    DATA_ROOT = "/lustre/fsmisc/dataset"   # torchvision looks for <root>/cifar-100-python
    DOWNLOAD_DATASET = False               # dataset is pre-staged on the server
    OFFICIAL_TRAIN_SPLIT = "train"         # 50,000 images
    OFFICIAL_TEST_SPLIT = "test"           # 10,000 images, never split
    VALIDATION_RATIO = 0.1

    BATCH_SIZE = 128
    NUM_EPOCHS = 60
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-4
    WARMUP_EPOCHS = 6
    EARLY_STOPPING_PATIENCE = 12

    MODELS = ["vgg16"]

    @classmethod
    def get_num_classes(cls):
        """CIFAR-100 fine labels."""
        return 100

    @classmethod
    def validate_dataset(cls):
        if not cls.DATA_ROOT:
            raise ValueError("DATA_ROOT must not be empty")

        if not cls.DOWNLOAD_DATASET and not os.path.isdir(
            os.path.join(cls.DATA_ROOT, "cifar-100-python")
        ):
            raise ValueError(
                f"CIFAR-100 not found at {os.path.join(cls.DATA_ROOT, 'cifar-100-python')}. "
                "Set --data-root to the folder that contains 'cifar-100-python'."
            )

        if not 0.0 < cls.VALIDATION_RATIO < 1.0:
            raise ValueError("VALIDATION_RATIO must be strictly between 0 and 1")

    @classmethod
    def describe_source(cls):
        return [
            ("data_root", cls.DATA_ROOT),
            ("dataset_source", "torchvision.datasets.CIFAR100 (download=False)"),
            ("official_train_split", cls.OFFICIAL_TRAIN_SPLIT),
            ("official_test_split", cls.OFFICIAL_TEST_SPLIT),
            ("validation_ratio", cls.VALIDATION_RATIO),
        ]
