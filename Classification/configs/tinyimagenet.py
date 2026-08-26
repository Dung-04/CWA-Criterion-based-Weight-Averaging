"""Tiny ImageNet configuration (Hugging Face ``zh-plus/tiny-imagenet``)."""
from .base import BaseConfig


class TinyImageNetConfig(BaseConfig):
    """Tiny ImageNet.

    The official ``valid`` split is kept untouched as the final test set;
    validation is a stratified holdout from the official train split
    (90,000 / 10,000).
    """

    DATASET = "tinyimagenet"
    LOADER = "tinyimagenet"

    HF_DATASET_NAME = "zh-plus/tiny-imagenet"
    HF_TRAIN_SPLIT = "train"
    HF_TEST_SPLIT = "valid"
    VALIDATION_RATIO = 0.1

    BATCH_SIZE = 1024
    NUM_EPOCHS = 60
    LEARNING_RATE = 2e-4
    WEIGHT_DECAY = 1e-4
    WARMUP_EPOCHS = 6
    EARLY_STOPPING_PATIENCE = 10

    MODELS = ["vit_base_patch16_224"]

    @classmethod
    def get_num_classes(cls):
        return 200

    @classmethod
    def validate_dataset(cls):
        if not cls.HF_DATASET_NAME:
            raise ValueError("HF_DATASET_NAME must not be empty")

        if not 0.0 < cls.VALIDATION_RATIO < 1.0:
            raise ValueError("VALIDATION_RATIO must be strictly between 0 and 1")

    @classmethod
    def describe_source(cls):
        return [
            ("dataset_name", cls.HF_DATASET_NAME),
            ("dataset_source", "huggingface datasets.load_dataset"),
            ("official_train_split", cls.HF_TRAIN_SPLIT),
            ("official_test_split", cls.HF_TEST_SPLIT),
            ("validation_ratio", cls.VALIDATION_RATIO),
        ]
