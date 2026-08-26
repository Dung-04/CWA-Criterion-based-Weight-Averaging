"""Train/eval image transforms.

Two augmentation policies, selected by ``Config.AUGMENTATION``:

``pretrain_224``
    Resize -> horizontal flip -> normalize -> RandomErasing. Used by CIFAR-100
    and Tiny ImageNet, which additionally apply Mixup/CutMix at batch level
    inside the training loop.

``agri``
    Resize -> horizontal + vertical flip -> rotation -> colour jitter ->
    normalize. Used by the agricultural leaf-disease datasets. No erasing and
    no Mixup, matching the reported agricultural runs.

Everything each policy uses comes from config, so a dataset changes
augmentation strength without touching this file.
"""
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from configs import Config

_INTERPOLATION = {
    "bicubic": InterpolationMode.BICUBIC,
    "bilinear": InterpolationMode.BILINEAR,
    "nearest": InterpolationMode.NEAREST,
}


def _resize():
    """Square resize to ``Config.IMAGE_SIZE`` using the configured filter."""
    mode = _INTERPOLATION.get(str(Config.RESIZE_INTERPOLATION).lower())
    if mode is None:
        raise ValueError(
            f"Unsupported RESIZE_INTERPOLATION '{Config.RESIZE_INTERPOLATION}'. "
            f"Choose one of: {', '.join(sorted(_INTERPOLATION))}"
        )
    return transforms.Resize(
        (Config.IMAGE_SIZE, Config.IMAGE_SIZE),
        interpolation=mode,
        antialias=True,
    )


def _normalize():
    return transforms.Normalize(mean=Config.IMAGE_MEAN, std=Config.IMAGE_STD)


def _pretrain_224(split):
    """Policy used by CIFAR-100 and Tiny ImageNet."""
    if split != "train":
        return transforms.Compose([_resize(), transforms.ToTensor(), _normalize()])

    return transforms.Compose(
        [
            _resize(),
            transforms.RandomHorizontalFlip(p=Config.HORIZONTAL_FLIP_PROB),
            transforms.ToTensor(),
            _normalize(),
            transforms.RandomErasing(
                p=Config.RANDOM_ERASING_PROB,
                scale=Config.RANDOM_ERASING_SCALE,
                ratio=Config.RANDOM_ERASING_RATIO,
                value=Config.RANDOM_ERASING_VALUE,
            ),
        ]
    )


def _agri(split):
    """Policy used by the agricultural leaf-disease datasets."""
    if split != "train":
        return transforms.Compose([_resize(), transforms.ToTensor(), _normalize()])

    return transforms.Compose(
        [
            _resize(),
            transforms.RandomHorizontalFlip(p=Config.HORIZONTAL_FLIP_PROB),
            transforms.RandomVerticalFlip(p=Config.VERTICAL_FLIP_PROB),
            transforms.RandomRotation(degrees=Config.ROTATION_DEGREES),
            transforms.ColorJitter(
                brightness=Config.COLOR_JITTER_BRIGHTNESS,
                contrast=Config.COLOR_JITTER_CONTRAST,
            ),
            transforms.ToTensor(),
            _normalize(),
        ]
    )


_POLICIES = {
    "pretrain_224": _pretrain_224,
    "agri": _agri,
}


def get_transforms(split="train"):
    """Build the transform pipeline for ``split`` under the active policy.

    Args:
        split: 'train', 'val', or 'test'. Only 'train' is augmented.
    """
    policy = _POLICIES.get(Config.AUGMENTATION)
    if policy is None:
        raise ValueError(
            f"Unknown AUGMENTATION '{Config.AUGMENTATION}'. "
            f"Choose one of: {', '.join(sorted(_POLICIES))}"
        )
    return policy(split)
