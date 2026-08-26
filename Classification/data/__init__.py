"""Dataset registry and the shared DataLoader builder.

``Config.LOADER`` selects one of the loader modules below. Each exposes:

    load_splits(random_seed) -> (train_data, train_labels, val_data, val_labels,
                                 test_data, test_labels, class_names)
    build_dataset(split, transform) -> torch.utils.data.Dataset
    verify() -> (total_images, [(item, error), ...])

Everything after that — transforms, sampler, DataLoader settings — is shared,
so a new dataset only needs a loader module and a config entry.
"""
import random

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from configs import Config
from . import cifar100 as _cifar100
from . import image_folder as _image_folder
from . import tinyimagenet as _tinyimagenet
from .splits import ArraySplit, PathSplit, concatenate_datasets
from .transforms import get_transforms

LOADERS = {
    "image_folder": _image_folder,
    "cifar100": _cifar100,
    "tinyimagenet": _tinyimagenet,
}


def get_loader():
    """Return the loader module for the active dataset."""
    loader = LOADERS.get(Config.LOADER)
    if loader is None:
        raise ValueError(
            f"Unknown LOADER '{Config.LOADER}'. "
            f"Choose one of: {', '.join(sorted(LOADERS))}"
        )
    return loader


def load_dataset(random_seed=42):
    """Load and split the active dataset."""
    return get_loader().load_splits(random_seed=random_seed)


def verify_dataset():
    """Decode every image in the active dataset and report failures."""
    return get_loader().verify()


def worker_init_fn_seed(worker_id):
    """Seed Python's RNG in every DataLoader worker.

    Module-level (not a lambda) so Windows can pickle it.
    """
    if Config.WORKER_SEED_MODE == "config_seed":
        random.seed(Config.RANDOM_SEED + worker_id)
    else:
        random.seed(torch.initial_seed() % (2 ** 32) + worker_id)


def _build_sampler(train_labels):
    """WeightedRandomSampler over inverse class frequency, when enabled."""
    if not Config.USE_WEIGHTED_SAMPLER:
        print("  WeightedRandomSampler: DISABLED (using default shuffle)")
        return None

    class_sample_counts = torch.bincount(torch.tensor(train_labels))
    if torch.any(class_sample_counts == 0):
        raise ValueError("Weighted sampler cannot handle a class with zero samples")
    class_weights = 1.0 / class_sample_counts.float()
    sample_weights = class_weights[torch.tensor(train_labels)]
    print("  WeightedRandomSampler: ENABLED")
    return WeightedRandomSampler(
        weights=sample_weights.double(),
        num_samples=len(sample_weights),
        replacement=True,
    )


def create_dataloaders(
    train_data,
    train_labels,
    val_data,
    test_data,
    batch_size,
    num_workers=4,
):
    """Create reproducible train/val/test DataLoaders for the active dataset.

    Only ``train_labels`` is needed beyond the splits themselves — it feeds the
    optional WeightedRandomSampler.
    """
    loader = get_loader()

    train_dataset = loader.build_dataset(train_data, get_transforms("train"))
    val_dataset = loader.build_dataset(val_data, get_transforms("val"))
    test_dataset = loader.build_dataset(test_data, get_transforms("test"))

    sampler = _build_sampler(train_labels)
    use_shuffle = sampler is None

    use_cuda = torch.cuda.is_available()
    common_loader_args = {
        "num_workers": num_workers,
        "pin_memory": Config.PIN_MEMORY and use_cuda,
        "persistent_workers": Config.PERSISTENT_WORKERS and num_workers > 0,
    }
    if num_workers > 0:
        common_loader_args["prefetch_factor"] = Config.PREFETCH_FACTOR

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=use_shuffle,
        drop_last=Config.TRAIN_DROP_LAST,
        worker_init_fn=worker_init_fn_seed,
        generator=torch.Generator().manual_seed(Config.RANDOM_SEED),
        **common_loader_args,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        **common_loader_args,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        **common_loader_args,
    )

    return train_loader, val_loader, test_loader


__all__ = [
    "ArraySplit",
    "LOADERS",
    "PathSplit",
    "concatenate_datasets",
    "create_dataloaders",
    "get_loader",
    "get_transforms",
    "load_dataset",
    "verify_dataset",
    "worker_init_fn_seed",
]
