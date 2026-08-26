"""Split containers shared by the dataset loaders.

Every loader returns splits exposing the same small read-only API, so training,
cross-validation and the statistics helpers do not care which dataset is
active:

* ``len(split)``
* ``split["image"]`` / ``split["label"]`` — column access
* ``split[i]`` — one item
* ``split.select(indices)`` — a new split with only those rows

Hugging Face ``datasets.Dataset`` already satisfies this API, so the Tiny
ImageNet loader returns it unchanged.
"""
import numpy as np
from PIL import Image


class PathSplit:
    """Split backed by image file paths on disk (agricultural datasets).

    ``split[i]`` returns the path rather than a decoded image, so iterating for
    statistics stays lazy and the dataset wrapper controls when files open.
    """

    def __init__(self, image_paths, labels):
        self.image_paths = list(image_paths)
        self.labels = [int(label) for label in labels]
        if len(self.image_paths) != len(self.labels):
            raise ValueError("image_paths and labels must have the same length")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, key):
        if isinstance(key, str):
            if key == "image":
                return list(self.image_paths)
            if key == "label":
                return list(self.labels)
            raise KeyError(f"Unknown column: {key}")

        return self.image_paths[int(key)]

    def select(self, indices):
        indices = [int(idx) for idx in indices]
        return PathSplit(
            [self.image_paths[idx] for idx in indices],
            [self.labels[idx] for idx in indices],
        )


class ArraySplit:
    """In-memory split backed by a uint8 numpy image array (CIFAR-100)."""

    def __init__(self, images, labels):
        self.images = np.asarray(images)
        self.labels = [int(label) for label in labels]
        if len(self.images) != len(self.labels):
            raise ValueError("images and labels must have the same length")

    def __len__(self):
        return len(self.labels)

    def _to_pil(self, index):
        return Image.fromarray(self.images[index])

    def __getitem__(self, key):
        if isinstance(key, str):
            if key == "image":
                return [self._to_pil(idx) for idx in range(len(self))]
            if key == "label":
                return list(self.labels)
            raise KeyError(f"Unknown column: {key}")

        index = int(key)
        return {"image": self._to_pil(index), "label": self.labels[index]}

    def select(self, indices):
        """Return a new split containing only ``indices`` (used by K-Fold CV)."""
        indices = [int(idx) for idx in indices]
        return ArraySplit(
            self.images[indices],
            [self.labels[idx] for idx in indices],
        )


def concatenate_datasets(splits):
    """Concatenate splits of the same kind into one split."""
    splits = list(splits)
    if not splits:
        raise ValueError("concatenate_datasets requires at least one split")

    first = splits[0]

    if isinstance(first, PathSplit):
        return PathSplit(
            [path for split in splits for path in split.image_paths],
            [label for split in splits for label in split.labels],
        )

    if isinstance(first, ArraySplit):
        return ArraySplit(
            np.concatenate([split.images for split in splits], axis=0),
            [label for split in splits for label in split.labels],
        )

    # Hugging Face Dataset
    from datasets import concatenate_datasets as hf_concatenate_datasets

    return hf_concatenate_datasets(splits)
