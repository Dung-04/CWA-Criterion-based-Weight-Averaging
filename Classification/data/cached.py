"""RAM-cached image dataset used by CIFAR-100 and Tiny ImageNet.

Both are small enough to hold decoded PIL images in memory, which removes
per-item file/Arrow reads from the training loop.
"""
from torch.utils.data import Dataset


class CachedImageDataset(Dataset):
    """Decode every image once at init, then index in O(1)."""

    def __init__(self, split, transform=None):
        self.dataset = split
        self.transform = transform
        self.labels = [int(label) for label in split["label"]]
        print("Caching images to RAM...")
        # Column access reads the whole image column in one pass.
        all_images = split["image"]
        self._cache = [img.convert("RGB") for img in all_images]

    def __len__(self):
        return len(self._cache)

    def __getitem__(self, idx):
        image = self._cache[idx]
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label
