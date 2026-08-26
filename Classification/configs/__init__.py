"""Dataset config registry.

Every module in the pipeline does ``from configs import Config`` and reads
``Config.SOMETHING``. ``Config`` is a thin proxy onto whichever dataset config
class was selected by :func:`select_dataset`, so importing it before selection
is safe and CLI overrides applied later are visible everywhere.

Adding a dataset means: write a config class, register it in ``DATASET_CONFIGS``
below, and add a loader in ``data/``.
"""
from .agricultural import BurmeseConfig, PotatoConfig, TomatoConfig
from .base import BaseConfig
from .cifar100 import CIFAR100Config
from .tinyimagenet import TinyImageNetConfig

DATASET_CONFIGS = {
    "burmese": BurmeseConfig,
    "potato": PotatoConfig,
    "tomato": TomatoConfig,
    "cifar100": CIFAR100Config,
    "tinyimagenet": TinyImageNetConfig,
}

DATASET_CHOICES = tuple(DATASET_CONFIGS)

_state = {"active": None}


class _ActiveConfig:
    """Attribute proxy onto the currently selected dataset config class."""

    def __getattr__(self, name):
        active = _state["active"]
        if active is None:
            raise RuntimeError(
                "No dataset selected. Call configs.select_dataset(<name>) "
                "before reading Config."
            )
        return getattr(active, name)

    def __setattr__(self, name, value):
        active = _state["active"]
        if active is None:
            raise RuntimeError(
                "No dataset selected. Call configs.select_dataset(<name>) "
                "before writing Config."
            )
        setattr(active, name, value)

    def __repr__(self):
        active = _state["active"]
        return f"<Config for {active.DATASET if active else 'no dataset'}>"


Config = _ActiveConfig()


def select_dataset(name):
    """Make ``name`` the active dataset and return its config class."""
    key = str(name).strip().lower()
    if key not in DATASET_CONFIGS:
        raise ValueError(
            f"Unknown dataset '{name}'. Choose one of: {', '.join(DATASET_CHOICES)}"
        )
    _state["active"] = DATASET_CONFIGS[key]
    return _state["active"]


def active_config():
    """Return the selected config class itself (not the proxy)."""
    active = _state["active"]
    if active is None:
        raise RuntimeError("No dataset selected. Call configs.select_dataset() first.")
    return active


def config_items():
    """Yield ``(name, value)`` for every uppercase setting on the active config."""
    active = active_config()
    for name in sorted(dir(active)):
        if name.isupper():
            yield name, getattr(active, name)


__all__ = [
    "BaseConfig",
    "Config",
    "DATASET_CHOICES",
    "DATASET_CONFIGS",
    "active_config",
    "config_items",
    "select_dataset",
]
