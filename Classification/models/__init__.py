"""Pretrained backbones with a custom classifier head.

Every supported model follows the same three steps: build the pretrained
backbone, set the backbone's ``requires_grad``, then replace its head. The
per-model differences are only *which* attribute holds the head and how the
backbone parameters are addressed, so they live in ``_SPECS`` rather than in
ten copies of the same code.

Adding a model means adding one ``_ModelSpec`` entry.
"""
import timm
from torchvision import models as tv_models

from configs import Config
from .heads import CustomClassifier, TransformerClassifier


class _ModelSpec:
    """How to build one backbone and where its classifier head lives.

    Args:
        build: Callable returning the pretrained backbone.
        head_path: Dotted attribute path to the head, e.g. "fc" or "head.fc".
        head_cls: CustomClassifier (CNNs) or TransformerClassifier (ViTs).
        backbone: Attribute holding the feature extractor, e.g. "features".
            When None, backbone parameters are selected by name prefix instead.
        in_features_from: Optional dotted path to read ``in_features`` from,
            when it is not the head module itself.
    """

    def __init__(self, build, head_path, head_cls, backbone=None,
                 in_features_from=None):
        self.build = build
        self.head_path = head_path
        self.head_cls = head_cls
        self.backbone = backbone
        self.in_features_from = in_features_from


def _resolve(model, dotted_path):
    """Return the attribute at ``dotted_path`` on ``model``."""
    target = model
    for part in dotted_path.split("."):
        target = getattr(target, part)
    return target


def _assign(model, dotted_path, value):
    """Assign ``value`` at ``dotted_path`` on ``model``."""
    parts = dotted_path.split(".")
    parent = model
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], value)


def _set_backbone_trainable(model, spec, trainable):
    """Freeze or unfreeze every backbone parameter.

    Uses the module attribute when the spec names one, otherwise selects by
    name prefix — matching how each architecture separates head from backbone.
    """
    if spec.backbone is not None:
        for param in _resolve(model, spec.backbone).parameters():
            param.requires_grad = trainable
        return

    head_prefix = spec.head_path.split(".")[0]
    for name, param in model.named_parameters():
        if not name.startswith(head_prefix):
            param.requires_grad = trainable


def _vit():
    return timm.create_model(
        Config.VIT_PRETRAINED_MODEL_ID,
        pretrained=Config.PRETRAINED,
        drop_rate=Config.MODEL_DROP_RATE,
        drop_path_rate=Config.MODEL_DROP_PATH_RATE,
        attn_drop_rate=Config.MODEL_ATTN_DROP_RATE,
    )


_SPECS = {
    'vgg16': _ModelSpec(
        build=lambda: tv_models.vgg16(weights=tv_models.VGG16_Weights.IMAGENET1K_V1),
        head_path='classifier',
        head_cls=CustomClassifier,
        backbone='features',
        in_features_from='classifier.0',
    ),
    'resnet18': _ModelSpec(
        build=lambda: tv_models.resnet18(weights=tv_models.ResNet18_Weights.IMAGENET1K_V1),
        head_path='fc',
        head_cls=CustomClassifier,
    ),
    'resnet101': _ModelSpec(
        build=lambda: tv_models.resnet101(weights=tv_models.ResNet101_Weights.IMAGENET1K_V1),
        head_path='fc',
        head_cls=CustomClassifier,
    ),
    'mobilenet_v2': _ModelSpec(
        build=lambda: tv_models.mobilenet_v2(weights=tv_models.MobileNet_V2_Weights.IMAGENET1K_V1),
        head_path='classifier',
        head_cls=CustomClassifier,
        backbone='features',
        in_features_from='classifier.1',
    ),
    'densenet121': _ModelSpec(
        build=lambda: tv_models.densenet121(weights=tv_models.DenseNet121_Weights.IMAGENET1K_V1),
        head_path='classifier',
        head_cls=CustomClassifier,
    ),
    'efficientnet_b0': _ModelSpec(
        build=lambda: timm.create_model('efficientnet_b0', pretrained=True),
        head_path='classifier',
        head_cls=CustomClassifier,
    ),
    'convnext_tiny': _ModelSpec(
        build=lambda: timm.create_model('convnext_tiny', pretrained=True),
        head_path='head.fc',
        head_cls=CustomClassifier,
    ),
    'vit_base_patch16_224': _ModelSpec(
        build=_vit,
        head_path='head',
        head_cls=TransformerClassifier,
    ),
    # Swin has a ClassifierHead with a nested fc; replace only fc to keep global_pool.
    'swin_tiny_patch4_window7_224': _ModelSpec(
        build=lambda: timm.create_model('swin_tiny_patch4_window7_224', pretrained=True),
        head_path='head.fc',
        head_cls=TransformerClassifier,
    ),
    'convit_tiny': _ModelSpec(
        build=lambda: timm.create_model('convit_tiny', pretrained=True),
        head_path='head',
        head_cls=TransformerClassifier,
    ),
}

SUPPORTED_MODELS = list(_SPECS)

MODEL_ALIASES = {
    "vit_base": "vit_base_patch16_224",
    "vit_b16": "vit_base_patch16_224",
    "efficientnet_b0": "efficientnet_b0",
    "efficientnet-b0": "efficientnet_b0",
    "mobilenetv2": "mobilenet_v2",
    "mobilenet_v2": "mobilenet_v2",
    "swin_tiny": "swin_tiny_patch4_window7_224",
}


def get_model(model_name, num_classes, freeze_backbone=False):
    """Build a pretrained model with the configured classifier head.

    Args:
        model_name: Name of the pretrained model (see SUPPORTED_MODELS)
        num_classes: Number of output classes
        freeze_backbone: Whether to freeze backbone weights

    Returns:
        model: PyTorch model with custom classifier
    """
    spec = _SPECS.get(model_name)
    if spec is None:
        raise ValueError(f"Model {model_name} not supported")

    model = spec.build()

    source = spec.in_features_from or spec.head_path
    in_features = _resolve(model, source).in_features

    _set_backbone_trainable(model, spec, trainable=not freeze_backbone)

    _assign(model, spec.head_path, spec.head_cls(in_features, num_classes))

    return model


def count_parameters(model):
    """Count trainable and total parameters."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def parse_model_list(values):
    """Parse repeated, space-separated, or comma-separated model arguments."""
    if not values:
        return None

    parsed = []
    for value in values:
        for item in value.split(","):
            item = item.strip()
            if item:
                parsed.append(MODEL_ALIASES.get(item, item))

    if any(model.lower() == "all" for model in parsed):
        return SUPPORTED_MODELS.copy()

    invalid = [model for model in parsed if model not in _SPECS]
    if invalid:
        raise ValueError(
            "Unsupported model(s): "
            + ", ".join(invalid)
            + ". Supported models: "
            + ", ".join(SUPPORTED_MODELS)
        )

    return parsed


__all__ = [
    "MODEL_ALIASES",
    "SUPPORTED_MODELS",
    "CustomClassifier",
    "TransformerClassifier",
    "count_parameters",
    "get_model",
    "parse_model_list",
]
