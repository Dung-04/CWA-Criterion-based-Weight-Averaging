"""Custom classifier heads.

Neither head uses BatchNorm, which is why checkpoint averaging needs no BN
recalibration for the head itself — see ``methods/averaging.py``.
"""
import torch.nn as nn

from configs import Config


class CustomClassifier(nn.Module):
    """Fully connected head for CNN backbones (no BatchNorm)."""

    def __init__(self, in_features, num_classes):
        super(CustomClassifier, self).__init__()

        layers = []
        prev_dim = in_features

        for hidden_dim in Config.CLASSIFIER_CONFIG:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(inplace=False),  # inplace=False is safer for gradient computation
                nn.Dropout(Config.DROPOUT_RATE)
            ])
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, num_classes))

        self.classifier = nn.Sequential(*layers)

    def forward(self, x):
        return self.classifier(x)


class TransformerClassifier(nn.Module):
    """Head for Vision Transformers: LayerNorm + GELU, no BatchNorm."""

    def __init__(self, in_features, num_classes):
        super(TransformerClassifier, self).__init__()

        layers = []
        prev_dim = in_features

        for hidden_dim in Config.CLASSIFIER_CONFIG:
            layers.extend([
                nn.LayerNorm(prev_dim),  # LayerNorm is better for Transformers
                nn.Linear(prev_dim, hidden_dim),
                nn.GELU(),               # GELU is better for Transformers
                nn.Dropout(Config.DROPOUT_RATE)
            ])
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, num_classes))

        self.classifier = nn.Sequential(*layers)

    def forward(self, x):
        return self.classifier(x)
