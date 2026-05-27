"""
Model construction.

For the baseline we use ResNet-18 pretrained on ImageNet. The architecture is
held fixed across all 6 datasets per the proposal — only the final FC layer
is resized to match each dataset's number of classes.

We deliberately keep the first conv layer unchanged (it expects 3-channel
input) because data.py converts grayscale to 3-channel via replication.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models


def build_resnet18(num_classes: int, pretrained: bool = True,
                   dropout_p: float = 0.0) -> nn.Module:
    """
    Build a ResNet-18 with a new classification head.

    Args:
        num_classes: dataset's number of classes (we pass C even for binary,
                     so binary tasks become 2-way softmax — keeps the rest of
                     the pipeline uniform).
        pretrained: load ImageNet weights (recommended; required for the
                    benchmark accuracies in the proposal).
        dropout_p: if >0, inserts a Dropout layer before the FC. This matters
                   for the MC-Dropout TTA strategy (S4 in Phase 3). For the
                   baseline we keep it at 0; for the MC-Dropout experiments
                   pass e.g. 0.2.
    """
    weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.resnet18(weights=weights)

    in_features = model.fc.in_features
    if dropout_p > 0:
        model.fc = nn.Sequential(
            nn.Dropout(p=dropout_p),
            nn.Linear(in_features, num_classes),
        )
    else:
        model.fc = nn.Linear(in_features, num_classes)

    return model


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters — handy for sanity checking."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
