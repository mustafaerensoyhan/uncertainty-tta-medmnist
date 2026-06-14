"""
Model construction.

The Phase 1-4 baseline is ResNet-18 pretrained on ImageNet, with only the final
FC layer resized to each dataset's number of classes. Phase 5 (VMV plan) adds a
second backbone, EfficientNet-B0, with identical hyperparameters — only the
architecture changes — to show the calibration findings are backbone-independent.

Both are reached through build_model(arch, ...) so the rest of the codebase stays
architecture-agnostic. We deliberately keep the first conv layer unchanged (it
expects 3-channel input) because data.py converts grayscale to 3-channel via
replication, so the ImageNet-pretrained stem transfers for either backbone.
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


def build_efficientnet_b0(num_classes: int, pretrained: bool = True,
                          dropout_p: float = 0.0) -> nn.Module:
    """
    Build an EfficientNet-B0 with a new classification head (Phase 5 second
    backbone). Identical training/eval contract to build_resnet18: 3-channel
    ImageNet-normalized input, a single resized final Linear layer.

    EfficientNet-B0's head is `classifier = Sequential(Dropout(p=0.2), Linear)`.
    We resize the Linear and, by default, keep EfficientNet's built-in dropout
    slot untouched. If dropout_p > 0 we replace the head with our own
    Sequential(Dropout(dropout_p), Linear) so the MC-Dropout forward-hook (which
    injects dropout before the head) behaves identically to the ResNet case.

    Note (VMV plan): hyperparameters are identical to the ResNet-18 runs — only
    the architecture changes. Pass pretrained=True for the ImageNet weights the
    plan specifies; pretrained=False when loading a trained checkpoint.
    """
    weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.efficientnet_b0(weights=weights)

    in_features = model.classifier[1].in_features
    if dropout_p > 0:
        model.classifier = nn.Sequential(
            nn.Dropout(p=dropout_p),
            nn.Linear(in_features, num_classes),
        )
    else:
        model.classifier[1] = nn.Linear(in_features, num_classes)

    return model


# Registry of supported backbones. The rest of the codebase stays
# architecture-agnostic by going through build_model() / this dict.
ARCHITECTURES = {
    "resnet18": build_resnet18,
    "effb0": build_efficientnet_b0,
}

# Human-readable labels (tracker headers, figure titles, log lines).
ARCH_LABELS = {
    "resnet18": "ResNet-18",
    "effb0": "EfficientNet-B0",
}


def build_model(arch: str, num_classes: int, pretrained: bool = True,
                dropout_p: float = 0.0) -> nn.Module:
    """
    Build a backbone by name. arch ∈ {"resnet18", "effb0"}.

    This is the single entry point every script should use so adding a third
    backbone later is a one-line change to ARCHITECTURES.
    """
    if arch not in ARCHITECTURES:
        valid = ", ".join(ARCHITECTURES)
        raise ValueError(f"Unknown architecture '{arch}'. Valid: {valid}")
    return ARCHITECTURES[arch](num_classes, pretrained=pretrained,
                               dropout_p=dropout_p)
