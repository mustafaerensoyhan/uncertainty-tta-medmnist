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



def build_mobilenet_v3_small(num_classes: int, pretrained: bool = True,
                             dropout_p: float = 0.0) -> nn.Module:
    """
    Build a MobileNetV3-Small with a new classification head (third backbone,
    ~2.5M params). Identical training/eval contract to the other builders:
    3-channel ImageNet-normalized input, a single resized final Linear layer.

    MobileNetV3-Small's head is
        classifier = Sequential(Linear(576, 1024), Hardswish, Dropout(0.2),
                                Linear(1024, num_classes))
    so the final Linear is classifier[-1]. We resize that final Linear and,
    when dropout_p > 0, set the rate of MobileNet's existing internal Dropout
    (we do NOT rebuild the head, because its input is 576-d, not 1024-d). The
    MC-Dropout forward-hook injects dropout on the classifier's input features,
    exactly as for ResNet-18 / EfficientNet-B0, so MC-Dropout works unchanged.
    """
    weights = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.mobilenet_v3_small(weights=weights)

    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)
    if dropout_p > 0:
        # MobileNetV3-Small already contains a Dropout inside its classifier
        # (between the 576->1024 Linear and the final Linear); just set its rate.
        for mod in model.classifier:
            if isinstance(mod, nn.Dropout):
                mod.p = dropout_p

    return model



def build_deit_tiny(num_classes: int, pretrained: bool = True,
                    dropout_p: float = 0.0) -> nn.Module:
    """
    Build a DeiT-Tiny Vision Transformer (timm) as a backbone that crosses the
    CNN/Transformer boundary (~5.5M params). Same train/eval contract as the
    other builders: 3-channel ImageNet-normalized 64x64 input (timm interpolates
    the patch16 position embeddings to the 4x4 grid for us), a single linear head.

    The head is `model.head` (a Linear); timm resizes it to num_classes via the
    num_classes argument, and drop_rate sets the head dropout used when
    dropout_p>0. The MC-Dropout forward-hook injects dropout on the head's input
    regardless, exactly as for the CNN backbones (see mc_dropout._find_head,
    which now also recognizes a `.head` module).
    """
    import timm  # lazy import: only required when this backbone is used
    model = timm.create_model(
        "deit_tiny_patch16_224", pretrained=pretrained,
        num_classes=num_classes, img_size=64, drop_rate=dropout_p,
    )
    return model


# Registry of supported backbones. The rest of the codebase stays
# architecture-agnostic by going through build_model() / this dict.
ARCHITECTURES = {
    "resnet18": build_resnet18,
    "effb0": build_efficientnet_b0,
    "mnv3_small": build_mobilenet_v3_small,
    "deit_tiny": build_deit_tiny,
}

# Human-readable labels (tracker headers, figure titles, log lines).
ARCH_LABELS = {
    "resnet18": "ResNet-18",
    "effb0": "EfficientNet-B0",
    "mnv3_small": "MobileNetV3-Small",
    "deit_tiny": "DeiT-Tiny",
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
