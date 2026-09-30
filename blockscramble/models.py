"""Adaptation network + classification network, as compared in Table 3 of the paper."""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from .adaptation import ELEAdaptNet, LEAdaptNet
from .shakedrop import ShakePyramidNet

__all__ = ["ADAPTATIONS", "ScrambledImageClassifier", "build_adaptation", "build_model"]

#: ``train.py --adaptation`` names -> rows of Table 3
ADAPTATIONS = {
    "none": "No AdaptNet",
    "tanaka": "LE-AdaptNet (Tanaka 2018)",
    "proposed": "ELE-AdaptNet (proposed)",
}


class ScrambledImageClassifier(nn.Module):
    """``backbone(adaptation(x))``; the adaptation network is optional (``None`` = No AdaptNet).

    ``forward(x, return_feature=True)`` also returns the adaptation feature map, which the proposed
    loss needs: ``loss = cross_entropy(logits, y) + model.regularization(feature)``.
    """

    def __init__(self, backbone: nn.Module, adaptation: Optional[nn.Module] = None):
        super().__init__()
        self.adaptation = adaptation
        self.backbone = backbone

    def forward(self, x: torch.Tensor, return_feature: bool = False):
        feature = x if self.adaptation is None else self.adaptation(x)
        logits = self.backbone(feature)
        return (logits, feature) if return_feature else logits

    def regularization(self, feature: torch.Tensor) -> torch.Tensor:
        """Extra loss terms of the adaptation network (only ELE-AdaptNet has them; else 0)."""
        reg = getattr(self.adaptation, "regularization", None)
        return reg(feature) if reg is not None else feature.new_zeros(())


def build_adaptation(name: str, image_size: int = 32, block_size: int = 4,
                     lambda_u: float = 1e-3, lambda_s: float = 0.1) -> Optional[nn.Module]:
    """``"none"`` -> ``None``, ``"tanaka"`` -> :class:`LEAdaptNet`, ``"proposed"`` -> :class:`ELEAdaptNet`."""
    if name == "none":
        return None
    if name == "tanaka":
        return LEAdaptNet(block_size=block_size)
    if name == "proposed":
        return ELEAdaptNet(image_size=image_size, block_size=block_size,
                           lambda_u=lambda_u, lambda_s=lambda_s)
    raise ValueError(f"unknown adaptation network {name!r}; choose from {sorted(ADAPTATIONS)}")


def build_model(adaptation: str = "proposed", num_classes: int = 10, depth: int = 110,
                alpha: int = 270, image_size: int = 32, block_size: int = 4,
                lambda_u: float = 1e-3, lambda_s: float = 0.1) -> ScrambledImageClassifier:
    """The models of Table 3: adaptation network + Shake-PyramidNet-110 (alpha = 270)."""
    adapt = build_adaptation(adaptation, image_size, block_size, lambda_u, lambda_s)
    in_channels = 3 if adapt is None else adapt.out_channels
    backbone = ShakePyramidNet(depth=depth, alpha=alpha, num_classes=num_classes, in_channels=in_channels)
    return ScrambledImageClassifier(backbone, adapt)
