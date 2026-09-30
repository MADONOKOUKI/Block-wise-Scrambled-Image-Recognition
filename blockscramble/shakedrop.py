"""Shake-PyramidNet classification network (Yamada et al., "ShakeDrop regularization", 2018).

The paper uses the shakedrop network "with the same setting as that of the original implementation"
(Section "Experimental Validation"); the original code instantiates PyramidNet-110 with widening factor
alpha = 270 and ShakeDrop with the linear-decay rule (p_L = 0.5). This is a device-agnostic port of the
implementation used by the original code, owruby/shake-drop_pytorch
(https://github.com/owruby/shake-drop_pytorch).
Layer names are kept, so ``state_dict`` keys match the original ``ShakePyramidNet``.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["ShakeDrop", "ShakeBasicBlock", "ShakePyramidNet", "init_weights"]


class ShakeDropFunction(torch.autograd.Function):
    """ShakeDrop in training mode, as in owruby/shake-drop_pytorch.

    One Bernoulli gate ``b ~ B(1 - p_drop)`` per mini-batch; if ``b == 0`` the forward pass scales
    every sample by ``alpha ~ U(alpha_range)`` and the backward pass by ``beta ~ U(0, 1)``.
    The gate is drawn from the CPU generator (the original drew it on the GPU and read it back with
    ``.item()``): same distribution, but no device synchronisation in every block.
    """

    @staticmethod
    def forward(ctx, x, p_drop=0.5, alpha_range=(-1.0, 1.0)):
        ctx.shake = bool(torch.rand(()) < p_drop)  # gate b == 0
        if ctx.shake:
            alpha = x.new_empty(x.size(0)).uniform_(*alpha_range)
            return alpha.view(-1, 1, 1, 1).expand_as(x) * x
        return x

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.shake:
            beta = grad_output.new_empty(grad_output.size(0)).uniform_(0, 1)
            return beta.view(-1, 1, 1, 1).expand_as(grad_output) * grad_output, None, None
        return grad_output, None, None


class ShakeDrop(nn.Module):
    """ShakeDrop; at test time the branch is scaled by its expectation ``1 - p_drop``."""

    def __init__(self, p_drop: float = 0.5, alpha_range=(-1.0, 1.0)):
        super().__init__()
        self.p_drop = p_drop
        self.alpha_range = tuple(alpha_range)

    def forward(self, x):
        if self.training:
            return ShakeDropFunction.apply(x, self.p_drop, self.alpha_range)
        return (1 - self.p_drop) * x

    def extra_repr(self) -> str:
        return f"p_drop={self.p_drop:.4f}"


class ShakeBasicBlock(nn.Module):
    """PyramidNet basic block (BN-conv-BN-ReLU-conv-BN) with ShakeDrop and a zero-padded shortcut."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1, p_shakedrop: float = 1.0):
        super().__init__()
        self.downsampled = stride == 2
        self.branch = nn.Sequential(
            nn.BatchNorm2d(in_ch),
            nn.Conv2d(in_ch, out_ch, 3, padding=1, stride=stride, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(out_ch),
        )
        self.shortcut = nn.AvgPool2d(2) if self.downsampled else nn.Identity()
        self.shake_drop = ShakeDrop(p_shakedrop)

    def forward(self, x):
        h = self.shake_drop(self.branch(x))
        h0 = self.shortcut(x)
        pad = h0.new_zeros(h0.size(0), h.size(1) - h0.size(1), h0.size(2), h0.size(3))
        return h + torch.cat([h0, pad], dim=1)


def init_weights(module: nn.Module) -> None:
    """Initialisation loop of the original networks (He-normal convs, BN = (1, 0), zero FC bias)."""
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2.0 / n))
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.bias.data.zero_()


class ShakePyramidNet(nn.Module):
    """Shake-PyramidNet for 32x32 inputs (the classification network of the paper).

    Args:
        depth: network depth, ``(depth - 2) / 6`` basic blocks per stage (paper/code: 110).
        alpha: total widening of PyramidNet (paper/code: 270).
        num_classes: number of classes (``label`` in the original code).
        in_channels: input channels: 3 for images (no adaptation network), 16 behind the
            adaptation networks (their output feature map is 16 x 32 x 32, Fig. 4).
    """

    def __init__(self, depth: int = 110, alpha: int = 270, num_classes: int = 10, in_channels: int = 3):
        super().__init__()
        in_ch = 16
        n_units = (depth - 2) // 6
        in_chs = [in_ch] + [in_ch + math.ceil((alpha / (3 * n_units)) * (i + 1)) for i in range(3 * n_units)]
        self.in_chs, self.u_idx = in_chs, 0
        # linear decay rule: p_drop grows from 0.5 / L to 0.5 (expression kept from the original)
        self.ps_shakedrop = [1 - (1.0 - (0.5 / (3 * n_units)) * (i + 1)) for i in range(3 * n_units)]

        self.c_in = nn.Conv2d(in_channels, in_chs[0], 3, padding=1)
        self.bn_in = nn.BatchNorm2d(in_chs[0])
        self.layer1 = self._make_layer(n_units, 1)
        self.layer2 = self._make_layer(n_units, 2)
        self.layer3 = self._make_layer(n_units, 2)
        self.bn_out = nn.BatchNorm2d(in_chs[-1])
        self.fc_out = nn.Linear(in_chs[-1], num_classes)
        init_weights(self)

    def _make_layer(self, n_units: int, stride: int) -> nn.Sequential:
        layers = []
        for _ in range(int(n_units)):
            layers.append(
                ShakeBasicBlock(self.in_chs[self.u_idx], self.in_chs[self.u_idx + 1], stride,
                                self.ps_shakedrop[self.u_idx])
            )
            self.u_idx, stride = self.u_idx + 1, 1
        return nn.Sequential(*layers)

    def forward(self, x):
        h = self.bn_in(self.c_in(x))
        h = self.layer3(self.layer2(self.layer1(h)))
        h = F.relu(self.bn_out(h))
        h = F.adaptive_avg_pool2d(h, 1)  # == F.avg_pool2d(h, 8) of the original for 32x32 inputs
        return self.fc_out(h.flatten(1))
