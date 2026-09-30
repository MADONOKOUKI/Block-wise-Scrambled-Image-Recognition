"""Adaptation networks and the regularisers of the proposed method.

An adaptation network maps a block-wise scrambled image ``(N, 3, H, W)`` to a feature map
``(N, 16, H, W)`` that is fed to an ordinary CNN (Figs. 1, 3 and 4 of the paper):

* :class:`LEAdaptNet` - "LE-AdaptNet" (Tanaka, ICCE-TW 2018): ONE block-wise sub-network shared by
  all blocks (``tanaka_adaptation_network.py``; ``--adaptation tanaka``).
* :class:`ELEAdaptNet` - "ELE-AdaptNet", the proposed network: a DIFFERENT sub-network per block,
  a learnable pseudo permutation matrix ``U`` over the blocks and a pixel-shuffle layer
  (``proposed_adaptation_network.py``; ``--adaptation proposed``).

and the proposed loss (Eq. 5) is ``L = L_CE + lambda_U * L_U + lambda_s * L_s`` with the L1-2
penalty :func:`l12_penalty` (Eq. 7) and the smoothness penalty :func:`smoothness_penalty` (Eq. 8),
``lambda_U = 0.001`` and ``lambda_s = 0.1``.

Both networks process the blocks one at a time, exactly like the original code. This matters for
batch normalisation: each block is normalised with statistics over the mini-batch only, and in
LE-AdaptNet the shared BN's running statistics are updated once per block (N times per forward).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch.nn.utils.parametrizations import spectral_norm

from .shakedrop import init_weights

__all__ = ["LEAdaptNet", "ELEAdaptNet", "l12_penalty", "smoothness_penalty"]


def l12_penalty(u: torch.Tensor) -> torch.Tensor:
    """L1-2 penalty of the pseudo permutation matrix, Eq. (7).

    ``L_U = 1/(N*N) * [ sum_i (||u_i,:||_1 - ||u_i,:||_2) + sum_j (||u_:,j||_1 - ||u_:,j||_2) ]``,
    which is zero only for rows/columns with at most one non-zero entry. Same as the
    "doubly stochastic constraint" loop of the original ``proposed_*.py`` scripts.
    """
    rows = u.abs().sum(dim=1) - u.pow(2).sum(dim=1).sqrt()
    cols = u.abs().sum(dim=0) - u.pow(2).sum(dim=0).sqrt()
    return (rows.sum() + cols.sum()) / (u.size(0) * u.size(1))


def smoothness_penalty(feature: torch.Tensor, channels: int = 3, beta: float = 2.0) -> torch.Tensor:
    """Spatial smoothness penalty ``L_s`` of Eq. (8), exactly as computed by the original code.

    Original: ``total_variation_norm(feature) / batch_size`` (``util_norm.py``, adapted from
    utkuozbulak/pytorch-cnn-visualizations). With ``beta = 2`` it is the sum of squared vertical and
    horizontal forward differences, averaged over the mini-batch. It differs from Eq. (8) in three
    details that we keep for faithfulness: only the first ``channels = 3`` of the 16 feature channels
    are penalised; both differences are taken on the top-left ``(H-1) x (W-1)`` grid; and the sum is
    divided by ``(H-1) * W * C`` with ``C`` the total number of channels (``31 * 32 * 16`` for CIFAR).
    """
    _, c, h, w = feature.shape
    x = feature[:, :channels]
    center = x[:, :, :-1, :-1]
    below = x[:, :, 1:, :-1]
    right = x[:, :, :-1, 1:]
    tv = (((center - below) ** 2 + (center - right) ** 2) ** (beta / 2)).sum()
    return tv / ((h - 1) * w * c) / feature.size(0)


def _block(x: torch.Tensor, index: int, grid_w: int, b: int) -> torch.Tensor:
    i, j = divmod(index, grid_w)
    return x[:, :, i * b:(i + 1) * b, j * b:(j + 1) * b]


class LEAdaptNet(nn.Module):
    """LE-AdaptNet (Tanaka, 2018): one sub-network shared by every block.

    Each ``B x B`` block goes through ``Conv(B x B, stride B) -> BN -> LeakyReLU(0.1)`` giving
    ``out_channels * B * B`` features, which a pixel-shuffle layer turns back into an
    ``out_channels x B x B`` patch at the same position. Works for any ``H, W`` divisible by ``B``.
    """

    def __init__(self, block_size: int = 4, in_channels: int = 3, out_channels: int = 16,
                 negative_slope: float = 0.1):
        super().__init__()
        self.block_size, self.out_channels = block_size, out_channels
        width = out_channels * block_size * block_size
        self.conv = nn.Conv2d(in_channels, width, block_size, stride=block_size, bias=False)
        self.bn = nn.BatchNorm2d(width)
        self.act = nn.LeakyReLU(negative_slope)
        self.shuffle = nn.PixelShuffle(block_size)
        init_weights(self)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = self.block_size
        gh, gw = x.size(2) // b, x.size(3) // b
        rows = []
        for i in range(gh):
            cols = [self.shuffle(self.act(self.bn(self.conv(_block(x, i * gw + j, gw, b)))))
                    for j in range(gw)]
            rows.append(torch.cat(cols, dim=3))
        return torch.cat(rows, dim=2)


class ELEAdaptNet(nn.Module):
    """ELE-AdaptNet, the proposed adaptation network (Figs. 3 and 4).

    1. Block-wise sub-networks: block ``k`` goes through its own
       ``SpectralNorm(Conv(B x B, stride B)) -> BN -> LeakyReLU(0.2)`` giving ``out_channels * B * B``
       features (``f(x_ek; theta_k)``).
    2. The ``N`` feature vectors are stacked and multiplied by the learnable pseudo permutation
       matrix ``U`` (``N x N``): ``h[:, :, k] = sum_j f_j * U[j, k]``, which can learn to undo the
       block location shuffling.
    3. A pixel-shuffle layer reshapes the ``(out_channels * B * B) x (H/B) x (W/B)`` map into the
       ``out_channels x H x W`` input of the classification network.

    ``U`` is initialised like ``nn.Linear`` weights (Kaiming uniform) and left unconstrained, as in
    the original code (the paper notes that non-negativity / doubly-stochastic constraints are not
    used). :meth:`regularization` returns ``lambda_u * L_U + lambda_s * L_s`` of Eq. (5).

    Args:
        image_size: side of the (square) input image; fixes the number of blocks ``N``.
        block_size: ``B``.
        lambda_u, lambda_s: weights of Eq. (5) (paper and code: 0.001 and 0.1).
    """

    def __init__(self, image_size: int = 32, block_size: int = 4, in_channels: int = 3,
                 out_channels: int = 16, negative_slope: float = 0.2,
                 lambda_u: float = 1e-3, lambda_s: float = 0.1):
        super().__init__()
        if image_size % block_size:
            raise ValueError("image_size must be a multiple of block_size")
        self.image_size, self.block_size, self.out_channels = image_size, block_size, out_channels
        self.grid = image_size // block_size
        self.num_blocks = self.grid * self.grid
        self.lambda_u, self.lambda_s = lambda_u, lambda_s
        width = out_channels * block_size * block_size

        convs = []
        for _ in range(self.num_blocks):
            conv = nn.Conv2d(in_channels, width, block_size, stride=block_size, bias=False)
            init_weights(conv)  # He-normal, applied to the raw weight before spectral normalisation
            # parametrization API; the original used the legacy torch.nn.utils.spectral_norm, which starts
            # the power iteration from random vectors instead of 15 initial iterations (same normalisation)
            convs.append(spectral_norm(conv))
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(nn.BatchNorm2d(width) for _ in range(self.num_blocks))
        self.act = nn.LeakyReLU(negative_slope)
        self.permutation = nn.Parameter(torch.empty(self.num_blocks, self.num_blocks))
        nn.init.kaiming_uniform_(self.permutation, a=math.sqrt(5))
        self.shuffle = nn.PixelShuffle(block_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.size(2) != self.image_size or x.size(3) != self.image_size:
            raise ValueError(f"expected {self.image_size}x{self.image_size} inputs, got {tuple(x.shape)}")
        b = self.block_size
        feats = [self.act(bn(conv(_block(x, k, self.grid, b)))).flatten(1)
                 for k, (conv, bn) in enumerate(zip(self.convs, self.bns))]
        h = torch.stack(feats, dim=2)            # (batch, width, N), blocks in source order
        h = torch.matmul(h, self.permutation)    # pseudo permutation over the block axis
        h = h.view(x.size(0), -1, self.grid, self.grid)
        return self.shuffle(h)                   # (batch, out_channels, H, W)

    def regularization(self, feature: torch.Tensor) -> torch.Tensor:
        """``lambda_u * L_U(U) + lambda_s * L_s(feature)`` (Eqs. 5, 7, 8); ``feature = self(x)``."""
        return self.lambda_u * l12_penalty(self.permutation) + self.lambda_s * smoothness_penalty(feature)
