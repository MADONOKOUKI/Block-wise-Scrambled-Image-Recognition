"""Learning-rate schedule of the original training scripts."""
from __future__ import annotations

from typing import Sequence

__all__ = ["original_lr"]


def original_lr(epoch: int, base_lr: float = 0.1, milestones: Sequence[int] = (150, 225),
                gamma: float = 0.1) -> float:
    """Learning rate used during (1-based) ``epoch`` by the original scripts.

    The scripts built ``MultiStepLR(optimizer, milestones=[150, 225])`` (gamma 0.1) and called
    ``scheduler.step()`` at the *start* of every epoch, so the decays take effect at the start of
    epochs 150 and 225: 0.1 for epochs 1-149, 0.01 for 150-224 and 0.001 from 225 on (305 epochs in
    the original README command). The paper describes the same schedule as 0.1 / 0.01 / 0.001 for
    epochs 0-150 / 150-225 / 225-300.
    """
    return base_lr * gamma ** sum(epoch >= m for m in milestones)
