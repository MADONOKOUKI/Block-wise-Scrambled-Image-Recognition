"""blockscramble: official PyTorch implementation of "Block-wise Scrambled Image Recognition Using
Adaptation Network" (Madono, Tanaka, Onishi, Ogawa; AAAI-20 Workshop on AIoT, arXiv:2001.07761).

* scrambling schemes (Table 2): :class:`Plain`, :class:`LE`, :class:`ELE` (proposed), :class:`EtC`,
  :class:`BlockShuffle` - numpy / PIL / torchvision-transform compatible, keyed by ``seed``;
* adaptation networks: :class:`LEAdaptNet` (Tanaka 2018) and :class:`ELEAdaptNet` (proposed), with the
  regularisers :func:`l12_penalty` (Eq. 7) and :func:`smoothness_penalty` (Eq. 8);
* classification network :class:`ShakePyramidNet`, :func:`build_model` for the models of Table 3,
  and the learning-rate schedule :func:`original_lr`.
"""
from .adaptation import ELEAdaptNet, LEAdaptNet, l12_penalty, smoothness_penalty
from .models import ADAPTATIONS, ScrambledImageClassifier, build_adaptation, build_model
from .schedule import original_lr
from .scrambling import ELE, LE, SCRAMBLERS, BlockShuffle, EtC, Plain, Scrambler, get_scrambler
from .shakedrop import ShakeDrop, ShakePyramidNet

__version__ = "1.0.0"

__all__ = [
    "Scrambler", "Plain", "BlockShuffle", "LE", "ELE", "EtC", "SCRAMBLERS", "get_scrambler",
    "LEAdaptNet", "ELEAdaptNet", "l12_penalty", "smoothness_penalty",
    "ShakeDrop", "ShakePyramidNet",
    "ADAPTATIONS", "ScrambledImageClassifier", "build_adaptation", "build_model",
    "original_lr",
]
