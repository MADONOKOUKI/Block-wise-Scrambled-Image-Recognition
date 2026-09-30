"""blockscramble: official PyTorch implementation of "Block-wise Scrambled Image Recognition Using
Adaptation Network" (Madono, Tanaka, Onishi, Ogawa; AAAI-20 Workshop on AIoT, arXiv:2001.07761).

* scrambling schemes (Table 2): :class:`Plain`, :class:`LE`, :class:`ELE` (proposed), :class:`EtC`,
  :class:`BlockShuffle` - numpy / PIL / torchvision-transform compatible; by default they use the keys
  of the paper's experiments (:func:`paper_keys`), ``seed=...`` draws new keys;
* adaptation networks: :class:`LEAdaptNet` (Tanaka 2018) and :class:`ELEAdaptNet` (proposed), with the
  regularisers :func:`l12_penalty` (Eq. 7) and :func:`smoothness_penalty` (Eq. 8);
* classification network :class:`ShakePyramidNet`, :func:`build_model` for the models of Table 3,
  and the learning-rate schedule :func:`original_lr`.
"""
from .adaptation import ELEAdaptNet, LEAdaptNet, l12_penalty, smoothness_penalty
from .models import ADAPTATIONS, ScrambledImageClassifier, build_adaptation, build_model
from .schedule import original_lr
from .scrambling import (ELE, LE, PAPER_SEED, SCRAMBLERS, BlockShuffle, EtC, Plain, Scrambler, get_scrambler,
                         paper_keys)
from .shakedrop import ShakeDrop, ShakePyramidNet

__version__ = "1.0.0"

__all__ = [
    "Scrambler", "Plain", "BlockShuffle", "LE", "ELE", "EtC", "SCRAMBLERS", "get_scrambler",
    "paper_keys", "PAPER_SEED",
    "LEAdaptNet", "ELEAdaptNet", "l12_penalty", "smoothness_penalty",
    "ShakeDrop", "ShakePyramidNet",
    "ADAPTATIONS", "ScrambledImageClassifier", "build_adaptation", "build_model",
    "original_lr",
]
