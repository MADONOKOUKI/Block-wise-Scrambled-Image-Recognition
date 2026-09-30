# Original research code (archived)

This directory contains the original research code that was used for the experiments of the paper
*Block-wise Scrambled Image Recognition Using Adaptation Network* (AAAI-20 Workshop on AIoT, 2020).
It is kept **for reference and reproducibility only and is not maintained**. It targets PyTorch 1.3 with
CUDA (`torch.cuda.FloatTensor`, `.cuda()`), tensorboardX and scikit-learn (see `src/requirements.txt`).
The LE/ELE scripts read their keys from `key4/<i>_.pkl` relative to the working directory, so run them from
this directory. `key4/` holds the 64 key files (`[block size, key]` pickles written by
`BlockScramble.save`). They were not in this repository before; they are byte-identical copies of the
`key4/` files committed in the author's SIA-GAN and psivt23_scramblemix repositories. The package ships the
same keys as `blockscramble/resources/key4.npz`.

Use the package and the single training script at the repository root instead. The tests in
`tests/test_parity_original.py` import this code and check that the new implementation produces the
same outputs (scrambled images with the key files, adaptation-network features, regularisers,
learning-rate schedule).

## Old scripts -> new command

Each of the 24 scripts `{default,tanaka,proposed}_{cifar10,cifar100}_{plain,LE,ELE,EtC}.py` was run as
`python <script>.py --e=305 --tensorboard_name <name> --training_model_name <name>.t7 --json_file_name <name>.json`.
The equivalent command is

```bash
python train.py --dataset {cifar10,cifar100} --scramble {plain,le,ele,etc} --adaptation {none,tanaka,proposed}
```

| old script prefix | `--adaptation` | name in the paper (Table 3) |
|---|---|---|
| `default_*`  | `none`     | No AdaptNet |
| `tanaka_*`   | `tanaka`   | LE-AdaptNet (Tanaka 2018) |
| `proposed_*` | `proposed` | ELE-AdaptNet (proposed) |

| old script suffix | `--scramble` |
|---|---|
| `*_plain.py` | `plain` |
| `*_LE.py`    | `le`    |
| `*_ELE.py`   | `ele`   |
| `*_EtC.py`   | `etc`   |

For example, `python proposed_cifar100_ELE.py --e=305 ...` becomes
`python train.py --dataset cifar100 --scramble ele --adaptation proposed`.
`scripts/reproduce_table3.sh` lists all 24 commands.

## Old modules -> new code

| archived file | new code |
|---|---|
| `learnable_encryption.py` (`BlockScramble`, from mastnk/ICCE-TW2018) | `blockscramble.LE` / `blockscramble.ELE` (4-bit pixel operation) |
| `key4/<i>_.pkl` (LE/ELE keys) | `blockscramble.paper_keys()`, the default keys of `LE()` / `ELE()` |
| `Blockwise_scramble_LE.py` | `blockscramble.LE` |
| `Blockwise_scramble.py` + `Block_location_shuffle.py` | `blockscramble.ELE` (and `blockscramble.BlockShuffle`) |
| `etc_encryption.py` | `blockscramble.EtC` |
| `no_adaptation_network.py`, `shakedrop.py` (from owruby/shake-drop_pytorch) | `blockscramble.ShakePyramidNet`, `blockscramble.ShakeDrop` |
| `tanaka_adaptation_network.py` | `blockscramble.LEAdaptNet` (+ `ShakePyramidNet(in_channels=16)`) |
| `proposed_adaptation_network.py` | `blockscramble.ELEAdaptNet` (+ `ShakePyramidNet(in_channels=16)`) |
| `util_norm.py` (`total_variation_norm`, from utkuozbulak/pytorch-cnn-visualizations) | `blockscramble.smoothness_penalty` |
| "doubly stochastic constraint" loop in `proposed_*.py` | `blockscramble.l12_penalty` |
| `MultiStepLR` loop in the scripts | `blockscramble.original_lr` |
| `scheduler.py` (`CyclicLR`) | not ported (imported by the scripts but never used) |

Layers that the original networks defined but never used in `forward` (self-attention, two 4096x4096
linear layers, an MLP and extra 1x1 convolutions) are not ported; they received no gradients and did not
influence training.
