#!/usr/bin/env bash
# Table 3 of the paper: accuracy of scrambled image classification on CIFAR-10 / CIFAR-100.
# 24 runs = 2 datasets x 3 adaptation networks (rows) x 4 scrambling schemes (columns). Every run trains
# Shake-PyramidNet-110 (alpha = 270) for 305 epochs with the defaults of the original code and writes
# runs/<dataset>_<scramble>_<adaptation>/{metrics.csv,summary.json}; like the original scripts, summary.json
# holds the best test accuracy over all epochs ("best_test_acc"). CIFAR is downloaded to ./data on first use.
# The runs are independent: launch them in parallel on several GPUs (e.g. prefix CUDA_VISIBLE_DEVICES=i).
# Extra arguments are passed to every run, e.g.  bash scripts/reproduce_table3.sh --data-root /path/to/data
set -euo pipefail
cd "$(dirname "$0")/.."

# ---- CIFAR-10 ----------------------------------------------------------------- paper: plain / LE / EtC / ELE
# No AdaptNet                                                          96.70 / 94.94 / 85.94 / 67.10
python train.py --dataset cifar10 --adaptation none --scramble plain "$@"
python train.py --dataset cifar10 --adaptation none --scramble le "$@"
python train.py --dataset cifar10 --adaptation none --scramble etc "$@"
python train.py --dataset cifar10 --adaptation none --scramble ele "$@"
# LE-AdaptNet (Tanaka 2018)                                            95.64 / 94.49 / 80.16 / 48.39
python train.py --dataset cifar10 --adaptation tanaka --scramble plain "$@"
python train.py --dataset cifar10 --adaptation tanaka --scramble le "$@"
python train.py --dataset cifar10 --adaptation tanaka --scramble etc "$@"
python train.py --dataset cifar10 --adaptation tanaka --scramble ele "$@"
# ELE-AdaptNet (proposed)                                              85.32 / 87.28 / 89.09 / 83.06
python train.py --dataset cifar10 --adaptation proposed --scramble plain "$@"
python train.py --dataset cifar10 --adaptation proposed --scramble le "$@"
python train.py --dataset cifar10 --adaptation proposed --scramble etc "$@"
python train.py --dataset cifar10 --adaptation proposed --scramble ele "$@"

# ---- CIFAR-100 ---------------------------------------------------------------- paper: plain / LE / EtC / ELE
# No AdaptNet                                                          83.59 / 78.25 / 61.90 / 43.05
python train.py --dataset cifar100 --adaptation none --scramble plain "$@"
python train.py --dataset cifar100 --adaptation none --scramble le "$@"
python train.py --dataset cifar100 --adaptation none --scramble etc "$@"
python train.py --dataset cifar100 --adaptation none --scramble ele "$@"
# LE-AdaptNet (Tanaka 2018)                                            79.13 / 75.48 / 44.83 /  7.19
python train.py --dataset cifar100 --adaptation tanaka --scramble plain "$@"
python train.py --dataset cifar100 --adaptation tanaka --scramble le "$@"
python train.py --dataset cifar100 --adaptation tanaka --scramble etc "$@"
python train.py --dataset cifar100 --adaptation tanaka --scramble ele "$@"
# ELE-AdaptNet (proposed)                                              60.36 / 71.30 / 71.91 / 62.97
python train.py --dataset cifar100 --adaptation proposed --scramble plain "$@"
python train.py --dataset cifar100 --adaptation proposed --scramble le "$@"
python train.py --dataset cifar100 --adaptation proposed --scramble etc "$@"
python train.py --dataset cifar100 --adaptation proposed --scramble ele "$@"
