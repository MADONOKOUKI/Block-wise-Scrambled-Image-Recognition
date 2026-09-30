#!/usr/bin/env python
"""Train and evaluate a scrambled-image classifier (Table 3 of the paper).

One entry point replaces the 24 original scripts ``{default,tanaka,proposed}_{cifar10,cifar100}_
{plain,LE,ELE,EtC}.py``; all defaults are those of the original code (see README, "Reproducing").

    python train.py --dataset cifar10 --scramble ele --adaptation proposed   # ELE-AdaptNet on ELE images
    python train.py --dataset cifar100 --scramble le --adaptation tanaka     # LE-AdaptNet on LE images
    python train.py --dataset fake --epochs 1 --batch-size 16 --depth 20 --alpha 12 --fake-size 64  # smoke test

Outputs in ``--out-dir``: ``metrics.csv`` (one row per epoch), ``summary.json``, ``last.pth`` and
``best.pth`` (best test accuracy over all epochs, which the original scripts printed at the end).
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader

import blockscramble as bs


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = p.add_argument_group("experiment (one cell of Table 3)")
    g.add_argument("--dataset", default="cifar10", choices=["cifar10", "cifar100", "fake"],
                   help="'fake' = random images (torchvision FakeData) for smoke tests")
    g.add_argument("--scramble", default="ele", choices=sorted(bs.SCRAMBLERS),
                   help="scrambling scheme: plain | le (Tanaka 2018) | ele (proposed) | etc (Chuman et al. 2018)")
    g.add_argument("--adaptation", default="proposed", choices=list(bs.ADAPTATIONS),
                   help="none = No AdaptNet | tanaka = LE-AdaptNet | proposed = ELE-AdaptNet")
    g.add_argument("--block-size", type=int, default=4)
    g.add_argument("--key-seed", type=int, default=30,
                   help="scrambling key; 30 reproduces the EtC key and ELE block permutation of the original code")
    g.add_argument("--etc-channel-shuffle", default="original", choices=["original", "permute"],
                   help="EtC colour operation: 'original' as in the paper's code, 'permute' = true permutation")

    g = p.add_argument_group("optimisation (defaults = original code)")
    g.add_argument("--epochs", type=int, default=305, help="original README command: --e=305 (paper: 300)")
    g.add_argument("--batch-size", type=int, default=512, help="original DataLoader (the paper text says 128)")
    g.add_argument("--test-batch-size", type=int, default=512)
    g.add_argument("--lr", type=float, default=0.1)
    g.add_argument("--momentum", type=float, default=0.9)
    g.add_argument("--weight-decay", type=float, default=5e-4)
    g.add_argument("--no-nesterov", dest="nesterov", action="store_false", help="disable Nesterov momentum")
    g.add_argument("--milestones", type=int, nargs="+", default=[150, 225])
    g.add_argument("--gamma", type=float, default=0.1)
    g.add_argument("--lambda-u", type=float, default=1e-3, help="weight of the L1-2 penalty L_U (Eq. 5)")
    g.add_argument("--lambda-s", type=float, default=0.1, help="weight of the smoothness penalty L_s (Eq. 5)")

    g = p.add_argument_group("classification network (Shake-PyramidNet)")
    g.add_argument("--depth", type=int, default=110)
    g.add_argument("--alpha", type=int, default=270)

    g = p.add_argument_group("run")
    g.add_argument("--data-root", default="./data", help="CIFAR is downloaded here if missing")
    g.add_argument("--out-dir", default=None, help="default: runs/<dataset>_<scramble>_<adaptation>")
    g.add_argument("--device", default="auto", help="auto (cuda > mps > cpu), cuda, cuda:1, mps or cpu")
    g.add_argument("--data-parallel", action="store_true", help="nn.DataParallel over all visible GPUs")
    g.add_argument("--workers", type=int, default=4)
    g.add_argument("--seed", type=int, default=0, help="seed for initialisation and data order")
    g.add_argument("--resume", action="store_true", help="continue from <out-dir>/last.pth")
    g.add_argument("--fake-size", type=int, default=256, help="number of training images for --dataset fake")
    return p.parse_args(argv)


def pick_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_data(args, scrambler):
    # augmentation before scrambling, no normalisation (as in the original scripts)
    train_tf = T.Compose([T.RandomCrop(32, padding=4), T.RandomHorizontalFlip(), T.ToTensor(), scrambler])
    test_tf = T.Compose([T.ToTensor(), scrambler])
    if args.dataset == "fake":
        n_test = max(args.fake_size // 2, 2)
        train = torchvision.datasets.FakeData(args.fake_size, (3, 32, 32), 10, transform=train_tf)
        test = torchvision.datasets.FakeData(n_test, (3, 32, 32), 10, transform=test_tf,
                                             random_offset=args.fake_size)
        return train, test, 10
    cls, num_classes = {"cifar10": (torchvision.datasets.CIFAR10, 10),
                        "cifar100": (torchvision.datasets.CIFAR100, 100)}[args.dataset]
    train = cls(args.data_root, train=True, download=True, transform=train_tf)
    test = cls(args.data_root, train=False, download=True, transform=test_tf)
    return train, test, num_classes


def run_epoch(model, core, loader, device, optimizer=None):
    """One pass over ``loader``; trains if ``optimizer`` is given. Returns (CE loss, reg, accuracy %)."""
    training = optimizer is not None
    model.train(training)
    ce_sum = reg_sum = correct = total = 0.0
    with torch.set_grad_enabled(training):
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            logits, feature = model(x, return_feature=True)
            ce = F.cross_entropy(logits, y)
            if training:
                reg = core.regularization(feature)  # lambda_U * L_U + lambda_s * L_s (proposed only)
                optimizer.zero_grad(set_to_none=True)
                (ce + reg).backward()
                optimizer.step()
                reg_sum += reg.item() * y.size(0)
            ce_sum += ce.item() * y.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            total += y.size(0)
    return ce_sum / total, reg_sum / total, 100.0 * correct / total


def main(argv=None):
    args = parse_args(argv)
    out_dir = args.out_dir or os.path.join("runs", f"{args.dataset}_{args.scramble}_{args.adaptation}")
    os.makedirs(out_dir, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = pick_device(args.device)

    extra = {"channel_shuffle": args.etc_channel_shuffle} if args.scramble == "etc" else {}
    scrambler = bs.get_scrambler(args.scramble, block_size=args.block_size, seed=args.key_seed, **extra)
    train_set, test_set, num_classes = build_data(args, scrambler)
    loader_kw = dict(num_workers=args.workers, pin_memory=device.type == "cuda",
                     persistent_workers=args.workers > 0)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **loader_kw)
    test_loader = DataLoader(test_set, batch_size=args.test_batch_size, shuffle=False, **loader_kw)

    core = bs.build_model(args.adaptation, num_classes=num_classes, depth=args.depth, alpha=args.alpha,
                          block_size=args.block_size, lambda_u=args.lambda_u, lambda_s=args.lambda_s).to(device)
    optimizer = torch.optim.SGD(core.parameters(), lr=args.lr, momentum=args.momentum,
                                weight_decay=args.weight_decay, nesterov=args.nesterov)
    start_epoch, best_acc, best_epoch = 0, 0.0, 0
    last_path, best_path = os.path.join(out_dir, "last.pth"), os.path.join(out_dir, "best.pth")
    if args.resume and os.path.isfile(last_path):
        ckpt = torch.load(last_path, map_location=device)
        core.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch, best_acc, best_epoch = ckpt["epoch"], ckpt["best_acc"], ckpt["best_epoch"]
        print(f"resumed from {last_path} (epoch {start_epoch}, best {best_acc:.2f}%)")
    elif args.resume:
        print(f"no checkpoint at {last_path}; starting from scratch")
    model =nn.DataParallel(core) if args.data_parallel and torch.cuda.device_count() > 1 else core

    n_params = sum(p.numel() for p in core.parameters())
    print(f"{args.dataset} | scramble={scrambler!r} | adaptation={bs.ADAPTATIONS[args.adaptation]} | "
          f"Shake-PyramidNet-{args.depth} (alpha={args.alpha}) | {n_params / 1e6:.2f}M params | device={device}")

    metrics_path = os.path.join(out_dir, "metrics.csv")
    fields = ["epoch", "lr", "train_loss", "train_reg", "train_acc", "test_loss", "test_acc",
              "best_test_acc", "seconds"]
    if start_epoch == 0 or not os.path.isfile(metrics_path):
        with open(metrics_path, "w", newline="") as f:
            csv.writer(f).writerow(fields)

    test_acc = None
    for epoch in range(start_epoch + 1, args.epochs + 1):
        t0 = time.time()
        lr = bs.original_lr(epoch, args.lr, args.milestones, args.gamma)
        for group in optimizer.param_groups:
            group["lr"] = lr
        train_loss, train_reg, train_acc = run_epoch(model, core, train_loader, device, optimizer)
        test_loss, _, test_acc = run_epoch(model, core, test_loader, device)
        if test_acc > best_acc or best_epoch == 0:
            best_acc, best_epoch = test_acc, epoch
            torch.save({"model": core.state_dict(), "epoch": epoch, "test_acc": test_acc,
                        "args": vars(args)}, best_path)
        torch.save({"model": core.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch,
                    "best_acc": best_acc, "best_epoch": best_epoch, "args": vars(args)}, last_path)
        row = [epoch, lr, train_loss, train_reg, train_acc, test_loss, test_acc, best_acc, time.time() - t0]
        with open(metrics_path, "a", newline="") as f:
            csv.writer(f).writerow(row)
        print(f"epoch {epoch:3d}/{args.epochs} lr {lr:.4g} | train loss {train_loss:.4f} reg {train_reg:.4f} "
              f"acc {train_acc:.2f}% | test loss {test_loss:.4f} acc {test_acc:.2f}% | best {best_acc:.2f}% "
              f"| {row[-1]:.1f}s")

    summary = {"best_test_acc": best_acc, "best_epoch": best_epoch, "final_test_acc": test_acc,
               "epochs": args.epochs, "num_params": n_params, "device": str(device), "args": vars(args)}
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"best test accuracy {best_acc:.2f}% (epoch {best_epoch}); results in {out_dir}")
    return summary


if __name__ == "__main__":
    main()
