"""Smoke tests of the real training entry point on synthetic data (no downloads)."""
import csv
import json
import os
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FAST = ["--dataset", "fake", "--fake-size", "32", "--batch-size", "16", "--depth", "20", "--alpha", "12",
        "--workers", "0", "--device", "cpu"]


def run(out_dir, *args, expect=0):
    cmd = [sys.executable, os.path.join(ROOT, "train.py"), *FAST, "--out-dir", str(out_dir), *args]
    res = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT,
                         env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
    assert res.returncode == expect, res.stdout + res.stderr
    return res.stdout + res.stderr


def rows(out_dir):
    with open(os.path.join(out_dir, "metrics.csv")) as f:
        return list(csv.DictReader(f))


@pytest.mark.parametrize("scramble,adaptation", [("plain", "none"), ("le", "tanaka"), ("ele", "proposed"),
                                                 ("etc", "proposed")])
def test_train_cli(tmp_path, scramble, adaptation):
    run(tmp_path, "--scramble", scramble, "--adaptation", adaptation, "--epochs", "1")
    assert len(rows(tmp_path)) == 1
    summary = json.load(open(tmp_path / "summary.json"))
    assert 0 <= summary["best_test_acc"] <= 100 and summary["args"]["adaptation"] == adaptation
    assert (tmp_path / "last.pth").exists() and (tmp_path / "best.pth").exists()
    if adaptation == "proposed":
        assert float(rows(tmp_path)[0]["train_reg"]) > 0


def test_resume(tmp_path):
    run(tmp_path, "--epochs", "1")
    out = run(tmp_path, "--epochs", "2", "--resume")
    assert "resumed" in out
    assert [r["epoch"] for r in rows(tmp_path)] == ["1", "2"]


def test_key_options(tmp_path):
    out = run(tmp_path, "--scramble", "ele", "--adaptation", "none", "--keys", "seed", "--key-seed", "3",
              "--epochs", "1")
    assert "ELE(block_size=4, seed=3)" in out
    summary = json.load(open(tmp_path / "summary.json"))
    assert summary["args"]["keys"] == "seed" and summary["args"]["key_seed"] == 3
    assert "only used with --keys seed" in run(tmp_path, "--key-seed", "3", expect=2)
