"""The package reproduces the original research code kept in archive/ (same inputs -> same outputs)."""
import os
import pickle
import random
import sys
import warnings

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.nn.utils import parametrize

import blockscramble as bs

ARCHIVE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "archive")
if not os.path.isdir(ARCHIVE):  # pragma: no cover
    pytest.skip("archive/ not available", allow_module_level=True)
sys.path.insert(0, ARCHIVE)
sys.dont_write_bytecode = True  # keep archive/ free of __pycache__


@pytest.fixture
def images():  # ToTensor-like batch (N, C, H, W) of 8-bit values in [0, 1]
    return (np.random.RandomState(0).randint(0, 256, size=(5, 3, 32, 32)) / 255.0).astype(np.float32)


def test_etc(images):
    import etc_encryption as orig  # module-level random.seed(30), as when the scripts import it

    ref = orig.EtC_encryption(images.copy()).numpy()
    ours = bs.EtC()(torch.from_numpy(images)).numpy()
    assert np.allclose(ours, ref, atol=1e-6)
    p = bs.EtC().params(64)
    assert list(p["rotate"]) == orig._rotate and list(p["flip"]) == orig._reverse
    assert list(p["channel"]) == orig._channel and list(p["negaposi"]) == orig._negaposi
    assert list(p["permutation"]) == orig._shf


def _original_le_ele(images, shuffle_seed=30):
    """LE and ELE exactly as the original scripts compute them (keys read from ./key4/<i>_.pkl)."""
    import Blockwise_scramble
    import Blockwise_scramble_LE
    from Block_location_shuffle import block_location_shuffle

    le = np.transpose(Blockwise_scramble_LE.blockwise_scramble(images.copy()), (0, 3, 1, 2))
    random.seed(shuffle_seed)  # the training scripts call random.seed(30)
    shf = list(range(64))
    random.shuffle(shf)
    ele = block_location_shuffle(shf, np.transpose(Blockwise_scramble.blockwise_scramble(images.copy()), (0, 3, 1, 2)))
    return le, ele


def test_paper_keys_are_the_original_key_files():
    from learnable_encryption import BlockScramble

    for i in range(64):  # loaded with the original loader
        orig = BlockScramble(os.path.join(ARCHIVE, "key4", f"{i}_.pkl"))
        assert list(orig.blockSize) == [4, 4, 3]
        assert np.array_equal(orig.key.astype(np.int64), bs.paper_keys()[i])


def test_le_and_ele_with_the_original_key_files(images, monkeypatch):
    from learnable_encryption import BlockScramble

    monkeypatch.chdir(ARCHIVE)  # the original code reads key4/<i>_.pkl relative to the working directory
    ref_le, ref_ele = _original_le_ele(images)
    x = torch.from_numpy(images)
    assert np.array_equal(bs.LE()(x).numpy(), ref_le)    # default = the paper's keys
    assert np.array_equal(bs.ELE()(x).numpy(), ref_ele)
    orig = BlockScramble(os.path.join("key4", "0_.pkl"))
    ref_dec = np.transpose(orig.Decramble(np.transpose(ref_le, (0, 2, 3, 1))), (0, 3, 1, 2))
    our_dec = bs.LE().inverse(torch.from_numpy(ref_le)).numpy()
    assert np.array_equal(our_dec, ref_dec) and np.array_equal(our_dec, images)
    assert np.array_equal(bs.ELE().inverse(torch.from_numpy(ref_ele)).numpy(), images)


def test_seeded_keys_in_the_original_format(images, tmp_path, monkeypatch):
    ele = bs.ELE(seed=7)
    os.makedirs(tmp_path / "key4")
    for i, key in enumerate(ele.pixel_keys(64, 3)):  # same file format as BlockScramble.save
        with open(tmp_path / "key4" / f"{i}_.pkl", "wb") as f:
            pickle.dump([[4, 4, 3], key.astype(np.uint32)], f)
    monkeypatch.chdir(tmp_path)
    ref_le, ref_ele = _original_le_ele(images, shuffle_seed=7)
    x = torch.from_numpy(images)
    assert np.array_equal(bs.LE(seed=7)(x).numpy(), ref_le)
    assert np.array_equal(ele(x).numpy(), ref_ele)


def test_regularisers():
    from util_norm import total_variation_norm

    torch.manual_seed(0)
    feature = torch.randn(6, 16, 32, 32)
    assert torch.allclose(bs.smoothness_penalty(feature), total_variation_norm(feature) / feature.size(0))
    mat = torch.randn(64, 64)
    dsc = 0  # loop of proposed_*.py ("doubly stochastic constraint")
    for i in range(64):
        dsc += torch.abs(mat[i, :]).sum() - torch.sqrt((mat[i, :] * mat[i, :]).sum())
        dsc += torch.abs(mat[:, i]).sum() - torch.sqrt((mat[:, i] * mat[:, i]).sum())
    assert torch.allclose(bs.l12_penalty(mat), dsc / (64 * 64))


def test_lr_schedule():
    opt = torch.optim.SGD([torch.zeros(1, requires_grad=True)], lr=0.1, momentum=0.9)
    sched = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[150, 225])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for epoch in range(305):  # the original loop steps the scheduler at the start of each epoch
            sched.step()
            assert opt.param_groups[0]["lr"] == pytest.approx(bs.original_lr(epoch + 1))


# ----------------------------------------------------------------------------------- networks
class _Stop(Exception):
    pass


class _Capture(nn.Module):
    def forward(self, x):
        self.x = x
        raise _Stop


def _randomize_bn(module):
    for m in module.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.weight.data.uniform_(0.5, 1.5)
            m.bias.data.uniform_(-0.2, 0.2)
            m.running_mean.uniform_(-0.1, 0.1)
            m.running_var.uniform_(0.5, 1.5)


def _backbone_state(orig):
    keep = ("c_in", "bn_in", "layer1", "layer2", "layer3", "bn_out", "fc_out")
    return {k: v for k, v in orig.state_dict().items() if k.split(".")[0] in keep}


def _adaptation_feature(orig, x):
    """Feature map that the original monolithic network feeds to its classification part."""
    orig.c_in = _Capture()
    with pytest.raises(_Stop):
        orig(x)
    return orig.c_in.x


@pytest.fixture
def cpu_original(monkeypatch):
    monkeypatch.setattr(torch.Tensor, "cuda", lambda self, *a, **k: self)  # original hard-codes .cuda()
    torch.manual_seed(0)


def test_backbone_matches_original(cpu_original):
    import no_adaptation_network as orig_mod

    orig = orig_mod.ShakePyramidNet(depth=20, alpha=12, label=10)
    _randomize_bn(orig)
    ours = bs.build_model("none", num_classes=10, depth=20, alpha=12)
    ours.backbone.load_state_dict(_backbone_state(orig), strict=True)  # same parameter names
    x = torch.rand(6, 3, 32, 32)
    orig.eval()
    ours.eval()
    with torch.no_grad():
        assert torch.allclose(ours(x), orig(x), atol=1e-4)


def test_le_adaptnet_matches_original(cpu_original):
    import tanaka_adaptation_network as orig_mod

    orig = orig_mod.ShakePyramidNet(depth=20, alpha=12, label=10)
    _randomize_bn(orig)
    ours = bs.build_model("tanaka", num_classes=10, depth=20, alpha=12)
    ours.backbone.load_state_dict(_backbone_state(orig), strict=True)
    ours.adaptation.conv.weight.data.copy_(orig.conv0.weight.data)
    ours.adaptation.bn.load_state_dict(orig.bn0.state_dict())
    x = torch.rand(6, 3, 32, 32)
    orig.eval()
    ours.eval()
    with torch.no_grad():
        assert torch.allclose(ours(x), orig(x), atol=1e-4)
    orig.train()  # per-block BN batch statistics and 64 running-stat updates
    ours.train()
    assert torch.equal(ours.adaptation(x), _adaptation_feature(orig, x))
    assert torch.allclose(ours.adaptation.bn.running_mean, orig.bn0.running_mean)
    assert torch.allclose(ours.adaptation.bn.running_var, orig.bn0.running_var)


@pytest.mark.skipif(not hasattr(torch.nn.utils, "spectral_norm"), reason="legacy spectral_norm API removed")
def test_ele_adaptnet_matches_original(cpu_original):
    import proposed_adaptation_network as orig_mod

    orig = orig_mod.ShakePyramidNet(depth=20, alpha=12, label=10)
    _randomize_bn(orig)
    ours = bs.build_model("proposed", num_classes=10, depth=20, alpha=12)
    ours.backbone.load_state_dict(_backbone_state(orig), strict=True)
    adapt = ours.adaptation.eval()
    for k in range(64):  # same spectrally normalised weights on both sides
        parametrize.remove_parametrizations(adapt.convs[k], "weight", leave_parametrized=True)
        torch.nn.utils.remove_spectral_norm(orig.convs0[k])
        orig.convs0[k].weight.data.copy_(adapt.convs[k].weight.data)
        adapt.bns[k].load_state_dict(orig.bns0[k].state_dict())
    orig.matrix.weight.data.copy_(adapt.permutation.data)
    x = torch.rand(6, 3, 32, 32)
    orig.eval()
    ours.eval()
    with torch.no_grad():
        logits, mat, feature = orig(x)
        our_logits, our_feature = ours(x, return_feature=True)
    assert torch.allclose(our_logits, logits, atol=1e-4) and torch.equal(our_feature, feature)
    assert torch.equal(mat, adapt.permutation)  # the matrix the original penalises is U itself
    orig.train()
    ours.train()
    assert torch.equal(adapt(x), _adaptation_feature(orig, x))
    assert all(torch.allclose(adapt.bns[k].running_mean, orig.bns0[k].running_mean) for k in range(64))
