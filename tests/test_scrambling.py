import numpy as np
import pytest
import torch
import torchvision.transforms as T
from PIL import Image

import blockscramble as bs

INVERTIBLE = [bs.Plain(), bs.BlockShuffle(), bs.LE(), bs.ELE(), bs.EtC(channel_shuffle="permute")]
ALL = INVERTIBLE + [bs.EtC()]
KEYED = [bs.BlockShuffle, bs.LE, bs.ELE, bs.EtC]


@pytest.fixture
def batch_u8():
    return np.random.RandomState(0).randint(0, 256, size=(4, 32, 32, 3)).astype(np.uint8)


@pytest.mark.parametrize("s", ALL, ids=repr)
def test_shapes_and_dtypes(s, batch_u8):
    assert s(batch_u8).shape == batch_u8.shape and s(batch_u8).dtype == np.uint8
    assert s(batch_u8[0]).shape == (32, 32, 3)
    f = batch_u8.astype(np.float32) / 255
    assert s(f).dtype == np.float32 and s(f).shape == f.shape
    t = torch.from_numpy(f).permute(0, 3, 1, 2)
    assert s(t).shape == (4, 3, 32, 32) and s(t).dtype == torch.float32
    assert s(t[0]).shape == (3, 32, 32)
    assert s(t.to(torch.float64)).dtype == torch.float64
    out = s(Image.fromarray(batch_u8[0]))
    assert isinstance(out, Image.Image) and out.size == (32, 32) and out.mode == "RGB"


@pytest.mark.parametrize("s", ALL, ids=repr)
def test_same_result_for_every_input_type(s, batch_u8):
    ref = s(batch_u8)
    t = torch.from_numpy(batch_u8).permute(0, 3, 1, 2)
    assert np.array_equal(s(t).permute(0, 2, 3, 1).numpy(), ref)  # uint8 tensor
    f = s(t.float() / 255).permute(0, 2, 3, 1).numpy()           # ToTensor-style float tensor
    assert np.allclose(f * 255, ref, atol=1e-3)
    assert np.array_equal(np.asarray(s(Image.fromarray(batch_u8[1]))), ref[1])


@pytest.mark.parametrize("s", INVERTIBLE, ids=repr)
def test_inverse(s, batch_u8):
    assert np.array_equal(s.inverse(s(batch_u8)), batch_u8)
    t = torch.from_numpy(batch_u8).permute(0, 3, 1, 2).float() / 255
    assert torch.allclose(s.inverse(s(t)), t, atol=1e-6)


def test_etc_as_in_paper_is_not_invertible(batch_u8):
    with pytest.raises(NotImplementedError):
        bs.EtC().inverse(batch_u8)
    assert not bs.EtC().invertible and bs.EtC(channel_shuffle="permute").invertible
    # the original colour operation duplicates a channel in every block whose code is not 0
    etc = bs.EtC()
    out = etc(batch_u8).reshape(4, 8, 4, 8, 4, 3).transpose(1, 3, 0, 2, 4, 5).reshape(64, -1, 3)
    p = etc.params(64)
    has_dup = [any((b[:, i] == b[:, j]).all() for i, j in ((0, 1), (0, 2), (1, 2))) for b in out]
    assert has_dup == list(p["channel"][p["permutation"]] != 0)


@pytest.mark.parametrize("cls", KEYED)
def test_deterministic_given_seed(cls, batch_u8):
    a, b, c = cls(seed=1), cls(seed=1), cls(seed=2)
    assert np.array_equal(a(batch_u8), b(batch_u8))
    assert not np.array_equal(a(batch_u8), c(batch_u8))
    assert not np.array_equal(a(batch_u8), batch_u8)


def test_input_not_modified(batch_u8):
    ref = batch_u8.copy()
    for s in ALL:
        s(batch_u8)
    assert np.array_equal(batch_u8, ref)


def test_le_uses_one_key_ele_uses_one_key_per_block():
    tile = np.random.RandomState(1).randint(0, 256, size=(4, 4, 3)).astype(np.uint8)
    img = np.tile(tile, (8, 8, 1))  # 64 identical blocks
    blocks = lambda x: x.reshape(8, 4, 8, 4, 3).transpose(0, 2, 1, 3, 4).reshape(64, -1)
    le, ele = blocks(bs.LE()(img)), blocks(bs.ELE()(img))
    assert (le == le[0]).all()                          # common key -> identical scrambled blocks
    assert len({row.tobytes() for row in ele}) == 64    # different keys -> all blocks differ


def test_keys_and_permutations():
    le, ele = bs.LE(seed=5), bs.ELE(seed=5)
    assert sorted(le.pixel_key(3)) == list(range(96))           # 2 * 4 * 4 * 3 4-bit values per block
    assert ele.pixel_keys(64, 3).shape == (64, 96)
    assert np.array_equal(ele.pixel_keys(64, 3)[0], le.pixel_key(3))  # block 0 of ELE uses the LE key
    # the default block order is `_shf` of the original ELE scripts (random.seed(30))
    assert list(bs.ELE().permutation(64)[:12]) == [14, 43, 53, 26, 12, 57, 28, 32, 18, 36, 31, 48]
    assert np.array_equal(bs.BlockShuffle().permutation(64), bs.ELE().permutation(64))
    assert np.array_equal(bs.ELE(seed=30).permutation(64), bs.ELE().permutation(64))
    p = bs.EtC().params(64)
    assert sorted(p["permutation"]) == list(range(64)) and p["negaposi"].sum() == 32


def test_paper_keys_are_the_default():
    keys = bs.paper_keys()
    assert keys.shape == (64, 96) and all(sorted(k) == list(range(96)) for k in keys)
    assert len({k.tobytes() for k in keys}) == 64
    assert list(keys[0, :6]) == [47, 24, 42, 54, 68, 26]         # key4/0_.pkl
    assert np.array_equal(bs.LE().pixel_key(), keys[0]) and np.array_equal(bs.ELE().pixel_keys(64), keys)
    assert bs.LE().key == bs.ELE(key="paper").key == "paper" and bs.LE(seed=0).key == "seed"
    img = np.random.RandomState(5).randint(0, 256, size=(32, 32, 3)).astype(np.uint8)
    assert not np.array_equal(bs.LE()(img), bs.LE(seed=30)(img))  # seeded keys are new keys
    for bad in (lambda: bs.LE(key="paper", seed=1), lambda: bs.LE(block_size=8), lambda: bs.ELE(key="x"),
                lambda: bs.ELE()(np.zeros((48, 48, 3), np.uint8)), lambda: bs.LE()(img[..., 0])):
        with pytest.raises(ValueError):
            bad()


def test_block_shuffle_moves_whole_blocks(batch_u8):
    s = bs.BlockShuffle(seed=3)
    out, perm = s(batch_u8), s.permutation(64)
    for i in (0, 17, 63):
        r, c, pr, pc = i // 8, i % 8, perm[i] // 8, perm[i] % 8
        assert np.array_equal(out[:, 4 * r:4 * r + 4, 4 * c:4 * c + 4], batch_u8[:, 4 * pr:4 * pr + 4, 4 * pc:4 * pc + 4])


def test_other_block_and_image_sizes():
    img = np.random.RandomState(2).randint(0, 256, size=(48, 64, 3)).astype(np.uint8)
    for s in [bs.LE(block_size=8, seed=0), bs.ELE(block_size=16, seed=0),
              bs.EtC(block_size=8, channel_shuffle="permute")]:
        assert np.array_equal(s.inverse(s(img)), img)
    gray = img[..., 0]
    assert np.array_equal(bs.ELE(seed=0).inverse(bs.ELE(seed=0)(gray)), gray)
    with pytest.raises(ValueError):
        bs.ELE()(np.zeros((30, 32, 3), np.uint8))  # 30 is not a multiple of 4


def test_torchvision_transform():
    tf = T.Compose([T.RandomCrop(32, padding=4), T.RandomHorizontalFlip(), T.ToTensor(), bs.ELE()])
    img = Image.fromarray(np.random.RandomState(4).randint(0, 256, size=(32, 32, 3)).astype(np.uint8))
    x = tf(img)
    assert x.shape == (3, 32, 32) and x.dtype == torch.float32 and 0 <= x.min() and x.max() <= 1
    assert torch.equal(x * 255, (x * 255).round())  # 8-bit values


def test_get_scrambler():
    assert isinstance(bs.get_scrambler("ELE"), bs.ELE)
    assert bs.get_scrambler("etc", channel_shuffle="permute").invertible
    with pytest.raises(ValueError):
        bs.get_scrambler("foo")
