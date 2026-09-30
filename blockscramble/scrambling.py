"""Block-wise image scrambling schemes compared in the paper (Table 2, Eqs. 1-3).

An image is split into ``B x B`` pixel blocks (``B = 4`` for 32x32 CIFAR images, i.e. ``N = 64``
blocks) and every block is transformed with a secret key:

``Plain``
    No scrambling (the "plain image" column of Table 3).
``LE``
    Learnable image encryption (Tanaka, ICCE-TW 2018). Every 8-bit value is split into its lower
    and upper 4 bits, giving ``2 * B * B * C`` 4-bit values per block; they are shuffled and partly
    negative-positive transformed (``v -> 15 - v``) with ONE key that is common to all blocks.
``ELE``
    Extended learnable encryption, the block-wise scrambling proposed in the paper: the LE pixel
    operation with a DIFFERENT key for every block, followed by block location shuffling.
``EtC``
    Encryption-then-compression (Chuman et al., IEEE TIFS 2018) as implemented in the original code:
    per-block rotation, flip, negative-positive transform and colour-channel operation, followed by
    block location shuffling.
``BlockShuffle``
    Block location shuffling alone (the second step of ELE and EtC).

By default every scheme uses the keys of the paper's experiments (see "Keys" below); pass
``seed=...`` for a new, deterministic key. Every scheme accepts

* a ``PIL.Image`` (returns a ``PIL.Image``),
* a numpy array in channels-last layout, ``(H, W)``, ``(H, W, C)`` or ``(N, H, W, C)``, either
  ``uint8`` in ``[0, 255]`` or float in ``[0, 1]`` (returns the same layout and dtype),
* a ``torch.Tensor`` ``(C, H, W)`` or ``(N, C, H, W)``, float in ``[0, 1]`` (e.g. the output of
  ``torchvision.transforms.ToTensor``) or ``uint8`` (returns the same shape, dtype and device),

so a scheme can be used directly inside ``torchvision.transforms.Compose``. ``inverse`` undoes the
scrambling for every scheme except the EtC variant used in the paper (see :class:`EtC`).

Faithfulness notes (the new code follows the original code where the two differ):

* LE/ELE work on 8-bit values. Float inputs are quantised with ``round(255 * x)``; for 8-bit images
  (``ToTensor`` outputs ``k / 255``) this is identical to the original ``(x * 255).astype(uint8)``.
* The original ELE/EtC code applies the block-wise pixel operation first (the key of a block is
  indexed by its *source* position) and shuffles block locations afterwards; Fig. 2 of the paper
  draws the two steps in the opposite order. Both orders are the same scheme up to a relabelling of
  the per-block keys; we follow the code.
* Keys. The original scripts read the LE/ELE pixel keys from the 64 files ``key4/0_.pkl`` ...
  ``key4/63_.pkl``: LE uses ``key4/0_.pkl`` for every block, ELE gives the block in grid row ``r``,
  column ``c`` the key ``key4/<8r+c>_.pkl``. These keys ship with the package
  (``resources/key4.npz``, see :func:`paper_keys`) and are the default (``key="paper"``). The EtC
  parameters and the ELE block permutation were drawn after ``random.seed(30)``; ``random.Random``
  with the same calls reproduces them exactly (the default ``seed=None`` means 30). With
  ``seed=s`` (``key="seed"`` for LE/ELE) new keys are drawn in the same format: the pixel keys from
  ``numpy.random.RandomState(s)`` block by block (so ``ELE`` block 0 uses the ``LE`` key) and the
  block permutation / EtC parameters from ``random.Random(s)``.
"""
from __future__ import annotations

import functools
import os
import random
from typing import Callable, Dict, Optional, Tuple

import numpy as np
import torch
from PIL import Image

__all__ = [
    "Scrambler",
    "Plain",
    "BlockShuffle",
    "LE",
    "ELE",
    "EtC",
    "SCRAMBLERS",
    "get_scrambler",
    "paper_keys",
    "PAPER_SEED",
]

#: ``random.seed(30)`` of the original code: EtC parameters and ELE block permutation of the paper.
PAPER_SEED = 30
_RESOURCES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "resources")


@functools.lru_cache(maxsize=1)
def paper_keys() -> np.ndarray:
    """The 64 LE keys read by the original scripts, ``key4/0_.pkl ... key4/63_.pkl``; shape ``(64, 96)``.

    Row ``i`` is the key stored in ``key4/<i>_.pkl`` (4x4 RGB blocks: a permutation of ``range(96)``).
    ``Blockwise_scramble_LE.py`` (LE) uses row 0 for every block; ``Blockwise_scramble.py`` (ELE) uses
    row ``8 * r + c`` for the block in grid row ``r``, column ``c`` of a 32x32 image. The pickle files
    are kept in ``archive/key4/`` (the same files are committed in the author's SIA-GAN and
    psivt23_scramblemix repositories); ``tests/test_parity_original.py`` checks this array against them.
    """
    with np.load(os.path.join(_RESOURCES, "key4.npz")) as data:
        keys = data["keys"].astype(np.int64)
    keys.setflags(write=False)
    return keys


def _key_mode(key: Optional[str], seed: Optional[int]) -> str:
    """``key=None`` -> ``"paper"`` unless a seed is given."""
    if key is None:
        return "paper" if seed is None else "seed"
    if key not in ("paper", "seed"):
        raise ValueError("key must be 'paper' (the key files of the original code) or 'seed'")
    if key == "paper" and seed is not None:
        raise ValueError("seed is only used with key='seed'")
    return key

# EtC colour-channel operation, indexed by the per-block code 0..5: output channel c is taken from
# input channel TABLE[code][c].
#  * "original": what the original ``etc_encryption.channel_change`` computes. It assigns the
#    channels one after another *in place*, so later assignments read already-overwritten channels
#    and five of the six codes duplicate a channel (e.g. code 4 gives (B, B, B)). This is what
#    produced the EtC numbers of Table 3.
#  * "permute": the colour-component shuffling intended by the code / described in the paper, i.e.
#    the same six assignments performed simultaneously (a true permutation of R, G, B).
_ETC_CHANNEL_TABLES = {
    "original": ((0, 1, 2), (1, 1, 2), (2, 1, 2), (0, 2, 2), (2, 2, 2), (1, 2, 1)),
    "permute": ((0, 1, 2), (1, 0, 2), (2, 1, 0), (0, 2, 1), (2, 0, 1), (1, 2, 0)),
}


# --------------------------------------------------------------------------------------------
# block helpers (channels-last numpy arrays)
# --------------------------------------------------------------------------------------------
def _to_blocks(x: np.ndarray, block_size: int) -> np.ndarray:
    """``(N, H, W, C) -> (N, num_blocks, B, B, C)``, blocks in row-major order."""
    n, h, w, c = x.shape
    b = block_size
    x = x.reshape(n, h // b, b, w // b, b, c).transpose(0, 1, 3, 2, 4, 5)
    return x.reshape(n, (h // b) * (w // b), b, b, c)


def _from_blocks(blocks: np.ndarray, height: int, width: int) -> np.ndarray:
    """Inverse of :func:`_to_blocks`."""
    n, _, b, _, c = blocks.shape
    x = blocks.reshape(n, height // b, width // b, b, b, c).transpose(0, 1, 3, 2, 4, 5)
    return x.reshape(n, height, width, c)


def _nibble_scramble(blocks: np.ndarray, order: np.ndarray, rev: np.ndarray) -> np.ndarray:
    """LE pixel operation of Tanaka (2018) on flattened 8-bit blocks.

    ``blocks``: ``(N, num_blocks, D)`` uint8 with ``D = B * B * C`` (row, column, channel order).
    ``order``/``rev``: ``(2D,)`` (one key for all blocks) or ``(num_blocks, 2D)`` (one key per block).
    Mirrors ``BlockScramble.doScramble`` of the original ``learnable_encryption.py``: split into
    lower/upper nibbles, negative-positive transform the ``rev`` positions, gather with ``order``,
    negative-positive transform the ``rev`` positions again, merge the nibbles.
    """
    d = blocks.shape[-1]
    nib = np.concatenate([blocks & 0xF, blocks >> 4], axis=-1)
    nib = np.where(rev, 15 - nib, nib)
    nib = np.take_along_axis(nib, np.broadcast_to(order, nib.shape), axis=-1)
    nib = np.where(rev, 15 - nib, nib)
    return ((nib[..., d:] << 4) + nib[..., :d]).astype(np.uint8)


def _nibble_keys(seed: int, num_keys: int, length: int) -> np.ndarray:
    """``num_keys`` LE keys (permutations of ``range(length)``), drawn one after another."""
    rs = np.random.RandomState(seed)
    return np.stack([rs.permutation(length) for _ in range(num_keys)])


def _block_permutation(seed: int, num_blocks: int) -> np.ndarray:
    """Block location permutation of the original scripts: ``random.seed(seed); shuffle(range(N))``."""
    perm = list(range(num_blocks))
    random.Random(seed).shuffle(perm)
    return np.asarray(perm)


def _to_uint8(x: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(x * 255.0), 0, 255).astype(np.uint8)


# --------------------------------------------------------------------------------------------
# base class
# --------------------------------------------------------------------------------------------
class Scrambler:
    """Base class: type dispatch (PIL / numpy / torch) around ``_encrypt`` / ``_decrypt``.

    Subclasses implement ``_encrypt(x)`` and ``_decrypt(x)`` on channels-last numpy arrays of shape
    ``(N, H, W, C)`` whose height and width are multiples of ``block_size``.
    """

    #: whether :meth:`inverse` is defined
    invertible: bool = True
    #: whether the scheme works on 8-bit values (float inputs are quantised)
    requires_uint8: bool = False

    def __init__(self, block_size: int = 4, seed: Optional[int] = None):
        if int(block_size) < 1:
            raise ValueError("block_size must be a positive integer")
        self.block_size = int(block_size)
        self.seed = PAPER_SEED if seed is None else int(seed)  # None: the seed of the original code
        self._cache: Dict[Tuple, object] = {}

    # public API ------------------------------------------------------------------------------
    def __call__(self, img):
        """Scramble ``img`` (PIL image, numpy array or torch tensor; see the module docstring)."""
        return self._apply(img, self._encrypt)

    def inverse(self, img):
        """Undo :meth:`__call__` with the same key (exact for 8-bit data)."""
        if not self.invertible:
            raise NotImplementedError(f"{self!r} is not invertible")
        return self._apply(img, self._decrypt)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(block_size={self.block_size}, seed={self.seed})"

    # to be implemented by subclasses ---------------------------------------------------------
    def _encrypt(self, x: np.ndarray) -> np.ndarray:  # pragma: no cover - abstract
        raise NotImplementedError

    def _decrypt(self, x: np.ndarray) -> np.ndarray:  # pragma: no cover - abstract
        raise NotImplementedError

    # dispatch --------------------------------------------------------------------------------
    def _run(self, arr: np.ndarray, fn: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
        """Apply ``fn`` to an ``(N, H, W, C)`` array, handling dtypes."""
        _, h, w, _ = arr.shape
        if h % self.block_size or w % self.block_size:
            raise ValueError(
                f"image size {h}x{w} is not a multiple of the block size {self.block_size}"
            )
        if arr.dtype == np.uint8:
            return fn(arr)
        if np.issubdtype(arr.dtype, np.floating):
            if self.requires_uint8:
                out = fn(_to_uint8(arr)).astype(np.float32) / np.float32(255.0)
                return out.astype(arr.dtype, copy=False)
            return fn(arr)
        raise TypeError(f"unsupported dtype {arr.dtype}: use uint8 or float in [0, 1]")

    def _apply(self, img, fn):
        if isinstance(img, Image.Image):
            if img.mode not in ("L", "RGB"):
                img = img.convert("RGB")
            arr = np.asarray(img)
            if arr.ndim == 2:
                return Image.fromarray(self._run(arr[None, :, :, None], fn)[0, :, :, 0])
            return Image.fromarray(self._run(arr[None], fn)[0])
        if isinstance(img, torch.Tensor):
            if img.dim() not in (3, 4):
                raise ValueError("expected a tensor of shape (C, H, W) or (N, C, H, W)")
            t = img.detach().cpu()
            t = t if img.dim() == 4 else t.unsqueeze(0)
            out = self._run(t.permute(0, 2, 3, 1).numpy(), fn)
            res = torch.from_numpy(np.ascontiguousarray(out)).permute(0, 3, 1, 2).contiguous()
            res = res.to(device=img.device, dtype=img.dtype)
            return res if img.dim() == 4 else res[0]
        arr = np.asarray(img)
        if arr.ndim == 2:
            return self._run(arr[None, :, :, None], fn)[0, :, :, 0]
        if arr.ndim == 3:
            return self._run(arr[None], fn)[0]
        if arr.ndim == 4:
            return self._run(arr, fn)
        raise ValueError("expected an array of shape (H, W), (H, W, C) or (N, H, W, C)")


# --------------------------------------------------------------------------------------------
# schemes
# --------------------------------------------------------------------------------------------
class Plain(Scrambler):
    """No scrambling ("plain image" in Tables 2 and 3). Returns its input unchanged."""

    def _apply(self, img, fn):
        return img

    def _encrypt(self, x):
        return x

    def _decrypt(self, x):
        return x


class BlockShuffle(Scrambler):
    """Block location shuffling: output block ``i`` is input block ``permutation[i]``.

    The permutation is ``random.seed(seed); random.shuffle(list(range(N)))`` as in the original
    training scripts; the default ``seed=None`` (= 30) reproduces their ``_shf``
    (``Block_location_shuffle.py``).
    """

    def permutation(self, num_blocks: int) -> np.ndarray:
        key = ("perm", num_blocks)
        if key not in self._cache:
            self._cache[key] = _block_permutation(self.seed, num_blocks)
        return self._cache[key]

    def _encrypt(self, x):
        blocks = _to_blocks(x, self.block_size)
        return _from_blocks(blocks[:, self.permutation(blocks.shape[1])], x.shape[1], x.shape[2])

    def _decrypt(self, x):
        blocks = _to_blocks(x, self.block_size)
        inv = np.argsort(self.permutation(blocks.shape[1]))
        return _from_blocks(blocks[:, inv], x.shape[1], x.shape[2])


class _LEKeys(Scrambler):
    """Key handling shared by :class:`LE` and :class:`ELE`.

    ``key="paper"`` (the default when no seed is given) uses the key files of the original code
    (:func:`paper_keys`, 4x4 RGB blocks); ``key="seed"`` / ``seed=s`` draws new keys from ``s``.
    """

    requires_uint8 = True

    def __init__(self, block_size: int = 4, seed: Optional[int] = None, key: Optional[str] = None):
        super().__init__(block_size, seed)
        self.key = _key_mode(key, seed)
        if self.key == "paper" and self.block_size != 4:
            raise ValueError("key='paper' holds the keys of 4x4 blocks; use seed=... for other block sizes")

    def __repr__(self) -> str:
        key = "key='paper'" if self.key == "paper" else f"seed={self.seed}"
        return f"{type(self).__name__}(block_size={self.block_size}, {key})"

    def _keys(self, num_keys: int, channels: int) -> np.ndarray:
        """``num_keys`` pixel keys: rows 0.. of the paper keys, or drawn from the seed."""
        ck = ("keys", num_keys, channels)
        if ck not in self._cache:
            if self.key == "paper":
                if channels != 3 or num_keys > 64:
                    raise ValueError("key='paper' holds 64 keys for RGB images (32x32 for ELE); "
                                     "use seed=... for other images")
                self._cache[ck] = paper_keys()[:num_keys]
            else:
                self._cache[ck] = _nibble_keys(self.seed, num_keys, 2 * self.block_size**2 * channels)
        return self._cache[ck]


class LE(_LEKeys):
    """Learnable image encryption (Tanaka, ICCE-TW 2018): one key shared by all blocks.

    Key space ``(B^2 * 6)! * 2^(B^2 * 6)`` for RGB (Eq. 1, Table 2). Port of ``BlockScramble`` from
    ``learnable_encryption.py`` (mastnk/ICCE-TW2018) as used by ``Blockwise_scramble_LE.py``.
    ``LE()`` uses the paper's key ``key4/0_.pkl``; ``LE(seed=s)`` a key drawn from ``s``.
    """

    def pixel_key(self, channels: int = 3) -> np.ndarray:
        """The key: a permutation of ``range(2 * B * B * channels)``."""
        return self._keys(1, channels)[0]

    def _run_nibbles(self, x, inverse: bool):
        n, h, w, c = x.shape
        key = self.pixel_key(c)
        rev = key > key.size / 2  # as in BlockScramble.setKey
        order = np.argsort(key) if inverse else key
        blocks = _to_blocks(x, self.block_size)
        flat = blocks.reshape(n, blocks.shape[1], -1)
        out = _nibble_scramble(flat, order, rev).reshape(blocks.shape)
        return _from_blocks(out, h, w)

    def _encrypt(self, x):
        return self._run_nibbles(x, inverse=False)

    def _decrypt(self, x):
        return self._run_nibbles(x, inverse=True)


class ELE(_LEKeys):
    """Extended learnable encryption (ELE), the block-wise scrambling proposed in the paper.

    Block-wise pixel shuffling + negative-positive transform with a different key for every block
    (the LE operation of :class:`LE`), then block location shuffling (Fig. 2, Table 2, Eq. 3; key
    space ``{(B^2 * 6)! * 2^(B^2 * 6)}^N * N!``). Port of ``Blockwise_scramble.py`` followed by
    ``Block_location_shuffle.py``. ``ELE()`` uses the paper's keys (``key4/0..63_.pkl`` for the 64
    blocks of a 32x32 image, block order of ``random.seed(30)``); ``ELE(seed=s)`` keys drawn from ``s``.
    """

    def pixel_keys(self, num_blocks: int, channels: int = 3) -> np.ndarray:
        """Per-block keys, shape ``(num_blocks, 2 * B * B * channels)``; row ``k`` is for block ``k``
        (row-major source position). Row 0 equals the key of :class:`LE` with the same settings."""
        if self.key == "paper" and num_blocks != 64:
            raise ValueError("key='paper' holds the 64 per-block keys of 32x32 images; use seed=...")
        return self._keys(num_blocks, channels)

    def permutation(self, num_blocks: int) -> np.ndarray:
        """Block location permutation (``random.seed(30)`` of the original scripts by default)."""
        ck = ("perm", num_blocks)
        if ck not in self._cache:
            self._cache[ck] = _block_permutation(self.seed, num_blocks)
        return self._cache[ck]

    def _encrypt(self, x):
        n, h, w, c = x.shape
        blocks = _to_blocks(x, self.block_size)
        nb = blocks.shape[1]
        keys = self.pixel_keys(nb, c)
        flat = _nibble_scramble(blocks.reshape(n, nb, -1), keys, keys > keys.shape[1] / 2)
        flat = flat[:, self.permutation(nb)]
        return _from_blocks(flat.reshape(blocks.shape), h, w)

    def _decrypt(self, x):
        n, h, w, c = x.shape
        blocks = _to_blocks(x, self.block_size)
        nb = blocks.shape[1]
        keys = self.pixel_keys(nb, c)
        flat = blocks.reshape(n, nb, -1)[:, np.argsort(self.permutation(nb))]
        flat = _nibble_scramble(flat, np.argsort(keys, axis=1), keys > keys.shape[1] / 2)
        return _from_blocks(flat.reshape(blocks.shape), h, w)


class EtC(Scrambler):
    """Encryption-then-compression (Chuman et al., 2018) as implemented in the original code.

    For every block (with its own parameters): rotation by 90/180/270/0 degrees, negative-positive
    transform (half of the blocks), vertical/horizontal/no flip, colour-channel operation; then
    block location shuffling (Table 2, Eq. 2). Port of ``etc_encryption.py``: the default
    ``seed=None`` (= 30) reproduces its parameters exactly; ``seed=s`` draws new ones.

    Args:
        channel_shuffle: ``"original"`` (default) reproduces the original code, whose in-place channel
            assignment duplicates a colour channel in five of the six cases; this variant produced
            the EtC column of Table 3 but loses information, so it has no inverse. ``"permute"``
            applies a true permutation of the colour channels (the colour component shuffling
            described in the paper) and is invertible.
    """

    def __init__(self, block_size: int = 4, seed: Optional[int] = None, channel_shuffle: str = "original"):
        super().__init__(block_size, seed)
        if channel_shuffle not in _ETC_CHANNEL_TABLES:
            raise ValueError(f"channel_shuffle must be one of {sorted(_ETC_CHANNEL_TABLES)}")
        self.channel_shuffle = channel_shuffle
        self.invertible = channel_shuffle == "permute"

    def __repr__(self) -> str:
        return (
            f"EtC(block_size={self.block_size}, seed={self.seed}, "
            f"channel_shuffle={self.channel_shuffle!r})"
        )

    def params(self, num_blocks: int) -> Dict[str, np.ndarray]:
        """Per-block parameters, generated with the exact call sequence of ``etc_encryption.py``.

        ``rotate``: 0/1/2 = rot90 by k=1/2/3, 3 = none; ``negaposi``: 0 = invert (``255 - v``),
        1 = keep; ``flip``: 0 = up-down, 1 = left-right, 2 = none; ``channel``: code 0..5 (see
        ``channel_shuffle``); ``permutation``: output block i is block ``permutation[i]``.
        """
        ck = ("params", num_blocks)
        if ck not in self._cache:
            rng = random.Random(self.seed)
            rotate, flip, channel = [], [], []
            negaposi = [1 if i % 2 == 0 else 0 for i in range(num_blocks)]
            for _ in range(num_blocks):
                rotate.append(rng.randint(0, 3))
                flip.append(rng.randint(0, 2))
                channel.append(rng.randint(0, 5))
            perm = list(range(num_blocks))
            rng.shuffle(perm)
            rng.shuffle(negaposi)
            self._cache[ck] = {
                "rotate": np.asarray(rotate),
                "negaposi": np.asarray(negaposi),
                "flip": np.asarray(flip),
                "channel": np.asarray(channel),
                "permutation": np.asarray(perm),
            }
        return self._cache[ck]

    def _encrypt(self, x):
        n, h, w, c = x.shape
        if c != 3:
            raise ValueError("EtC needs RGB images (3 channels)")
        vmax = 255 if x.dtype == np.uint8 else 1.0
        out = _to_blocks(x, self.block_size).copy()
        p = self.params(out.shape[1])
        for code in (0, 1, 2):  # rotation
            idx = np.flatnonzero(p["rotate"] == code)
            out[:, idx] = np.rot90(out[:, idx], k=code + 1, axes=(2, 3))
        idx = np.flatnonzero(p["negaposi"] == 0)  # negative-positive transform
        out[:, idx] = vmax - out[:, idx]
        for code, axis in ((0, 2), (1, 3)):  # flip
            idx = np.flatnonzero(p["flip"] == code)
            out[:, idx] = np.flip(out[:, idx], axis=axis)
        table = np.asarray(_ETC_CHANNEL_TABLES[self.channel_shuffle])[p["channel"]]
        out = np.take_along_axis(out, table[None, :, None, None, :], axis=4)
        return _from_blocks(out[:, p["permutation"]], h, w)

    def _decrypt(self, x):
        n, h, w, c = x.shape
        if c != 3:
            raise ValueError("EtC needs RGB images (3 channels)")
        vmax = 255 if x.dtype == np.uint8 else 1.0
        blocks = _to_blocks(x, self.block_size)
        p = self.params(blocks.shape[1])
        out = blocks[:, np.argsort(p["permutation"])]
        table = np.asarray(_ETC_CHANNEL_TABLES[self.channel_shuffle])[p["channel"]]
        out = np.take_along_axis(out, np.argsort(table, axis=1)[None, :, None, None, :], axis=4)
        for code, axis in ((0, 2), (1, 3)):
            idx = np.flatnonzero(p["flip"] == code)
            out[:, idx] = np.flip(out[:, idx], axis=axis)
        idx = np.flatnonzero(p["negaposi"] == 0)
        out[:, idx] = vmax - out[:, idx]
        for code in (0, 1, 2):
            idx = np.flatnonzero(p["rotate"] == code)
            out[:, idx] = np.rot90(out[:, idx], k=-(code + 1), axes=(2, 3))
        return _from_blocks(out, h, w)


#: CLI name -> scheme class (``train.py --scramble``)
SCRAMBLERS = {"plain": Plain, "le": LE, "ele": ELE, "etc": EtC}


def get_scrambler(name: str, block_size: int = 4, seed: Optional[int] = None, **kwargs) -> Scrambler:
    """Build a scheme by name: ``"plain"``, ``"le"``, ``"ele"`` or ``"etc"`` (case-insensitive).

    ``seed=None`` gives the keys of the paper's experiments; extra keyword arguments go to the class
    (``key=`` for LE/ELE, ``channel_shuffle=`` for EtC).
    """
    try:
        cls = SCRAMBLERS[name.lower()]
    except KeyError:
        raise ValueError(f"unknown scheme {name!r}; choose from {sorted(SCRAMBLERS)}") from None
    return cls(block_size=block_size, seed=seed, **kwargs)
