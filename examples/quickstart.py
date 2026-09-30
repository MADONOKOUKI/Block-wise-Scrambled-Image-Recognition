"""Quick start: scramble an image with the four schemes of the paper, decrypt it with the key, and take
one training step of the proposed ELE-AdaptNet + Shake-PyramidNet-110 on block-wise scrambled images.

    pip install -e . matplotlib scikit-image
    python examples/quickstart.py          # writes assets/quickstart.png (a few seconds on a CPU)

No downloads: the cat image ships with scikit-image.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
import torchvision.transforms as T  # noqa: E402
from PIL import Image  # noqa: E402
from skimage import data  # noqa: E402

import blockscramble as bs  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "assets" / "quickstart.png"

# 1) A CIFAR-sized image: 32 x 32 pixels = 8 x 8 blocks of 4 x 4 pixels (B = 4, N = 64), as in the paper.
cat = Image.fromarray(data.chelsea()[:, 75:375]).resize((32, 32), Image.BICUBIC)
img = np.asarray(cat)  # (32, 32, 3) uint8

# 2) Scramble with every scheme of Table 2; the key is the seed (30 = the keys of the original code).
schemes = {
    "plain": bs.Plain(),
    "LE (Tanaka 2018)": bs.LE(seed=30),
    "ELE (proposed)": bs.ELE(seed=30),
    "EtC (Chuman et al. 2018)": bs.EtC(seed=30),
}
fig, axes = plt.subplots(2, 4, figsize=(10, 5.4))
for col, (name, scheme) in enumerate(schemes.items()):
    scrambled = scheme(img)
    axes[0, col].imshow(scrambled, interpolation="nearest")
    axes[0, col].set_title(name, fontsize=11)
    if scheme.invertible:
        recovered = scheme.inverse(scrambled)
        assert np.array_equal(recovered, img)
        axes[1, col].imshow(recovered, interpolation="nearest")
        print(f"{name:26s} scrambled -> recovered exactly with the key")
    else:  # the EtC of the original code overwrites colour channels (see bs.EtC)
        axes[1, col].text(0.5, 0.5, "no inverse:\nthe original EtC code\nduplicates colour channels\n\n"
                          "invertible variant:\nEtC(channel_shuffle=\n'permute')", ha="center", va="center",
                          fontsize=9, transform=axes[1, col].transAxes)
        print(f"{name:26s} scrambled (not invertible, see bs.EtC)")
axes[0, 0].set_ylabel("scrambled", fontsize=11)
axes[1, 0].set_ylabel("decrypted with the key", fontsize=11)
for ax in axes.flat:
    ax.set_xticks([])
    ax.set_yticks([])
fig.suptitle("Block-wise scrambling of a 32x32 image with 4x4 blocks (N = 64)", fontsize=12)
fig.tight_layout()
OUT.parent.mkdir(exist_ok=True)
fig.savefig(OUT, dpi=100)
print(f"saved {OUT}")

# 3) One training step of the proposed method: augmentation -> ELE scrambling -> ELE-AdaptNet -> classifier.
torch.manual_seed(0)
transform = T.Compose([T.RandomCrop(32, padding=4), T.RandomHorizontalFlip(), T.ToTensor(), bs.ELE(seed=30)])
x = torch.stack([transform(cat) for _ in range(8)])  # a mini-batch of scrambled views of the cat
y = torch.full((8,), 3)  # "cat" is class 3 in CIFAR-10
model = bs.build_model(adaptation="proposed", num_classes=10)  # ELE-AdaptNet + Shake-PyramidNet-110
optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4, nesterov=True)
logits, feature = model(x, return_feature=True)
ce, reg = F.cross_entropy(logits, y), model.regularization(feature)  # Eq. (5): L_CE + lambda_U L_U + lambda_s L_s
(ce + reg).backward()
optimizer.step()
n_params = sum(p.numel() for p in model.parameters()) / 1e6
print(f"model: ELE-AdaptNet + Shake-PyramidNet-110, {n_params:.2f}M parameters")
print(f"input {tuple(x.shape)} -> adaptation feature {tuple(feature.shape)} -> logits {tuple(logits.shape)}")
print(f"loss: cross-entropy {ce.item():.3f} + regularisation {reg.item():.4f}; one SGD step done")
