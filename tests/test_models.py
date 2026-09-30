import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

import blockscramble as bs

SMALL = dict(depth=20, alpha=12)  # tiny Shake-PyramidNet for fast tests (paper: depth 110, alpha 270)


@pytest.mark.parametrize("adaptation,channels", [("none", 3), ("tanaka", 16), ("proposed", 16)])
def test_shapes(adaptation, channels):
    model = bs.build_model(adaptation, num_classes=7, **SMALL)
    logits, feature = model(torch.rand(4, 3, 32, 32), return_feature=True)
    assert logits.shape == (4, 7) and feature.shape == (4, channels, 32, 32)
    model.eval()
    assert model(torch.rand(2, 3, 32, 32)).shape == (2, 7)


def test_default_model_is_the_papers():
    model = bs.build_model("proposed", num_classes=10)
    assert isinstance(model.adaptation, bs.ELEAdaptNet) and model.adaptation.num_blocks == 64
    assert model.adaptation.permutation.shape == (64, 64)
    assert len(model.backbone.layer1) == len(model.backbone.layer3) == 18  # (110 - 2) / 6
    assert model.backbone.in_chs[-1] == 16 + 270
    assert model.backbone.ps_shakedrop[-1] == pytest.approx(0.5)


def test_regularization():
    x = torch.rand(4, 3, 32, 32)
    for name in ("none", "tanaka"):
        model = bs.build_model(name, **SMALL)
        _, feature = model(x, return_feature=True)
        assert model.regularization(feature).item() == 0
    model = bs.build_model("proposed", lambda_u=0.5, lambda_s=2.0, **SMALL)
    _, feature = model(x, return_feature=True)
    expected = 0.5 * bs.l12_penalty(model.adaptation.permutation) + 2.0 * bs.smoothness_penalty(feature)
    assert torch.allclose(model.regularization(feature), expected)


def test_l12_penalty():
    perm = torch.eye(8)[torch.randperm(8)]
    assert bs.l12_penalty(perm).item() == pytest.approx(0, abs=1e-6)
    assert bs.l12_penalty(-2 * perm).item() == pytest.approx(0, abs=1e-6)  # one non-zero per row/col
    assert bs.l12_penalty(torch.full((8, 8), 1 / 8)).item() > 0


def test_smoothness_penalty():
    assert bs.smoothness_penalty(torch.ones(2, 16, 32, 32)).item() == 0
    x = torch.zeros(1, 16, 32, 32)
    x[0, 5:] = torch.rand(11, 32, 32)  # the original penalises only channels 0-2
    assert bs.smoothness_penalty(x).item() == 0
    x[0, 0, 10, 10] = 1.0  # a unit bump enters 4 squared forward differences; / (31 * 32 * 16)
    assert bs.smoothness_penalty(x).item() == pytest.approx(4 / (31 * 32 * 16))


def test_one_optimisation_step():
    torch.manual_seed(0)
    model = bs.build_model("proposed", **SMALL)
    opt = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, nesterov=True, weight_decay=5e-4)
    x, y = torch.rand(8, 3, 32, 32), torch.randint(0, 10, (8,))
    u0 = model.adaptation.permutation.detach().clone()
    logits, feature = model(x, return_feature=True)
    loss = F.cross_entropy(logits, y) + model.regularization(feature)
    loss.backward()
    assert torch.isfinite(loss) and model.adaptation.permutation.grad.abs().sum() > 0
    assert all(p.grad is not None for p in model.parameters())
    opt.step()
    assert not torch.equal(model.adaptation.permutation, u0)


def test_determinism():
    torch.manual_seed(1)
    a = bs.build_model("proposed", **SMALL).state_dict()
    torch.manual_seed(1)
    b = bs.build_model("proposed", **SMALL).state_dict()
    assert a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)
    model = bs.build_model("tanaka", **SMALL).eval()
    x = torch.rand(2, 3, 32, 32)
    assert torch.equal(model(x), model(x))


def test_shakedrop():
    x = torch.randn(16, 4, 3, 3, requires_grad=True)
    sd = bs.ShakeDrop(p_drop=0.3).eval()
    assert torch.allclose(sd(x), 0.7 * x)
    sd = bs.ShakeDrop(p_drop=0.0).train()  # gate always open: identity
    assert torch.equal(sd(x), x)
    sd = bs.ShakeDrop(p_drop=1.0).train()  # gate always closed: alpha * x, backward beta * grad
    y = sd(x)
    alpha = (y / x).flatten(1)
    assert torch.allclose(alpha, alpha[:, :1].expand_as(alpha), atol=1e-5) and alpha.abs().max() <= 1
    y.sum().backward()
    beta = x.grad.flatten(1)
    assert (beta >= 0).all() and (beta <= 1).all() and torch.allclose(beta, beta[:, :1].expand_as(beta))


def test_drop_in_in_front_of_any_backbone():
    torchvision = pytest.importorskip("torchvision")
    backbone = torchvision.models.resnet18(num_classes=10)
    backbone.conv1 = nn.Conv2d(16, 64, 3, 1, 1, bias=False)
    model = bs.ScrambledImageClassifier(backbone, bs.ELEAdaptNet())
    logits, feature = model(torch.rand(2, 3, 32, 32), return_feature=True)
    assert logits.shape == (2, 10) and model.regularization(feature).ndim == 0


def test_original_lr():
    lrs = {e: bs.original_lr(e) for e in (1, 149, 150, 224, 225, 305)}
    assert lrs[1] == lrs[149] == pytest.approx(0.1)
    assert lrs[150] == lrs[224] == pytest.approx(0.01)
    assert lrs[225] == lrs[305] == pytest.approx(0.001)
