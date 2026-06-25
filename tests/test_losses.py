"""Tests for the loss modules (``artvid/losses/*``).

All tests depend on the ``torch`` fixture from ``conftest.py``, which skips the
test when torch is not installed — so this file is collectable (and the rest of
the suite runnable) in the torch-less scaffold environment.

Parity references point back to ``artistic_video_core.lua`` line numbers.
"""

from __future__ import annotations

import math

import pytest


# --------------------------------------------------------------------------- #
# Gram matrix (artistic_video_core.lua:351-360, 380)
# --------------------------------------------------------------------------- #
def test_gram_matrix_shape_unbatched(torch):
    from artvid.losses.style import gram_matrix

    feat = torch.randn(4, 5, 6)  # C, H, W
    g = gram_matrix(feat)
    assert g.shape == (4, 4)


def test_gram_matrix_shape_batched(torch):
    from artvid.losses.style import gram_matrix

    feat = torch.randn(2, 4, 5, 6)  # N, C, H, W
    g = gram_matrix(feat)
    assert g.shape == (2, 4, 4)


def test_gram_matrix_symmetric(torch):
    from artvid.losses.style import gram_matrix

    feat = torch.randn(3, 7, 8)
    g = gram_matrix(feat)
    assert torch.allclose(g, g.t(), atol=1e-5)


def test_gram_matrix_values_match_legacy_formula(torch):
    """Gram == (F·Fᵀ) / nElement, with nElement = C*H*W (legacy :380)."""
    from artvid.losses.style import gram_matrix

    c, h, w = 3, 4, 5
    feat = torch.randn(c, h, w)
    flat = feat.reshape(c, h * w)
    expected = (flat @ flat.t()) / feat.numel()
    g = gram_matrix(feat)
    assert torch.allclose(g, expected, atol=1e-6)


def test_gram_matrix_batched_matches_per_sample(torch):
    from artvid.losses.style import gram_matrix

    feat = torch.randn(2, 3, 4, 5)
    g = gram_matrix(feat)
    g0 = gram_matrix(feat[0])
    assert torch.allclose(g[0], g0, atol=1e-6)


def test_gram_matrix_rejects_bad_rank(torch):
    from artvid.losses.style import gram_matrix

    with pytest.raises(ValueError):
        gram_matrix(torch.randn(10))


# --------------------------------------------------------------------------- #
# ContentLoss (artistic_video_core.lua:257-288)
# --------------------------------------------------------------------------- #
def test_content_loss_zero_at_target(torch):
    from artvid.losses.content import ContentLoss

    target = torch.randn(3, 8, 8)
    loss = ContentLoss(target, strength=2.0)
    out = loss(target.clone())
    assert out.dim() == 0
    assert out.item() == pytest.approx(0.0, abs=1e-6)


def test_content_loss_matches_scaled_mse(torch):
    from artvid.losses.content import ContentLoss

    target = torch.randn(3, 8, 8)
    inp = torch.randn(3, 8, 8)
    strength = 3.0
    loss = ContentLoss(target, strength=strength)
    expected = torch.nn.functional.mse_loss(inp, target) * strength
    assert loss(inp).item() == pytest.approx(expected.item(), rel=1e-5)


def test_content_loss_backward_produces_grad(torch):
    from artvid.losses.content import ContentLoss

    target = torch.randn(3, 4, 4)
    inp = torch.randn(3, 4, 4, requires_grad=True)
    loss = ContentLoss(target, strength=1.0)
    loss(inp).backward()
    assert inp.grad is not None
    assert torch.isfinite(inp.grad).all()


def test_content_loss_normalize_l1_grad(torch):
    """With normalize=True the input grad has unit L1 norm (legacy :282-283)."""
    from artvid.losses.content import ContentLoss

    target = torch.zeros(3, 4, 4)
    inp = torch.randn(3, 4, 4, requires_grad=True)
    loss = ContentLoss(target, strength=1.0, normalize=True)
    loss(inp).backward()
    assert inp.grad.abs().sum().item() == pytest.approx(1.0, rel=1e-4)


def test_content_loss_shape_mismatch_returns_zero(torch):
    from artvid.losses.content import ContentLoss

    target = torch.randn(3, 8, 8)
    loss = ContentLoss(target)
    out = loss(torch.randn(3, 4, 4))
    assert out.item() == pytest.approx(0.0)


# --------------------------------------------------------------------------- #
# StyleLoss (artistic_video_core.lua:364-397)
# --------------------------------------------------------------------------- #
def test_style_loss_zero_at_target_features(torch):
    from artvid.losses.style import StyleLoss

    feat = torch.randn(3, 6, 6)
    loss = StyleLoss(feat, strength=1.0)  # target built from these features
    out = loss(feat.clone())
    assert out.dim() == 0
    assert out.item() == pytest.approx(0.0, abs=1e-6)


def test_style_loss_matches_gram_mse(torch):
    from artvid.losses.style import StyleLoss, gram_matrix

    target_feat = torch.randn(3, 6, 6)
    inp = torch.randn(3, 6, 6)
    strength = 4.0
    loss = StyleLoss(target_feat, strength=strength)
    expected = (
        torch.nn.functional.mse_loss(gram_matrix(inp), gram_matrix(target_feat))
        * strength
    )
    assert loss(inp).item() == pytest.approx(expected.item(), rel=1e-5)


def test_style_loss_accepts_precomputed_gram(torch):
    from artvid.losses.style import StyleLoss, gram_matrix

    target_feat = torch.randn(3, 6, 6)
    g = gram_matrix(target_feat)
    loss = StyleLoss(g, strength=1.0, target_is_gram=True)
    assert torch.allclose(loss.target, g)


def test_style_loss_backward_produces_grad(torch):
    from artvid.losses.style import StyleLoss

    target_feat = torch.randn(3, 5, 5)
    inp = torch.randn(3, 5, 5, requires_grad=True)
    loss = StyleLoss(target_feat, strength=1.0)
    loss(inp).backward()
    assert inp.grad is not None
    assert torch.isfinite(inp.grad).all()


# --------------------------------------------------------------------------- #
# TVLoss (artistic_video_core.lua:400-430)
# --------------------------------------------------------------------------- #
def test_tv_loss_zero_on_flat_image(torch):
    from artvid.losses.tv import TVLoss

    flat = torch.full((3, 8, 8), 0.5)
    loss = TVLoss(strength=1.0)
    assert loss(flat).item() == pytest.approx(0.0, abs=1e-6)


def test_tv_loss_positive_on_varying_image(torch):
    from artvid.losses.tv import TVLoss

    img = torch.randn(3, 8, 8)
    loss = TVLoss(strength=1.0)
    assert loss(img).item() > 0.0


def test_tv_loss_gradient_matches_legacy_stencil(torch):
    """Autograd grad of the L2 TV energy == the legacy finite-difference stencil.

    Reproduces ``TVLoss:updateGradInput`` (artistic_video_core.lua:415-429):
        grad[:, :-1, :-1] += x_diff + y_diff
        grad[:, :-1, 1:]  += -x_diff
        grad[:, 1:,  :-1] += -y_diff
    with x_diff = in[:, :-1, :-1] - in[:, :-1, 1:],
         y_diff = in[:, :-1, :-1] - in[:, 1:, :-1].
    """
    from artvid.losses.tv import TVLoss

    strength = 1.5
    inp = torch.randn(3, 6, 7, requires_grad=True)
    TVLoss(strength=strength)(inp).backward()
    autograd_grad = inp.grad.clone()

    x = inp.detach()
    x_diff = x[:, :-1, :-1] - x[:, :-1, 1:]
    y_diff = x[:, :-1, :-1] - x[:, 1:, :-1]
    expected = torch.zeros_like(x)
    expected[:, :-1, :-1] += x_diff + y_diff
    expected[:, :-1, 1:] += -x_diff
    expected[:, 1:, :-1] += -y_diff
    expected *= strength

    assert torch.allclose(autograd_grad, expected, atol=1e-5)


def test_tv_loss_gradient_direction_smooths(torch):
    """A gradient step should reduce the TV energy (descent direction check)."""
    from artvid.losses.tv import TVLoss

    loss_mod = TVLoss(strength=1.0)
    inp = torch.randn(3, 8, 8, requires_grad=True)
    e0 = loss_mod(inp)
    e0.backward()
    with torch.no_grad():
        stepped = inp - 1e-2 * inp.grad
    e1 = loss_mod(stepped)
    assert e1.item() < e0.item()


def test_tv_loss_l1_form_positive(torch):
    from artvid.losses.tv import TVLoss

    img = torch.randn(3, 5, 5)
    loss = TVLoss(strength=1.0, form="l1")
    assert loss(img).item() > 0.0


def test_tv_loss_rejects_bad_form(torch):
    from artvid.losses.tv import TVLoss

    with pytest.raises(ValueError):
        TVLoss(form="huber")


# --------------------------------------------------------------------------- #
# WeightedContentLoss / temporal (artistic_video_core.lua:291-347)
# --------------------------------------------------------------------------- #
def test_temporal_loss_no_weights_equals_mse(torch):
    from artvid.losses.temporal import WeightedContentLoss

    target = torch.randn(3, 6, 6)
    inp = torch.randn(3, 6, 6)
    strength = 2.0
    loss = WeightedContentLoss(target, weights=None, strength=strength)
    expected = torch.nn.functional.mse_loss(inp, target) * strength
    assert loss(inp).item() == pytest.approx(expected.item(), rel=1e-5)


def test_temporal_loss_weighted_semantics(torch):
    """Weighted MSE realizes w * err^2 (legacy sqrt trick, :296-301,321-322)."""
    from artvid.losses.temporal import WeightedContentLoss

    target = torch.randn(3, 6, 6)
    inp = torch.randn(3, 6, 6)
    weights = torch.rand(3, 6, 6) + 0.1  # strictly positive
    strength = 1.0
    loss = WeightedContentLoss(
        target, weights=weights, strength=strength, criterion="mse"
    )

    # Expected: mean( w * (inp - target)^2 ) * strength, since MSE divides by N
    # and the sqrt-weighting on both operands yields (sqrt(w)*(inp-target))^2.
    err = inp - target
    expected = (weights * err.pow(2)).mean() * strength
    assert loss(inp).item() == pytest.approx(expected.item(), rel=1e-5)


def test_temporal_loss_zero_at_target(torch):
    from artvid.losses.temporal import WeightedContentLoss

    target = torch.randn(3, 5, 5)
    weights = torch.rand(3, 5, 5)
    loss = WeightedContentLoss(target, weights=weights, strength=1.0)
    assert loss(target.clone()).item() == pytest.approx(0.0, abs=1e-6)


def test_temporal_loss_smoothl1_criterion(torch):
    from artvid.losses.temporal import WeightedContentLoss

    target = torch.randn(3, 5, 5)
    inp = torch.randn(3, 5, 5)
    loss = WeightedContentLoss(
        target, weights=None, strength=1.0, criterion="smoothl1"
    )
    expected = torch.nn.functional.smooth_l1_loss(inp, target)
    assert loss(inp).item() == pytest.approx(expected.item(), rel=1e-5)


def test_temporal_loss_unknown_criterion_falls_back_to_mse(torch):
    from artvid.losses.temporal import WeightedContentLoss

    target = torch.randn(3, 4, 4)
    loss = WeightedContentLoss(target, criterion="nonsense")
    assert loss.criterion == "mse"


def test_temporal_loss_backward_produces_grad(torch):
    from artvid.losses.temporal import WeightedContentLoss

    target = torch.randn(3, 5, 5)
    weights = torch.rand(3, 5, 5)
    inp = torch.randn(3, 5, 5, requires_grad=True)
    loss = WeightedContentLoss(target, weights=weights, strength=1.0)
    loss(inp).backward()
    assert inp.grad is not None
    assert torch.isfinite(inp.grad).all()


def test_temporal_loss_shape_mismatch_returns_zero(torch):
    from artvid.losses.temporal import WeightedContentLoss

    target = torch.randn(3, 8, 8)
    loss = WeightedContentLoss(target)
    assert loss(torch.randn(3, 4, 4)).item() == pytest.approx(0.0)
