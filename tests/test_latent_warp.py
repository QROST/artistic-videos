"""Tests for latent-space optical-flow warp + reliability
(``artvid/diffusion/latent_warp.py``).

These tests require torch (``grid_sample`` / ``interpolate``), so they depend on
the ``torch`` fixture from ``conftest.py`` and skip automatically when torch is
not installed in the environment.

The load-bearing property pinned here (design doc §5 item 1, the highest-risk
correctness item) is the **pixel→latent scale handling**: a pixel flow of
``vae_factor`` pixels must move the latent by exactly ONE latent cell. Getting
either the spatial resize or the per-axis magnitude rescale wrong is silently
off by a constant factor, so we assert it directly against a manual roll.
"""

from __future__ import annotations

from artvid.diffusion.latent_warp import (
    LatentWarpResult,
    flow_to_latent,
    latent_reliability,
    warp_latent,
)


def _latent_ramp(torch, c: int = 4, h: int = 5, w: int = 7):
    """A (1, C, h, w) latent whose channels encode (x, y, x+y, const).

    A wrong axis order / sign / scale in the latent warp is then directly
    visible in which channel / direction moves and by how much.
    """
    ys, xs = torch.meshgrid(
        torch.arange(h, dtype=torch.float32),
        torch.arange(w, dtype=torch.float32),
        indexing="ij",
    )
    chans = [xs, ys, xs + ys, torch.full_like(xs, 3.0)][:c]
    return torch.stack(chans, dim=0).unsqueeze(0)  # (1, C, h, w)


def test_flow_to_latent_scales_magnitude_and_size(torch):
    """A constant 8-px pixel flow becomes a constant 1-cell latent flow at f=8."""
    H, W, f = 40, 56, 8
    h, w = H // f, W // f
    flow_px = torch.zeros((2, H, W), dtype=torch.float32)
    flow_px[0, :, :] = float(f)  # u = +8 px == +1 latent cell at f=8
    flow_px[1, :, :] = float(2 * f)  # v = +16 px == +2 latent cells

    flow_lat = flow_to_latent(flow_px, (h, w))

    assert flow_lat.shape == (2, h, w)
    torch.testing.assert_close(
        flow_lat[0], torch.ones((h, w)), rtol=0, atol=1e-5
    )
    torch.testing.assert_close(
        flow_lat[1], torch.full((h, w), 2.0), rtol=0, atol=1e-5
    )


def test_warp_latent_zero_flow_is_identity(torch):
    """A zero pixel flow warps the latent to itself (all cells valid)."""
    z = _latent_ramp(torch)
    h, w = z.shape[-2:]
    flow_px = torch.zeros((2, h * 8, w * 8), dtype=torch.float32)

    result = warp_latent(z, flow_px, vae_factor=8)

    assert isinstance(result, LatentWarpResult)
    assert result.image.shape == z.shape
    assert result.valid.shape == (1, 1, h, w)
    assert torch.all(result.valid)
    torch.testing.assert_close(result.image, z, rtol=0, atol=1e-5)


def test_warp_latent_integer_cell_shift_x(torch):
    """A +f pixel u-flow shifts the latent by exactly +1 cell in x.

    Backward warp: ``out(x) = z(x + u_latent)`` with ``u_latent = +1``. Interior
    latent columns 0..w-2 must equal the original columns 1..w-1, matching a
    manual roll. This is the pixel→latent scale correctness check.
    """
    z = _latent_ramp(torch)
    h, w = z.shape[-2:]
    flow_px = torch.zeros((2, h * 8, w * 8), dtype=torch.float32)
    flow_px[0, :, :] = 8.0  # +8 px == +1 latent cell at f=8

    result = warp_latent(z, flow_px, vae_factor=8)

    interior = result.image[:, :, :, : w - 1]
    expected = z[:, :, :, 1:]
    torch.testing.assert_close(interior, expected, rtol=0, atol=1e-5)
    # Last latent column samples cell w (out of border) -> invalid.
    assert torch.all(~result.valid[:, :, :, w - 1])
    assert torch.all(result.valid[:, :, :, : w - 1])


def test_warp_latent_integer_cell_shift_y(torch):
    """A +f pixel v-flow shifts the latent by exactly +1 cell in y."""
    z = _latent_ramp(torch)
    h, w = z.shape[-2:]
    flow_px = torch.zeros((2, h * 8, w * 8), dtype=torch.float32)
    flow_px[1, :, :] = 8.0  # +8 px == +1 latent cell in y

    result = warp_latent(z, flow_px, vae_factor=8)

    interior = result.image[:, :, : h - 1, :]
    expected = z[:, :, 1:, :]
    torch.testing.assert_close(interior, expected, rtol=0, atol=1e-5)
    assert torch.all(~result.valid[:, :, h - 1, :])


def test_warp_latent_no_vgg_fill_keeps_zeros(torch):
    """Out-of-border latent cells stay at the zero pad (no VGG mean fill)."""
    z = _latent_ramp(torch) + 1.0  # shift away from 0 so zero-pad is distinct
    h, w = z.shape[-2:]
    flow_px = torch.zeros((2, h * 8, w * 8), dtype=torch.float32)
    flow_px[0, :, :] = 8.0  # last latent column goes out of border

    result = warp_latent(z, flow_px, vae_factor=8)

    torch.testing.assert_close(
        result.image[:, :, :, w - 1],
        torch.zeros((1, z.shape[1], h), dtype=torch.float32),
        rtol=0,
        atol=1e-5,
    )


def test_latent_reliability_all_reliable_zero_flow(torch):
    """Zero flow (no occlusion) yields reliability ~1 over all valid cells."""
    H, W, f = 40, 56, 8
    h, w = H // f, W // f
    zero = torch.zeros((2, H, W), dtype=torch.float32)
    valid = torch.ones((1, 1, h, w), dtype=torch.bool)

    rel = latent_reliability(zero, zero, valid, latent_hw=(h, w))

    assert rel.shape == (1, 1, h, w)
    assert rel.dtype == torch.float32
    # No motion -> consistent everywhere -> reliability close to 1.
    assert torch.all(rel >= 0.9)


def test_latent_reliability_invalid_cells_zeroed(torch):
    """valid_latent gates the output: invalid cells -> 0 reliability."""
    H, W, f = 40, 56, 8
    h, w = H // f, W // f
    zero = torch.zeros((2, H, W), dtype=torch.float32)
    valid = torch.ones((1, 1, h, w), dtype=torch.bool)
    valid[:, :, :, -1] = False  # mark last column out-of-border

    rel = latent_reliability(zero, zero, valid, latent_hw=(h, w))

    assert torch.all(rel[:, :, :, -1] == 0.0)


def test_latent_reliability_gamma_erodes(torch):
    """gamma > 1 lowers partial reliability more than gamma == 1.

    With a partially-reliable downsampled cell (value in (0,1)), raising to
    gamma>1 must not increase it; a strictly-interior partial value decreases.
    """
    # Build a forward/backward flow pair with an occlusion so some cells are
    # partially reliable after downsampling: a divergent flow at a seam.
    H, W, f = 40, 56, 8
    h, w = H // f, W // f
    fwd = torch.zeros((2, H, W), dtype=torch.float32)
    bwd = torch.zeros((2, H, W), dtype=torch.float32)
    # Inconsistent region: forward says +5 px right, backward says +5 px right
    # too (so round-trip error is large -> occluded) in the left half.
    fwd[0, :, : W // 2] = 5.0
    bwd[0, :, : W // 2] = 5.0
    valid = torch.ones((1, 1, h, w), dtype=torch.bool)

    rel1 = latent_reliability(fwd, bwd, valid, latent_hw=(h, w), gamma=1.0)
    rel2 = latent_reliability(fwd, bwd, valid, latent_hw=(h, w), gamma=2.0)

    # Erosion never increases reliability.
    assert torch.all(rel2 <= rel1 + 1e-6)
    # And at least one partially-reliable cell strictly decreases.
    partial = (rel1 > 1e-3) & (rel1 < 1.0 - 1e-3)
    if partial.any():
        assert torch.all(rel2[partial] < rel1[partial] + 1e-6)
        assert (rel2[partial] < rel1[partial]).any()


def test_warp_latent_batched(torch):
    """Batched (N, C, h, w) latent returns batched output of matching rank."""
    z = _latent_ramp(torch)
    batch = torch.cat([z, z + 0.5], dim=0)  # (2, C, h, w)
    h, w = z.shape[-2:]
    flow_px = torch.zeros((2, h * 8, w * 8), dtype=torch.float32)

    result = warp_latent(batch, flow_px, vae_factor=8)

    assert result.image.shape == batch.shape
    assert result.valid.shape == (2, 1, h, w)
    torch.testing.assert_close(result.image, batch, rtol=0, atol=1e-5)


def test_warp_latent_rejects_3d(torch):
    """A 3-D (C, h, w) latent is rejected: latents must be (N, C, h, w)."""
    import pytest

    z = _latent_ramp(torch)[0]  # (C, h, w)
    flow_px = torch.zeros((2, z.shape[-2] * 8, z.shape[-1] * 8))
    with pytest.raises(ValueError):
        warp_latent(z, flow_px)
