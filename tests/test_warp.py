"""Tests for backward optical-flow warping (``artvid/flow/warp.py``).

These tests require torch (grid_sample), so they depend on the ``torch``
fixture from ``conftest.py`` and skip automatically when torch is not installed
in the environment.

The two load-bearing properties pinned here:

1. **Zero flow is identity** — a zero displacement field must reproduce the
   input exactly (modulo the fill on a fully-valid sample).
2. **Integer-shift flow shifts correctly** — a constant integer flow must shift
   image content by exactly that many pixels in the direction implied by the
   backward-warp / ``grid_sample`` convention (the #1 bug source). We assert the
   *sign* explicitly: ``flow[0] = u = +1`` samples the source one pixel to the
   right, so output content moves one pixel left.
"""

from __future__ import annotations

import numpy as np

from artvid.flow.warp import VGG_MEAN_PIXEL_RGB_01, flow_to_grid, warp_image


def _ramp_image(torch, h: int = 5, w: int = 7):
    """A (3, H, W) image whose channels encode (x, y, x+y) for shift detection.

    Channel 0 increases along x (columns), channel 1 along y (rows). A wrong
    axis order or sign in the warp is then directly visible in which channel /
    direction moves.
    """
    ys, xs = torch.meshgrid(
        torch.arange(h, dtype=torch.float32),
        torch.arange(w, dtype=torch.float32),
        indexing="ij",
    )
    img = torch.stack([xs, ys, xs + ys], dim=0)  # (3, H, W)
    return img


def test_zero_flow_is_identity(torch):
    """Warping by a zero flow returns the input unchanged."""
    img = _ramp_image(torch)
    flow = torch.zeros((2, img.shape[1], img.shape[2]), dtype=torch.float32)

    result = warp_image(img, flow)

    assert result.image.shape == img.shape
    assert result.valid.shape == (1, img.shape[1], img.shape[2])
    # Every pixel maps exactly onto itself -> all valid, exact reproduction.
    assert torch.all(result.valid)
    torch.testing.assert_close(result.image, img, rtol=0, atol=1e-5)


def test_integer_shift_flow_x(torch):
    """A constant u=+1 flow samples one pixel to the right (content shifts left).

    Backward warp: ``out(x, y) = img(x + u, y + v)``. With ``u = +1`` the output
    column ``x`` equals input column ``x + 1`` for all interior columns. This
    nails the x-axis sign and that channel 0 of the flow is the horizontal (x)
    component.
    """
    img = _ramp_image(torch)
    h, w = img.shape[1], img.shape[2]
    flow = torch.zeros((2, h, w), dtype=torch.float32)
    flow[0, :, :] = 1.0  # u = +1 (horizontal)

    result = warp_image(img, flow)

    # Interior columns 0..w-2 should equal the original columns 1..w-1.
    interior = result.image[:, :, : w - 1]
    expected = img[:, :, 1:]
    torch.testing.assert_close(interior, expected, rtol=0, atol=1e-5)

    # The last column samples x = w (out of border) -> invalid + mean-filled.
    assert torch.all(~result.valid[:, :, w - 1])
    fill = torch.tensor(VGG_MEAN_PIXEL_RGB_01, dtype=torch.float32).view(3, 1)
    torch.testing.assert_close(
        result.image[:, :, w - 1], fill.expand(3, h), rtol=0, atol=1e-5
    )
    # Interior pixels stay valid.
    assert torch.all(result.valid[:, :, : w - 1])


def test_integer_shift_flow_y(torch):
    """A constant v=+1 flow samples one pixel down (content shifts up).

    Pins the y-axis: channel 1 of the flow is the vertical (y) component and the
    sign matches the backward-warp convention.
    """
    img = _ramp_image(torch)
    h, w = img.shape[1], img.shape[2]
    flow = torch.zeros((2, h, w), dtype=torch.float32)
    flow[1, :, :] = 1.0  # v = +1 (vertical)

    result = warp_image(img, flow)

    interior = result.image[:, : h - 1, :]
    expected = img[:, 1:, :]
    torch.testing.assert_close(interior, expected, rtol=0, atol=1e-5)

    # Last row samples y = h (out of border) -> invalid + mean-filled.
    assert torch.all(~result.valid[:, h - 1, :])


def test_negative_shift_flow_x(torch):
    """A constant u=-1 flow samples one pixel to the left (content shifts right)."""
    img = _ramp_image(torch)
    h, w = img.shape[1], img.shape[2]
    flow = torch.zeros((2, h, w), dtype=torch.float32)
    flow[0, :, :] = -1.0  # u = -1

    result = warp_image(img, flow)

    # out(x) = img(x - 1): interior columns 1..w-1 equal original 0..w-2.
    interior = result.image[:, :, 1:]
    expected = img[:, :, : w - 1]
    torch.testing.assert_close(interior, expected, rtol=0, atol=1e-5)
    # Column 0 samples x = -1 -> out of border, invalid.
    assert torch.all(~result.valid[:, :, 0])


def test_batched_warp(torch):
    """Batched (N,3,H,W) input returns batched output of matching rank."""
    img = _ramp_image(torch)
    batch = torch.stack([img, img + 0.5], dim=0)  # (2, 3, H, W)
    flow = torch.zeros((2, 2, img.shape[1], img.shape[2]), dtype=torch.float32)

    result = warp_image(batch, flow)

    assert result.image.shape == batch.shape
    assert result.valid.shape == (2, 1, img.shape[1], img.shape[2])
    torch.testing.assert_close(result.image, batch, rtol=0, atol=1e-5)


def test_fractional_flow_bilinear(torch):
    """A half-pixel shift interpolates linearly between neighbours.

    For the channel-0 ramp (value == column index), out(x) = img(x + 0.5)
    should equal x + 0.5 at interior pixels under bilinear sampling.
    """
    img = _ramp_image(torch)
    h, w = img.shape[1], img.shape[2]
    flow = torch.zeros((2, h, w), dtype=torch.float32)
    flow[0, :, :] = 0.5

    result = warp_image(img, flow)

    # Channel 0 encodes the column index; after +0.5 sampling it reads x+0.5.
    cols = torch.arange(w, dtype=torch.float32) + 0.5
    expected_row = cols[: w - 1]  # interior, last col is partly out of border
    got = result.image[0, 0, : w - 1]
    torch.testing.assert_close(got, expected_row, rtol=0, atol=1e-5)


def test_fill_disabled_keeps_zero_pad(torch):
    """fill=None leaves out-of-border pixels at the grid_sample zero value."""
    img = _ramp_image(torch) + 1.0  # shift away from 0 so zero-pad is distinct
    h, w = img.shape[1], img.shape[2]
    flow = torch.zeros((2, h, w), dtype=torch.float32)
    flow[0, :, :] = 1.0

    result = warp_image(img, flow, fill=None)

    # Last column is out of border; with zeros padding and no fill it stays ~0.
    torch.testing.assert_close(
        result.image[:, :, w - 1],
        torch.zeros((3, h), dtype=torch.float32),
        rtol=0,
        atol=1e-5,
    )


def test_flow_to_grid_zero_is_base_grid(torch):
    """Zero flow yields the canonical identity grid in (x, y) last-axis order."""
    h, w = 4, 6
    flow = torch.zeros((1, 2, h, w), dtype=torch.float32)
    grid = flow_to_grid(flow, align_corners=True)

    assert grid.shape == (1, h, w, 2)
    # Top-left pixel -> (-1, -1); bottom-right -> (+1, +1).
    torch.testing.assert_close(
        grid[0, 0, 0], torch.tensor([-1.0, -1.0]), rtol=0, atol=1e-6
    )
    torch.testing.assert_close(
        grid[0, h - 1, w - 1], torch.tensor([1.0, 1.0]), rtol=0, atol=1e-6
    )
    # Last axis is (x, y): along a row x varies, y is constant.
    assert torch.all(grid[0, 0, :, 1] == grid[0, 0, 0, 1])  # y constant in a row
    assert not torch.all(grid[0, 0, :, 0] == grid[0, 0, 0, 0])  # x varies


def test_flo_convention_end_to_end(torch, tmp_path):
    """A flow round-tripped through flow_io warps with the expected sign.

    Guards the full path: ``flow_io`` writes/reads (u, v) order, and warp_image
    interprets channel 0 as horizontal. A u=+2 flow must sample 2 px to the
    right, independent of how the .flo file stored its axes.
    """
    from artvid.io.flow_io import read_flo, write_flo

    img = _ramp_image(torch)
    h, w = img.shape[1], img.shape[2]
    flow_np = np.zeros((2, h, w), dtype=np.float32)
    flow_np[0, :, :] = 2.0  # u = +2 (horizontal)
    path = tmp_path / "shift.flo"
    write_flo(path, flow_np)

    flow = torch.from_numpy(read_flo(path))
    result = warp_image(img, flow)

    interior = result.image[:, :, : w - 2]
    expected = img[:, :, 2:]
    torch.testing.assert_close(interior, expected, rtol=0, atol=1e-5)
