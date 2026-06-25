"""Tests for Middlebury ``.flo`` read/write (``artvid/io/flow_io.py``).

These tests use NumPy only and therefore run without torch (the ``.flo`` I/O
layer is framework-agnostic).
"""

from __future__ import annotations

import struct

import numpy as np
import pytest

from artvid.io.flow_io import (
    FLO_TAG_BYTES,
    FLO_TAG_FLOAT,
    read_flo,
    write_flo,
)


def _synthetic_flow(height: int = 3, width: int = 4) -> np.ndarray:
    """Build a small deterministic (2, H, W) flow with distinct u/v patterns.

    ``u`` (channel 0) ramps along x, ``v`` (channel 1) ramps along y plus a
    fractional offset, so a u/v swap or an x/y transpose would be detectable.
    """
    ys, xs = np.mgrid[0:height, 0:width]
    u = xs.astype(np.float32) - width / 2.0  # horizontal, varies with x
    v = ys.astype(np.float32) * 0.5 + 0.25  # vertical, varies with y
    return np.stack([u, v], axis=0).astype(np.float32)


def test_roundtrip_chw(tmp_path):
    flow = _synthetic_flow()
    path = tmp_path / "flow.flo"
    write_flo(path, flow)
    out = read_flo(path)

    assert out.shape == flow.shape
    assert out.dtype == np.float32
    np.testing.assert_array_equal(out, flow)


def test_roundtrip_hwc_input(tmp_path):
    """A (H, W, 2) input should round-trip to the canonical (2, H, W)."""
    flow_chw = _synthetic_flow()
    flow_hwc = np.transpose(flow_chw, (1, 2, 0)).copy()  # (H, W, 2)
    path = tmp_path / "flow_hwc.flo"
    write_flo(path, flow_hwc)
    out = read_flo(path)

    assert out.shape == flow_chw.shape  # reader always returns (2, H, W)
    np.testing.assert_array_equal(out, flow_chw)


def test_channel_order_is_u_v(tmp_path):
    """Channel 0 must be u (dx, along x); channel 1 must be v (dy, along y).

    Guards against the legacy (y, x) swap leaking into the I/O layer.
    """
    flow = _synthetic_flow()
    path = tmp_path / "order.flo"
    write_flo(path, flow)
    out = read_flo(path)

    # Channel 0 (u) varies across columns (x) and is constant down rows.
    assert np.all(out[0, 0, :] == out[0, 1, :])  # same across rows
    assert not np.all(out[0, :, 0] == out[0, :, 1])  # differs across columns
    # Channel 1 (v) varies down rows (y) and is constant across columns.
    assert np.all(out[1, :, 0] == out[1, :, 1])  # same across columns
    assert not np.all(out[1, 0, :] == out[1, 1, :])  # differs across rows


def test_on_disk_layout(tmp_path):
    """Verify the exact header bytes and interleaved (u, v) row-major data."""
    height, width = 2, 3
    flow = _synthetic_flow(height, width)
    path = tmp_path / "layout.flo"
    write_flo(path, flow)

    raw = path.read_bytes()
    assert raw[:4] == FLO_TAG_BYTES
    assert struct.unpack_from("<f", raw, 0)[0] == FLO_TAG_FLOAT
    assert struct.unpack_from("<i", raw, 4)[0] == width
    assert struct.unpack_from("<i", raw, 8)[0] == height

    data = np.frombuffer(raw, dtype="<f4", offset=12)
    # First pixel (row 0, col 0): u then v, interleaved.
    assert data[0] == flow[0, 0, 0]  # u at (0,0)
    assert data[1] == flow[1, 0, 0]  # v at (0,0)
    # Second pixel (row 0, col 1).
    assert data[2] == flow[0, 0, 1]
    assert data[3] == flow[1, 0, 1]


def test_bad_magic_raises(tmp_path):
    path = tmp_path / "bad.flo"
    path.write_bytes(b"\x00\x00\x00\x00" + struct.pack("<ii", 1, 1) + b"\x00" * 8)
    with pytest.raises(ValueError, match="magic tag"):
        read_flo(path)


def test_truncated_raises(tmp_path):
    path = tmp_path / "trunc.flo"
    # Valid header claiming 4x4 flow but no data section.
    path.write_bytes(FLO_TAG_BYTES + struct.pack("<ii", 4, 4))
    with pytest.raises(ValueError, match="truncated"):
        read_flo(path)


def test_too_short_raises(tmp_path):
    path = tmp_path / "short.flo"
    path.write_bytes(b"PIE")
    with pytest.raises(ValueError, match="too short"):
        read_flo(path)


def test_invalid_dims_raise(tmp_path):
    path = tmp_path / "dims.flo"
    path.write_bytes(FLO_TAG_BYTES + struct.pack("<ii", 0, 5))
    with pytest.raises(ValueError, match="invalid dimensions"):
        read_flo(path)


def test_write_rejects_wrong_shape(tmp_path):
    path = tmp_path / "x.flo"
    with pytest.raises(ValueError):
        write_flo(path, np.zeros((3, 4, 5), dtype=np.float32))
    with pytest.raises(ValueError):
        write_flo(path, np.zeros((4,), dtype=np.float32))


def test_non_float_input_is_cast(tmp_path):
    """Integer-typed flow input is written as float32 and reads back equal."""
    flow = (_synthetic_flow() * 2).astype(np.int32)
    path = tmp_path / "int.flo"
    write_flo(path, flow)
    out = read_flo(path)
    assert out.dtype == np.float32
    np.testing.assert_array_equal(out, flow.astype(np.float32))
