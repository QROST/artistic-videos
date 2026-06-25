"""Middlebury ``.flo`` optical-flow read/write.

Ports ``flowFileLoader_load`` from ``flowFileLoader.lua:14-34`` (the legacy
reader) and adds a matching writer.

File format (Middlebury ``.flo``)
---------------------------------
========  ====================================================================
bytes     contents
========  ====================================================================
0-3       tag ``"PIEH"`` in ASCII -- in little-endian this is the float
          ``202021.25`` (a sanity check that floats are stored correctly)
4-7       width  ``W`` as a little-endian 32-bit int
8-11      height ``H`` as a little-endian 32-bit int
12-end    data: ``H * W * 2`` little-endian float32, row-major, with the two
          flow components ``(u, v)`` interleaved per pixel
========  ====================================================================

Axis convention (IMPORTANT -- this is the easy-to-get-wrong part)
-----------------------------------------------------------------
The on-disk Middlebury layout stores, for each pixel in row-major order, the
pair ``(u, v)`` where:

* ``u`` = horizontal displacement ``dx`` (along the image width / x axis),
* ``v`` = vertical displacement ``dy`` (along the image height / y axis).

The legacy Lua loader returned a ``2 x H x W`` tensor but **swapped** the
channels to ``(v, u)`` (channel 0 = ``v``/``dy``, channel 1 = ``u``/``dx``),
because Torch's ``image.warp`` expects ``(y, x)`` ordering
(``flowFileLoader.lua:20-30``).

This module deliberately does **not** carry that swap. ``read_flo`` returns a
standard array shaped ``(2, H, W)`` with::

    flow[0] = u = dx   (horizontal, along width  / x)
    flow[1] = v = dy   (vertical,   along height / y)

i.e. channel order ``(u, v)``. The ``(y, x)`` re-ordering that
``torch.nn.functional.grid_sample`` (or any warp backend) needs is the
responsibility of ``artvid/flow/warp.py``, performed once at the warp
boundary -- not baked into the on-disk loader. Keeping the I/O layer in the
plain ``(u, v)`` convention makes ``.flo`` files interoperable with every
other standard tool (RAFT, Middlebury utilities, ``cv2.readOpticalFlow``,
etc.).

``read_flo``/``write_flo`` round-trip exactly: ``read_flo(write_flo(x)) == x``.

This module is framework-agnostic and depends only on NumPy (no torch, no
MPS-specific calls), so flow files can be read/written without a GPU backend.
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

#: ASCII tag at the start of every Middlebury ``.flo`` file. Stored as the
#: float ``202021.25`` in little-endian; the four bytes happen to spell
#: ``"PIEH"``. Mirrors ``flowFileLoader.lua:8``.
FLO_TAG_FLOAT = 202021.25
FLO_TAG_BYTES = b"PIEH"

# Sanity-check that the documented tag float really encodes to the PIEH bytes.
assert struct.pack("<f", FLO_TAG_FLOAT) == FLO_TAG_BYTES

__all__ = ["read_flo", "write_flo", "FLO_TAG_FLOAT", "FLO_TAG_BYTES"]


def read_flo(path: str | Path) -> np.ndarray:
    """Read a Middlebury ``.flo`` file.

    Ports ``flowFileLoader_load`` (``flowFileLoader.lua:14-34``) but returns
    flow in the **standard ``(u, v)`` channel order** (no ``(y, x)`` swap; see
    the module docstring).

    Parameters
    ----------
    path:
        Path to a ``.flo`` file.

    Returns
    -------
    numpy.ndarray
        ``float32`` array of shape ``(2, H, W)`` where ``flow[0]`` is the
        horizontal displacement ``u`` (``dx``, along width) and ``flow[1]`` is
        the vertical displacement ``v`` (``dy``, along height).

    Raises
    ------
    ValueError
        If the magic tag is missing/incorrect, the dimensions are invalid, or
        the data section is truncated.
    """
    path = Path(path)
    with open(path, "rb") as f:
        raw = f.read()

    if len(raw) < 12:
        raise ValueError(
            f"{path}: file too short to be a .flo (got {len(raw)} bytes, "
            "need at least a 12-byte header)"
        )

    tag = struct.unpack_from("<f", raw, 0)[0]
    if tag != FLO_TAG_FLOAT:
        raise ValueError(
            f"{path}: bad .flo magic tag (got {tag!r}, expected "
            f"{FLO_TAG_FLOAT!r} / {FLO_TAG_BYTES!r}); wrong file or byte order"
        )

    width = struct.unpack_from("<i", raw, 4)[0]
    height = struct.unpack_from("<i", raw, 8)[0]
    if width <= 0 or height <= 0:
        raise ValueError(
            f"{path}: invalid dimensions W={width} H={height}"
        )

    expected = 12 + width * height * 2 * 4
    if len(raw) < expected:
        raise ValueError(
            f"{path}: truncated data section (got {len(raw)} bytes, "
            f"expected {expected} for {width}x{height} flow)"
        )

    # On disk the components are interleaved per pixel as (u, v) in row-major
    # order. Reshape to (H, W, 2) then move the channel axis to the front to
    # get (2, H, W) with channel 0 = u (dx) and channel 1 = v (dy).
    data = np.frombuffer(raw, dtype="<f4", count=width * height * 2, offset=12)
    flow_hwc = data.reshape(height, width, 2)
    flow = np.ascontiguousarray(np.transpose(flow_hwc, (2, 0, 1)), dtype=np.float32)
    return flow


def write_flo(path: str | Path, flow: np.ndarray) -> None:
    """Write a Middlebury ``.flo`` file.

    Inverse of :func:`read_flo`. Accepts flow in the standard ``(u, v)``
    channel order described in the module docstring.

    Parameters
    ----------
    path:
        Destination path for the ``.flo`` file.
    flow:
        Flow array shaped either ``(2, H, W)`` (channel 0 = ``u``/``dx``,
        channel 1 = ``v``/``dy``) or ``(H, W, 2)`` (last axis = ``(u, v)``).
        Written as little-endian ``float32`` regardless of the input dtype.

    Raises
    ------
    ValueError
        If ``flow`` does not have a recognizable 2-channel shape.
    """
    arr = np.asarray(flow)
    if arr.ndim != 3:
        raise ValueError(
            f"flow must be 3-D ((2,H,W) or (H,W,2)); got shape {arr.shape}"
        )

    if arr.shape[0] == 2:
        # (2, H, W) -> (H, W, 2)
        height, width = arr.shape[1], arr.shape[2]
        flow_hwc = np.transpose(arr, (1, 2, 0))
    elif arr.shape[2] == 2:
        # already (H, W, 2)
        height, width = arr.shape[0], arr.shape[1]
        flow_hwc = arr
    else:
        raise ValueError(
            f"flow must have a size-2 channel axis; got shape {arr.shape}"
        )

    flow_hwc = np.ascontiguousarray(flow_hwc, dtype="<f4")

    path = Path(path)
    with open(path, "wb") as f:
        f.write(FLO_TAG_BYTES)
        f.write(struct.pack("<i", int(width)))
        f.write(struct.pack("<i", int(height)))
        f.write(flow_hwc.tobytes())
