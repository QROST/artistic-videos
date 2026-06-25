"""Backward optical-flow warping with disocclusion fill.

Ports ``warpImage`` (``artistic_video.lua:291-304``) to PyTorch using
:func:`torch.nn.functional.grid_sample`. The legacy implementation relied on
Torch7's ``image.warp(img, flow, 'bilinear', true, 'pad', -1)`` followed by a
per-pixel loop that replaced fully out-of-border pixels (all channels ``== -1``)
with the VGG mean pixel. This module reproduces that behaviour vectorised on the
GPU/MPS backend and additionally returns an explicit validity mask.

What "backward warp" means here
-------------------------------
Given a *backward* flow (the flow from the **current** frame to the
**previous** frame), we sample the previous frame/output into the current
frame's coordinate system. For each output pixel at integer location
``(x, y)`` we read the source image at::

    src_x = x + u(x, y)
    src_y = y + v(x, y)

where ``flow[0] = u = dx`` (horizontal) and ``flow[1] = v = dy`` (vertical), the
standard ``(u, v)`` channel order produced by :mod:`artvid.io.flow_io` and
:mod:`artvid.flow.raft` (no legacy ``(y, x)`` swap). This matches the legacy
``image.warp`` semantics, which is itself a backward/inverse warp:
``result(y, x) = img(y + flow_y, x + flow_x)``.

grid_sample convention (the easy-to-get-wrong part)
---------------------------------------------------
:func:`torch.nn.functional.grid_sample` expects a sampling grid of shape
``(N, H, W, 2)`` whose **last axis is ordered ``(x, y)``** (NOT ``(y, x)``) and
whose values are **normalised to ``[-1, 1]``** over the *input* spatial extent.
With ``align_corners=True`` the mapping from absolute pixel coordinate to
normalised coordinate is::

    x_norm = 2 * src_x / (W - 1) - 1
    y_norm = 2 * src_y / (H - 1) - 1

so ``-1`` is the centre of the first pixel and ``+1`` the centre of the last
pixel. We use ``align_corners=True`` because the flow displacements are defined
in absolute pixel units relative to pixel *centres* (consistent with the legacy
``image.warp`` integer-shift behaviour, which the unit tests pin down).

Disocclusion / out-of-border fill
----------------------------------
Pixels whose sampling location falls outside the source image are *disoccluded*
(no valid source). The legacy code padded such pixels with ``-1`` and then
overwrote them with the VGG mean pixel. We instead sample with
``padding_mode='zeros'`` and, in parallel, warp an all-ones mask: any output
pixel whose warped mask value drops below 1 had part of its bilinear support
outside the border and is treated as invalid. Invalid pixels are filled with the
VGG mean pixel. The boolean validity mask is returned so downstream temporal
losses can down-weight disoccluded regions (cf. ``flow/consistency.py``).

This module is framework-agnostic: it only uses ``torch`` tensor ops and
``torch.nn.functional`` (no MPS-specific calls), so it runs unchanged on
``mps | cuda | cpu`` per :mod:`artvid.device`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

if TYPE_CHECKING:  # pragma: no cover - typing only; torch is optional at import
    import torch

#: VGG mean pixel in **RGB** order, scaled to ``[0, 1]``. This mirrors the
#: legacy fill value ``{123.68/256, 116.779/256, 103.939/256}`` from
#: ``artistic_video.lua:292`` (R, G, B). The caffe means are
#: ``[103.939, 116.779, 123.68]`` in BGR; here they are RGB-ordered and divided
#: by 256, matching the deprocessed/[0,1] image space the warp operates in.
VGG_MEAN_PIXEL_RGB_01: tuple[float, float, float] = (
    123.68 / 256.0,
    116.779 / 256.0,
    103.939 / 256.0,
)


class WarpResult(NamedTuple):
    """Result of :func:`warp_image`.

    Attributes:
        image: ``(N, 3, H, W)`` (or ``(3, H, W)`` for unbatched input) warped
            image, with disoccluded/out-of-border pixels filled by the VGG mean
            pixel. Same dtype/device as the input image (float32).
        valid: ``(N, 1, H, W)`` (or ``(1, H, W)``) boolean mask, ``True`` where
            the sample came fully from inside the source image and ``False`` for
            disoccluded / out-of-border pixels that were mean-filled.
    """

    image: "torch.Tensor"
    valid: "torch.Tensor"


def flow_to_grid(flow: "torch.Tensor", *, align_corners: bool = True) -> "torch.Tensor":
    """Convert a ``(N, 2, H, W)`` flow into a ``grid_sample`` sampling grid.

    Builds an absolute-coordinate sampling grid (each output pixel ``(x, y)``
    maps to source ``(x + u, y + v)``) and normalises it to ``[-1, 1]`` in the
    ``(x, y)`` last-axis order that :func:`torch.nn.functional.grid_sample`
    expects.

    Args:
        flow: ``(N, 2, H, W)`` flow in ``(u, v) = (dx, dy)`` channel order
            (channel 0 = horizontal/x, channel 1 = vertical/y), in absolute
            pixel-displacement units.
        align_corners: Normalisation convention; must match the value later
            passed to ``grid_sample``. ``True`` (the default) treats flow
            displacements as pixel-centre offsets.

    Returns:
        ``(N, H, W, 2)`` float grid with last axis ordered ``(x_norm, y_norm)``.
    """
    import torch

    if flow.dim() != 4 or flow.shape[1] != 2:
        raise ValueError(
            f"flow must have shape (N, 2, H, W); got {tuple(flow.shape)}."
        )

    n, _, h, w = flow.shape
    device = flow.device
    dtype = flow.dtype

    # Base grid of absolute pixel coordinates. ``xs`` varies along width (x),
    # ``ys`` along height (y). indexing='ij' gives ys, xs each of shape (H, W).
    ys, xs = torch.meshgrid(
        torch.arange(h, device=device, dtype=dtype),
        torch.arange(w, device=device, dtype=dtype),
        indexing="ij",
    )

    u = flow[:, 0, :, :]  # (N, H, W) horizontal displacement (dx)
    v = flow[:, 1, :, :]  # (N, H, W) vertical displacement (dy)

    src_x = xs.unsqueeze(0) + u  # (N, H, W)
    src_y = ys.unsqueeze(0) + v  # (N, H, W)

    # Normalise to [-1, 1]. grid_sample's last axis is (x, y) in this order.
    if align_corners:
        # Guard W==1 / H==1 to avoid division by zero (degenerate but valid).
        denom_x = max(w - 1, 1)
        denom_y = max(h - 1, 1)
        x_norm = 2.0 * src_x / denom_x - 1.0
        y_norm = 2.0 * src_y / denom_y - 1.0
    else:
        x_norm = (2.0 * src_x + 1.0) / w - 1.0
        y_norm = (2.0 * src_y + 1.0) / h - 1.0

    grid = torch.stack((x_norm, y_norm), dim=-1)  # (N, H, W, 2)
    return grid


def warp_image(
    img: "torch.Tensor",
    flow: "torch.Tensor",
    *,
    fill: tuple[float, float, float] | None = VGG_MEAN_PIXEL_RGB_01,
    align_corners: bool = True,
) -> WarpResult:
    """Backward-warp ``img`` according to ``flow`` (ports ``warpImage``).

    Samples ``img`` (a previous frame or previous stylised output) into the
    current frame's coordinates using a *backward* flow, filling disoccluded /
    out-of-border pixels with the VGG mean pixel, and returns a validity mask.

    Args:
        img: Image to warp. RGB float32 in ``[0, 1]`` space, shape ``(3, H, W)``
            or ``(N, 3, H, W)``. (The warp itself is range-agnostic; the default
            ``fill`` value assumes the deprocessed ``[0, 1]`` RGB space.)
        flow: Backward flow (current → previous), shape ``(2, H, W)`` or
            ``(N, 2, H, W)`` in ``(u, v) = (dx, dy)`` channel order, in absolute
            pixel units. Spatial size must match ``img``.
        fill: RGB triple used to fill disoccluded/out-of-border pixels. Defaults
            to the VGG mean pixel (:data:`VGG_MEAN_PIXEL_RGB_01`), matching
            ``artistic_video.lua:297-299``. Pass ``None`` to leave those pixels
            at the ``grid_sample`` zero-pad value instead.

    Returns:
        A :class:`WarpResult` ``(image, valid)``. Output rank matches the input
        image rank (unbatched in → unbatched out).

    Raises:
        ValueError: If ``img``/``flow`` ranks or spatial sizes are incompatible.
    """
    import torch
    import torch.nn.functional as F

    was_3d = img.dim() == 3
    img_b = img.unsqueeze(0) if was_3d else img
    flow_b = flow.unsqueeze(0) if flow.dim() == 3 else flow

    if img_b.dim() != 4 or img_b.shape[1] != 3:
        raise ValueError(
            f"img must have shape (3, H, W) or (N, 3, H, W); got {tuple(img.shape)}."
        )
    if flow_b.dim() != 4 or flow_b.shape[1] != 2:
        raise ValueError(
            f"flow must have shape (2, H, W) or (N, 2, H, W); got {tuple(flow.shape)}."
        )
    if img_b.shape[0] != flow_b.shape[0]:
        raise ValueError(
            "img and flow batch sizes differ: "
            f"{img_b.shape[0]} vs {flow_b.shape[0]}."
        )
    if img_b.shape[-2:] != flow_b.shape[-2:]:
        raise ValueError(
            "img and flow spatial sizes differ: "
            f"{tuple(img_b.shape[-2:])} vs {tuple(flow_b.shape[-2:])}."
        )

    grid = flow_to_grid(flow_b.to(img_b.dtype), align_corners=align_corners)

    # Bilinear backward warp. zeros padding lets us detect out-of-border samples
    # via a companion ones-mask warp (legacy used pad=-1 then mean-filled).
    warped = F.grid_sample(
        img_b,
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=align_corners,
    )

    ones = torch.ones(
        (img_b.shape[0], 1, img_b.shape[2], img_b.shape[3]),
        device=img_b.device,
        dtype=img_b.dtype,
    )
    mask = F.grid_sample(
        ones,
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=align_corners,
    )
    # A pixel is fully valid only if its entire bilinear support was inside the
    # source. Allow a tiny tolerance for float rounding at exact pixel hits.
    valid = mask >= (1.0 - 1e-6)  # (N, 1, H, W) bool

    if fill is not None:
        fill_t = torch.tensor(
            fill, device=img_b.device, dtype=img_b.dtype
        ).view(1, 3, 1, 1)
        warped = torch.where(valid, warped, fill_t)

    if was_3d:
        return WarpResult(image=warped[0], valid=valid[0])
    return WarpResult(image=warped, valid=valid)
