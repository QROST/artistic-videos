"""Latent-space optical-flow warp and reliability masking (Phase 2 core reuse).

This module implements §2.3 (``warp_latent``) and §2.4 (``latent_reliability``)
of ``docs/07-phase2-design.md``: it lifts the 2016 optical-flow temporal-
consistency idea — "warp the previous stylized frame into the current frame via
optical flow, trust it only where the forward-backward consistency check says
it's reliable" — from pixel space into the VAE *latent* grid, so the warp
directly constrains the diffusion denoising trajectory.

It is the layer that REUSES the Phase 1 flow stack:

* :func:`artvid.flow.warp.flow_to_grid` — builds the ``grid_sample`` sampling
  grid from a ``(N, 2, H, W)`` flow. It is **resolution-agnostic** (it
  normalizes to ``[-1, 1]`` over whatever H×W the flow has) and uses the same
  ``align_corners`` convention as :func:`artvid.flow.warp.warp_image`. We call it
  at *latent* resolution and ``grid_sample`` the latent ourselves — we do **not**
  reuse ``warp_image`` directly, because its default ``fill=VGG_MEAN_PIXEL_RGB_01``
  is a pixel-space RGB constant that is meaningless in latent space (we keep the
  zero pad and let the mask handle out-of-border cells).
* :func:`artvid.flow.consistency.consistency_mask` — forward-backward occlusion
  reliability, computed at full pixel resolution (most accurate; reuses the
  validated Phase 1 code) and then conservatively downsampled to the latent grid.
* :func:`artvid.flow.consistency.combine_longterm_weights` — optional multi-
  reference (prev + anchor) reliability combination, lifted to latent space.

No ``diffusers`` import is needed here: this operates purely on latent tensors
that are passed in by the engine. Only framework-agnostic ``torch`` /
``torch.nn.functional`` ops are used (lazy-imported inside the functions), so the
module is ``py_compile``-able and unit-testable on CPU with synthetic
latents/flows once ``torch`` is present, exactly like the Phase 1 flow modules.

THE #1 CORRECTNESS TRAP — pixel vs latent scale
-----------------------------------------------
RAFT flow is expressed in **pixel displacement at image resolution** ``(H, W)``.
To warp a latent at ``(h, w) = (H / f, W / f)`` (VAE downsample factor ``f``,
typically 8 for SD/SDXL) we must BOTH:

1. **resize** the flow field spatially from ``(H, W)`` to ``(h, w)``; and
2. **rescale the displacement magnitudes** by the per-axis ratio ``h / H`` and
   ``w / W`` — a 16-px motion is a 2-latent-cell motion at ``f = 8``.

This is exactly the operation :func:`artvid.flow.raft._postprocess_flow`
performs when it rescales flow vectors after a resolution change; we deliberately
use the **per-axis** ratios ``(h / H, w / W)`` rather than a single ``1 / f`` so
the warp stays exact even when ``H`` / ``W`` is not an exact multiple of ``f``
(e.g. odd / padded sizes). Getting either of these two steps wrong produces a
latent warp that is silently off by a constant factor — the worst kind of bug
because the output still "looks plausible" but temporal consistency degrades.

Conventions inherited from Phase 1 (must hold for correctness):

* Flow is ``(2, H, W)`` float in ``(u, v) = (dx, dy)`` order; ``flow[:, y, x]``
  maps pixel ``(x, y)`` in ``img1`` to ``img2``.
* :func:`warp_image` / :func:`flow_to_grid` are a **backward** warp: given the
  *backward* flow (current → previous) they sample the previous content into the
  current frame's coordinates. ``warp_latent`` therefore expects the
  current→previous flow as ``backward_flow_px``.
* ``consistency_mask(primary, crosscheck, ...)``: to validate the **backward**
  warp we call ``consistency_mask(backward, forward, ...)`` — matching
  ``cli.cmd_flow``'s ``rel_back`` argument order.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

if TYPE_CHECKING:  # pragma: no cover - typing only; torch is optional at import
    import torch

from artvid.flow import consistency as _consistency
from artvid.flow import warp as _warp


class LatentWarpResult(NamedTuple):
    """Result of :func:`warp_latent`.

    Attributes:
        image: ``(1, C, h, w)`` (or ``(N, C, h, w)``) warped latent. Out-of-border
            / disoccluded latent cells are left at the ``grid_sample`` zero pad
            value (we do **not** mean-fill latents — see module docstring);
            callers gate them with ``valid`` / the reliability mask. Same
            dtype/device as the input latent.
        valid: ``(1, 1, h, w)`` (or ``(N, 1, h, w)``) boolean mask, ``True`` where
            the bilinear sample came fully from inside the source latent and
            ``False`` for out-of-border cells.
    """

    image: "torch.Tensor"
    valid: "torch.Tensor"


def flow_to_latent(
    flow_px: "torch.Tensor",
    latent_hw: tuple[int, int],
    *,
    image_hw: tuple[int, int] | None = None,
) -> "torch.Tensor":
    """Resize a pixel-resolution flow to the latent grid and rescale magnitudes.

    Performs the two-step pixel→latent conversion described in the module
    docstring (the #1 correctness trap):

    1. bilinearly resize the flow field from image resolution ``(H, W)`` to the
       latent resolution ``(h, w)``; and
    2. rescale the ``u`` (x) and ``v`` (y) displacement components by the
       **per-axis** ratios ``w / W`` and ``h / H`` respectively, so the
       displacements are expressed in *latent cells* rather than pixels.

    This mirrors :func:`artvid.flow.raft._postprocess_flow` exactly (resize +
    per-axis vector rescale), just in the downscaling direction.

    Args:
        flow_px: ``(2, H, W)`` or ``(N, 2, H, W)`` flow in ``(u, v) = (dx, dy)``
            channel order, in absolute pixel-displacement units at image
            resolution.
        latent_hw: Target latent spatial size ``(h, w)``.
        image_hw: Source image spatial size ``(H, W)`` used for the magnitude
            rescale. Defaults to the flow's own spatial size, which is the
            normal case (the flow was computed at image resolution). Pass this
            explicitly only if the flow has already been cropped/padded relative
            to the resolution its displacements are measured in.

    Returns:
        ``(2, h, w)`` (or ``(N, 2, h, w)``) flow whose displacements are in
        latent cells, ready to feed to :func:`artvid.flow.warp.flow_to_grid` at
        latent resolution.

    Raises:
        ValueError: If ``flow_px`` does not have 2 flow channels.
    """
    import torch.nn.functional as F

    was_3d = flow_px.dim() == 3
    flow_b = flow_px.unsqueeze(0) if was_3d else flow_px
    if flow_b.dim() != 4 or flow_b.shape[1] != 2:
        raise ValueError(
            f"flow must have shape (2, H, W) or (N, 2, H, W); got {tuple(flow_px.shape)}."
        )

    src_h, src_w = (image_hw if image_hw is not None else flow_b.shape[-2:])
    h, w = latent_hw

    # (1) Spatial resize of the flow FIELD. We use align_corners=False to match
    # how artvid.flow.raft._postprocess_flow resizes flow fields. (The separate
    # align_corners used to BUILD the sampling grid in flow_to_grid is the
    # warp.warp_image convention — True by default — and is independent of this
    # field-resize choice; do not conflate the two.)
    flow_b = F.interpolate(
        flow_b, size=(h, w), mode="bilinear", align_corners=False
    )

    # (2) Per-axis magnitude rescale: u (channel 0, x/width) by w/W, v (channel
    # 1, y/height) by h/H. Per-axis (not a single 1/f) keeps the warp exact when
    # H/W are not exact multiples of the VAE factor. Operate out-of-place so we
    # never mutate the caller's flow tensor.
    scale_x = float(w) / float(src_w)
    scale_y = float(h) / float(src_h)
    flow_b = flow_b.clone()
    flow_b[:, 0, :, :] = flow_b[:, 0, :, :] * scale_x
    flow_b[:, 1, :, :] = flow_b[:, 1, :, :] * scale_y

    return flow_b[0] if was_3d else flow_b


def warp_latent(
    z_prev: "torch.Tensor",
    backward_flow_px: "torch.Tensor",
    *,
    vae_factor: int = 8,
    image_hw: tuple[int, int] | None = None,
    align_corners: bool = True,
) -> LatentWarpResult:
    """Backward-warp a previous-frame latent into the current frame's grid.

    Reuses :func:`artvid.flow.warp.flow_to_grid` (the validated Phase 1 grid
    math) at *latent* resolution, after converting the pixel-resolution flow to
    latent cells with :func:`flow_to_latent`. Then ``grid_sample``s the latent
    with zero padding and builds a companion ones-mask ``valid`` exactly as
    :func:`artvid.flow.warp.warp_image` does — but WITHOUT the VGG mean fill,
    which is a pixel-space constant meaningless for latents.

    Args:
        z_prev: Previous frame's latent, ``(1, C, h, w)`` or ``(N, C, h, w)``
            (the latent to pull forward into the current frame, in the current
            frame's denoise loop). ``C`` is the VAE latent channel count (4 for
            SD/SDXL); it is treated as opaque.
        backward_flow_px: RAFT **backward** flow (current → previous), ``(2, H, W)``
            or ``(N, 2, H, W)`` in ``(u, v) = (dx, dy)`` pixel units at image
            resolution. This is the flow that pulls the previous frame INTO the
            current frame (see module docstring on backward-warp semantics).
        vae_factor: VAE downsample factor ``f`` (default 8). Used only as a
            fallback to infer ``image_hw`` when it is not given (``H = h * f``,
            ``W = w * f``); the actual magnitude rescale always uses the exact
            per-axis ratios ``h / H`` and ``w / W`` (see :func:`flow_to_latent`),
            so a slightly inexact ``vae_factor`` only affects the inferred image
            size, not the rescale exactness.
        image_hw: Image resolution ``(H, W)`` the flow was computed at. Strongly
            recommended; defaults to ``(h * vae_factor, w * vae_factor)``.
        align_corners: ``grid_sample`` / ``flow_to_grid`` normalization
            convention. Must match what Phase 1 uses elsewhere; default ``True``
            (treats displacements as pixel-centre offsets), matching
            :func:`artvid.flow.warp.warp_image`.

    Returns:
        :class:`LatentWarpResult` ``(image, valid)`` at latent resolution. Output
        rank matches the input latent rank (unbatched 3-D input is NOT supported;
        latents are always ``(N, C, h, w)``).

    Raises:
        ValueError: If ``z_prev`` is not 4-D, or shapes/batch sizes are
            incompatible with ``backward_flow_px``.
    """
    import torch
    import torch.nn.functional as F

    if z_prev.dim() != 4:
        raise ValueError(
            "z_prev must have shape (N, C, h, w) (a 4-D latent); got "
            f"{tuple(z_prev.shape)}."
        )

    n, c, h, w = z_prev.shape

    flow_b = (
        backward_flow_px.unsqueeze(0)
        if backward_flow_px.dim() == 3
        else backward_flow_px
    )
    if flow_b.dim() != 4 or flow_b.shape[1] != 2:
        raise ValueError(
            "backward_flow_px must have shape (2, H, W) or (N, 2, H, W); got "
            f"{tuple(backward_flow_px.shape)}."
        )
    if flow_b.shape[0] not in (1, n):
        raise ValueError(
            "backward_flow_px batch size must be 1 or match z_prev's "
            f"({n}); got {flow_b.shape[0]}."
        )
    if flow_b.shape[0] == 1 and n > 1:
        flow_b = flow_b.expand(n, -1, -1, -1)

    # Resolve the image resolution the flow's displacements are measured at.
    if image_hw is None:
        image_hw = (h * vae_factor, w * vae_factor)

    # (1)+(2) pixel-resolution flow -> latent-cell flow at (h, w).
    flow_lat = flow_to_latent(
        flow_b.to(dtype=z_prev.dtype), (h, w), image_hw=image_hw
    )

    # (3) Build the sampling grid at LATENT resolution (reuse Phase 1 grid math)
    # and bilinearly backward-warp the latent with zero padding.
    grid = _warp.flow_to_grid(flow_lat, align_corners=align_corners)
    warped = F.grid_sample(
        z_prev,
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=align_corners,
    )

    # (4) Companion ones-mask warp -> out-of-border / disoccluded latent cells.
    # Identical logic to warp_image's `mask`/`valid`, but on a single channel and
    # with NO VGG mean fill (zeros are kept; the mask gates downstream use).
    ones = torch.ones((n, 1, h, w), device=z_prev.device, dtype=z_prev.dtype)
    mask = F.grid_sample(
        ones,
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=align_corners,
    )
    valid = mask >= (1.0 - 1e-6)  # (N, 1, h, w) bool

    return LatentWarpResult(image=warped, valid=valid)


def latent_reliability(
    forward_flow_px: "torch.Tensor",
    backward_flow_px: "torch.Tensor",
    valid_latent: "torch.Tensor",
    *,
    latent_hw: tuple[int, int],
    smooth_sigma: float = _consistency._DEFAULT_SMOOTH_SIGMA,
    check_motion_boundaries: bool = True,
    gamma: float = 2.0,
) -> "torch.Tensor":
    """Per-latent-cell reliability mask for the warped latent, in ``[0, 1]``.

    Computes the forward-backward consistency mask at FULL pixel resolution
    (reusing the validated :func:`artvid.flow.consistency.consistency_mask`),
    then conservatively downsamples it to the latent grid, erodes it, and ANDs in
    the latent warp's ``valid`` (out-of-border) mask. The result is the per-cell
    weight used by the engine to (a) blend the warped previous latent into the
    init and (b) fuse it between denoising steps in reliable regions only.

    The argument order ``consistency_mask(backward, forward)`` validates the
    **backward** warp (the one ``warp_latent`` performs), matching
    ``cli.cmd_flow``'s ``rel_back``.

    Conservative downsample + erosion (the most important tuning knob, §2.4):
    we area-average the full-resolution reliability into each latent cell
    (``adaptive_avg_pool2d`` to ``(h, w)`` so non-divisible sizes are handled),
    then raise it to ``gamma`` (``gamma > 1`` erodes: a cell stays near 1 only if
    *most* of its pixels are reliable; partially-occluded cells are pushed toward
    0). Finally we multiply by ``valid_latent`` so out-of-border cells are 0.

    Args:
        forward_flow_px: ``(2, H, W)`` (or ``(1, 2, H, W)``) forward flow
            (previous → current), pixel units. (Cross-check for the consistency
            test.)
        backward_flow_px: ``(2, H, W)`` backward flow (current → previous), pixel
            units — the flow ``warp_latent`` warps with. Primary argument to
            ``consistency_mask``.
        valid_latent: ``(1, 1, h, w)`` (or ``(N, 1, h, w)`` / ``(h, w)``) boolean
            (or float) out-of-border mask from :func:`warp_latent`.
        latent_hw: Target latent spatial size ``(h, w)``.
        smooth_sigma: Gaussian sigma passed through to ``consistency_mask`` (px).
            Default matches the Phase 1 default (0.8).
        check_motion_boundaries: Forwarded to ``consistency_mask`` (default True).
        gamma: Erosion exponent applied to the area-downsampled reliability.
            ``> 1`` erodes (more conservative), ``1`` = plain area mean, ``< 1``
            dilates. Default 2.0.

    Returns:
        ``(N, 1, h, w)`` float32 reliability in ``[0, 1]`` on the inputs' device,
        where ``N`` matches ``valid_latent`` (1 if ``valid_latent`` was 2-D).
    """
    import torch
    import torch.nn.functional as F

    h, w = latent_hw

    # (1) Full-resolution forward-backward reliability. Argument order validates
    # the BACKWARD warp (== cli.cmd_flow rel_back).
    rel_px = _consistency.consistency_mask(
        backward_flow_px,
        forward_flow_px,
        smooth_sigma=smooth_sigma,
        check_motion_boundaries=check_motion_boundaries,
    )  # (H, W) float32 in [0, 1]

    # (2) Conservative area downsample to the latent grid. adaptive_avg_pool2d
    # handles non-divisible (H, W) -> (h, w); for exactly divisible sizes it is
    # equivalent to avg_pool2d with kernel=stride=f.
    rel_lat = F.adaptive_avg_pool2d(
        rel_px.unsqueeze(0).unsqueeze(0), output_size=(h, w)
    )  # (1, 1, h, w)

    # (3) Erode: gamma > 1 pushes partially-reliable cells toward 0.
    rel_lat = rel_lat.clamp(0.0, 1.0)
    if gamma != 1.0:
        rel_lat = rel_lat ** float(gamma)

    # (4) AND in the out-of-border validity from the latent warp.
    valid = valid_latent
    if valid.dim() == 2:  # (h, w) -> (1, 1, h, w)
        valid = valid.unsqueeze(0).unsqueeze(0)
    elif valid.dim() == 3:  # (1, h, w) or (C, h, w) -> add batch
        valid = valid.unsqueeze(0)
    valid_f = valid.to(dtype=rel_lat.dtype)

    # Broadcast rel_lat (1,1,h,w) against valid (N,1,h,w) so the output batch
    # follows valid_latent.
    rel = rel_lat * valid_f
    return rel.to(dtype=torch.float32)


def combine_latent_reliability(
    reliabilities: "list[torch.Tensor]",
    method: str = "closestFirst",
    *,
    invert: bool = False,
) -> "list[torch.Tensor]":
    """Combine several latent reliability masks (prev + anchor) into weights.

    Thin latent-space wrapper over
    :func:`artvid.flow.consistency.combine_longterm_weights` for the optional
    long-term / anchor consistency path (§2.6): warp BOTH the previous latent and
    an anchor (keyframe) latent into the current frame, compute a
    :func:`latent_reliability` for each, and pass them here
    **closest-previous-frame first** to obtain the per-frame blend weights. This
    is the Phase 1 long-term-weight scheme (``processFlowWeights``) lifted to the
    latent grid; default ``method="closestFirst"`` gives priority to the closest
    (previous) frame and lets the anchor only claim cells the previous frame does
    not reliably see.

    Args:
        reliabilities: Reliability masks (each ``(N, 1, h, w)`` from
            :func:`latent_reliability`), ordered closest-previous-frame first.
        method: ``'closestFirst'`` | ``'normalize'`` | ``'none'`` — see
            :func:`artvid.flow.consistency.combine_longterm_weights`.
        invert: Forwarded; ``True`` if the masks encode *un*reliability.

    Returns:
        A new list of combined latent weight tensors, same length/shapes.
    """
    # combine_longterm_weights is shape-agnostic (broadcasting tensor ops); the
    # (N, 1, h, w) latent masks flow through unchanged.
    return _consistency.combine_longterm_weights(
        reliabilities, method, invert=invert
    )


def combine_anchor_reliability(
    prev_rel: "torch.Tensor",
    anchor_rel: "torch.Tensor",
    *,
    method: str = "closestFirst",
) -> "tuple[torch.Tensor, torch.Tensor]":
    """Combine the previous-frame and anchor-keyframe reliabilities (§2.6).

    Two-reference convenience wrapper over :func:`combine_latent_reliability`
    (and therefore over :func:`artvid.flow.consistency.combine_longterm_weights`)
    for the common long-term-consistency case: the deepened video loop warps
    BOTH the previous stylized latent and a fixed anchor (frame-0 / keyframe)
    latent into the current frame, computes a :func:`latent_reliability` mask for
    each, and calls this to obtain the per-reference blend weights used when the
    engine fuses the two warped references into the init / per-step latent.

    **Ordering (closest-first, REQUIRED).** The references must be supplied in
    increasing temporal distance from the current frame: ``prev_rel`` is the
    immediately-previous frame (closest), ``anchor_rel`` is the farther keyframe.
    The default ``method="closestFirst"`` then gives the previous frame priority
    — the anchor only claims latent cells the previous frame does *not* reliably
    see (e.g. disocclusions, regions that have drifted) — which is exactly the
    Phase 1 long-term-weight scheme (``processFlowWeights``) lifted to the latent
    grid. Returning a fixed ``(prev_weight, anchor_weight)`` tuple (rather than a
    list) makes the caller's unpacking unambiguous about which weight is which.

    Args:
        prev_rel: ``(N, 1, h, w)`` reliability of the warped *previous-frame*
            latent (the closest reference), from :func:`latent_reliability`.
        anchor_rel: ``(N, 1, h, w)`` reliability of the warped *anchor* latent
            (the farther reference), from :func:`latent_reliability`.
        method: ``'closestFirst'`` (default) | ``'normalize'`` | ``'none'`` — see
            :func:`artvid.flow.consistency.combine_longterm_weights`.
            ``'closestFirst'`` prioritizes ``prev_rel``; ``'normalize'`` instead
            splits each cell proportionally between the two references.

    Returns:
        ``(prev_weight, anchor_weight)`` — the combined per-cell weights for the
        previous and anchor references respectively, same shapes as the inputs.

    Notes:
        TODO(tuning): on M5 Max, verify on long clips whether anchor warp stays
        reliable enough over many frames under ``closestFirst`` (the anchor flow
        is composed/accumulated and may degrade), or whether keyframe
        re-anchoring (Rerender-style keyframe + interpolation) is needed (§2.6).
        Also compare ``closestFirst`` vs ``normalize`` for drift vs ghosting.
    """
    prev_weight, anchor_weight = combine_latent_reliability(
        [prev_rel, anchor_rel], method=method
    )
    return prev_weight, anchor_weight


def warp_previous_pixel(
    prev_rgb: "torch.Tensor",
    flow_px: "torch.Tensor",
    *,
    vae_factor: int = 8,
) -> "_warp.WarpResult":
    """Backward-warp the previous frame in PIXEL space (``warp_space='pixel'``).

    The alternative to :func:`warp_latent` for ``config.warp_space == 'pixel'``.
    Instead of warping in the VAE latent grid, this warps the previous *decoded*
    stylized RGB frame at FULL image resolution by directly reusing the validated
    Phase 1 :func:`artvid.flow.warp.warp_image`, and returns the warped RGB ready
    to be re-encoded by the engine (``DiffusionEngine.encode``) into a latent.
    **This function does NOT encode** — encoding is the engine's responsibility
    (it owns the VAE); we only produce the pixel-space warped frame + valid mask.

    Pixel vs latent warp — the trade-off (§2.5 / 3.2):

    * **Accuracy.** Warping at full resolution applies the flow at its native
      pixel scale and lets ``grid_sample`` interpolate the high-frequency RGB
      detail directly, then the VAE re-encodes a coherent image. The latent warp
      (:func:`warp_latent`) instead downsamples the flow to the coarse latent
      grid (``H/f × W/f``) and warps 4-channel latents whose cells each cover an
      ``f×f`` pixel block, so sub-block motion and fine structure are blurred /
      quantized. The pixel path is therefore the more accurate, less "smeared"
      option, especially for small or fast motions.
    * **Cost.** The pixel path needs an EXTRA VAE decode (to get ``prev_rgb``
      from the previous latent) *and* an extra VAE encode (of this warped RGB
      back to a latent) per frame — two full VAE passes the latent warp avoids
      entirely (it stays in latent space). On a long clip that is the dominant
      added cost. ``vae_factor`` is accepted only for signature symmetry with
      :func:`warp_latent` / call-site uniformity; the pixel warp is performed at
      ``prev_rgb``'s own resolution and does not use it.

    The warp is a **backward** warp: pass the RAFT *backward* flow (current →
    previous), matching :func:`artvid.flow.warp.warp_image`'s convention, so the
    previous frame's content is pulled into the current frame's coordinates.
    Disoccluded / out-of-border pixels are left at the ``grid_sample`` zero pad
    (``fill=None``) rather than the VGG mean: the companion reliability/``valid``
    mask gates them downstream, and we do not want a constant RGB bias leaking
    into the VAE encode.

    Args:
        prev_rgb: Previous stylized frame, RGB float in ``[0, 1]``, ``(3, H, W)``
            or ``(N, 3, H, W)`` — the engine's VAE-decoded previous output.
        flow_px: RAFT **backward** flow (current → previous), ``(2, H, W)`` or
            ``(N, 2, H, W)`` in ``(u, v) = (dx, dy)`` pixel units at ``prev_rgb``
            resolution.
        vae_factor: Accepted for call-site symmetry with :func:`warp_latent`;
            unused here (the warp runs at full pixel resolution). See the cost
            note above.

    Returns:
        :class:`artvid.flow.warp.WarpResult` ``(image, valid)`` in PIXEL space:
        ``image`` is the warped RGB ``[0, 1]`` (zero-padded outside the source),
        ``valid`` the ``(N, 1, H, W)`` (or ``(1, H, W)``) in-border mask. The
        engine encodes ``image`` to a latent; reliability for fusion is computed
        from the flows + this ``valid`` via :func:`latent_reliability` after the
        encode (the latent-grid mask), exactly as in the latent-warp path.
    """
    # Full-resolution backward warp via the Phase 1 pixel warp. fill=None keeps
    # the zero pad (no VGG mean constant) so out-of-border pixels stay neutral
    # for the subsequent VAE encode; the valid mask gates them downstream.
    del vae_factor  # accepted for symmetry; see docstring.
    return _warp.warp_image(prev_rgb, flow_px, fill=None)
