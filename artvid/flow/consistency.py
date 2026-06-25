"""Forward-backward optical-flow consistency / occlusion reliability mask.

This module replaces two pieces of the legacy pipeline:

* the standalone C++ ``consistencyChecker`` binary
  (``consistencyChecker/consistencyChecker.cpp``), which read a forward and a
  backward ``.flo`` file and wrote a per-pixel reliability ``.pgm`` in
  ``[0, 255]``; and
* ``processFlowWeights`` (``artistic_video.lua:307-329``), which combined the
  reliability masks of several previous frames into long-term temporal weights
  using the ``normalize`` / ``closestFirst`` schemes (and an optional invert).

Both are ported to pure PyTorch tensor ops here so the whole flow path runs
on-device (see ``docs/02-migration-map.md`` section 1: ``consistencyChecker/``
→ ``flow/consistency.py``, and ``artistic_video.lua:307-329`` →
``flow/consistency.py:combine_longterm_weights``).

Occlusion / consistency criterion
----------------------------------
Given the forward flow ``w = (u, v)`` (mapping pixel ``p`` in frame ``t`` to
frame ``t+1``) and the backward flow ``w' = (u', v')`` (frame ``t+1`` → ``t``),
a pixel ``p`` is *reliable* when following ``w`` then ``w'`` lands back close
to ``p`` (the forward-backward consistency check of Sundaram et al. 2010, as
used by Ruder et al. 2016). Concretely, with ``p' = p + w(p)`` and
``w'(p')`` bilinearly sampled at the sub-pixel location ``p'``:

    |w(p) + w'(p')|^2  >  0.01 * (|w(p)|^2 + |w'(p')|^2)  +  0.5
        =>  occluded / inconsistent  (reliability 0)

The right-hand-side is a tolerance that grows with the flow magnitude (1% of
the summed squared magnitudes) plus a 0.5 px floor, so large but consistent
motions are not penalised. Pixels whose forward flow leaves the image domain
are also marked unreliable.

A second test flags **motion boundaries**: where the spatial gradient of the
forward flow is large,

    |∇w|^2  >  0.01 * |w(p)|^2  +  0.002   =>  motion boundary (reliability 0)

These reproduce the exact thresholds in ``consistencyChecker.cpp:77,83``.

As in the C++ original, occluded pixels are seeded to a *negative* value
(``-1`` here, ``-255`` there) before optional Gaussian smoothing so the
smoothing erodes reliability outward across the occlusion edge; the result is
then clipped to ``[0, 1]``. Motion-boundary pixels are seeded to ``0``.

Output scale
------------
Reliability is returned in ``[0, 1]`` (the C++ wrote ``[0, 255]`` PGMs which
``image.load`` in the Lua then rescaled to ``[0, 1]``). This is the value
consumed as the temporal-loss weight in ``losses/temporal.py`` (which then
takes its ``sqrt`` per the parity note in ``docs/01-architecture.md`` §losses).

This module uses only framework-agnostic ``torch`` tensor ops (no MPS-specific
calls); device placement follows whatever device the input flows live on.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

if TYPE_CHECKING:  # pragma: no cover - typing only, torch is optional at import
    import torch


# Tolerance constants, ported verbatim from consistencyChecker.cpp.
#: Relative tolerance on the forward-backward round-trip error (cpp:77).
_FB_REL_TOL = 0.01
#: Absolute (px^2) floor on the round-trip error tolerance (cpp:77).
_FB_ABS_TOL = 0.5
#: Relative tolerance for the motion-boundary test (cpp:83).
_MB_REL_TOL = 0.01
#: Absolute floor for the motion-boundary test (cpp:83).
_MB_ABS_TOL = 0.002
#: Default Gaussian smoothing sigma applied to the reliability map
#: (``SMOOTH_STRENGH`` in consistencyChecker.cpp:15).
_DEFAULT_SMOOTH_SIGMA = 0.8
#: Seed value for occluded pixels before smoothing (negative so smoothing
#: erodes outward), scaled to the [0, 1] output range (cpp uses -255).
_OCCLUDED_SEED = -1.0
#: Seed value for motion-boundary pixels (``MOTION_BOUNDARIE_VALUE`` cpp:12).
_MOTION_BOUNDARY_SEED = 0.0


__all__ = [
    "consistency_mask",
    "combine_longterm_weights",
]


def _as_2hw(flow: "torch.Tensor") -> "torch.Tensor":
    """Validate and return a ``(2, H, W)`` float flow tensor.

    Accepts ``(2, H, W)`` or ``(1, 2, H, W)``; channel 0 is ``u`` (dx, along
    width / x) and channel 1 is ``v`` (dy, along height / y), matching the
    ``(u, v)`` convention of ``io/flow_io.py`` and ``flow/raft.py``.
    """
    if flow.dim() == 4 and flow.shape[0] == 1:
        flow = flow[0]
    if flow.dim() != 3 or flow.shape[0] != 2:
        raise ValueError(
            f"flow must be (2, H, W) or (1, 2, H, W); got {tuple(flow.shape)}"
        )
    return flow


def _flow_gradient_sq(flow: "torch.Tensor") -> "torch.Tensor":
    """Squared spatial-gradient magnitude of the flow, per pixel.

    Reproduces the ``motionEdge`` accumulation in
    ``consistencyChecker.cpp:43-54``: sum over both flow components of the
    squared x-derivative and squared y-derivative. The C++ used a 3-tap
    central-difference derivative filter (``CDerivative<float>(3)``); we use the
    plain central difference ``(f[i+1] - f[i-1]) / 2`` with replicate padding at
    the borders, which is the same operator.

    Args:
        flow: ``(2, H, W)`` flow.

    Returns:
        ``(H, W)`` tensor of squared gradient magnitude.
    """
    import torch
    import torch.nn.functional as F

    # Pad with edge replication so border derivatives are defined (the C++
    # filter handles borders implicitly; replicate is the closest analogue and
    # keeps boundary gradients small rather than spuriously large).
    padded = F.pad(flow.unsqueeze(0), (1, 1, 1, 1), mode="replicate")[0]
    # Central differences along width (x) and height (y).
    dx = (padded[:, 1:-1, 2:] - padded[:, 1:-1, :-2]) * 0.5  # (2, H, W)
    dy = (padded[:, 2:, 1:-1] - padded[:, :-2, 1:-1]) * 0.5  # (2, H, W)
    grad_sq = (dx * dx).sum(dim=0) + (dy * dy).sum(dim=0)
    return grad_sq


def _sample_bilinear(flow: "torch.Tensor", x: "torch.Tensor", y: "torch.Tensor"):
    """Bilinearly sample a ``(2, H, W)`` flow at float coordinates ``(x, y)``.

    Mirrors the manual bilinear interpolation in
    ``consistencyChecker.cpp:60-72`` (``floor`` + four-corner blend). Returns the
    sampled ``(u, v)`` plus a boolean mask of in-bounds samples (the C++ marks
    out-of-bounds forward targets as fully unreliable, cpp:64-65).

    Args:
        flow: ``(2, H, W)`` flow to sample.
        x: ``(H, W)`` float x (width) coordinates to sample at.
        y: ``(H, W)`` float y (height) coordinates to sample at.

    Returns:
        ``(u, v, valid)`` each ``(H, W)``; ``valid`` is a bool tensor.
    """
    import torch

    _, height, width = flow.shape

    x1 = torch.floor(x)
    y1 = torch.floor(y)
    x2 = x1 + 1
    y2 = y1 + 1

    # In-bounds iff the whole 2x2 interpolation stencil lies inside the image,
    # matching the C++ guard ``x1<0 || x2>=xSize || y1<0 || y2>=ySize``.
    valid = (x1 >= 0) & (x2 <= width - 1) & (y1 >= 0) & (y2 <= height - 1)

    alpha_x = x - x1
    alpha_y = y - y1

    # Clamp indices so gather is always legal; invalid pixels are masked out via
    # ``valid`` afterwards, so their sampled value is irrelevant.
    x1c = x1.clamp(0, width - 1).long()
    x2c = x2.clamp(0, width - 1).long()
    y1c = y1.clamp(0, height - 1).long()
    y2c = y2.clamp(0, height - 1).long()

    def bilerp(channel: "torch.Tensor") -> "torch.Tensor":
        f11 = channel[y1c, x1c]
        f21 = channel[y1c, x2c]
        f12 = channel[y2c, x1c]
        f22 = channel[y2c, x2c]
        top = (1.0 - alpha_x) * f11 + alpha_x * f21
        bot = (1.0 - alpha_x) * f12 + alpha_x * f22
        return (1.0 - alpha_y) * top + alpha_y * bot

    u = bilerp(flow[0])
    v = bilerp(flow[1])
    return u, v, valid


def _gaussian_smooth(img: "torch.Tensor", sigma: float) -> "torch.Tensor":
    """Apply separable Gaussian smoothing to an ``(H, W)`` map.

    Replaces the ``CSmooth<float>(SMOOTH_STRENGH, 2.0f)`` filter applied to the
    reliability map (``consistencyChecker.cpp:106-108``). The C++ used a
    truncation radius of ``2.0 * sigma``; we build a matching truncated kernel.
    """
    import torch
    import torch.nn.functional as F

    if sigma <= 0:
        return img

    radius = max(1, int(round(2.0 * sigma)))
    coords = torch.arange(
        -radius, radius + 1, dtype=img.dtype, device=img.device
    )
    kernel_1d = torch.exp(-(coords * coords) / (2.0 * sigma * sigma))
    kernel_1d = kernel_1d / kernel_1d.sum()

    x = img.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
    # Horizontal then vertical pass (separable), with replicate padding.
    kx = kernel_1d.view(1, 1, 1, -1)
    ky = kernel_1d.view(1, 1, -1, 1)
    x = F.pad(x, (radius, radius, 0, 0), mode="replicate")
    x = F.conv2d(x, kx)
    x = F.pad(x, (0, 0, radius, radius), mode="replicate")
    x = F.conv2d(x, ky)
    return x[0, 0]


def consistency_mask(
    forward_flow: "torch.Tensor",
    backward_flow: "torch.Tensor",
    *,
    smooth_sigma: float = _DEFAULT_SMOOTH_SIGMA,
    check_motion_boundaries: bool = True,
) -> "torch.Tensor":
    """Compute a forward-backward consistency reliability mask.

    Ports ``checkConsistency`` + the smoothing/clipping in ``main``
    (``consistencyChecker.cpp:40-112``). The returned mask is high (``~1``)
    where the forward flow is consistent with the backward flow and low (``0``)
    where it is inconsistent (occlusions, disocclusions) or — optionally — on
    motion boundaries.

    The occlusion criterion is (see module docstring for the derivation): with
    ``p' = p + forward(p)`` and ``w' = backward(p')`` bilinearly sampled,

        |forward(p) + w'|^2 > 0.01 * (|forward(p)|^2 + |w'|^2) + 0.5  →  occluded.

    Args:
        forward_flow: ``(2, H, W)`` (or ``(1, 2, H, W)``) flow from frame ``t``
            to frame ``t+1``, in ``(u, v)`` pixel order (channel 0 = dx/width,
            channel 1 = dy/height).
        backward_flow: ``(2, H, W)`` flow from frame ``t+1`` back to ``t``, same
            convention. Must match ``forward_flow``'s spatial size.
        smooth_sigma: Gaussian sigma for post-smoothing the mask (px). ``0``
            disables smoothing. Defaults to ``0.8`` (the C++ ``SMOOTH_STRENGH``).
        check_motion_boundaries: If ``True`` (default, matching the C++), also
            zero out reliability on motion boundaries (large flow gradient).

    Returns:
        ``(H, W)`` float32 reliability mask in ``[0, 1]`` on the same device as
        the inputs. ``1`` = fully reliable, ``0`` = occluded / inconsistent.
    """
    import torch

    fwd = _as_2hw(forward_flow)
    bwd = _as_2hw(backward_flow)
    if fwd.shape[1:] != bwd.shape[1:]:
        raise ValueError(
            "forward and backward flow must share spatial size; got "
            f"{tuple(fwd.shape)} vs {tuple(bwd.shape)}"
        )

    fwd = fwd.to(dtype=torch.float32)
    bwd = bwd.to(dtype=torch.float32)
    _, height, width = fwd.shape
    device = fwd.device

    # Per-pixel integer coordinate grids (x = width, y = height).
    ys, xs = torch.meshgrid(
        torch.arange(height, device=device, dtype=torch.float32),
        torch.arange(width, device=device, dtype=torch.float32),
        indexing="ij",
    )

    u = fwd[0]
    v = fwd[1]

    # Target location p' = p + forward(p) in frame t+1.
    bx = xs + u
    by = ys + v

    # Backward flow sampled at p' (sub-pixel, bilinear). Out-of-bounds targets
    # are flagged and become fully unreliable.
    u_back, v_back, in_bounds = _sample_bilinear(bwd, bx, by)

    # Round-trip landing point: p' + backward(p'). Squared distance to p.
    cx = bx + u_back
    cy = by + v_back
    fb_err_sq = (cx - xs) ** 2 + (cy - ys) ** 2

    fwd_mag_sq = u * u + v * v
    back_mag_sq = u_back * u_back + v_back * v_back

    # Forward-backward consistency test (cpp:77).
    occluded = fb_err_sq >= (
        _FB_REL_TOL * (fwd_mag_sq + back_mag_sq) + _FB_ABS_TOL
    )
    # Out-of-bounds forward targets are unreliable regardless (cpp:64-65).
    occluded = occluded | (~in_bounds)

    # Seed reliability: 1 everywhere, occluded pixels to a negative value so the
    # subsequent Gaussian smoothing erodes reliability outward across the edge
    # (the C++ uses -255 on a [0,255] scale; we use -1 on a [0,1] scale).
    reliable = torch.ones((height, width), dtype=torch.float32, device=device)
    reliable = torch.where(
        occluded,
        torch.full_like(reliable, _OCCLUDED_SEED),
        reliable,
    )

    if check_motion_boundaries:
        # Motion-boundary test (cpp:83). Only applied where not already occluded
        # (the C++ ``continue`` skips the boundary test on occluded pixels).
        grad_sq = _flow_gradient_sq(fwd)
        motion_boundary = grad_sq > (_MB_REL_TOL * fwd_mag_sq + _MB_ABS_TOL)
        motion_boundary = motion_boundary & (~occluded)
        reliable = torch.where(
            motion_boundary,
            torch.full_like(reliable, _MOTION_BOUNDARY_SEED),
            reliable,
        )

    if smooth_sigma > 0:
        reliable = _gaussian_smooth(reliable, smooth_sigma)

    # Clip to the valid [0, 1] range (cpp:110 clips to [0, 255]).
    reliable = reliable.clamp(0.0, 1.0)
    return reliable


def combine_longterm_weights(
    weights: Sequence["torch.Tensor"],
    method: str = "closestFirst",
    *,
    invert: bool = False,
) -> list["torch.Tensor"]:
    """Combine per-frame reliability masks into long-term temporal weights.

    Ports ``processFlowWeights`` (``artistic_video.lua:307-329``). Given the
    reliability masks for several previous frames — ordered from the closest
    previous frame to the farthest, matching the legacy ``J`` /
    ``flow_relative_indices`` ordering (``artistic_video.lua:160-215``) — this
    produces the weight each previous frame's warped contribution receives in
    the temporal loss.

    Args:
        weights: Sequence of reliability masks, each broadcastable but typically
            ``(H, W)`` or ``(1, H, W)`` / ``(3, H, W)``, ordered
            **closest-previous-frame first**. Values are expected in ``[0, 1]``.
        method: Combination scheme:

            * ``'normalize'`` — divide each weight by the per-pixel sum of all
              weights (clamped so the sum is at least 1), so the weights sum to
              at most 1 per pixel (``artistic_video.lua:313-319``).
            * ``'closestFirst'`` — give priority to the closest frame: from each
              farther frame's weight subtract every closer frame's weight, then
              clamp to ``>= 0``, so each pixel is "claimed" by the closest frame
              that reliably sees it (``artistic_video.lua:320-327``).
            * ``'none'`` — leave the (optionally inverted) weights unchanged.
        invert: If ``True``, replace each weight ``x`` with ``1 - x`` before
            combining (``artistic_video.lua:308-311``; the legacy
            ``-invert_flowWeights`` option, used when the loaded masks encode
            *un*reliability).

    Returns:
        A new list of combined weight tensors, same length and shapes as the
        input. The inputs are not modified in place.

    Raises:
        ValueError: If ``weights`` is empty or ``method`` is unknown.
    """
    import torch

    if len(weights) == 0:
        raise ValueError("weights must contain at least one reliability mask")

    # Work on float clones so we never mutate the caller's tensors (the Lua
    # mutated the table in place; we deliberately do not).
    combined = [w.to(dtype=torch.float32).clone() for w in weights]

    if invert:
        combined = [1.0 - w for w in combined]

    if method == "none":
        return combined

    if method == "normalize":
        # Sum across all frames, floor at 1, divide -> per-pixel sum <= 1.
        total = combined[0].clone()
        for w in combined[1:]:
            total = total + w
        total = torch.clamp(total, min=1.0)
        return [w / total for w in combined]

    if method == "closestFirst":
        # weights[0] is the closest previous frame and is left untouched. For
        # each farther frame j, subtract every closer frame's (already combined)
        # weight, then clamp at 0. This mirrors the in-place Lua loop
        #   for j=2..n: for k=1..j-1: w[j] -= w[j-k]; w[j]:cmax(0)
        # Because the Lua subtracts the *running* (partially updated) closer
        # weights, we read the closer entries from the list as we go.
        result = [combined[0]]
        for j in range(1, len(combined)):
            acc = combined[j].clone()
            for k in range(j):
                acc = acc - result[k]
            acc = torch.clamp(acc, min=0.0)
            result.append(acc)
        return result

    raise ValueError(
        f"Unknown combine method {method!r}; "
        "expected 'normalize', 'closestFirst', or 'none'."
    )
