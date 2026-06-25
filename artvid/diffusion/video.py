"""Per-frame diffusion video stylization with latent temporal consistency.

What this module does
---------------------
Implements :func:`stylize_video_diffusion`, the §2.5 per-frame loop of
``docs/07-phase2-design.md``. It is the Phase 2 analogue of the Phase 1
single-pass loop (:func:`artvid.pipeline.singlepass.stylize_video`): it iterates
the *content* frames in order and stylizes each with the diffusion engine,
grafting the 2016 optical-flow temporal-consistency idea into VAE **latent**
space so the result is temporally stable.

The algorithm, per frame ``t``:

* **first frame** (no predecessor) — plain ControlNet + IP-Adapter generation
  (the anchor): denoise from pure noise, carry the resulting latent forward.
* **subsequent frames** — compute RAFT optical flow between the *content* frames
  (motion is a property of the scene, not the stylization), backward-warp the
  **previous frame's latent** into the current frame's grid via
  :func:`artvid.diffusion.latent_warp.warp_latent`, compute a latent-grid
  reliability mask via :func:`artvid.diffusion.latent_warp.latent_reliability`,
  and feed both into :meth:`DiffusionEngine.denoise_frame`:

    1. **warped-latent init** (mechanism 1): renoise the warped previous latent
       to the start timestep and start denoising from it in reliable regions.
    2. **per-step warped-latent fusion** (mechanism 2): at selected early/mid
       steps, blend the running latent toward the warped previous latent in
       reliable regions — the latent analogue of the 2016 temporal loss applied
       as a hard, reliability-masked blend.

Each decoded frame is written via :func:`artvid.io.image.save_image` using the
**same** output naming as the optim single-pass engine
(:func:`artvid.pipeline.singlepass.build_out_filename`), so ``cli.cmd_run``'s
re-encode step (``encode_video`` over the ``out-%0Nd.ext`` pattern) works
**unchanged**.

Which Phase 1 / Phase 2 modules this builds on
----------------------------------------------
* :mod:`artvid.diffusion.engine` — ``DiffusionEngine`` (SDXL + depth ControlNet +
  IP-Adapter single-frame stylizer; VAE encode/decode; ``denoise_frame``).
* :mod:`artvid.diffusion.latent_warp` — ``warp_latent`` / ``latent_reliability``
  (the latent-space reuse of the Phase 1 flow stack).
* :mod:`artvid.flow.raft` — ``compute_flow_pair`` (flow between content frames).
* :mod:`artvid.pipeline.singlepass` — reused, framework-light helpers for frame
  discovery / output naming / precomputed-vs-RAFT flow loading
  (``discover_num_images``, ``_content_frame_path``, ``build_out_filename``,
  ``select_previous_indices`` is *not* reused — see §2.6 note below).
* :mod:`artvid.io.image` — ``load_image`` / ``save_image`` (RGB ``[0,1]`` CHW).
* :mod:`artvid.device` — device / MPS-fallback policy.

Hard constraints honoured here
------------------------------
* ``torch`` / ``diffusers`` are **lazy-imported inside functions**, mirroring the
  Phase 1 pattern (``artvid/pipeline/singlepass.py``, ``artvid/cli.py``). This
  module is therefore ``py_compile``-able and importable without torch; only the
  *call* to :func:`stylize_video_diffusion` needs the frameworks.
* This is FOUNDATION/scaffolding written against the documented engine API; it is
  meant to be **run and tuned on the user's M5 Max**. Every numerically- or
  quality-sensitive choice is marked ``TODO(tuning)`` with what to verify.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, List, Optional

if TYPE_CHECKING:  # pragma: no cover - typing only; torch/diffusers optional
    import torch

    from artvid.config import Config
    from artvid.diffusion.engine import DiffusionEngine, StyleReference


# ---------------------------------------------------------------------------
# Defaults mirroring the documented ``Config`` "diffusion" field group
# (docs/07-phase2-design.md §4.1). ``Config`` is owned by the config agent; we
# read every value via ``_get(cfg, name, DEFAULT)`` so this loop works whether or
# not those fields have landed yet, and so the contract is self-documenting here.
# ---------------------------------------------------------------------------

#: Per-step fusion blend cap (multiplied by the reliability mask). The single
#: most important temporal knob alongside the fuse window. TODO(tuning): too high
#: / too late -> over-smoothed, washed detail; too low / too early -> flicker
#: (docs §5.3). Sweep on a fast-motion and a slow-pan clip.
DEFAULT_TEMPORAL_STRENGTH = 0.6
#: Fraction-of-steps window in which the per-step fusion is active. We fuse only
#: in early/mid steps so the UNet can still synthesize fresh high-frequency
#: detail late (fusing at the very end reintroduces the warped frame's VAE
#: artifacts). TODO(tuning): sweep this window (docs §2.5 / §5.3).
DEFAULT_TEMPORAL_FUSE_START = 0.0
DEFAULT_TEMPORAL_FUSE_END = 0.7
#: img2img-style denoising strength for the warped-latent init on subsequent
#: frames: <1 keeps more of the warped previous latent (more coherent, less
#: fresh synthesis). TODO(tuning): the main fidelity/coherence knob for the init
#: mechanism (docs §5.3 item 1). 1.0 would ignore the warped init entirely.
DEFAULT_TEMPORAL_INIT_STRENGTH = 0.6
#: Erosion exponent for the downsampled latent reliability (docs §2.4). >1 erodes
#: (more conservative). TODO(tuning): mean+gamma vs min-pool vs threshold
#: (docs §5.2). Forwarded to ``latent_reliability``.
DEFAULT_RELIABILITY_GAMMA = 2.0
#: VAE spatial downsample factor (SDXL/SD1.5 both use 8). Exposed so the loop
#: rescales flow to the latent grid consistently with the engine.
DEFAULT_VAE_FACTOR = 8
#: Long-term anchor (frame-0) warp + reliability combination. Default off; enable
#: for long clips to fight drift (docs §2.6). TODO(tuning): whether single-prev
#: warp suffices or keyframe re-anchoring is needed for long clips (docs §5.10).
DEFAULT_USE_ANCHOR = False
#: Base-noise seed strategy. ``None`` => derive a fixed-across-frames seed from
#: ``Config.seed`` when >= 0 (more coherent base noise, less flicker). TODO(tuning):
#: fixed-seed vs warped-noise vs per-frame random (docs §5.5).
DEFAULT_SEED = None


def _get(cfg: Any, name: str, default: Any) -> Any:
    """Read ``cfg.<name>`` falling back to ``default`` (defensive, see engine)."""
    if cfg is None:
        return default
    val = getattr(cfg, name, default)
    return default if val is None else val


# ---------------------------------------------------------------------------
# Per-frame result (mirrors singlepass.FrameResult so callers / cli treat the
# diffusion and optim engines uniformly).
# ---------------------------------------------------------------------------


@dataclass
class FrameResult:
    """Per-frame outcome of :func:`stylize_video_diffusion`.

    Mirrors :class:`artvid.pipeline.singlepass.FrameResult` so the CLI and any
    downstream consumer can treat the optim and diffusion engines uniformly.

    Attributes:
        frame_idx: Absolute content-frame index.
        output_path: Where the stylized frame was written.
        previous_indices: The previous-frame indices used for the latent temporal
            constraint on this frame (empty for the anchor / first frame; the
            immediate predecessor, plus the anchor when ``use_anchor`` is on).
        used_temporal: Whether the latent temporal init/fusion was applied (False
            for the anchor frame and whenever flow/reliability were unavailable).
    """

    frame_idx: int
    output_path: str
    previous_indices: List[int] = field(default_factory=list)
    used_temporal: bool = False


# ---------------------------------------------------------------------------
# fuse-step window helper (torch-free, unit-testable)
# ---------------------------------------------------------------------------


def fuse_step_set(num_steps: int, start_frac: float, end_frac: float) -> set:
    """Indices of denoising steps in the ``[start_frac, end_frac)`` window.

    Converts the fraction-of-steps fuse window (docs §2.5: fuse early/mid only)
    into the concrete set of integer step indices ``denoise_frame`` checks via
    ``i in fuse_steps``. Pure-python so it is testable without torch.

    Args:
        num_steps: Total denoising steps ``K``.
        start_frac: Lower bound (inclusive) as a fraction of ``K`` in ``[0, 1]``.
        end_frac: Upper bound (exclusive) as a fraction of ``K`` in ``[0, 1]``.

    Returns:
        The set ``{i : start_frac*K <= i < end_frac*K}`` clamped to
        ``[0, num_steps)``. Empty when the window is empty or degenerate.
    """
    if num_steps <= 0:
        return set()
    lo = max(0, int(round(start_frac * num_steps)))
    hi = min(num_steps, int(round(end_frac * num_steps)))
    return {i for i in range(lo, hi)}


# ---------------------------------------------------------------------------
# Output naming — reuse the optim engine's exact scheme so cli.cmd_run's
# re-encode (encode_video over out-%0Nd.ext) works unchanged.
# ---------------------------------------------------------------------------


def _output_path_for(config: "Config", frame_idx: int) -> str:
    """Per-frame output path, identical to the optim single-pass naming.

    Delegates to :func:`artvid.pipeline.singlepass.build_out_filename` with the
    *relative* frame number ``abs(frame_idx - start_number + 1)`` (legacy
    convention), so the diffusion engine writes ``<folder><base>-<NNN><ext>``
    files that ``cli.cmd_run`` re-encodes with no changes.
    """
    from artvid.pipeline.singlepass import build_out_filename

    image_number = abs(frame_idx - config.start_number + 1)
    return build_out_filename(config, image_number)


# ---------------------------------------------------------------------------
# Flow loading — reuse the single-pass precomputed-vs-RAFT helpers so the
# diffusion engine consumes the SAME .flo / reliability plumbing as the optim
# engine (docs §3.3). We need BOTH directions of flow (backward to warp, forward
# to cross-check), so we read/compute the pair explicitly here.
# ---------------------------------------------------------------------------


def _flow_pair_for(
    config: "Config",
    frame_idx: int,
    prev_index: int,
    content_rgb: "torch.Tensor",
    prev_content_rgb: "torch.Tensor",
    device: Any,
    flow_source: str,
):
    """Return ``(backward_flow, forward_flow)`` ``(2, H, W)`` for prev->current.

    * ``backward`` = current -> previous (the flow ``warp_latent`` warps with).
    * ``forward``  = previous -> current (the consistency cross-check).

    Precomputed path: read the ``.flo`` files the ``artvid flow`` step wrote
    (``Config.backward_flow_pattern`` / ``forward_flow_pattern``), reusing the
    single-pass ``.flo`` reader and filename resolver. RAFT path: one
    :func:`artvid.flow.raft.compute_flow_pair` call on the *content* frames.

    The ``flow_source`` semantics (``auto`` | ``precomputed`` | ``raft``) and the
    "flow on content frames, not stylized outputs" decision match the optim
    single-pass engine exactly (``singlepass._get_backward_flow``).
    """
    from artvid.pipeline.singlepass import (
        _read_flo_tensor,
        _use_precomputed,
        format_flow_filename,
    )

    bwd_path = Path(
        format_flow_filename(
            config.backward_flow_pattern, abs(prev_index), abs(frame_idx)
        )
    )
    # cmd_flow writes the forward (prev->cur) flow as ``forward_<prev>_<cur>``
    # (_flow_filename("forward", i=prev, j=cur)). format_flow_filename fills
    # ``{...}``<-from and ``[...]``<-to, and the default pattern is
    # ``forward_[%d]_{%d}`` == ``forward_<to>_<from>``; so to resolve to
    # ``forward_<prev>_<cur>`` we must pass from=cur, to=prev (the reverse of the
    # backward call). See docs/06 / the Phase 2 review note.
    fwd_path = Path(
        format_flow_filename(
            config.forward_flow_pattern, abs(frame_idx), abs(prev_index)
        )
    )

    # Use precomputed only when BOTH directions are available/selected; otherwise
    # fall back to RAFT for the pair (auto-mode only "auto-selects" precomputed
    # when the files actually exist).
    if _use_precomputed(flow_source, bwd_path) and _use_precomputed(
        flow_source, fwd_path
    ):
        backward = _read_flo_tensor(bwd_path, device)
        forward = _read_flo_tensor(fwd_path, device)
        return backward, forward

    from artvid.flow.raft import compute_flow_pair

    # compute_flow_pair(img1=prev, img2=cur): .forward = prev->cur, .backward =
    # cur->prev. So forward-warp flow == pair.forward, backward-warp flow (the one
    # that pulls prev INTO cur) == pair.backward. (Mind the naming: the design
    # doc §2.5 names variables from the warp's perspective; here we name from the
    # flow's source->target perspective and map explicitly.)
    pair = compute_flow_pair(
        prev_content_rgb.to(device), content_rgb.to(device), device=device
    )
    backward = pair.backward  # current -> previous (warp flow)
    forward = pair.forward  # previous -> current (cross-check)
    return backward, forward


# ---------------------------------------------------------------------------
# Main per-frame loop
# ---------------------------------------------------------------------------


def stylize_video_diffusion(
    config: Optional["Config"] = None,
    *,
    engine: Optional["DiffusionEngine"] = None,
    device: Any = None,
    flow_source: str = "auto",
) -> List[FrameResult]:
    """Diffusion video stylization with latent optical-flow temporal consistency.

    The Phase 2 counterpart to :func:`artvid.pipeline.singlepass.stylize_video`.
    Iterates content frames ``start_number .. start_number + num_images - 1``,
    stylizes the first frame plainly (anchor) and every subsequent frame with the
    warped-previous-latent init + per-step reliability-masked fusion described in
    ``docs/07-phase2-design.md`` §2.5.

    Args:
        config: Run parameters (:class:`artvid.config.Config`). The content
            sequence is ``content_pattern`` / ``start_number`` / ``num_images``;
            the IP-Adapter style reference is ``style_image``; the diffusion +
            temporal knobs are the §4.1 "diffusion" field group (read defensively
            via :func:`_get`, defaulting to the constants above when absent).
        engine: Optional pre-built :class:`DiffusionEngine`. When ``None`` one is
            built from ``config`` via ``DiffusionEngine.from_config(config)``.
            Passing a shared engine avoids rebuilding the (heavy) pipeline.
        device: Optional torch device / string override; ``None`` lets the engine
            autodetect via :mod:`artvid.device`.
        flow_source: ``"auto"`` (precomputed ``.flo`` when present, else RAFT) |
            ``"precomputed"`` | ``"raft"`` — identical semantics to the optim
            engine, so the SAME flow precompute artifacts are reused (docs §3.3).

    Returns:
        A list of :class:`FrameResult`, one per stylized frame, in frame order.

    Notes:
        TODO(tuning): this whole loop is the P2-M2 milestone — it must be run and
        tuned on the M5 Max (temporal_strength × fuse window, init strength,
        reliability gamma, seed strategy; docs §5 items 1-5). Defaults above are
        starting points only.
    """
    # Lazy framework imports (this function is the only torch-touching entry).
    import torch

    from artvid.config import Config
    from artvid.device import enable_mps_fallback, get_device
    from artvid.diffusion.engine import DiffusionEngine
    from artvid.diffusion.latent_warp import (
        combine_latent_reliability,
        latent_reliability,
        warp_latent,
    )
    from artvid.io.image import load_image, save_image
    from artvid.pipeline.singlepass import _content_frame_path, discover_num_images

    config = config or Config()

    if flow_source not in ("auto", "precomputed", "raft"):
        raise ValueError(
            f"flow_source must be 'auto', 'precomputed' or 'raft'; got "
            f"{flow_source!r}."
        )

    enable_mps_fallback()
    if device is None:
        device = get_device(_get(config, "device", None))
    elif isinstance(device, str):
        device = torch.device(device)

    # --- engine (build once; reused for every frame) -----------------------
    if engine is None:
        engine = DiffusionEngine.from_config(config)
    engine.load()

    # --- temporal / diffusion knobs (defensive config reads) ---------------
    vae_factor = int(_get(config, "vae_factor", DEFAULT_VAE_FACTOR))
    temporal_strength = float(_get(config, "temporal_strength", DEFAULT_TEMPORAL_STRENGTH))
    fuse_start = float(_get(config, "temporal_fuse_start", DEFAULT_TEMPORAL_FUSE_START))
    fuse_end = float(_get(config, "temporal_fuse_end", DEFAULT_TEMPORAL_FUSE_END))
    init_strength = float(
        _get(config, "temporal_init_strength", DEFAULT_TEMPORAL_INIT_STRENGTH)
    )
    gamma = float(_get(config, "latent_reliability_gamma", DEFAULT_RELIABILITY_GAMMA))
    use_anchor = bool(_get(config, "use_anchor", DEFAULT_USE_ANCHOR))
    steps = int(getattr(engine, "steps", 30))
    fuse_steps = fuse_step_set(steps, fuse_start, fuse_end)

    # Base-noise seed strategy (docs §5.5): a fixed seed across frames gives a
    # coherent base-noise field (less flicker). Use Config.seed when set (>= 0),
    # else the diffusion default. TODO(tuning): try warped-noise / per-frame.
    cfg_seed = _get(config, "seed", DEFAULT_SEED)
    base_seed = int(cfg_seed) if (cfg_seed is not None and int(cfg_seed) >= 0) else None

    # --- frame discovery (reuse the optim engine's autodetect) -------------
    num_images = discover_num_images(config)
    if num_images == 0:
        raise FileNotFoundError(
            "No content frames found for pattern "
            f"{config.content_pattern!r} starting at frame {config.start_number}."
        )

    start = int(config.start_number)
    continue_with = int(getattr(config, "continue_with", 1))
    first_idx = start + continue_with - 1
    last_idx = start + num_images - 1

    # --- style reference, encoded ONCE (frame-invariant) -------------------
    style_paths = [s.strip() for s in str(config.style_image).split(",") if s.strip()]
    if not style_paths:
        raise ValueError(
            "Config.style_image must name the IP-Adapter style reference image."
        )
    # IP-Adapter takes a single style image; if several are given we use the
    # first and note the rest are ignored. TODO(tuning): multi-image IP-Adapter
    # (a list of references) is supported by some diffusers versions — wire it in
    # if multi-style blending is wanted (mirrors the optim engine's blend list).
    style_ref: "StyleReference" = engine.encode_style(style_paths[0])

    # Carry-forward state across frames.
    prev_latent: "torch.Tensor | None" = None
    prev_content_rgb: "torch.Tensor | None" = None
    anchor_latent: "torch.Tensor | None" = None
    anchor_content_rgb: "torch.Tensor | None" = None

    results: List[FrameResult] = []

    for frame_idx in range(first_idx, last_idx + 1):
        content_path = _content_frame_path(config, frame_idx)
        if not content_path.is_file():
            break

        content_rgb = load_image(content_path)  # (3, H, W) [0,1] RGB
        _, H, W = content_rgb.shape
        h, w = H // vae_factor, W // vae_factor

        # Per-frame structure (ControlNet) conditioning. ``denoise_frame``
        # requires a concrete ``control_image`` (it does not build one from
        # ``None``), so we build it here from the content frame via the engine's
        # own ``_build_control`` (which delegates to ``artvid.diffusion.preprocess``
        # for the configured signal kind: depth/lineart/canny, docs §3.4). Keeping
        # the build on the engine means the structure-signal choice stays a single
        # config-driven decision owned by the engine/preprocess agents.
        control_image = engine._build_control(content_rgb)

        seed = base_seed  # fixed-across-frames (docs §5.5); TODO(tuning)

        if prev_latent is None:
            # ---- anchor / first frame: plain ControlNet + IP-Adapter -------
            latent = engine.denoise_frame(
                content_rgb,
                control_image=control_image,
                style=style_ref,
                seed=seed,
                steps=steps,
            )
            used_temporal = False
            prev_indices: List[int] = []
            # Seed the anchor for optional long-term consistency.
            if use_anchor:
                anchor_latent = latent
                anchor_content_rgb = content_rgb
        else:
            # ---- subsequent frame: warped-latent init + per-step fusion ----
            prev_index = frame_idx - 1
            backward_flow, forward_flow = _flow_pair_for(
                config,
                frame_idx,
                prev_index,
                content_rgb,
                prev_content_rgb,
                device,
                flow_source,
            )

            # Backward-warp the PREVIOUS latent into the current frame's grid.
            warp = warp_latent(
                prev_latent,
                backward_flow.to(device),
                vae_factor=vae_factor,
                image_hw=(H, W),
            )
            warped_latent = warp.image  # (1, C, h, w)

            # Latent-grid reliability for the (prev -> current) warp.
            reliability = latent_reliability(
                forward_flow.to(device),
                backward_flow.to(device),
                warp.valid,
                latent_hw=(h, w),
                gamma=gamma,
            )  # (1, 1, h, w) [0,1]
            prev_indices = [prev_index]

            # ---- optional long-term anchor (docs §2.6) --------------------
            if use_anchor and anchor_latent is not None:
                a_backward, a_forward = _flow_pair_for(
                    config,
                    frame_idx,
                    start,  # anchor is the first/start frame
                    content_rgb,
                    anchor_content_rgb,
                    device,
                    flow_source,
                )
                a_warp = warp_latent(
                    anchor_latent,
                    a_backward.to(device),
                    vae_factor=vae_factor,
                    image_hw=(H, W),
                )
                a_rel = latent_reliability(
                    a_forward.to(device),
                    a_backward.to(device),
                    a_warp.valid,
                    latent_hw=(h, w),
                    gamma=gamma,
                )
                # Combine closest-previous-frame first (the anchor only claims
                # cells the previous frame does not reliably see).
                rel_prev, rel_anchor = combine_latent_reliability(
                    [reliability, a_rel], method="closestFirst"
                )
                # Fuse the anchor contribution into the warped target weighted by
                # its (post-combination) reliability, then renormalize the blend
                # weight to the union of reliable cells. This keeps a single
                # (warped_latent, reliability) pair for the engine's mechanisms.
                denom = (rel_prev + rel_anchor).clamp_min(1e-6)
                warped_latent = (
                    rel_prev * warped_latent + rel_anchor * a_warp.image
                ) / denom
                reliability = (rel_prev + rel_anchor).clamp(0.0, 1.0)
                prev_indices = [prev_index, start]

            # MECHANISM 1: init from the warped previous latent (renoised to the
            # start timestep) in reliable regions; img2img strength keeps more of
            # the warped init the lower it is. MECHANISM 2: per-step fusion in
            # reliable regions over the early/mid fuse window. Both handed to the
            # engine's single denoise loop.
            latent = engine.denoise_frame(
                content_rgb,
                control_image=control_image,
                style=style_ref,
                init_latents=warped_latent,
                reliability=reliability,
                warped_latent=warped_latent,
                strength=init_strength,
                steps=steps,
                seed=seed,
                fuse_steps=fuse_steps,
                temporal_strength=temporal_strength,
            )
            used_temporal = True

        # --- decode + save (same naming as the optim engine) ---------------
        out_rgb = engine.decode(latent)  # (3, H, W) [0,1] RGB
        out_path = _output_path_for(config, frame_idx)
        # save_image deprocesses for a VGG-preprocessed tensor by default; our
        # decoded frame is already plain RGB [0,1], so use the torchvision-style
        # passthrough? No: save_image always deprocesses. We therefore write the
        # RGB tensor directly via the io.image low-level path (see _save_rgb).
        _save_rgb(out_rgb, out_path)

        results.append(
            FrameResult(
                frame_idx=frame_idx,
                output_path=out_path,
                previous_indices=prev_indices,
                used_temporal=used_temporal,
            )
        )

        # Carry state forward.
        prev_latent = latent
        prev_content_rgb = content_rgb

    return results


def _save_rgb(image_rgb: "torch.Tensor", path: str) -> None:
    """Write a plain RGB ``[0,1]`` CHW tensor to ``path`` (no VGG deprocess).

    :func:`artvid.io.image.save_image` always runs :func:`~artvid.io.image.deprocess`
    (it assumes a VGG-*pre*processed tensor, the optim engine's optimized
    variable). Our decoded diffusion frame is already plain RGB ``[0,1]``, so we
    must NOT deprocess it; we reuse the same Pillow encode path ``save_image``
    uses, minus the deprocess step.

    Implemented by clamping + writing directly, matching ``save_image``'s
    CHW[0,1]->HWC uint8 encode so on-disk output is byte-identical in format to
    the optim engine's frames (which keeps ``cli.cmd_run``'s re-encode happy).
    """
    import torch
    from PIL import Image

    # Move to CPU + float32 before the numpy/PIL hop: MPS tensors cannot be
    # ``.numpy()``-ed directly, and float64 is unsupported on MPS — float32 here
    # is both MPS-safe and the right precision for an 8-bit RGB encode.
    disp = image_rgb.detach().to("cpu", dtype=torch.float32).clamp(0.0, 1.0)
    if disp.dim() == 4:
        disp = disp[0]
    arr = disp.mul(255.0).round().permute(1, 2, 0).to(dtype=torch.uint8).numpy()
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr, mode="RGB").save(out_path)
