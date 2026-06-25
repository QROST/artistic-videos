"""Multi-pass video stylization (forward/backward alternating passes).

Ports the multi-pass algorithm from ``artistic_video_multiPass.lua`` — the main
loop (``:145-274``) plus its two warp helpers ``readPrevImageWarped``
(``:280-302``) and ``readNextImageWarped`` (``:306-327``). The warping itself is
delegated to :func:`artvid.flow.warp.warp_image` (the M1 port of ``warpImage``);
flow + reliability sourcing reuses the same precomputed/RAFT backends as
:mod:`artvid.pipeline.singlepass`, and the style/Gram setup reuses the helpers
in :mod:`artvid.pipeline.stylize_image`.

Algorithm (mirroring the Lua loop)
----------------------------------
The whole sequence is processed ``num_passes`` times, **alternating direction**:

* odd passes (``run`` odd, ``flag == 1``): **forward**, frame ``start .. end``;
* even passes (``run`` even, ``flag == 0``): **backward**, frame ``end .. start``.

For each frame in a pass (``:152-260``):

* **Warped neighbour** (``:159-172``). The previous output of this frame is
  warped from a neighbour: on a forward pass we warp the *previous* frame's
  result by the **backward** flow (``frameIdx-1 -> frameIdx``); on a backward
  pass we warp the *next* frame's result by the **forward** flow
  (``frameIdx+1 -> frameIdx``). The neighbour result is taken from the
  appropriate already-finished pass (``run - (1 - flag)`` for the previous
  neighbour, ``run - flag`` for the next neighbour).
* **Temporal loss gate** (``:174``). Temporal loss is enabled only when
  ``run >= use_temporal_loss_after`` *and* a warped neighbour exists.
* **Init / blend** (``:210-249``). On pass 1, frames are initialised
  independently (``random`` | ``image`` | ``prevWarped``). On subsequent passes
  the frame is initialised by blending **this frame's previous-pass result**
  with the flow-warped neighbour result(s) using per-pixel reliability weights
  scaled by ``blend_weight`` (for the neighbour in the pass direction) and
  ``blend_weight_last_pass`` (for the opposite-direction neighbour), then
  normalised by the accumulated divisor.
* **Optimize + save** (``:259``). One :func:`artvid.optim.runner.run_optimization`
  call per frame; result saved per pass as ``<basename>-<frame>_<pass>.<ext>``.

Blend-weight selection (parity-critical, ``:234`` & ``:243``)
-------------------------------------------------------------
The previous-neighbour weight is scaled by ``blend_weight`` on forward passes
(``flag == 1``) and by ``blend_weight_last_pass`` on backward passes; the
next-neighbour weight is scaled by ``blend_weight`` on backward passes
(``flag == 0``) and by ``blend_weight_last_pass`` on forward passes. In other
words the neighbour lying *behind* the sweep direction (the one already
re-stylised this pass) gets the full ``blend_weight``; the neighbour *ahead*
gets ``blend_weight_last_pass``. See :func:`blend_weight_for_neighbour`.

Framework-agnostic core: only torch tensor ops, no MPS-specific calls. Device
selection goes through :mod:`artvid.device`. ``torch`` is imported lazily inside
the functions that need it so the direction/blend helpers (and their tests) stay
importable without torch.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

from artvid.config import Config

# Reuse the torch-free filename/flow helpers from the single-pass module so the
# two pipelines resolve content / flow / output names identically.
from artvid.pipeline.singlepass import (
    _content_frame_path,
    discover_num_images,
    format_flow_filename,
)

# ---------------------------------------------------------------------------
# Pure (torch-free) helpers: pass direction + blend-weight selection
# ---------------------------------------------------------------------------

def pass_direction(run: int, start_number: int, end_image_idx: int) -> Tuple[int, int, int]:
    """Return ``(start, endp, incr)`` for pass ``run`` (ports ``:147-150``).

    The multi-pass scheme alternates sweep direction by pass parity (legacy
    ``flag = run % 2``):

    * **odd** ``run`` (``flag == 1``) → **forward**: ``start_number ..
      end_image_idx`` step ``+1``;
    * **even** ``run`` (``flag == 0``) → **backward**: ``end_image_idx ..
      start_number`` step ``-1``.

    Args:
        run: 1-based pass index.
        start_number: Absolute index of the first frame.
        end_image_idx: Absolute index of the last frame.

    Returns:
        ``(start, endp, incr)`` suitable for an inclusive range walk:
        iterate ``frame_idx`` from ``start`` to ``endp`` inclusive stepping by
        ``incr``.
    """
    flag = run % 2
    if flag == 0:  # even pass -> backward
        return end_image_idx, start_number, -1
    return start_number, end_image_idx, 1  # odd pass -> forward


def pass_frame_order(run: int, start_number: int, end_image_idx: int) -> List[int]:
    """Concrete list of frame indices visited in pass ``run`` (uses :func:`pass_direction`).

    A convenience wrapper that materialises the inclusive range from
    :func:`pass_direction`, so callers/tests can assert the exact sequencing
    (forward on odd passes, backward on even passes).
    """
    start, endp, incr = pass_direction(run, start_number, end_image_idx)
    return list(range(start, endp + incr, incr))


def is_forward_pass(run: int) -> bool:
    """``True`` when pass ``run`` sweeps forward (odd pass, legacy ``flag == 1``)."""
    return run % 2 == 1


def blend_weight_for_neighbour(
    config: Config, run: int, *, neighbour: str
) -> float:
    """Pick the blend weight for a warped neighbour (ports ``:234`` / ``:243``).

    On each pass the neighbour lying *behind* the sweep direction — the one
    already re-stylised earlier in this same pass — receives the full
    ``blend_weight``; the neighbour *ahead* of the sweep receives the smaller
    ``blend_weight_last_pass``.

    Concretely, with ``flag = run % 2``:

    * ``"prev"`` (previous-frame neighbour): ``blend_weight`` when ``flag == 1``
      (forward pass), else ``blend_weight_last_pass`` (legacy ``:234``).
    * ``"next"`` (next-frame neighbour): ``blend_weight`` when ``flag == 0``
      (backward pass), else ``blend_weight_last_pass`` (legacy ``:243``).

    Args:
        config: Run config (provides ``blend_weight`` / ``blend_weight_last_pass``).
        run: 1-based pass index.
        neighbour: ``"prev"`` or ``"next"``.

    Returns:
        The scalar blend weight to multiply the neighbour's reliability mask by.

    Raises:
        ValueError: If ``neighbour`` is not ``"prev"`` or ``"next"``.
    """
    forward = is_forward_pass(run)
    if neighbour == "prev":
        return config.blend_weight if forward else config.blend_weight_last_pass
    if neighbour == "next":
        return config.blend_weight_last_pass if forward else config.blend_weight
    raise ValueError(f"neighbour must be 'prev' or 'next'; got {neighbour!r}.")


def temporal_loss_enabled(
    run: int, use_temporal_loss_after: int, has_warped_neighbour: bool
) -> bool:
    """Whether the temporal loss is active this frame (ports ``:174``).

    Legacy ``temporalLossEnabled = run >= params.use_temporalLoss_after and
    imageWarped ~= nil``: the temporal term only kicks in once enough passes
    have run *and* there is a warped neighbour to compare against (i.e. not the
    very first frame in the sweep direction).
    """
    return run >= use_temporal_loss_after and has_warped_neighbour


def build_pass_out_filename(config: Config, frame_idx: int, run: int) -> str:
    """Per-(frame, pass) output filename (ports ``build_OutFilename`` multi-pass form).

    The multi-pass legacy ``build_OutFilename(params, frameIdx, run)`` names each
    intermediate result ``<basename>-<frame>_<pass><ext>`` (the run number is
    appended so passes do not overwrite each other; cf.
    ``artistic_video_core.lua:558-569`` with ``iterationOrRun = run``). The frame
    number uses ``Config.number_format`` and is the *absolute* ``frame_idx``
    (the multi-pass tool passes the absolute index, unlike single-pass which uses
    a relative number).

    Args:
        config: Run config (``output_image`` / ``output_folder`` /
            ``number_format``).
        frame_idx: Absolute content-frame index.
        run: 1-based pass index.

    Returns:
        The output path string.
    """
    out_image = Path(config.output_image)
    ext = out_image.suffix
    basename = out_image.stem
    number = config.number_format % frame_idx
    return f"{config.output_folder}{basename}-{number}_{run}{ext}"


# ---------------------------------------------------------------------------
# Flow / weight filename resolution (multi-pass forward+backward patterns)
# ---------------------------------------------------------------------------

def _backward_flow_path(config: Config, frame_idx: int) -> Path:
    """Backward-flow ``.flo`` path for warping the *previous* frame into ``frame_idx``.

    Legacy ``getFormatedFlowFileName(params.backwardFlow_pattern, idx-1, idx)``
    (``:281``): ``{...}`` = from = ``idx-1``, ``[...]`` = to = ``idx``.
    """
    return Path(
        format_flow_filename(config.backward_flow_pattern, frame_idx - 1, frame_idx)
    )


def _forward_flow_path(config: Config, frame_idx: int) -> Path:
    """Forward-flow ``.flo`` path for warping the *next* frame into ``frame_idx``.

    Legacy ``getFormatedFlowFileName(params.forwardFlow_pattern, idx+1, idx)``
    (``:307``): ``{...}`` = from = ``idx+1``, ``[...]`` = to = ``idx``.
    """
    return Path(
        format_flow_filename(config.forward_flow_pattern, frame_idx + 1, frame_idx)
    )


def _backward_weight_path(config: Config, frame_idx: int) -> Path:
    """Backward reliability ``.pgm`` path for the previous-frame warp into ``frame_idx``.

    Legacy ``getFormatedFlowFileName(params.backwardFlow_weight_pattern,
    frameIdx-1, frameIdx)`` (``:192`` / ``:230``).
    """
    return Path(
        format_flow_filename(
            config.backward_flow_weight_pattern, frame_idx - 1, frame_idx
        )
    )


def _forward_weight_path(config: Config, frame_idx: int) -> Path:
    """Forward reliability ``.pgm`` path for the next-frame warp into ``frame_idx``.

    Legacy ``getFormatedFlowFileName(params.forwardFlow_weight_pattern,
    frameIdx+1, frameIdx)`` (``:194`` / ``:239``).
    """
    return Path(
        format_flow_filename(
            config.forward_flow_weight_pattern, frame_idx + 1, frame_idx
        )
    )


def _use_precomputed(flow_source: str, path: Path) -> bool:
    """Decide whether to read a precomputed file vs compute on the fly."""
    if flow_source == "precomputed":
        return True
    if flow_source == "raft":
        return False
    return path.is_file()  # auto


# ---------------------------------------------------------------------------
# Run result
# ---------------------------------------------------------------------------

@dataclass
class PassFrameResult:
    """Per-(frame, pass) outcome of :func:`stylize_video_multipass`.

    Attributes:
        run: 1-based pass index.
        frame_idx: Absolute content-frame index.
        output_path: Where the stylized frame was written for this pass.
        forward: ``True`` if this was a forward (odd) pass.
        temporal_enabled: Whether the temporal loss was active for this frame.
        num_iterations: Optimizer iterations actually run.
    """

    run: int
    frame_idx: int
    output_path: str
    forward: bool
    temporal_enabled: bool
    num_iterations: int = 0


# ---------------------------------------------------------------------------
# Main multi-pass loop
# ---------------------------------------------------------------------------

def stylize_video_multipass(
    config: Optional[Config] = None,
    *,
    device: Optional[object] = None,
    flow_source: str = "auto",
) -> List[PassFrameResult]:
    """Run multi-pass forward/backward temporal-consistency video style transfer.

    Direct port of ``artistic_video_multiPass.lua:145-274``. The full sequence is
    processed ``config.num_passes`` times, alternating direction each pass
    (forward on odd passes, backward on even). Each subsequent pass blends the
    previous-pass result of a frame with its flow-warped neighbour(s), and a
    reliability-weighted temporal loss is added once ``run >=
    config.use_temporal_loss_after``.

    Args:
        config: Run parameters (:class:`~artvid.config.Config`). The multi-pass
            scheme uses ``num_passes`` / ``continue_with_pass`` /
            ``blend_weight`` / ``blend_weight_last_pass`` /
            ``use_temporal_loss_after`` and the forward/backward flow + weight
            patterns. Note the multi-pass ``temporal_weight`` default differs
            from single-pass (5e2 vs 1e3); pass an explicit value to override.
        device: Optional torch device / string; ``None`` autodetects via
            :func:`artvid.device.get_device`.
        flow_source: ``"precomputed"`` | ``"raft"`` | ``"auto"`` (default), as in
            :func:`artvid.pipeline.singlepass.stylize_video`.

    Returns:
        A list of :class:`PassFrameResult`, one per (frame, pass) in execution
        order.
    """
    import torch

    from artvid.device import enable_mps_fallback, get_device
    from artvid.io.image import deprocess, load_image, preprocess, save_image
    from artvid.losses.content import ContentLoss
    from artvid.losses.style import StyleLoss
    from artvid.losses.temporal import WeightedContentLoss
    from artvid.losses.tv import TVLoss
    from artvid.models.vgg import build_feature_net, split_activations
    from artvid.optim.runner import run_optimization
    from artvid.pipeline.stylize_image import (
        _build_style_targets,
        _normalized_blend_weights,
        _preprocess_mode_for,
    )

    config = config or Config()

    enable_mps_fallback()
    if device is None:
        device = get_device(config.device)
    elif isinstance(device, str):
        device = torch.device(device)

    if flow_source not in ("auto", "precomputed", "raft"):
        raise ValueError(
            f"flow_source must be 'auto', 'precomputed' or 'raft'; got "
            f"{flow_source!r}."
        )

    mode = _preprocess_mode_for(config)

    num_images = discover_num_images(config)
    if num_images == 0:
        raise FileNotFoundError(
            "No content frames found for pattern "
            f"{config.content_pattern!r} starting at frame {config.start_number}."
        )
    print(f"Detected {num_images} content images.")

    start = config.start_number
    end_image_idx = num_images + start - 1

    # --- Feature network + style targets (built once, reused everywhere) ------
    net = build_feature_net(
        content_layers=config.content_layers,
        style_layers=config.style_layers,
        pooling=config.pooling,
        weights=config.vgg_weights,
    ).to(device)
    net.eval()

    first_content_rgb = load_image(_content_frame_path(config, start))
    _, first_h, first_w = first_content_rgb.shape

    style_paths = [s.strip() for s in str(config.style_image).split(",") if s.strip()]
    blend_weights = _normalized_blend_weights(config.style_blend_weights, len(style_paths))
    style_targets = _build_style_targets(
        style_paths, blend_weights, (first_h, first_w), config, net, device
    )
    style_losses = {
        layer: StyleLoss(
            style_targets[layer],
            strength=config.style_weight,
            normalize=config.normalize_gradients,
            target_is_gram=True,
        ).to(device)
        for layer in config.style_layers
    }

    # Cache of stylized outputs in [0,1] RGB CHW space, keyed by (frame_idx, run).
    # The legacy re-loaded each neighbour from disk (build_OutFilename(... run));
    # we keep them in memory and fall back to disk for passes from an earlier
    # continue_with_pass run.
    outputs_rgb: dict[Tuple[int, int], "torch.Tensor"] = {}

    results: List[PassFrameResult] = []

    for run in range(config.continue_with_pass, config.num_passes + 1):
        flag = run % 2
        forward = flag == 1
        frame_order = pass_frame_order(run, start, end_image_idx)
        print(
            f"=== Pass {run}/{config.num_passes} "
            f"({'forward' if forward else 'backward'}) ==="
        )

        for frame_idx in frame_order:
            if config.seed >= 0:
                torch.manual_seed(config.seed)

            content_rgb = load_image(_content_frame_path(config, frame_idx))
            content_pre = preprocess(content_rgb, mode=mode).unsqueeze(0).to(device)

            with torch.no_grad():
                content_acts = net(content_pre)
            content_losses = {
                layer: ContentLoss(
                    content_acts[layer].detach(),
                    strength=config.content_weight,
                    normalize=config.normalize_gradients,
                ).to(device)
                for layer in config.content_layers
            }

            # --- Warp neighbour result(s) used for blend + temporal loss ------
            # prev neighbour exists for any frame past the sequence start; its
            # source pass is run-(1-flag) (legacy :166). next neighbour exists
            # from pass 2 onward and before the sequence end; source pass
            # run-flag (legacy :169).
            prev_warp = None  # (warped_rgb, reliability) for frameIdx-1
            next_warp = None  # (warped_rgb, reliability) for frameIdx+1
            if frame_idx > start:
                prev_warp = _warp_neighbour(
                    config,
                    frame_idx,
                    neighbour="prev",
                    source_run=run - (1 - flag),
                    content_rgb=content_rgb,
                    outputs_rgb=outputs_rgb,
                    device=device,
                    flow_source=flow_source,
                )
            if run > 1 and frame_idx < end_image_idx:
                next_warp = _warp_neighbour(
                    config,
                    frame_idx,
                    neighbour="next",
                    source_run=run - flag,
                    content_rgb=content_rgb,
                    outputs_rgb=outputs_rgb,
                    device=device,
                    flow_source=flow_source,
                )

            # The temporal target is the neighbour in the *sweep* direction
            # (legacy :171-172): forward pass -> prev neighbour; backward -> next.
            image_warped = prev_warp if forward else next_warp
            temporal_on = temporal_loss_enabled(
                run, config.use_temporal_loss_after, image_warped is not None
            )

            # --- Temporal loss (pixel space) ---------------------------------
            temporal_losses: List[WeightedContentLoss] = []
            if temporal_on:
                warped_rgb, reliab = image_warped
                warped_pre = preprocess(warped_rgb, mode=mode).to(device)
                w3 = reliab.unsqueeze(0).expand(3, reliab.shape[-2], reliab.shape[-1])
                temporal_losses.append(
                    WeightedContentLoss(
                        warped_pre,
                        weights=w3.to(device),
                        strength=config.temporal_weight,
                        criterion=config.temporal_criterion,
                        normalize=config.normalize_gradients,
                    ).to(device)
                )

            tv_loss = (
                TVLoss(strength=config.tv_weight).to(device)
                if config.tv_weight > 0
                else None
            )

            # --- Initialization (pass 1: independent; else: blend) -----------
            if run == 1:
                image_var = _init_first_pass(
                    config,
                    frame_idx=frame_idx,
                    content_pre=content_pre.squeeze(0),
                    prev_warp=prev_warp,
                    mode=mode,
                    device=device,
                )
            else:
                image_var = _init_blended(
                    config,
                    run=run,
                    frame_idx=frame_idx,
                    end_image_idx=end_image_idx,
                    content_rgb=content_rgb,
                    prev_warp=prev_warp,
                    next_warp=next_warp,
                    outputs_rgb=outputs_rgb,
                    mode=mode,
                    device=device,
                )

            # --- Loss closure ------------------------------------------------
            def loss_fn():
                acts = net(
                    image_var.unsqueeze(0) if image_var.dim() == 3 else image_var
                )
                content_acts_i, style_acts_i = split_activations(
                    acts, config.content_layers, config.style_layers
                )
                terms = {}
                for layer in config.content_layers:
                    terms[f"content[{layer}]"] = content_losses[layer](
                        content_acts_i[layer]
                    )
                for layer in config.style_layers:
                    terms[f"style[{layer}]"] = style_losses[layer](style_acts_i[layer])
                for k, tloss in enumerate(temporal_losses):
                    terms[f"temporal[{k}]"] = tloss(image_var)
                if tv_loss is not None:
                    terms["tv"] = tv_loss(image_var)
                return terms

            out_path = build_pass_out_filename(config, frame_idx, run)

            if config.save_init:
                init_path = (
                    f"{config.output_folder}init-"
                    f"{config.number_format % frame_idx}_{run}.png"
                )
                save_image(image_var.detach(), init_path, mode=mode)

            def save_fn(iteration: int, is_end: bool) -> None:  # noqa: ARG001
                save_image(image_var.detach(), out_path, mode=mode)

            print(
                f"Pass {run} frame {frame_idx} (-> {out_path}); "
                f"temporal={'on' if temporal_on else 'off'}"
            )
            run_result = run_optimization(
                image_var,
                loss_fn,
                max_iter=config.num_iterations[1],
                optimizer=config.optimizer,
                tol_loss_relative=config.tol_loss_relative,
                tol_loss_relative_interval=config.tol_loss_relative_interval,
                learning_rate=config.learning_rate,
                print_iter=config.print_iter,
                save_iter=config.save_iter,
                save_fn=save_fn,
            )

            outputs_rgb[(frame_idx, run)] = (
                deprocess(image_var.detach(), mode=mode).clamp(0.0, 1.0).to(device)
            )

            results.append(
                PassFrameResult(
                    run=run,
                    frame_idx=frame_idx,
                    output_path=out_path,
                    forward=forward,
                    temporal_enabled=temporal_on,
                    num_iterations=run_result.num_iterations,
                )
            )

    return results


# ---------------------------------------------------------------------------
# Internal: neighbour warp + reliability sourcing
# ---------------------------------------------------------------------------

def _output_rgb(config: Config, frame_idx: int, run: int, outputs_rgb, device):
    """Return the stylized output (``[0,1]`` RGB) of ``frame_idx`` at pass ``run``.

    Prefers the in-memory cache; otherwise re-loads it from disk via the
    per-pass output filename (a previous ``continue_with_pass`` run), mirroring
    the legacy ``image.load(build_OutFilename(params, idx, run), 3)``.
    """
    key = (frame_idx, run)
    if key in outputs_rgb:
        return outputs_rgb[key]
    from artvid.io.image import load_image

    path = build_pass_out_filename(config, frame_idx, run)
    return load_image(path).to(device)


def _read_flo_tensor(path, device):
    """Read a ``.flo`` file as a ``(2, H, W)`` torch float32 tensor on ``device``."""
    import torch

    from artvid.io.flow_io import read_flo

    flow_np = read_flo(path)  # (2, H, W) numpy (u, v)
    return torch.from_numpy(flow_np).to(device=device, dtype=torch.float32)


def _read_reliability(path, device):
    """Read a reliability ``.pgm``/``.png`` mask as an ``(H, W)`` tensor in [0,1]."""
    from artvid.io.image import load_image

    img = load_image(path).to(device)
    return img[0]  # grayscale -> channel 0


def _neighbour_flow(
    config, frame_idx, neighbour, content_rgb, device, flow_source
):
    """Flow (current -> neighbour) used to warp the neighbour into this frame.

    Prev neighbour: backward flow ``frameIdx-1 -> frameIdx`` warps the previous
    frame's result into the current frame (legacy ``readPrevImageWarped``).
    Next neighbour: forward flow ``frameIdx+1 -> frameIdx`` warps the next
    frame's result into the current frame (legacy ``readNextImageWarped``).

    Precomputed path reads the ``.flo``; RAFT path computes flow between the
    *content* frames (optical flow is a property of the source scene motion).
    """
    if neighbour == "prev":
        flo_path = _backward_flow_path(config, frame_idx)
        neighbour_idx = frame_idx - 1
    else:
        flo_path = _forward_flow_path(config, frame_idx)
        neighbour_idx = frame_idx + 1

    if _use_precomputed(flow_source, flo_path):
        return _read_flo_tensor(flo_path, device)

    from artvid.flow.raft import compute_flow
    from artvid.io.image import load_image

    neighbour_content = load_image(
        _content_frame_path(config, neighbour_idx)
    ).to(device)
    # Backward/forward warp both map current-frame pixels into the neighbour:
    # compute_flow(img_current, img_neighbour).
    return compute_flow(content_rgb.to(device), neighbour_content, device=device)


def _neighbour_reliability(
    config, frame_idx, neighbour, content_rgb, primary_flow, device, flow_source
):
    """Per-pixel reliability mask for warping a neighbour into ``frame_idx``.

    Precomputed path loads the ``.pgm`` (backward weight pattern for the prev
    neighbour, forward weight pattern for the next neighbour; legacy ``:192-194``
    / ``:230,:239``). RAFT path computes the opposite-direction flow and runs
    :func:`artvid.flow.consistency.consistency_mask` on the pair.
    """
    import torch

    if neighbour == "prev":
        pgm_path = _backward_weight_path(config, frame_idx)
        neighbour_idx = frame_idx - 1
    else:
        pgm_path = _forward_weight_path(config, frame_idx)
        neighbour_idx = frame_idx + 1

    if _use_precomputed(flow_source, pgm_path):
        return _read_reliability(pgm_path, device)

    from artvid.flow.consistency import consistency_mask
    from artvid.flow.raft import compute_flow
    from artvid.io.image import load_image

    neighbour_content = load_image(
        _content_frame_path(config, neighbour_idx)
    ).to(device)
    # The opposite-direction flow (neighbour -> current) validates primary_flow.
    other_flow = compute_flow(neighbour_content, content_rgb.to(device), device=device)
    return consistency_mask(
        primary_flow.to(device),
        other_flow.to(device),
    ).to(dtype=torch.float32)


def _warp_neighbour(
    config,
    frame_idx,
    *,
    neighbour,
    source_run,
    content_rgb,
    outputs_rgb,
    device,
    flow_source,
):
    """Warp a neighbour's stylized output into ``frame_idx`` (ports the warp helpers).

    Ports ``readPrevImageWarped`` (``:280-302``) / ``readNextImageWarped``
    (``:306-327``): load the neighbour's result from the given ``source_run``
    pass, warp it by the appropriate flow into the current frame's coordinates
    (mean-filling disocclusions), and return ``(warped_rgb, reliability)`` where
    the reliability mask is combined with the warp validity so disoccluded
    pixels are down-weighted.

    Returns ``None`` when the source pass output is unavailable (e.g. a guard
    failed upstream), so callers can treat a missing neighbour as ``nil``.
    """
    neighbour_idx = frame_idx - 1 if neighbour == "prev" else frame_idx + 1

    prev_out_rgb = _output_rgb(config, neighbour_idx, source_run, outputs_rgb, device)
    flow = _neighbour_flow(
        config, frame_idx, neighbour, content_rgb, device, flow_source
    )

    from artvid.flow.warp import warp_image

    warp_res = warp_image(prev_out_rgb, flow)

    reliab = _neighbour_reliability(
        config, frame_idx, neighbour, content_rgb, flow, device, flow_source
    )
    # Down-weight disoccluded / out-of-border pixels (warp invalid).
    reliab = reliab.to(device) * warp_res.valid.to(reliab.dtype).squeeze(0)
    return warp_res.image, reliab


# ---------------------------------------------------------------------------
# Internal: initialization (pass 1 independent vs subsequent-pass blend)
# ---------------------------------------------------------------------------

def _init_first_pass(
    config, *, frame_idx, content_pre, prev_warp, mode, device
):
    """Initialise a frame on pass 1 (ports ``:210-222``).

    Pass-1 frames are processed independently:

    * first frame, or ``init == 'random'`` → ``randn * 0.001``;
    * ``init == 'image'`` → the preprocessed content frame;
    * ``init == 'prevWarped'`` → the previous frame's pass-1 result warped (with
      mean-pixel pad) into this frame.

    ``config.init`` is a ``(first, subsequent)`` pair; the legacy multi-pass tool
    used a single ``-init`` value, so we take ``init[0]`` as that single value.
    """
    import torch

    from artvid.io.image import preprocess

    init_mode = config.init[0]
    start = config.start_number

    if frame_idx == start or init_mode == "random":
        img = torch.randn_like(content_pre, dtype=torch.float32).mul_(0.001)
    elif init_mode == "image":
        img = content_pre.detach().clone().to(torch.float32)
    elif init_mode == "prevWarped" and prev_warp is not None:
        # prev_warp[0] is the previous frame warped into this frame already.
        img = preprocess(prev_warp[0], mode=mode).to(torch.float32)
    else:
        raise ValueError(
            f"Unknown initialization method {init_mode!r} for multi-pass pass 1 "
            f"(frame {frame_idx})."
        )

    # Optimized image stays float32 on all backends (L-BFGS stability;
    # MPS lacks float64) — see artvid.device.image_optim_dtype().
    img = img.to(device=device, dtype=torch.float32).contiguous()
    img.requires_grad_(True)
    return img


def _init_blended(
    config,
    *,
    run,
    frame_idx,
    end_image_idx,
    content_rgb,
    prev_warp,
    next_warp,
    outputs_rgb,
    mode,
    device,
):
    """Blend this frame's previous-pass result with warped neighbours (ports ``:223-249``).

    Starts from this frame's previous-pass output (``run-1``) with unit divisor,
    then for each available neighbour adds ``warped_neighbour * (reliability *
    blend_scale)`` to the numerator and ``reliability * blend_scale`` to the
    divisor, finally dividing. The blend scale is :func:`blend_weight_for_neighbour`
    (``blend_weight`` for the in-direction neighbour, ``blend_weight_last_pass``
    for the other). The result is preprocessed into the optimizer's space.
    """
    import torch

    from artvid.io.image import preprocess

    start = config.start_number

    # img starts as this frame's previous-pass result in [0,1] RGB (legacy :225
    # image.load(build_OutFilename(params, frameIdx, run-1))).
    img = _output_rgb(config, frame_idx, run - 1, outputs_rgb, device).to(
        device=device, dtype=torch.float32
    ).clone()
    divisor = torch.ones_like(img)

    if frame_idx > start and prev_warp is not None:
        warped_rgb, reliab = prev_warp
        scale = blend_weight_for_neighbour(config, run, neighbour="prev")
        w3 = (
            reliab.unsqueeze(0)
            .expand(3, reliab.shape[-2], reliab.shape[-1])
            .to(device=device, dtype=torch.float32)
            * scale
        )
        img = img + warped_rgb.to(device) * w3
        divisor = divisor + w3

    if frame_idx < end_image_idx and next_warp is not None:
        warped_rgb, reliab = next_warp
        scale = blend_weight_for_neighbour(config, run, neighbour="next")
        w3 = (
            reliab.unsqueeze(0)
            .expand(3, reliab.shape[-2], reliab.shape[-1])
            .to(device=device, dtype=torch.float32)
            * scale
        )
        img = img + warped_rgb.to(device) * w3
        divisor = divisor + w3

    img = img / divisor
    # Optimized image stays float32 on all backends (L-BFGS stability;
    # MPS lacks float64) — see artvid.device.image_optim_dtype().
    img = preprocess(img, mode=mode).to(device=device, dtype=torch.float32).contiguous()
    img.requires_grad_(True)
    return img
