"""Single-pass video stylization (temporal-consistency main loop).

Ports the single-pass frame loop of ``artistic_video.lua:137-286`` plus its two
helpers ``warpImage`` (``:291-304``) and ``processFlowWeights`` (``:307-329``).
The warping and long-term weight combination are **not** re-implemented here;
they are delegated to :func:`artvid.flow.warp.warp_image` and
:func:`artvid.flow.consistency.combine_longterm_weights` respectively (the
legacy helpers map onto those M1 modules per ``docs/02-migration-map.md`` §1).

What this module does (mirroring the Lua loop)
----------------------------------------------
For each frame ``frameIdx`` in ``start_number .. start_number + N - 1``:

* **Init** (``:231-254``). First frame: ``Config.init[0]`` (``random`` |
  ``image``). Subsequent frames: ``Config.init[1]`` (``prevWarped`` — warp the
  *previous output* by the backward flow ``frameIdx-1 -> frameIdx``; or
  ``prev`` / ``image`` / ``random`` / ``first``).
* **Long-term frame selection** (``:159-187``). From ``flow_relative_indices``
  (and ``use_flow_every``) pick the previous frame indices ``J`` that supply
  temporal targets, sorted descending. See :func:`select_previous_indices`.
* **Losses**. Style Gram targets built once (reused across frames, like the Lua
  ``style_losses`` baked into the net). Per-frame content loss on the frame's
  content image, a TV loss, and one temporal
  :class:`~artvid.losses.temporal.WeightedContentLoss` per selected previous
  frame ``j`` — its target is the warped previous output and its weight is the
  long-term-combined reliability mask (``:206-227`` ``prevPlusFlowWeighted``).
* **Optimize + save** (``:261-262``). One
  :func:`artvid.optim.runner.run_optimization` call per frame; result saved as
  ``<basename>-<n>.<ext>`` via :func:`artvid.io.image.save_image`.

Flow sourcing (two interchangeable backends)
--------------------------------------------
The flow needed to warp previous outputs and to build reliability masks can be
either:

* **precomputed** — read ``.flo`` files (and ``.pgm`` reliability masks) named
  by ``Config.flow_pattern`` / ``Config.flow_weight_pattern`` exactly as the
  legacy loop did (``flowFile.load`` + ``image.load``); or
* **on-the-fly RAFT** — compute forward/backward flow between content frames
  with :func:`artvid.flow.raft.compute_flow_pair` and derive the reliability
  mask with :func:`artvid.flow.consistency.consistency_mask`.

The backend is chosen by ``flow_source`` (``"precomputed"`` | ``"raft"`` |
``"auto"``; ``auto`` uses precomputed files when they exist, else RAFT).

Framework-agnostic core: only torch tensor ops, no MPS-specific calls. Device
selection goes through :mod:`artvid.device`. ``torch`` is imported lazily inside
the functions that need it so the filename / index helpers (and their tests)
stay importable without torch.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from artvid.config import Config


# ---------------------------------------------------------------------------
# Pure (torch-free) helpers: frame/flow filenames + long-term index selection
# ---------------------------------------------------------------------------

def select_previous_indices(
    frame_idx: int,
    start_number: int,
    flow_relative_indices: Sequence[int],
    use_flow_every: int = -1,
    temporal_weight: float = 1.0,
) -> List[int]:
    """Select the previous-frame indices ``J`` used for the temporal loss.

    Direct port of ``artistic_video.lua:159-187`` (the block that builds the
    ``J`` table). For each relative step ``s`` in ``flow_relative_indices`` the
    candidate ``frame_idx - s`` is added when it is ``>= start_number``. When
    ``use_flow_every > 0``, additional previous frames at a fixed stride
    (``frame_idx - use_flow_every``, ``... - 2*use_flow_every`` ...) down to
    ``start_number`` are appended if not already present. The result is sorted
    **descending** (closest previous frame first), which is the ordering the
    long-term weight combination (``processFlowWeights`` /
    :func:`artvid.flow.consistency.combine_longterm_weights`) expects.

    For the first frame (``frame_idx == start_number``) or when
    ``temporal_weight == 0`` there are no temporal targets and an empty list is
    returned (legacy guard at ``:159``).

    Args:
        frame_idx: Absolute index of the current frame.
        start_number: Absolute index of the first frame in the sequence.
        flow_relative_indices: Relative steps back (e.g. ``(1,)`` for the
            immediately previous frame, ``(1, 2, 4)`` for long-term).
        use_flow_every: If ``> 0``, also include every ``use_flow_every``-th
            previous frame down to ``start_number`` (legacy ``-use_flow_every``).
            ``-1`` (default) disables this.
        temporal_weight: The temporal-loss strength; ``0`` disables the temporal
            term entirely (no previous frames selected).

    Returns:
        A list of absolute previous-frame indices, sorted descending. Empty when
        no temporal target applies.
    """
    if frame_idx <= start_number or temporal_weight == 0:
        return []

    j: List[int] = []
    for step in flow_relative_indices:
        prev_index = frame_idx - int(step)
        if prev_index >= start_number:
            # The legacy code inserts the index even if it duplicates an earlier
            # one (it only de-dups against the use_flow_every additions). We
            # preserve that: duplicates from flow_relative_indices are kept.
            j.append(prev_index)

    if use_flow_every > 0:
        prev_index = frame_idx - use_flow_every
        while prev_index >= start_number:
            if prev_index not in j:
                j.append(prev_index)
            prev_index -= use_flow_every

    # Sort descending (closest previous frame first) — used by the long-term
    # weight combination (artistic_video.lua:173-174).
    j.sort(reverse=True)
    return j


# ``getFormatedFlowFileName`` uses ``{...}`` for the *from* index and ``[...]``
# for the *to* index, where the inner text is a printf format (e.g. ``%d``).
_BRACE_RE = re.compile(r"\{(.*?)\}")
_BRACKET_RE = re.compile(r"\[(.*?)\]")


def format_flow_filename(pattern: str, from_index: int, to_index: int) -> str:
    """Resolve a flow/weight filename pattern (ports ``getFormatedFlowFileName``).

    Port of ``artistic_video_core.lua:571-578``: substitute the ``{...}``
    placeholder with ``from_index`` and the ``[...]`` placeholder with
    ``to_index``, where each placeholder's inner text is a ``printf`` integer
    format (``%d``, ``%02d`` ...). E.g. the default flow pattern
    ``backward_[%d]_{%d}.flo`` with ``from_index=6, to_index=7`` →
    ``backward_7_6.flo``.

    Args:
        pattern: Pattern containing a ``{...}`` (from) and ``[...]`` (to)
            placeholder, each wrapping a printf integer conversion.
        from_index: The value substituted into ``{...}``.
        to_index: The value substituted into ``[...]``.

    Returns:
        The resolved filename string.
    """
    out = _BRACE_RE.sub(lambda m: m.group(1) % from_index, pattern)
    out = _BRACKET_RE.sub(lambda m: m.group(1) % to_index, out)
    return out


def build_out_filename(config: Config, image_number: int) -> str:
    """Build the per-frame output filename (ports ``build_OutFilename``).

    Port of ``artistic_video_core.lua:558-569`` for the ``iterationOrRun == -1``
    (final) case: ``<output_folder><basename>-<number_format><ext>`` where
    ``basename`` and ``ext`` come from ``Config.output_image`` and the number is
    formatted with ``Config.number_format``. ``image_number`` is the *relative*
    frame number ``abs(frameIdx - start_number + 1)`` (legacy ``:181``,
    ``:258``).

    Args:
        config: The run config (provides ``output_image``, ``output_folder``,
            ``number_format``).
        image_number: 1-based relative frame number.

    Returns:
        The output path string (folder + basename + number + extension).
    """
    out_image = Path(config.output_image)
    ext = out_image.suffix  # includes leading dot, e.g. ".png"
    basename = out_image.stem
    number = config.number_format % image_number
    return f"{config.output_folder}{basename}-{number}{ext}"


def _content_frame_path(config: Config, frame_idx: int) -> Path:
    """Resolve the content-image path for an absolute frame index.

    Mirrors ``getContentImage`` (``artistic_video_core.lua:580-583``):
    ``string.format(content_pattern, frameIdx)``.
    """
    return Path(config.content_pattern % frame_idx)


def discover_num_images(config: Config) -> int:
    """Count contiguous content frames (ports ``calcNumberOfContentImages``).

    When ``Config.num_images == 0`` we autodetect: count frames starting at
    ``start_number`` until the first missing one (legacy
    ``artistic_video_core.lua:547-556``). Otherwise the explicit count is
    returned.
    """
    if config.num_images > 0:
        return config.num_images
    count = 0
    idx = config.start_number
    while _content_frame_path(config, idx).is_file():
        count += 1
        idx += 1
    return count


# ---------------------------------------------------------------------------
# Run result
# ---------------------------------------------------------------------------

@dataclass
class FrameResult:
    """Per-frame outcome of :func:`stylize_video`.

    Attributes:
        frame_idx: Absolute content-frame index.
        output_path: Where the stylized frame was written.
        previous_indices: The ``J`` previous-frame indices used for the temporal
            loss on this frame (empty for the first frame).
        num_iterations: Optimizer iterations actually run for this frame.
    """

    frame_idx: int
    output_path: str
    previous_indices: List[int] = field(default_factory=list)
    num_iterations: int = 0


# ---------------------------------------------------------------------------
# Main single-pass loop
# ---------------------------------------------------------------------------

def stylize_video(
    config: Optional[Config] = None,
    *,
    device: Optional[object] = None,
    flow_source: str = "auto",
) -> List[FrameResult]:
    """Run single-pass temporal-consistency video style transfer.

    Direct port of the single-pass frame loop ``artistic_video.lua:137-286``.
    Iterates content frames in order, initializing each frame from the previous
    stylized output (warped by backward flow) and adding a reliability-weighted
    temporal loss against one or more previous outputs (long-term consistency).

    Args:
        config: Run parameters (:class:`~artvid.config.Config`). The content
            sequence is ``config.content_pattern`` /  ``start_number`` /
            ``num_images``; the style image(s) ``config.style_image``; per-frame
            iteration counts ``config.num_iterations`` (first vs subsequent) and
            init modes ``config.init``.
        device: Optional torch device / string; ``None`` autodetects via
            :func:`artvid.device.get_device`.
        flow_source: Where flow + reliability come from:

            * ``"precomputed"`` — read ``.flo`` (``config.flow_pattern``) and
              ``.pgm`` reliability (``config.flow_weight_pattern``) files.
            * ``"raft"`` — compute flow with
              :func:`artvid.flow.raft.compute_flow_pair` and the reliability mask
              with :func:`artvid.flow.consistency.consistency_mask`.
            * ``"auto"`` (default) — use precomputed files when the required
              ``.flo`` exists, otherwise fall back to RAFT.

    Returns:
        A list of :class:`FrameResult`, one per stylized frame, in frame order.
    """
    import torch

    from artvid.device import enable_mps_fallback, get_device
    from artvid.io.image import load_image, preprocess, save_image

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

    # Preprocessing mode (caffe vs torchvision) follows the chosen VGG weights.
    from artvid.pipeline.stylize_image import _preprocess_mode_for

    mode = _preprocess_mode_for(config)

    num_images = discover_num_images(config)
    if num_images == 0:
        raise FileNotFoundError(
            "No content frames found for pattern "
            f"{config.content_pattern!r} starting at frame {config.start_number}."
        )
    print(f"Detected {num_images} content images.")

    # --- Feature network + style targets (built once, reused per frame) -------
    from artvid.models.vgg import build_feature_net, split_activations

    net = build_feature_net(
        content_layers=config.content_layers,
        style_layers=config.style_layers,
        pooling=config.pooling,
        weights=config.vgg_weights,
    ).to(device)
    net.eval()

    # Style Gram targets are scaled against the *first* content frame's size
    # (legacy getStyleImages uses params.start_number's content image).
    first_content_rgb = load_image(_content_frame_path(config, config.start_number))
    _, first_h, first_w = first_content_rgb.shape

    from artvid.pipeline.stylize_image import (
        _build_style_targets,
        _normalized_blend_weights,
    )

    style_paths = [s.strip() for s in str(config.style_image).split(",") if s.strip()]
    blend_weights = _normalized_blend_weights(config.style_blend_weights, len(style_paths))
    style_targets = _build_style_targets(
        style_paths, blend_weights, (first_h, first_w), config, net, device
    )

    from artvid.losses.content import ContentLoss
    from artvid.losses.style import StyleLoss
    from artvid.losses.temporal import WeightedContentLoss
    from artvid.losses.tv import TVLoss
    from artvid.optim.runner import run_optimization

    style_losses = {
        layer: StyleLoss(
            style_targets[layer],
            strength=config.style_weight,
            normalize=config.normalize_gradients,
            target_is_gram=True,
        ).to(device)
        for layer in config.style_layers
    }

    start = config.start_number
    first_idx = start + config.continue_with - 1
    last_idx = start + num_images - 1

    # Cache of stylized *outputs* in [0,1] RGB CHW space, keyed by absolute frame
    # index, so we can warp previous outputs without re-reading from disk. The
    # legacy code re-loaded them from disk (image.load(build_OutFilename(...)));
    # we keep them in memory but fall back to disk for frames produced in a
    # previous (continue_with) run.
    outputs_rgb: dict[int, "torch.Tensor"] = {}

    results: List[FrameResult] = []

    for frame_idx in range(first_idx, last_idx + 1):
        if config.seed >= 0:
            torch.manual_seed(config.seed)

        content_path = _content_frame_path(config, frame_idx)
        if not content_path.is_file():
            print("No more frames.")
            break

        content_rgb = load_image(content_path)  # (3, H, W) [0,1] RGB
        _, ch, cw = content_rgb.shape
        content_pre = preprocess(content_rgb, mode=mode).unsqueeze(0).to(device)

        is_first = frame_idx == first_idx
        num_iters = config.num_iterations[0] if frame_idx == start else config.num_iterations[1]
        init_mode = config.init[0] if frame_idx == start else config.init[1]

        # --- Content target for this frame ---
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

        # --- Long-term previous-frame selection (J) ---
        prev_indices = select_previous_indices(
            frame_idx,
            start,
            config.flow_relative_indices,
            use_flow_every=config.use_flow_every,
            temporal_weight=config.temporal_weight,
        )

        # --- Build temporal losses: warp each previous output + reliability ---
        temporal_losses: List[WeightedContentLoss] = []
        if prev_indices:
            warped_targets: List["torch.Tensor"] = []
            raw_weights: List["torch.Tensor"] = []
            for prev_index in prev_indices:
                prev_out_rgb = _get_previous_output_rgb(
                    config, prev_index, outputs_rgb, device
                )
                backward_flow = _get_backward_flow(
                    config,
                    frame_idx,
                    prev_index,
                    content_rgb,
                    outputs_rgb,
                    device,
                    flow_source,
                )
                warp_res = _warp_previous_output(prev_out_rgb, backward_flow)
                # Preprocess the warped previous output into VGG space (legacy
                # imgWarped = preprocess(warpImage(...))).
                warped_pre = preprocess(warp_res.image, mode=mode).to(device)
                warped_targets.append(warped_pre)

                reliab = _get_reliability_mask(
                    config,
                    frame_idx,
                    prev_index,
                    content_rgb,
                    backward_flow,
                    device,
                    flow_source,
                )
                # Combine the consistency mask with the warp validity (warp
                # invalid / disoccluded pixels are unreliable too).
                reliab = reliab.to(device) * warp_res.valid.to(reliab.dtype).squeeze(0)
                raw_weights.append(reliab)

            # Long-term weight combination (processFlowWeights). Inputs ordered
            # closest-previous-frame first, matching prev_indices (descending).
            from artvid.flow.consistency import combine_longterm_weights

            combined = combine_longterm_weights(
                raw_weights,
                method=config.combine_flow_weights_method,
                invert=config.invert_flow_weights,
            )
            for target_pre, weight in zip(warped_targets, combined):
                # Expand (H, W) weight to (3, H, W) so it broadcasts over the
                # pixel-space target (legacy flowWeights:expand(3, ...)).
                w3 = weight.unsqueeze(0).expand(3, weight.shape[-2], weight.shape[-1])
                temporal_losses.append(
                    WeightedContentLoss(
                        target_pre,
                        weights=w3.to(device),
                        strength=config.temporal_weight,
                        criterion=config.temporal_criterion,
                        normalize=config.normalize_gradients,
                    ).to(device)
                )

        tv_loss = TVLoss(strength=config.tv_weight).to(device) if config.tv_weight > 0 else None

        # --- Initialization ---
        image_var = _init_frame_image(
            init_mode,
            is_first=is_first,
            frame_idx=frame_idx,
            config=config,
            content_pre=content_pre.squeeze(0),
            content_rgb=content_rgb,
            outputs_rgb=outputs_rgb,
            device=device,
            mode=mode,
            flow_source=flow_source,
        )

        # --- Loss closure (re-runs the feature net each optimizer step) ---
        def loss_fn():
            acts = net(image_var.unsqueeze(0) if image_var.dim() == 3 else image_var)
            content_acts_i, style_acts_i = split_activations(
                acts, config.content_layers, config.style_layers
            )
            terms = {}
            for layer in config.content_layers:
                terms[f"content[{layer}]"] = content_losses[layer](content_acts_i[layer])
            for layer in config.style_layers:
                terms[f"style[{layer}]"] = style_losses[layer](style_acts_i[layer])
            # Temporal losses act on the *image* (pixel space), not features
            # (legacy getWeightedContentLossModuleForLayer is inserted near the
            # input). We compare the current image to the warped previous output.
            for k, tloss in enumerate(temporal_losses):
                terms[f"temporal[{k}]"] = tloss(image_var)
            if tv_loss is not None:
                terms["tv"] = tv_loss(image_var)
            return terms

        out_path = build_out_filename(config, abs(frame_idx - start + 1))

        if config.save_init:
            init_path = (
                f"{config.output_folder}init-"
                f"{config.number_format % abs(frame_idx - start + 1)}.png"
            )
            save_image(image_var.detach(), init_path, mode=mode)

        def save_fn(iteration: int, is_end: bool) -> None:  # noqa: ARG001
            save_image(image_var.detach(), out_path, mode=mode)

        print(f"Stylizing frame {frame_idx} (-> {out_path}); J={prev_indices}")
        run_result = run_optimization(
            image_var,
            loss_fn,
            max_iter=num_iters,
            optimizer=config.optimizer,
            tol_loss_relative=config.tol_loss_relative,
            tol_loss_relative_interval=config.tol_loss_relative_interval,
            learning_rate=config.learning_rate,
            print_iter=config.print_iter,
            save_iter=config.save_iter,
            save_fn=save_fn,
        )

        # Cache the stylized output in [0,1] RGB space for warping later frames.
        from artvid.io.image import deprocess

        outputs_rgb[frame_idx] = (
            deprocess(image_var.detach(), mode=mode).clamp(0.0, 1.0).to(device)
        )

        results.append(
            FrameResult(
                frame_idx=frame_idx,
                output_path=out_path,
                previous_indices=list(prev_indices),
                num_iterations=run_result.num_iterations,
            )
        )

    return results


# ---------------------------------------------------------------------------
# Internal: flow / warp / reliability sourcing
# ---------------------------------------------------------------------------

def _load_output_rgb_from_disk(config: Config, prev_index: int, device):
    """Load a previously-stylized output frame from disk in [0,1] RGB CHW.

    Used when a previous frame was produced in an earlier ``continue_with`` run
    and is not in the in-memory cache (legacy re-loaded it via
    ``image.load(build_OutFilename(...))``).
    """
    from artvid.io.image import load_image

    rel = abs(prev_index - config.start_number + 1)
    path = build_out_filename(config, rel)
    return load_image(path).to(device)


def _get_previous_output_rgb(config: Config, prev_index: int, outputs_rgb, device):
    """Return the previous stylized output (``[0,1]`` RGB) for ``prev_index``."""
    if prev_index in outputs_rgb:
        return outputs_rgb[prev_index]
    return _load_output_rgb_from_disk(config, prev_index, device)


def _read_flo_tensor(path, device):
    """Read a ``.flo`` file as a ``(2, H, W)`` torch float32 tensor on ``device``."""
    import torch

    from artvid.io.flow_io import read_flo

    flow_np = read_flo(path)  # (2, H, W) numpy (u, v) order
    return torch.from_numpy(flow_np).to(device=device, dtype=torch.float32)


def _read_reliability_pgm(path, device):
    """Read a reliability ``.pgm``/``.png`` mask as an ``(H, W)`` tensor in [0,1]."""
    from artvid.io.image import load_image

    # load_image returns (3, H, W) in [0,1]; the reliability map is grayscale so
    # all channels are equal — take channel 0.
    img = load_image(path).to(device)
    return img[0]


def _backward_flow_path(config: Config, frame_idx: int, prev_index: int):
    """Resolve the precomputed backward-flow ``.flo`` path for (prev -> current).

    Legacy ``getFormatedFlowFileName(flow_pattern, abs(prevIndex), abs(frameIdx))``
    (``artistic_video.lua:178``): ``{...}`` = from = ``prev_index``, ``[...]`` =
    to = ``frame_idx``. The default pattern ``backward_[%d]_{%d}.flo`` thus
    resolves to ``backward_<frame_idx>_<prev_index>.flo``.
    """
    return Path(
        format_flow_filename(config.flow_pattern, abs(prev_index), abs(frame_idx))
    )


def _reliability_path(config: Config, frame_idx: int, prev_index: int):
    """Resolve the precomputed reliability ``.pgm`` path for (prev -> current).

    Legacy ``getFormatedFlowFileName(flowWeight_pattern, J[j], abs(frameIdx))``
    (``artistic_video.lua:210``).
    """
    return Path(
        format_flow_filename(config.flow_weight_pattern, abs(prev_index), abs(frame_idx))
    )


def _use_precomputed(flow_source: str, path: Path) -> bool:
    """Decide whether to read a precomputed file vs compute on the fly."""
    if flow_source == "precomputed":
        return True
    if flow_source == "raft":
        return False
    # auto
    return path.is_file()


def _get_backward_flow(
    config, frame_idx, prev_index, content_rgb, outputs_rgb, device, flow_source
):
    """Backward flow (current -> previous) used to warp the previous output.

    Reads the precomputed ``.flo`` when available/selected; otherwise computes
    RAFT flow between the *content* frames ``prev_index`` and ``frame_idx``. We
    use content frames (not stylized outputs) for RAFT because optical flow is a
    property of the underlying scene motion (the legacy precomputed flow was
    likewise computed on the source frames).
    """
    flo_path = _backward_flow_path(config, frame_idx, prev_index)
    if _use_precomputed(flow_source, flo_path):
        return _read_flo_tensor(flo_path, device)

    from artvid.flow.raft import compute_flow

    prev_content = load_content_or_cached(config, prev_index, content_rgb, device)
    # backward flow maps current-frame pixels to the previous frame:
    # compute_flow(img_current, img_previous).
    return compute_flow(content_rgb.to(device), prev_content, device=device)


def _get_reliability_mask(
    config, frame_idx, prev_index, content_rgb, backward_flow, device, flow_source
):
    """Per-pixel reliability mask for warping ``prev_index`` into ``frame_idx``.

    Precomputed path: load the ``.pgm`` written by ``artvid flow`` /
    ``consistencyChecker``. RAFT path: compute the forward flow (previous ->
    current) and run :func:`artvid.flow.consistency.consistency_mask` on the
    (backward, forward) pair.
    """
    import torch

    pgm_path = _reliability_path(config, frame_idx, prev_index)
    if _use_precomputed(flow_source, pgm_path):
        return _read_reliability_pgm(pgm_path, device)

    from artvid.flow.consistency import consistency_mask
    from artvid.flow.raft import compute_flow

    prev_content = load_content_or_cached(config, prev_index, content_rgb, device)
    # forward flow: previous -> current.
    forward_flow = compute_flow(prev_content, content_rgb.to(device), device=device)
    # backward is the primary flow being validated (mirrors the cli `flow`
    # reliable_<j>_<i> arg order: backward validated against forward).
    return consistency_mask(
        backward_flow.to(device),
        forward_flow.to(device),
    ).to(dtype=torch.float32)


def load_content_or_cached(config: Config, frame_idx: int, current_rgb, device):
    """Load the content frame ``frame_idx`` in [0,1] RGB (for RAFT flow)."""
    from artvid.io.image import load_image

    return load_image(_content_frame_path(config, frame_idx)).to(device)


def _warp_previous_output(prev_out_rgb, backward_flow):
    """Backward-warp a previous output (``[0,1]`` RGB) into the current frame.

    Delegates to :func:`artvid.flow.warp.warp_image` (the M1 port of
    ``warpImage``); returns its :class:`~artvid.flow.warp.WarpResult` (warped
    image + validity mask).
    """
    from artvid.flow.warp import warp_image

    return warp_image(prev_out_rgb, backward_flow)


def _init_frame_image(
    init_mode,
    *,
    is_first,
    frame_idx,
    config,
    content_pre,
    content_rgb,
    outputs_rgb,
    device,
    mode,
    flow_source,
):
    """Build the initial optimized image for a frame (ports ``:231-254``).

    Returns a float32 ``requires_grad`` preprocessed-space image tensor.

    Modes:

    * ``random`` — ``randn * 0.001`` (legacy ``:233-234``).
    * ``image``  — clone the preprocessed content frame (legacy ``:235-236``).
    * ``prevWarped`` (subsequent frames only) — warp the previous output by the
      backward flow ``frameIdx-1 -> frameIdx`` (legacy ``:237-243``).
    * ``prev`` (subsequent frames only) — the previous output, preprocessed
      (legacy ``:244-247``).
    * ``first`` — clone the first stylized frame (legacy ``:248-249``).
    """
    import torch

    from artvid.io.image import preprocess

    start = config.start_number

    if init_mode == "random":
        img = torch.randn_like(content_pre, dtype=torch.float32).mul_(0.001)
    elif init_mode == "image":
        img = content_pre.detach().clone().to(torch.float32)
    elif init_mode == "prevWarped" and frame_idx > start:
        prev_index = frame_idx - 1
        prev_out_rgb = _get_previous_output_rgb(config, prev_index, outputs_rgb, device)
        backward_flow = _get_backward_flow(
            config, frame_idx, prev_index, content_rgb, outputs_rgb, device, flow_source
        )
        warp_res = _warp_previous_output(prev_out_rgb, backward_flow)
        img = preprocess(warp_res.image, mode=mode).to(torch.float32)
    elif init_mode == "prev" and frame_idx > start:
        prev_index = frame_idx - 1
        prev_out_rgb = _get_previous_output_rgb(config, prev_index, outputs_rgb, device)
        img = preprocess(prev_out_rgb, mode=mode).to(torch.float32)
    elif init_mode == "first":
        first_out_rgb = _get_previous_output_rgb(config, start, outputs_rgb, device)
        img = preprocess(first_out_rgb, mode=mode).to(torch.float32)
    elif init_mode in ("prevWarped", "prev") and frame_idx <= start:
        # The very first frame of the sequence (frame_idx == start_number) has no
        # previous output to warp; legacy gates these branches on
        # frameIdx > params.start_number (artistic_video.lua:237,244), so here we
        # fall back to random. NOTE: this is keyed on frame_idx > start (not on
        # is_first), so a resumed run with continue_with>1 — whose first iterated
        # frame has frame_idx > start — correctly warps the prior on-disk output.
        img = torch.randn_like(content_pre, dtype=torch.float32).mul_(0.001)
    else:
        raise ValueError(
            f"Invalid initialization method {init_mode!r} for frame {frame_idx}."
        )

    img = img.to(device=device, dtype=torch.float32).contiguous()
    img.requires_grad_(True)
    return img
