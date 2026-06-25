"""``artvid`` command-line entry point.

Dispatches the three subcommands described in ``docs/01-architecture.md`` §3.9:

* ``flow``    — compute forward/backward RAFT optical flow + reliability masks
                for adjacent (and long-term) frame pairs. **Replaces
                ``makeOptFlow.sh`` + ``run-deepflow.sh`` + the C++
                ``consistencyChecker``** (see ``docs/02-migration-map.md`` §1).
* ``stylize`` — per-frame style transfer (single/multi pass). Dispatches to
                :func:`artvid.pipeline.singlepass.stylize_video` (default) or
                :func:`artvid.pipeline.multipass.stylize_video_multipass`
                (``--multipass`` / ``--passes``). Ports the CLI front-end of
                ``artistic_video.lua`` / ``artistic_video_multiPass.lua``
                (the ``cmd:option`` parsing + main-loop entry; see
                ``docs/02-migration-map.md`` §1).
* ``run``     — end-to-end ``video → stylized video`` (replaces
                ``stylizeVideo.sh``): extract frames (:mod:`artvid.io.video`),
                compute flow (``flow`` logic / :mod:`artvid.flow`), stylize,
                re-encode (:mod:`artvid.io.video`).

``flow`` lands in milestone **M1**; ``stylize`` (single/multi pass) in **M2/M3**
and ``run`` (end-to-end) in **M4**. ``main`` parses ``argv``, selects a
subcommand and dispatches.

The ``--engine`` flag selects the style-transfer engine. ``optim`` (the Phase 1
default, fully intact) is the L-BFGS/Adam optimization method ported here;
``diffusion`` (Phase 2) dispatches ``stylize`` / ``run`` to the diffusion video
pipeline (:func:`artvid.diffusion.video.stylize_video_diffusion` — SDXL +
ControlNet + IP-Adapter with latent optical-flow temporal consistency, reusing
the Phase 1 flow stack). The diffusion path lazily imports ``torch`` /
``diffusers`` and downloads model weights from Hugging Face on first run (see
``docs/07-phase2-design.md`` §1/§7); a one-line banner warns about this before
the heavy pipeline is built. ``flow`` is engine-agnostic and unchanged.

The ``flow`` subcommand wires together the already-implemented flow modules:

* :func:`artvid.flow.raft.compute_flow_pair` — forward + backward RAFT flow
  (ports the two ``eval $flowCommandLine`` calls per pair in
  ``makeOptFlow.sh:44-47``);
* :func:`artvid.flow.consistency.consistency_mask` — forward/backward
  consistency reliability mask (ports ``consistencyChecker.cpp`` /
  ``makeOptFlow.sh:49-50``);
* :func:`artvid.io.flow_io.write_flo` — Middlebury ``.flo`` writer;
* :func:`artvid.io.image.load_image` / ``Pillow`` — frame load / mask save.

This module is intentionally thin: argument parsing + orchestration only. The
heavy lifting (and all device handling) lives in the wired modules and
:mod:`artvid.device`. ``torch`` is imported lazily inside the command handlers
so ``--help`` works without torch installed.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence


# ---------------------------------------------------------------------------
# Frame-filename helpers
# ---------------------------------------------------------------------------

def format_frame_path(pattern: str, index: int) -> Path:
    """Resolve a ``printf``-style frame pattern for a single frame index.

    Mirrors the legacy ``string.format(filePattern, i)`` in ``makeOptFlow.sh``
    and ``getContentImage`` (``artistic_video_core.lua:580-587``), e.g.
    ``"frames/frame_%04d.ppm"`` with ``index=7`` → ``frames/frame_0007.ppm``.

    Args:
        pattern: A C/``printf``-style pattern containing exactly one integer
            conversion (e.g. ``%d``, ``%02d``, ``%04d``).
        index: The 1-based frame index to substitute.

    Returns:
        The resolved :class:`pathlib.Path`.
    """
    return Path(pattern % index)


def discover_frame_count(
    pattern: str,
    start_number: int,
    num_images: int,
) -> int:
    """Count contiguous existing frames for ``pattern`` starting at ``start_number``.

    Reproduces the ``[ -a $file2 ]`` existence loop in ``makeOptFlow.sh:42`` and
    the ``num_images == 0`` autodetect in the legacy pipeline: when
    ``num_images`` is ``0`` we count frames until the first missing one;
    otherwise we trust the caller's explicit count.

    Args:
        pattern: ``printf``-style frame pattern.
        start_number: Index of the first frame.
        num_images: Explicit frame count, or ``0`` to autodetect.

    Returns:
        The number of frames available (``>= 0``).
    """
    if num_images > 0:
        return num_images
    count = 0
    idx = start_number
    while format_frame_path(pattern, idx).is_file():
        count += 1
        idx += 1
    return count


def _flow_filename(prefix: str, from_index: int, to_index: int, ext: str) -> str:
    """Build a legacy-compatible flow/mask filename.

    The legacy ``makeOptFlow.sh`` wrote ``forward_<i>_<j>.flo`` (flow from frame
    ``i`` to frame ``j``), ``backward_<j>_<i>.flo`` and ``reliable_<j>_<i>.pgm``
    (``makeOptFlow.sh:44-50``). The single-/multi-pass config patterns
    (``Config.flow_pattern`` etc., ``backward_[%d]_{%d}.flo``) resolve via
    ``getFormatedFlowFileName`` (``artistic_video_core.lua:571-578``) where the
    ``{...}`` placeholder is the *from* index and ``[...]`` the *to* index, so a
    backward flow stored as ``backward_<to>_<from>`` matches
    ``backward_[to]_{from}``. We follow that exact ``<prefix>_<from>_<to>``
    layout so produced files are consumable by the stylize pipeline unchanged.

    Args:
        prefix: ``"forward"``, ``"backward"`` or ``"reliable"``.
        from_index: Source frame index (the ``_<a>_`` slot).
        to_index: Target frame index (the ``_<b>`` slot).
        ext: File extension *with* leading dot (e.g. ``".flo"``, ``".pgm"``).

    Returns:
        The bare filename (no directory).
    """
    return f"{prefix}_{from_index}_{to_index}{ext}"


# ---------------------------------------------------------------------------
# `flow` subcommand
# ---------------------------------------------------------------------------

def _save_reliability(mask, path: Path) -> None:
    """Save a ``(H, W)`` reliability mask in ``[0, 1]`` as an 8-bit grayscale image.

    The legacy ``consistencyChecker`` wrote a ``[0, 255]`` ``.pgm``
    (``consistencyChecker.cpp``); the Lua then rescaled it back to ``[0, 1]`` on
    load. We mirror that on-disk scale (``round(mask * 255)``) so the saved masks
    are byte-compatible with the legacy ``.pgm`` reliability weights and with
    ``image.load``-style readers. The extension of ``path`` selects the encoder
    (``.pgm`` or ``.png`` both work via Pillow).

    Args:
        mask: ``(H, W)`` float tensor in ``[0, 1]`` on any device.
        path: Output path; parent directories are created if needed.
    """
    import torch  # noqa: F401  (kept explicit for the typed .to / .mul chain)
    from PIL import Image

    arr = (
        mask.detach()
        .to("cpu", dtype=torch.float32)
        .clamp(0.0, 1.0)
        .mul(255.0)
        .round()
        .to(torch.uint8)
        .numpy()
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr, mode="L").save(path)


def cmd_flow(args: argparse.Namespace) -> int:
    """Compute and save forward/backward flow + reliability for a frame sequence.

    This is the Python replacement for ``makeOptFlow.sh``. For every frame ``i``
    and every long-term step ``s`` in ``--steps`` (default ``[1]`` = adjacent
    frames only), with ``j = i + s``:

    1. forward flow  ``i → j``  and backward flow ``j → i`` are computed in one
       :func:`artvid.flow.raft.compute_flow_pair` call (RAFT replaces the two
       DeepFlow invocations in ``makeOptFlow.sh:44-47``);
    2. two reliability masks are produced via
       :func:`artvid.flow.consistency.consistency_mask` — one for each warp
       direction, matching the two ``consistencyChecker`` calls in
       ``makeOptFlow.sh:49-50``:
         * ``reliable_<j>_<i>`` validates the **backward** flow against the
           forward flow (used to warp frame ``i`` into frame ``j`` — the common
           case for the stylize temporal loss);
         * ``reliable_<i>_<j>`` validates the **forward** flow against the
           backward flow (used by the multi-pass backward sweep).
    3. all four artifacts are written under ``--out`` with legacy-compatible
       names (see :func:`_flow_filename`).

    Multiple ``--steps`` reproduce running ``makeOptFlow.sh`` once per step size
    to build the long-term flow consumed by ``Config.flow_relative_indices``.

    Args:
        args: Parsed CLI namespace (see :func:`build_parser`).

    Returns:
        Process exit code (``0`` on success).
    """
    # Lazy imports so `--help` / unrelated subcommands work without torch.
    import torch

    from artvid import device as _device
    from artvid.flow import raft as _raft
    from artvid.flow import consistency as _consistency
    from artvid.io import flow_io as _flow_io
    from artvid.io import image as _image

    _device.enable_mps_fallback()
    dev = _device.get_device(args.device)

    pattern = args.frames
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    steps = sorted(set(int(s) for s in args.steps))
    if any(s <= 0 for s in steps):
        print("error: --steps must be positive integers", file=sys.stderr)
        return 2

    total = discover_frame_count(pattern, args.start_number, args.num_images)
    if total < 2:
        print(
            f"error: need at least 2 frames matching {pattern!r} "
            f"(found {total}); nothing to do.",
            file=sys.stderr,
        )
        return 2

    last_index = args.start_number + total - 1
    mask_ext = args.mask_ext if args.mask_ext.startswith(".") else "." + args.mask_ext

    written = 0
    skipped = 0
    for step in steps:
        for i in range(args.start_number, last_index + 1):
            j = i + step
            if j > last_index:
                break

            path_i = format_frame_path(pattern, i)
            path_j = format_frame_path(pattern, j)
            if not (path_i.is_file() and path_j.is_file()):
                # A gap in the sequence; mirror makeOptFlow.sh's break-on-missing.
                continue

            fwd_path = out_dir / _flow_filename("forward", i, j, ".flo")
            bwd_path = out_dir / _flow_filename("backward", j, i, ".flo")
            # reliable_<j>_<i>: trustworthiness of the backward (j->i) warp.
            rel_back_path = out_dir / _flow_filename("reliable", j, i, mask_ext)
            # reliable_<i>_<j>: trustworthiness of the forward (i->j) warp.
            rel_fwd_path = out_dir / _flow_filename("reliable", i, j, mask_ext)

            if (
                not args.overwrite
                and fwd_path.is_file()
                and bwd_path.is_file()
                and rel_back_path.is_file()
                and rel_fwd_path.is_file()
            ):
                # Same idempotency guard as makeOptFlow.sh's `[ ! -f ... ]`.
                skipped += 1
                continue

            # Frames are RGB [0,1] CHW — exactly what RAFT expects (NOT the
            # caffe/VGG preprocessing; see flow/raft.py docstring).
            img_i = _image.load_image(path_i).to(dev)
            img_j = _image.load_image(path_j).to(dev)

            pair = _raft.compute_flow_pair(img_i, img_j, device=dev)
            forward = pair.forward    # i -> j, (2, H, W)
            backward = pair.backward  # j -> i, (2, H, W)

            # Reliability of the backward warp: backward is the "primary" flow,
            # forward is its cross-check (mirrors consistencyChecker arg order
            # `backward forward reliable_<j>_<i>` in makeOptFlow.sh:49).
            rel_back = _consistency.consistency_mask(
                backward,
                forward,
                smooth_sigma=args.smooth_sigma,
            )
            # Reliability of the forward warp (makeOptFlow.sh:50).
            rel_fwd = _consistency.consistency_mask(
                forward,
                backward,
                smooth_sigma=args.smooth_sigma,
            )

            # .flo writer is NumPy-based; move flow to CPU first.
            _flow_io.write_flo(fwd_path, forward.detach().to("cpu").numpy())
            _flow_io.write_flo(bwd_path, backward.detach().to("cpu").numpy())
            _save_reliability(rel_back, rel_back_path)
            _save_reliability(rel_fwd, rel_fwd_path)

            written += 1
            if args.verbose:
                print(
                    f"[flow] step={step} {i}->{j}: "
                    f"{fwd_path.name}, {bwd_path.name}, "
                    f"{rel_back_path.name}, {rel_fwd_path.name}"
                )

    print(
        f"[flow] done: {written} pair(s) written, {skipped} skipped, "
        f"steps={steps}, out={out_dir}"
    )
    return 0


# ---------------------------------------------------------------------------
# Config construction (CLI flags -> artvid.config.Config)
# ---------------------------------------------------------------------------

def _is_diffusion(args: argparse.Namespace) -> bool:
    """Return True when ``--engine diffusion`` (the Phase 2 engine) is selected."""
    return getattr(args, "engine", "optim") == "diffusion"


def _diffusion_banner() -> None:
    """Print a one-line first-run notice before the diffusion pipeline is built.

    The diffusion engine lazily imports ``torch`` / ``diffusers`` and downloads
    the SDXL base + ControlNet + IP-Adapter weights from Hugging Face on first
    run (multiple GB; cached afterwards under ``HF_HOME``). It also acknowledges
    the model licenses (CreativeML Open RAIL++-M for SDXL; see
    ``docs/07-phase2-design.md`` §7). Surfacing this up front avoids a silent
    long pause on the first invocation.
    """
    print(
        "[diffusion] Phase 2 engine: SDXL + ControlNet + IP-Adapter with latent "
        "optical-flow temporal consistency.\n"
        "[diffusion] First run downloads model weights from Hugging Face "
        "(several GB; cached under HF_HOME). Requires torch + diffusers and a "
        "device with enough memory (tuned for Apple Silicon / MPS).\n"
        "[diffusion] Models carry their own licenses (SDXL: CreativeML Open "
        "RAIL++-M); see docs/07-phase2-design.md §7.",
        file=sys.stderr,
    )


# CLI-flag attribute name -> Config field name. Only attributes that are present
# (not None) on the namespace are applied, so unset flags keep Config defaults.
_STYLIZE_CONFIG_MAP: dict[str, str] = {
    "style_blend_weights": "style_blend_weights",
    "num_images": "num_images",
    "start_number": "start_number",
    "continue_with": "continue_with",
    "number_format": "number_format",
    # flow (single-pass) patterns
    "flow_pattern": "flow_pattern",
    "flow_weight_pattern": "flow_weight_pattern",
    "flow_relative_indices": "flow_relative_indices",
    "use_flow_every": "use_flow_every",
    "invert_flow_weights": "invert_flow_weights",
    # flow (multi-pass) patterns
    "forward_flow_pattern": "forward_flow_pattern",
    "backward_flow_pattern": "backward_flow_pattern",
    "forward_flow_weight_pattern": "forward_flow_weight_pattern",
    "backward_flow_weight_pattern": "backward_flow_weight_pattern",
    # multi-pass options
    "blend_weight": "blend_weight",
    "blend_weight_last_pass": "blend_weight_last_pass",
    "use_temporal_loss_after": "use_temporal_loss_after",
    "passes": "num_passes",
    "continue_with_pass": "continue_with_pass",
    # optimization
    "content_weight": "content_weight",
    "style_weight": "style_weight",
    "temporal_weight": "temporal_weight",
    "tv_weight": "tv_weight",
    "temporal_criterion": "temporal_criterion",
    "num_iterations": "num_iterations",
    "tol_loss_relative": "tol_loss_relative",
    "tol_loss_relative_interval": "tol_loss_relative_interval",
    "normalize_gradients": "normalize_gradients",
    "init": "init",
    "optimizer": "optimizer",
    "learning_rate": "learning_rate",
    # output
    "print_iter": "print_iter",
    "save_iter": "save_iter",
    "output_image": "output_image",
    "output_folder": "output_folder",
    "save_init": "save_init",
    # model / other
    "style_scale": "style_scale",
    "pooling": "pooling",
    "seed": "seed",
    "content_layers": "content_layers",
    "style_layers": "style_layers",
    "combine_flow_weights_method": "combine_flow_weights_method",
    "vgg_weights": "vgg_weights",
    "device": "device",
    # --- Diffusion engine (Phase 2; only consulted when --engine diffusion).
    # These map onto the Config "diffusion" field group (docs/07-phase2-design.md
    # §4.1) and are ignored by the optim path (they are not referenced by it).
    "diff_base_model": "diff_base_model",
    "controlnet_model": "controlnet_model",
    "controlnet_kind": "controlnet_kind",
    "controlnet_scale": "controlnet_scale",
    "ip_adapter_repo": "ip_adapter_repo",
    "ip_adapter_subfolder": "ip_adapter_subfolder",
    "ip_adapter_weight": "ip_adapter_weight",
    "ip_adapter_scale": "ip_adapter_scale",
    "diff_prompt": "diff_prompt",
    "diff_negative_prompt": "diff_negative_prompt",
    "diff_steps": "diff_steps",
    "guidance_scale": "guidance_scale",
    "denoise_strength": "denoise_strength",
    "diff_scheduler": "diff_scheduler",
    "temporal_strength": "temporal_strength",
    "temporal_fuse_start": "temporal_fuse_start",
    "temporal_fuse_end": "temporal_fuse_end",
    "latent_consistency_weight": "latent_consistency_weight",
    "latent_reliability_gamma": "latent_reliability_gamma",
    "warp_space": "warp_space",
    "use_anchor": "use_anchor",
    "vae_factor": "vae_factor",
}


def build_config(args: argparse.Namespace):
    """Construct a :class:`artvid.config.Config` from parsed CLI ``args``.

    Maps the ``stylize`` / ``run`` flags onto Config fields (positional
    ``frames``→``content_pattern`` and ``style``→``style_image``), applying only
    the flags the user actually provided so unset options fall back to the legacy
    Config defaults. ``--args`` legacy parameter files are layered first (lowest
    priority) via :func:`artvid.config.load_args_file` so explicit CLI flags win.

    The multi-pass ``temporal_weight`` default (5e2, vs single-pass 1e3) is
    applied only when the user did not pass ``--temporal-weight`` and the run is
    multi-pass (mirrors ``docs/02-migration-map.md`` §3).
    """
    from artvid.config import Config, load_args_file

    overrides: dict[str, object] = {}

    # 1. Legacy -args file(s), lowest priority.
    for args_file in getattr(args, "args_file", None) or []:
        overrides.update(load_args_file(args_file))

    # 2. Positionals.
    if getattr(args, "frames", None) is not None:
        overrides["content_pattern"] = args.frames
    if getattr(args, "style", None) is not None:
        overrides["style_image"] = args.style

    # 3. Explicit flags (only those the user set, i.e. not None).
    for attr, field_name in _STYLIZE_CONFIG_MAP.items():
        if not hasattr(args, attr):
            continue
        value = getattr(args, attr)
        if value is None:
            continue
        overrides[field_name] = value

    # Multi-pass temporal_weight default differs from single-pass.
    multipass = bool(getattr(args, "multipass", False)) or (
        getattr(args, "passes", None) is not None
    )
    if multipass and "temporal_weight" not in overrides:
        overrides["temporal_weight"] = 5e2

    return Config(**overrides)


def _flow_source(args: argparse.Namespace) -> str:
    """Resolve the pipeline ``flow_source`` ("auto" | "precomputed" | "raft")."""
    return getattr(args, "flow_source", None) or "auto"


# ---------------------------------------------------------------------------
# `stylize` subcommand
# ---------------------------------------------------------------------------

def cmd_stylize(args: argparse.Namespace) -> int:
    """Per-frame style transfer (single- or multi-pass).

    Ports the CLI front-end of ``artistic_video.lua`` /
    ``artistic_video_multiPass.lua``: builds a :class:`~artvid.config.Config`
    from the CLI flags and dispatches to the matching pipeline:

    * default → :func:`artvid.pipeline.singlepass.stylize_video` (M2);
    * ``--multipass`` / ``--passes N`` →
      :func:`artvid.pipeline.multipass.stylize_video_multipass` (M3).

    Device selection goes through :mod:`artvid.device` (via ``Config.device`` /
    ``--device``), replacing the legacy ``-gpu`` / ``-backend`` options.

    With ``--engine diffusion`` (Phase 2) this instead dispatches to
    :func:`artvid.diffusion.video.stylize_video_diffusion` — the SDXL +
    ControlNet + IP-Adapter pipeline with latent optical-flow temporal
    consistency. The diffusion engine has no forward/backward multi-pass notion,
    so ``--multipass`` / ``--passes`` are rejected for it; the positional
    ``style`` argument is reused as the IP-Adapter style reference.

    Args:
        args: Parsed CLI namespace (see :func:`build_parser`).

    Returns:
        Process exit code (``0`` on success).
    """
    flow_source = _flow_source(args)

    if _is_diffusion(args):
        if bool(args.multipass) or (args.passes is not None):
            print(
                "error: --multipass / --passes are not supported with "
                "--engine diffusion (the diffusion engine is single-pass; "
                "temporal coherence comes from latent flow consistency, not "
                "forward/backward passes).",
                file=sys.stderr,
            )
            return 2
        config = build_config(args)
        _diffusion_banner()
        from artvid.diffusion.video import stylize_video_diffusion

        results = stylize_video_diffusion(config, flow_source=flow_source)
        print(f"[stylize] diffusion done: {len(results)} frame(s) stylized.")
        if args.verbose:
            for r in results:
                print(f"  {r.output_path}")
        return 0

    config = build_config(args)
    multipass = bool(args.multipass) or (args.passes is not None)

    if multipass:
        from artvid.pipeline.multipass import stylize_video_multipass

        results = stylize_video_multipass(config, flow_source=flow_source)
        print(
            f"[stylize] multi-pass done: {len(results)} (frame, pass) result(s), "
            f"{config.num_passes} pass(es)."
        )
    else:
        from artvid.pipeline.singlepass import stylize_video

        results = stylize_video(config, flow_source=flow_source)
        print(f"[stylize] single-pass done: {len(results)} frame(s) stylized.")

    if args.verbose:
        for r in results:
            print(f"  {r.output_path}")
    return 0


# ---------------------------------------------------------------------------
# `run` subcommand (end-to-end, replaces stylizeVideo.sh)
# ---------------------------------------------------------------------------

def _compute_flow_for_run(
    content_pattern: str,
    flow_dir: Path,
    *,
    start_number: int,
    num_images: int,
    steps,
    smooth_sigma: float,
    mask_ext: str,
    device,
    overwrite: bool,
    verbose: bool,
) -> None:
    """Compute forward/backward flow + reliability for the extracted frames.

    Thin reuse of the ``flow`` subcommand machinery (the same RAFT +
    consistency pipeline driven by :func:`cmd_flow`), invoked in-process so
    ``run`` does not shell out to itself. Mirrors the
    ``bash makeOptFlow.sh ...`` step of ``stylizeVideo.sh:81``.
    """
    flow_args = argparse.Namespace(
        frames=content_pattern,
        out=str(flow_dir),
        start_number=start_number,
        num_images=num_images,
        steps=list(steps),
        smooth_sigma=smooth_sigma,
        mask_ext=mask_ext,
        device=None,  # cmd_flow re-derives via artvid.device; keep autodetect
        overwrite=overwrite,
        verbose=verbose,
    )
    # cmd_flow handles its own device selection through artvid.device.
    cmd_flow(flow_args)


def cmd_run(args: argparse.Namespace) -> int:
    """End-to-end ``video → stylized video`` (replaces ``stylizeVideo.sh``).

    Ports the orchestration of ``stylizeVideo.sh``:

    1. **extract frames** from the input video with
       :func:`artvid.io.video.extract_frames` (``stylizeVideo.sh:58-64``);
    2. **compute optical flow** + reliability with the RAFT/consistency pipeline
       (``flow`` logic; replaces ``makeOptFlow.sh`` at ``stylizeVideo.sh:81``) —
       skipped with ``--no-flow``;
    3. **stylize** the frame sequence single- or multi-pass
       (``stylizeVideo.sh:84-95``), wiring the precomputed flow patterns;
    4. **re-encode** the stylized frames into a video with
       :func:`artvid.io.video.encode_video` (``stylizeVideo.sh:98``).

    The legacy interactive ``-backend`` / ``-gpu`` prompts are replaced by
    :mod:`artvid.device` (``--device``); the resolution prompt by
    ``--resolution``.

    With ``--engine diffusion`` (Phase 2) steps 1 (extract), 2 (flow precompute —
    **fully reused**: the diffusion engine consumes the same ``.flo`` /
    reliability files) and 4 (re-encode) are unchanged; only step 3 swaps the
    optim single/multi-pass stylize for
    :func:`artvid.diffusion.video.stylize_video_diffusion`. The diffusion engine
    is single-pass (no ``--multipass`` / ``--passes``).

    Args:
        args: Parsed CLI namespace (see :func:`build_parser`).

    Returns:
        Process exit code (``0`` on success).
    """
    diffusion = _is_diffusion(args)

    from artvid.io.video import extract_frames, encode_video

    video_path = Path(args.video)
    if not video_path.is_file():
        print(f"error: input video not found: {video_path}", file=sys.stderr)
        return 2

    multipass = bool(args.multipass) or (args.passes is not None)
    if diffusion and multipass:
        print(
            "error: --multipass / --passes are not supported with "
            "--engine diffusion (the diffusion engine is single-pass).",
            file=sys.stderr,
        )
        return 2

    # Work folder, derived from the video basename like stylizeVideo.sh:22-29.
    if args.work_dir is not None:
        work_dir = Path(args.work_dir)
    else:
        work_dir = Path(video_path.stem.replace("%", "x"))
    frames_dir = work_dir / "frames"
    flow_dir = work_dir / "flow"
    out_dir = work_dir / "out"
    work_dir.mkdir(parents=True, exist_ok=True)

    # --- 1. Extract frames -------------------------------------------------
    frame_pattern = args.frame_pattern  # ffmpeg pattern, e.g. frame_%04d.ppm
    content_pattern = extract_frames(
        video_path,
        frames_dir,
        pattern=frame_pattern,
        resolution=args.resolution,
    )
    print(f"[run] extracted frames -> {content_pattern}")

    # --- 2. Optical flow ----------------------------------------------------
    if not args.no_flow:
        from artvid import device as _device

        _device.enable_mps_fallback()
        dev = _device.get_device(args.device)
        print(f"[run] computing optical flow on {dev} ...")
        steps = sorted(set(int(s) for s in (args.steps or [1])))
        _compute_flow_for_run(
            content_pattern,
            flow_dir,
            start_number=args.start_number or 1,
            num_images=args.num_images or 0,
            steps=steps,
            smooth_sigma=args.smooth_sigma,
            mask_ext=args.mask_ext,
            device=dev,
            overwrite=args.overwrite_flow,
            verbose=args.verbose,
        )

    # --- 3. Stylize ---------------------------------------------------------
    from artvid.config import Config, load_args_file

    overrides: dict[str, object] = {}
    for args_file in args.args_file or []:
        overrides.update(load_args_file(args_file))
    overrides["content_pattern"] = content_pattern
    overrides["style_image"] = args.style
    overrides["output_folder"] = str(out_dir) + "/"
    overrides["number_format"] = args.number_format
    if args.start_number is not None:
        overrides["start_number"] = args.start_number
    if args.num_images:
        overrides["num_images"] = args.num_images

    # Wire the precomputed flow file patterns to the flow output folder, using
    # the legacy [from]/{to} placeholder convention (matches `artvid flow`
    # output names and getFormatedFlowFileName).
    mask_ext = args.mask_ext if args.mask_ext.startswith(".") else "." + args.mask_ext
    overrides["flow_pattern"] = str(flow_dir / "backward_[%d]_{%d}.flo")
    overrides["flow_weight_pattern"] = str(flow_dir / f"reliable_[%d]_{{%d}}{mask_ext}")
    overrides["forward_flow_pattern"] = str(flow_dir / "forward_[%d]_{%d}.flo")
    overrides["backward_flow_pattern"] = str(flow_dir / "backward_[%d]_{%d}.flo")
    overrides["forward_flow_weight_pattern"] = str(
        flow_dir / f"reliable_[%d]_{{%d}}{mask_ext}"
    )
    overrides["backward_flow_weight_pattern"] = str(
        flow_dir / f"reliable_[%d]_{{%d}}{mask_ext}"
    )

    # Explicit optimization / model flags (only those the user set).
    for attr, field_name in _STYLIZE_CONFIG_MAP.items():
        if attr in ("num_images", "start_number", "number_format"):
            continue  # already handled above
        if not hasattr(args, attr):
            continue
        value = getattr(args, attr)
        if value is None:
            continue
        overrides[field_name] = value

    if multipass and "temporal_weight" not in overrides:
        overrides["temporal_weight"] = 5e2

    out_dir.mkdir(parents=True, exist_ok=True)
    config = Config(**overrides)

    flow_source = "precomputed" if not args.no_flow else _flow_source(args)
    if args.no_flow and getattr(args, "flow_source", None) is None:
        flow_source = "raft"

    if diffusion:
        # Phase 2: swap only the stylize step. The diffusion engine writes frames
        # with the SAME single-pass naming (artvid.diffusion.video._output_path_for
        # -> pipeline.singlepass.build_out_filename), so the single-pass
        # encode_pattern below re-encodes them unchanged.
        _diffusion_banner()
        from artvid.diffusion.video import stylize_video_diffusion

        results = stylize_video_diffusion(config, flow_source=flow_source)
        out_stem = Path(config.output_image).stem
        out_ext = Path(config.output_image).suffix
        encode_pattern = (
            f"{config.output_folder}{out_stem}-{config.number_format}{out_ext}"
        )
    elif multipass:
        from artvid.pipeline.multipass import stylize_video_multipass

        results = stylize_video_multipass(config, flow_source=flow_source)
        last_pass = config.num_passes
        # Multi-pass final frames are named <basename>-<frame>_<pass><ext>.
        out_stem = Path(config.output_image).stem
        out_ext = Path(config.output_image).suffix
        encode_pattern = (
            f"{config.output_folder}{out_stem}-{config.number_format}"
            f"_{last_pass}{out_ext}"
        )
    else:
        from artvid.pipeline.singlepass import stylize_video

        results = stylize_video(config, flow_source=flow_source)
        out_stem = Path(config.output_image).stem
        out_ext = Path(config.output_image).suffix
        # Single-pass frames are named <basename>-<number><ext>, number is the
        # relative frame index formatted with number_format.
        encode_pattern = (
            f"{config.output_folder}{out_stem}-{config.number_format}{out_ext}"
        )
    print(f"[run] stylized {len(results)} frame(s).")

    # --- 4. Re-encode -------------------------------------------------------
    extension = video_path.suffix.lstrip(".") or "mp4"
    out_video = (
        Path(args.output)
        if args.output is not None
        else work_dir.parent / f"{work_dir.name}-stylized.{extension}"
    )
    encode_video(encode_pattern, out_video, framerate=args.framerate)
    print(f"[run] wrote stylized video -> {out_video}")
    return 0


# ---------------------------------------------------------------------------
# Argument parsing / dispatch
# ---------------------------------------------------------------------------

def _add_stylize_flags(p: argparse.ArgumentParser) -> None:
    """Add the shared ``Config`` / pipeline flags used by ``stylize`` and ``run``.

    Every flag defaults to ``None`` (or ``store_true``/``False`` for booleans) so
    that :func:`build_config` only applies the options the user explicitly set,
    leaving everything else at the legacy :class:`~artvid.config.Config`
    defaults. The flag names mirror the Config field names (and thus the legacy
    ``cmd:option`` names) per ``docs/02-migration-map.md`` §3.
    """
    # --- pipeline selection -------------------------------------------------
    p.add_argument(
        "--multipass",
        action="store_true",
        help="Use the multi-pass (forward/backward) pipeline instead of single-pass.",
    )
    p.add_argument(
        "--passes",
        type=int,
        default=None,
        help="Number of multi-pass passes (-> Config.num_passes; implies --multipass).",
    )
    p.add_argument(
        "--flow-source",
        choices=("auto", "precomputed", "raft"),
        default=None,
        help="Where flow + reliability come from: auto (default), precomputed (.flo/.pgm), or raft (on-the-fly).",
    )
    p.add_argument(
        "--args",
        dest="args_file",
        action="append",
        default=None,
        metavar="FILE",
        help="Legacy -args parameter file (repeatable, applied lowest priority).",
    )

    # --- basic / sequence ---------------------------------------------------
    p.add_argument("--style-blend-weights", default=None,
                   help="Comma-separated blend weights for multiple style images.")
    p.add_argument("--num-images", type=int, default=None,
                   help="Number of frames; 0 = autodetect (-> Config.num_images).")
    p.add_argument("--start-number", type=int, default=None,
                   help="Index of the first frame (-> Config.start_number).")
    p.add_argument("--continue-with", type=int, default=None,
                   help="Resume from this (1-based) frame in the sequence.")
    p.add_argument("--number-format", default=None,
                   help="printf format for output frame numbers (-> Config.number_format).")

    # --- flow patterns (single-pass) ---------------------------------------
    p.add_argument("--flow-pattern", default=None,
                   help="Backward flow .flo pattern (-> Config.flow_pattern).")
    p.add_argument("--flow-weight-pattern", default=None,
                   help="Reliability .pgm pattern (-> Config.flow_weight_pattern).")
    p.add_argument("--flow-relative-indices", default=None,
                   help="Comma-separated long-term step sizes (-> Config.flow_relative_indices).")
    p.add_argument("--use-flow-every", type=int, default=None,
                   help="Include every Nth previous frame; -1 disables (-> Config.use_flow_every).")
    p.add_argument("--invert-flow-weights", action="store_const", const=True, default=None,
                   help="Invert reliability weights (-> Config.invert_flow_weights).")

    # --- flow patterns (multi-pass) ----------------------------------------
    p.add_argument("--forward-flow-pattern", default=None,
                   help="Forward flow .flo pattern (-> Config.forward_flow_pattern).")
    p.add_argument("--backward-flow-pattern", default=None,
                   help="Backward flow .flo pattern (-> Config.backward_flow_pattern).")
    p.add_argument("--forward-flow-weight-pattern", default=None,
                   help="Forward reliability pattern (-> Config.forward_flow_weight_pattern).")
    p.add_argument("--backward-flow-weight-pattern", default=None,
                   help="Backward reliability pattern (-> Config.backward_flow_weight_pattern).")

    # --- multi-pass options -------------------------------------------------
    p.add_argument("--blend-weight", type=float, default=None,
                   help="Multi-pass neighbour blend weight (-> Config.blend_weight).")
    p.add_argument("--blend-weight-last-pass", type=float, default=None,
                   help="Multi-pass opposite-direction blend weight (-> Config.blend_weight_last_pass).")
    p.add_argument("--use-temporal-loss-after", type=int, default=None,
                   help="Enable temporal loss from this pass onward (-> Config.use_temporal_loss_after).")
    p.add_argument("--continue-with-pass", type=int, default=None,
                   help="Resume multi-pass from this pass (-> Config.continue_with_pass).")

    # --- optimization -------------------------------------------------------
    p.add_argument("--content-weight", type=float, default=None,
                   help="Content reconstruction weight (-> Config.content_weight).")
    p.add_argument("--style-weight", type=float, default=None,
                   help="Style reconstruction weight (-> Config.style_weight).")
    p.add_argument("--temporal-weight", type=float, default=None,
                   help="Temporal consistency weight (-> Config.temporal_weight; multi-pass default 5e2).")
    p.add_argument("--tv-weight", type=float, default=None,
                   help="Total-variation weight (-> Config.tv_weight).")
    p.add_argument("--temporal-criterion", choices=("mse", "smoothl1"), default=None,
                   help="Temporal loss criterion (-> Config.temporal_criterion).")
    p.add_argument("--num-iterations", default=None,
                   help="Iterations 'first,subsequent' or single value (-> Config.num_iterations).")
    p.add_argument("--tol-loss-relative", type=float, default=None,
                   help="Relative-loss stopping tolerance (-> Config.tol_loss_relative).")
    p.add_argument("--tol-loss-relative-interval", type=int, default=None,
                   help="Iterations between relative-loss checks (-> Config.tol_loss_relative_interval).")
    p.add_argument("--normalize-gradients", action="store_const", const=True, default=None,
                   help="Normalize loss gradients (-> Config.normalize_gradients).")
    p.add_argument("--init", default=None,
                   help="Init mode 'first,subsequent' (random|image|prev|prevWarped|first; -> Config.init).")
    p.add_argument("--optimizer", choices=("lbfgs", "adam"), default=None,
                   help="Optimizer (-> Config.optimizer).")
    p.add_argument("--learning-rate", type=float, default=None,
                   help="Adam learning rate (-> Config.learning_rate).")

    # --- output -------------------------------------------------------------
    p.add_argument("--print-iter", type=int, default=None,
                   help="Print loss every N iterations (-> Config.print_iter).")
    p.add_argument("--save-iter", type=int, default=None,
                   help="Save intermediate every N iterations; 0 = only final (-> Config.save_iter).")
    p.add_argument("--output-image", default=None,
                   help="Output image basename+ext, e.g. out.png (-> Config.output_image).")
    p.add_argument("--output-folder", default=None,
                   help="Output folder prefix (-> Config.output_folder).")
    p.add_argument("--save-init", action="store_const", const=True, default=None,
                   help="Save the per-frame initialization image (-> Config.save_init).")

    # --- model / other ------------------------------------------------------
    p.add_argument("--style-scale", type=float, default=None,
                   help="Style-image scale relative to content (-> Config.style_scale).")
    p.add_argument("--pooling", choices=("max", "avg"), default=None,
                   help="VGG pooling type (-> Config.pooling).")
    p.add_argument("--seed", type=int, default=None,
                   help="RNG seed; -1 disables (-> Config.seed).")
    p.add_argument("--content-layers", default=None,
                   help="Comma-separated content layers (-> Config.content_layers).")
    p.add_argument("--style-layers", default=None,
                   help="Comma-separated style layers (-> Config.style_layers).")
    p.add_argument("--combine-flow-weights-method", choices=("normalize", "closestFirst"), default=None,
                   help="Long-term weight combination method (-> Config.combine_flow_weights_method).")
    p.add_argument("--vgg-weights", default=None,
                   help="'torchvision' or a path to caffe VGG-19 weights (-> Config.vgg_weights).")
    p.add_argument(
        "--device",
        choices=("mps", "cuda", "cpu"),
        default=None,
        help="Compute device; default autodetect mps>cuda>cpu (-> Config.device).",
    )

    # --- diffusion engine (Phase 2; only used with --engine diffusion) ------
    # All default=None so build_config applies them only when the user set them,
    # leaving Config's diffusion-group defaults (docs/07-phase2-design.md §4.1)
    # intact. Ignored by the optim engine.
    _add_diffusion_flags(p)

    p.add_argument(
        "-v", "--verbose", action="store_true",
        help="Print each output path as it is produced.",
    )


def _add_diffusion_flags(p: argparse.ArgumentParser) -> None:
    """Add the Phase 2 diffusion-engine flags (only consulted with --engine diffusion).

    Every flag defaults to ``None`` so :func:`build_config` only applies the ones
    the user explicitly set, leaving the :class:`~artvid.config.Config`
    "diffusion" field-group defaults (docs/07-phase2-design.md §4.1) intact. These
    flags are ignored by the optim engine. Numerically/qualitatively sensitive
    knobs (controlnet_scale, ip_adapter_scale, guidance_scale, the temporal-flow
    consistency strength/window, reliability gamma) are TODO(tuning) on the M5 Max
    — the CLI just surfaces them; their defaults live in Config.
    """
    g = p.add_argument_group(
        "diffusion engine (Phase 2; only with --engine diffusion)"
    )
    # --- model stack (HF ids) ----------------------------------------------
    g.add_argument("--diff-base-model", default=None,
                   help="Base T2I HF id (-> Config.diff_base_model; default SDXL base).")
    g.add_argument("--controlnet-model", default=None,
                   help="Structure ControlNet HF id (-> Config.controlnet_model).")
    g.add_argument("--controlnet-kind", choices=("depth", "canny", "lineart", "hed", "tile"),
                   default=None,
                   help="Structure preprocessor signal (-> Config.controlnet_kind).")
    g.add_argument("--controlnet-scale", type=float, default=None,
                   help="ControlNet conditioning scale (-> Config.controlnet_scale; TODO tuning).")
    g.add_argument("--ip-adapter-repo", default=None,
                   help="IP-Adapter HF repo (-> Config.ip_adapter_repo).")
    g.add_argument("--ip-adapter-subfolder", default=None,
                   help="IP-Adapter weights subfolder (-> Config.ip_adapter_subfolder).")
    g.add_argument("--ip-adapter-weight", default=None,
                   help="IP-Adapter weight filename (-> Config.ip_adapter_weight).")
    g.add_argument("--ip-adapter-scale", type=float, default=None,
                   help="Style-from-reference strength (-> Config.ip_adapter_scale; TODO tuning).")
    # --- sampling / prompting ----------------------------------------------
    g.add_argument("--diff-prompt", default=None,
                   help="Optional text prompt / style hints (-> Config.diff_prompt).")
    g.add_argument("--diff-negative-prompt", default=None,
                   help="Optional negative prompt (-> Config.diff_negative_prompt).")
    g.add_argument("--diff-steps", type=int, default=None,
                   help="Denoising steps K (-> Config.diff_steps; TODO tuning per scheduler).")
    g.add_argument("--guidance-scale", type=float, default=None,
                   help="Classifier-free guidance scale (-> Config.guidance_scale; TODO tuning).")
    g.add_argument("--denoise-strength", type=float, default=None,
                   help="img2img denoise fraction, 1.0=full (-> Config.denoise_strength; TODO tuning).")
    g.add_argument("--diff-scheduler", choices=("euler", "ddim", "dpm"), default=None,
                   help="Diffusion scheduler (-> Config.diff_scheduler).")
    # --- temporal latent-flow consistency (the Phase 2 differentiator) -----
    g.add_argument("--temporal-strength", type=float, default=None,
                   help="Per-step warped-latent fusion blend cap (-> Config.temporal_strength; TODO tuning).")
    g.add_argument("--temporal-fuse-start", type=float, default=None,
                   help="Fraction of steps where fusion begins (-> Config.temporal_fuse_start; TODO tuning).")
    g.add_argument("--temporal-fuse-end", type=float, default=None,
                   help="Fraction of steps where fusion ends (-> Config.temporal_fuse_end; TODO tuning).")
    g.add_argument("--latent-consistency-weight", type=float, default=None,
                   help="Latent-consistency pull weight (-> Config.latent_consistency_weight; TODO tuning).")
    g.add_argument("--latent-reliability-gamma", type=float, default=None,
                   help="Erosion exponent for downsampled reliability (-> Config.latent_reliability_gamma; TODO tuning).")
    g.add_argument("--warp-space", choices=("latent", "pixel"), default=None,
                   help="Warp the previous latent (default) or VAE-decode->warp->encode (-> Config.warp_space).")
    g.add_argument("--use-anchor", action="store_const", const=True, default=None,
                   help="Enable long-term anchor (frame-0) warp for drift (-> Config.use_anchor; §2.6).")
    g.add_argument("--vae-factor", type=int, default=None,
                   help="VAE spatial downsample factor (-> Config.vae_factor; change only if model differs).")


def build_parser() -> argparse.ArgumentParser:
    """Construct the top-level ``argparse`` parser with all subcommands."""
    parser = argparse.ArgumentParser(
        prog="artvid",
        description=(
            "Artistic video style transfer (PyTorch port). Subcommands: "
            "flow (optical flow + reliability), stylize (per-frame transfer), "
            "run (end-to-end video)."
        ),
    )
    parser.add_argument(
        "--engine",
        choices=("optim", "diffusion"),
        default="optim",
        help="Style-transfer engine (Phase 1 default 'optim'; 'diffusion' is Phase 2).",
    )
    sub = parser.add_subparsers(dest="command", metavar="<command>")
    sub.required = True

    # --- flow ---
    p_flow = sub.add_parser(
        "flow",
        help="Compute forward/backward optical flow + reliability masks (replaces makeOptFlow.sh).",
        description=(
            "Compute RAFT forward+backward optical flow and forward-backward "
            "consistency reliability masks for adjacent (and long-term) frame "
            "pairs, writing legacy-compatible .flo and reliability files."
        ),
    )
    p_flow.add_argument(
        "frames",
        help=(
            "printf-style frame pattern, e.g. 'frames/frame_%%04d.ppm' "
            "(one integer conversion)."
        ),
    )
    p_flow.add_argument(
        "-o",
        "--out",
        required=True,
        help="Output folder for .flo / reliability files.",
    )
    p_flow.add_argument(
        "--start-number",
        type=int,
        default=1,
        help="Index of the first frame (default: 1).",
    )
    p_flow.add_argument(
        "--num-images",
        type=int,
        default=0,
        help="Number of frames; 0 = autodetect by existence (default: 0).",
    )
    p_flow.add_argument(
        "--steps",
        type=int,
        nargs="+",
        default=[1],
        help=(
            "Long-term flow step sizes (frame index deltas). Default [1] = "
            "adjacent frames only. e.g. '--steps 1 2 4' for flow_relative_indices."
        ),
    )
    p_flow.add_argument(
        "--smooth-sigma",
        type=float,
        default=0.8,
        help="Gaussian sigma for reliability-mask smoothing (0 disables; default: 0.8).",
    )
    p_flow.add_argument(
        "--mask-ext",
        default=".pgm",
        help="Reliability mask file extension (default: .pgm; .png also works).",
    )
    p_flow.add_argument(
        "--device",
        choices=("mps", "cuda", "cpu"),
        default=None,
        help="Compute device; default autodetect (mps>cuda>cpu).",
    )
    p_flow.add_argument(
        "--overwrite",
        action="store_true",
        help="Recompute even if output files already exist (default: skip existing).",
    )
    p_flow.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Print each frame pair as it is written.",
    )
    p_flow.set_defaults(func=cmd_flow)

    # --- stylize ---
    p_stylize = sub.add_parser(
        "stylize",
        help="Per-frame style transfer, single- or multi-pass (M2/M3).",
        description=(
            "Per-frame style transfer over a frame sequence. Single-pass by "
            "default (ports artistic_video.lua); --multipass / --passes N "
            "selects the forward/backward multi-pass pipeline "
            "(ports artistic_video_multiPass.lua)."
        ),
    )
    p_stylize.add_argument(
        "frames",
        help="printf-style content frame pattern (-> Config.content_pattern), e.g. 'frames/frame_%%04d.ppm'.",
    )
    p_stylize.add_argument("style", help="Path to the style image (comma-separate for multiple).")
    _add_stylize_flags(p_stylize)
    p_stylize.set_defaults(func=cmd_stylize)

    # --- run (end-to-end) ---
    p_run = sub.add_parser(
        "run",
        help="End-to-end video -> stylized video (replaces stylizeVideo.sh).",
        description=(
            "End-to-end pipeline: extract frames, compute RAFT optical flow + "
            "reliability, stylize (single- or multi-pass), and re-encode the "
            "stylized frames into a video. Ports stylizeVideo.sh."
        ),
    )
    p_run.add_argument("video", help="Input video file.")
    p_run.add_argument("style", help="Path to the style image (comma-separate for multiple).")
    p_run.add_argument(
        "--work-dir",
        default=None,
        help="Working folder for frames/flow/outputs (default: derived from video basename).",
    )
    p_run.add_argument(
        "-o",
        "--output",
        default=None,
        help="Output video path (default: <work_dir>-stylized.<ext>).",
    )
    p_run.add_argument(
        "--frame-pattern",
        default="frame_%04d.ppm",
        help="ffmpeg frame-extraction filename pattern (default: frame_%%04d.ppm).",
    )
    p_run.add_argument(
        "--resolution",
        default=None,
        help="Optional 'w:h' to rescale frames during extraction (default: original).",
    )
    p_run.add_argument(
        "--framerate",
        type=int,
        default=None,
        help="Output video framerate (default: ffmpeg default, 25fps).",
    )
    p_run.add_argument(
        "--no-flow",
        action="store_true",
        help="Skip flow precompute; stylize uses on-the-fly RAFT flow instead.",
    )
    p_run.add_argument(
        "--steps",
        type=int,
        nargs="+",
        default=[1],
        help="Long-term flow step sizes for the flow precompute (default: [1]).",
    )
    p_run.add_argument(
        "--smooth-sigma",
        type=float,
        default=0.8,
        help="Gaussian sigma for reliability-mask smoothing in flow precompute (default: 0.8).",
    )
    p_run.add_argument(
        "--mask-ext",
        default=".pgm",
        help="Reliability mask file extension (default: .pgm).",
    )
    p_run.add_argument(
        "--overwrite-flow",
        action="store_true",
        help="Recompute flow even if output files already exist.",
    )
    _add_stylize_flags(p_run)
    p_run.set_defaults(func=cmd_run)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point: parse ``argv``, dispatch to the selected subcommand.

    Args:
        argv: Argument list (excluding the program name). Defaults to
            ``sys.argv[1:]``.

    Returns:
        Process exit code. ``NotImplementedError`` from stub subcommands is
        caught and surfaced as a clean message with exit code ``2`` rather than
        a traceback.
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except NotImplementedError as exc:
        print(f"artvid {args.command}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
