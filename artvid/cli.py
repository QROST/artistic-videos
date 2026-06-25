"""``artvid`` command-line entry point.

Dispatches the three subcommands described in ``docs/01-architecture.md`` §3.9:

* ``flow``    — compute forward/backward RAFT optical flow + reliability masks
                for adjacent (and long-term) frame pairs. **Replaces
                ``makeOptFlow.sh`` + ``run-deepflow.sh`` + the C++
                ``consistencyChecker``** (see ``docs/02-migration-map.md`` §1).
* ``stylize`` — per-frame style transfer (single/multi pass). *Stub until M2–M3.*
* ``run``     — end-to-end ``video → stylized video`` (replaces
                ``stylizeVideo.sh``). *Stub until M4.*

Only ``flow`` is implemented in this phase (milestone **M1**); ``stylize`` and
``run`` raise :class:`NotImplementedError` pointing at the milestone that lands
them. ``main`` parses ``argv``, selects a subcommand and dispatches.

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
# Stub subcommands (later milestones)
# ---------------------------------------------------------------------------

def cmd_stylize(args: argparse.Namespace) -> int:
    """Per-frame style transfer — **not yet implemented** (milestones M2–M3).

    Will drive :mod:`artvid.pipeline.singlepass` (single pass) and
    :mod:`artvid.pipeline.multipass` (``--passes`` / ``--multipass``), porting
    the main frame loop of ``artistic_video.lua`` /
    ``artistic_video_multiPass.lua``.

    Raises:
        NotImplementedError: Always, until the M2/M3 pipelines land.
    """
    raise NotImplementedError(
        "`artvid stylize` is not implemented yet. The single-pass pipeline "
        "lands in milestone M2 (artvid/pipeline/singlepass.py) and the "
        "multi-pass pipeline in M3 (artvid/pipeline/multipass.py). "
        "Use `artvid flow ...` (M1) to precompute optical flow in the meantime."
    )


def cmd_run(args: argparse.Namespace) -> int:
    """End-to-end ``video → stylized video`` — **not yet implemented** (milestone M4).

    Will chain frame extraction (:mod:`artvid.io.video`), ``flow`` and
    ``stylize`` and re-encode, replacing ``stylizeVideo.sh``.

    Raises:
        NotImplementedError: Always, until the M4 end-to-end pipeline lands.
    """
    raise NotImplementedError(
        "`artvid run` (end-to-end video pipeline, replacing stylizeVideo.sh) "
        "lands in milestone M4. It will chain frame extraction (artvid/io/"
        "video.py), `artvid flow`, `artvid stylize`, and ffmpeg re-encode."
    )


# ---------------------------------------------------------------------------
# Argument parsing / dispatch
# ---------------------------------------------------------------------------

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

    # --- stylize (stub) ---
    p_stylize = sub.add_parser(
        "stylize",
        help="Per-frame style transfer (NOT YET IMPLEMENTED — milestones M2-M3).",
        description="Per-frame style transfer. Stub: implemented in M2 (single) / M3 (multi).",
    )
    p_stylize.add_argument("frames", help="printf-style frame pattern.")
    p_stylize.add_argument("style", help="Path to the style image.")
    p_stylize.add_argument(
        "--multipass",
        action="store_true",
        help="Use the multi-pass pipeline (M3) instead of single-pass (M2).",
    )
    p_stylize.set_defaults(func=cmd_stylize)

    # --- run (stub) ---
    p_run = sub.add_parser(
        "run",
        help="End-to-end video stylization (NOT YET IMPLEMENTED — milestone M4).",
        description="End-to-end video -> stylized video. Stub: implemented in M4 (replaces stylizeVideo.sh).",
    )
    p_run.add_argument("video", help="Input video file.")
    p_run.add_argument("style", help="Path to the style image.")
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
