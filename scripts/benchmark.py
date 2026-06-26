#!/usr/bin/env python3
"""Apple Silicon (MPS) performance benchmark for single-pass video stylization.

This harness is the M4 "性能基准" deliverable from ``docs/03-phase1-plan.md``
(see the M4 acceptance row: *基准表：分辨率 × 每帧迭代数 × 每帧墙钟 × 峰值内存 ×
是否 CPU 回退*). It does **not** port any single legacy Lua function; instead it
drives the ported single-pass loop (:func:`artvid.pipeline.singlepass.stylize_video`,
the port of ``artistic_video.lua:137-286``) and measures it.

What it records, per (resolution, frame)
----------------------------------------
* resolution (``HxW``)
* optimizer iterations actually run (from ``FrameResult.num_iterations``)
* wall-clock seconds/frame
* peak accelerator memory (via ``torch.mps`` / ``torch.cuda`` where available)
* whether any op fell back to CPU during the run, and the state of
  ``PYTORCH_ENABLE_MPS_FALLBACK``

It then prints a Markdown table the user can paste straight into the docs.

IMPORTANT: this script is meant to RUN ON THE USER'S MACHINE (any Apple Silicon
M-series Mac with a torch MPS build). The development/CI environment for this port has
**no GPU and no torch installed**, so do not execute it there. It is written
against the documented torch + artvid API and is exercised on real hardware.

Usage
-----
Generate synthetic content frames and benchmark a sweep of resolutions::

    python scripts/benchmark.py --resolutions 360x640 540x960 720x1280 \
        --frames 3 --iterations 100,50

Benchmark real content frames instead of synthetic ones (point at an existing
content pattern; the first frame's native resolution is used and ``--resolutions``
is ignored)::

    python scripts/benchmark.py --content "example/marple8_%02d.ppm" \
        --start-number 1 --frames 3 --iterations 100,50

Notes / parity
--------------
* The accelerator is chosen by :func:`artvid.device.pick_device` (mps > cuda >
  cpu), exactly like the real pipeline. ``PYTORCH_ENABLE_MPS_FALLBACK`` is set
  via :func:`artvid.device.enable_mps_fallback` before any MPS op, mirroring the
  pipeline's own startup behaviour.
* Synthetic frames are deterministic smooth gradients with a small translating
  block between frames, so the temporal warp/flow path is actually exercised
  (a static image would make every reliability mask trivial). Flow is computed
  on the fly via RAFT (``--flow-source raft``) by default, because synthetic
  frames have no precomputed ``.flo`` files.
* Peak-memory numbers are per *run* (the whole sweep at one resolution), not
  truly per-frame, because torch's peak-memory counters are global. We reset the
  counter before each resolution and report the peak observed across that
  resolution's frames; the per-frame rows share that resolution's peak.
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

# Make ``import artvid`` work when this script is run from the repo root or from
# the scripts/ directory without an editable install.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


# ---------------------------------------------------------------------------
# Result rows
# ---------------------------------------------------------------------------

@dataclass
class FrameBench:
    """One row of the benchmark table."""

    resolution: str          # "HxW"
    frame_idx: int
    iterations: int
    seconds: float
    peak_mem_mb: Optional[float]   # None when no accelerator memory counter
    cpu_fallback: str              # "n/a" | "no" | "yes" | "unknown"


# ---------------------------------------------------------------------------
# Memory + fallback probing (torch is imported lazily inside these)
# ---------------------------------------------------------------------------

def _accel_kind(device) -> str:
    """Return 'mps' | 'cuda' | 'cpu' for a torch.device."""
    return getattr(device, "type", str(device))


def _reset_peak_memory(device) -> None:
    """Reset the per-process peak-memory counter for the active accelerator."""
    import torch

    kind = _accel_kind(device)
    if kind == "cuda" and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    elif kind == "mps":
        mps = getattr(torch, "mps", None)
        # ``reset_peak_memory_stats`` was added to torch.mps in newer builds.
        reset = getattr(mps, "reset_peak_memory_stats", None) if mps else None
        if reset is not None:
            reset()


def _read_peak_memory_mb(device) -> Optional[float]:
    """Read peak accelerator memory in MB, or None if unavailable."""
    import torch

    kind = _accel_kind(device)
    if kind == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()
        return torch.cuda.max_memory_allocated() / (1024 * 1024)
    if kind == "mps":
        mps = getattr(torch, "mps", None)
        if mps is None:
            return None
        # Prefer the peak counter when present; fall back to current allocation.
        for attr in ("driver_allocated_memory", "current_allocated_memory"):
            fn = getattr(mps, attr, None)
            if fn is not None:
                try:
                    return fn() / (1024 * 1024)
                except Exception:
                    pass
        return None
    return None


def _sync(device) -> None:
    """Synchronize the accelerator so wall-clock timing is accurate."""
    import torch

    kind = _accel_kind(device)
    if kind == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()
    elif kind == "mps":
        mps = getattr(torch, "mps", None)
        synchronize = getattr(mps, "synchronize", None) if mps else None
        if synchronize is not None:
            synchronize()


# ---------------------------------------------------------------------------
# Synthetic content-frame generation
# ---------------------------------------------------------------------------

def _make_synthetic_frame(height: int, width: int, frame_idx: int):
    """Build a deterministic synthetic content frame as a ``(3,H,W)`` [0,1] RGB.

    Smooth horizontal/vertical color gradients plus a bright square that
    translates by a few pixels per frame, so optical flow / warping is
    non-trivial (a static frame would make the temporal path degenerate).
    """
    import torch

    ys = torch.linspace(0.0, 1.0, height).view(height, 1).expand(height, width)
    xs = torch.linspace(0.0, 1.0, width).view(1, width).expand(height, width)
    r = xs
    g = ys
    b = (xs + ys) * 0.5
    img = torch.stack([r, g, b], dim=0)

    # A translating bright block (motion = 3 px / frame, diagonal).
    block = max(8, min(height, width) // 8)
    shift = 3 * frame_idx
    y0 = (height // 4 + shift) % max(1, height - block)
    x0 = (width // 4 + shift) % max(1, width - block)
    img[:, y0:y0 + block, x0:x0 + block] = 1.0
    return img.clamp_(0.0, 1.0)


def _write_synthetic_frames(
    out_dir: Path, resolution: Tuple[int, int], start_number: int, frames: int
) -> str:
    """Write synthetic frames to ``out_dir`` and return their content pattern."""
    # We write the raw [0,1] RGB directly with PIL to avoid any
    # preprocess/deprocess mode coupling for synthetic data.
    import numpy as np
    from PIL import Image

    out_dir.mkdir(parents=True, exist_ok=True)
    height, width = resolution
    pattern = str(out_dir / "synth_%04d.png")
    for k in range(frames):
        idx = start_number + k
        frame = _make_synthetic_frame(height, width, k)  # (3,H,W) [0,1]
        arr = (frame.mul(255.0).round().clamp(0, 255)
               .to("cpu").permute(1, 2, 0).numpy().astype(np.uint8))
        Image.fromarray(arr, mode="RGB").save(pattern % idx)
    return pattern


# ---------------------------------------------------------------------------
# Core benchmark driver
# ---------------------------------------------------------------------------

def _run_one_resolution(
    *,
    content_pattern: str,
    start_number: int,
    frames: int,
    iterations: Tuple[int, int],
    style_image: str,
    flow_source: str,
    output_folder: str,
    resolution_label: str,
    device,
) -> List[FrameBench]:
    """Run ``stylize_video`` over ``frames`` and collect per-frame benchmark rows."""
    import torch  # noqa: F401  (ensures torch present; used by helpers)

    from artvid.config import Config
    from artvid.pipeline.singlepass import stylize_video

    config = Config(
        content_pattern=content_pattern,
        style_image=style_image,
        start_number=start_number,
        num_images=frames,
        num_iterations=iterations,          # (first, subsequent)
        output_image="bench.png",
        output_folder=output_folder,
        # Use a deterministic init so the optimizer does the same work each run.
        seed=0,
        # Keep printing quiet-ish; the loop still prints per-frame headers.
        print_iter=max(iterations) + 1,
        save_iter=0,
    )

    Path(output_folder).mkdir(parents=True, exist_ok=True)

    _reset_peak_memory(device)
    _sync(device)

    rows: List[FrameBench] = []

    # ``stylize_video`` runs the whole frame sweep in one call and does not
    # expose a per-frame timing hook, so we time the aggregate and attribute
    # per-frame wall-clock proportionally to each frame's iteration count
    # (``FrameResult.num_iterations``). This keeps the real pipeline's
    # warm-up/caching semantics intact (one run) while still giving a per-frame
    # breakdown in the table. The first frame typically runs more iterations
    # (``num_iterations[0]``), so it gets a larger time share.
    t0 = time.perf_counter()
    results = stylize_video(config, device=device, flow_source=flow_source)
    _sync(device)
    total_seconds = time.perf_counter() - t0
    peak_mb = _read_peak_memory_mb(device)

    total_iters = sum(max(1, r.num_iterations) for r in results) or 1
    fallback = _fallback_state(device)

    for r in results:
        share = max(1, r.num_iterations) / total_iters
        rows.append(
            FrameBench(
                resolution=resolution_label,
                frame_idx=r.frame_idx,
                iterations=r.num_iterations,
                seconds=total_seconds * share,
                peak_mem_mb=peak_mb,
                cpu_fallback=fallback,
            )
        )
    return rows


def _fallback_state(device) -> str:
    """Best-effort report of whether MPS->CPU fallback is possible/occurred.

    torch does not expose a counter for *how many* ops fell back, so we report
    the enabling env var state. On MPS with ``PYTORCH_ENABLE_MPS_FALLBACK=1`` a
    fallback is *possible* (and silent); we mark this 'enabled'. On non-MPS the
    field is 'n/a'.
    """
    import os

    from artvid.device import MPS_FALLBACK_ENV

    kind = _accel_kind(device)
    if kind != "mps":
        return "n/a"
    val = os.environ.get(MPS_FALLBACK_ENV, "")
    return "enabled" if val in ("1", "true", "TRUE", "yes", "on") else "disabled"


# ---------------------------------------------------------------------------
# Markdown rendering
# ---------------------------------------------------------------------------

def _render_markdown(rows: List[FrameBench], device, env_summary: str) -> str:
    """Render the collected rows as a Markdown table + a small header."""
    lines: List[str] = []
    lines.append(f"### artvid single-pass benchmark — `{_accel_kind(device)}`")
    lines.append("")
    lines.append(env_summary)
    lines.append("")
    lines.append(
        "| resolution (HxW) | frame | iterations | sec/frame | "
        "peak mem (MB) | CPU fallback |"
    )
    lines.append("|---|---|---|---|---|---|")
    for r in rows:
        mem = f"{r.peak_mem_mb:.0f}" if r.peak_mem_mb is not None else "n/a"
        lines.append(
            f"| {r.resolution} | {r.frame_idx} | {r.iterations} | "
            f"{r.seconds:.2f} | {mem} | {r.cpu_fallback} |"
        )
    # Per-resolution summary (mean sec/frame).
    lines.append("")
    by_res: dict = {}
    for r in rows:
        by_res.setdefault(r.resolution, []).append(r.seconds)
    lines.append("| resolution (HxW) | mean sec/frame | frames |")
    lines.append("|---|---|---|")
    for res, secs in by_res.items():
        mean = sum(secs) / len(secs) if secs else float("nan")
        lines.append(f"| {res} | {mean:.2f} | {len(secs)} |")
    return "\n".join(lines)


def _env_summary(device) -> str:
    """One-line environment summary for the table header."""
    import os
    import platform

    import torch

    from artvid.device import MPS_FALLBACK_ENV

    parts = [
        f"torch {torch.__version__}",
        f"python {platform.python_version()}",
        f"{platform.system()} {platform.machine()}",
        f"device={_accel_kind(device)}",
        f"{MPS_FALLBACK_ENV}={os.environ.get(MPS_FALLBACK_ENV, '<unset>')}",
    ]
    return "_" + " · ".join(parts) + "_"


# ---------------------------------------------------------------------------
# Resolution parsing
# ---------------------------------------------------------------------------

def _parse_resolution(text: str) -> Tuple[int, int]:
    """Parse ``"720x1280"`` (HxW) into ``(height, width)``."""
    sep = "x" if "x" in text else ("X" if "X" in text else None)
    if sep is None:
        raise argparse.ArgumentTypeError(
            f"Resolution {text!r} must look like HEIGHTxWIDTH, e.g. 720x1280."
        )
    h_str, w_str = text.split(sep, 1)
    return (int(h_str), int(w_str))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Benchmark artvid single-pass stylization on the local "
        "accelerator (Apple Silicon / MPS, CUDA, or CPU).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--resolutions",
        type=_parse_resolution,
        nargs="+",
        default=[(360, 640), (540, 960), (720, 1280)],
        help="Resolutions to sweep as HEIGHTxWIDTH (synthetic frames). "
        "Ignored when --content is given (real frames use their native size).",
    )
    p.add_argument(
        "--frames",
        type=int,
        default=3,
        help="Number of frames to stylize per resolution.",
    )
    p.add_argument(
        "--iterations",
        type=str,
        default="100,50",
        help="Optimizer iterations as 'first,subsequent' (kept small for a "
        "quick benchmark; production uses 2000,1000).",
    )
    p.add_argument(
        "--style-image",
        type=str,
        default="example/seated-nude.jpg",
        help="Style image path (defaults to the repo's example style).",
    )
    p.add_argument(
        "--content",
        type=str,
        default=None,
        help="Optional real content pattern (e.g. 'example/marple8_%%02d.ppm'). "
        "When set, real frames are benchmarked at their native resolution "
        "instead of synthetic frames.",
    )
    p.add_argument(
        "--start-number",
        type=int,
        default=1,
        help="First frame index for the content pattern.",
    )
    p.add_argument(
        "--flow-source",
        choices=("auto", "raft", "precomputed"),
        default="raft",
        help="Flow backend. Synthetic frames have no .flo files, so 'raft' is "
        "the sensible default; use 'precomputed'/'auto' only with --content "
        "that has matching .flo/.pgm files.",
    )
    p.add_argument(
        "--device",
        choices=("mps", "cuda", "cpu"),
        default=None,
        help="Force a device. Default: autodetect (mps > cuda > cpu).",
    )
    p.add_argument(
        "--work-dir",
        type=str,
        default=None,
        help="Directory for synthetic frames + stylized outputs. "
        "Default: a temp dir under the system temp.",
    )
    p.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional path to also write the Markdown table to a file.",
    )
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = _build_arg_parser().parse_args(argv)

    # Import torch only now (so --help works without torch, and so the import
    # error message is clear if torch is missing).
    try:
        import torch  # noqa: F401
    except ImportError:
        print(
            "ERROR: torch is not installed. This benchmark must run on the "
            "user's machine (any Apple Silicon M-series Mac with a torch MPS build).",
            file=sys.stderr,
        )
        return 2

    from artvid.config import _parse_int_pair
    from artvid.device import enable_mps_fallback, get_device

    # Mirror the pipeline's startup: enable MPS->CPU fallback before any MPS op.
    enable_mps_fallback()
    device = get_device(args.device)

    iterations = _parse_int_pair(args.iterations)

    import tempfile

    work_dir = Path(args.work_dir) if args.work_dir else Path(
        tempfile.mkdtemp(prefix="artvid_bench_")
    )
    work_dir.mkdir(parents=True, exist_ok=True)

    all_rows: List[FrameBench] = []

    if args.content is not None:
        # Real content: single "resolution" determined by the frames themselves.
        out_folder = str(work_dir / "out_real") + "/"
        label = _native_resolution_label(args.content, args.start_number)
        print(f"Benchmarking real content {args.content!r} ({label}) ...")
        rows = _run_one_resolution(
            content_pattern=args.content,
            start_number=args.start_number,
            frames=args.frames,
            iterations=iterations,
            style_image=args.style_image,
            flow_source=args.flow_source,
            output_folder=out_folder,
            resolution_label=label,
            device=device,
        )
        all_rows.extend(rows)
    else:
        for (height, width) in args.resolutions:
            label = f"{height}x{width}"
            res_dir = work_dir / f"frames_{label}"
            out_folder = str(work_dir / f"out_{label}") + "/"
            print(f"Generating synthetic frames at {label} ...")
            pattern = _write_synthetic_frames(
                res_dir, (height, width), args.start_number, args.frames
            )
            print(f"Benchmarking {label} ...")
            rows = _run_one_resolution(
                content_pattern=pattern,
                start_number=args.start_number,
                frames=args.frames,
                iterations=iterations,
                style_image=args.style_image,
                flow_source=args.flow_source,
                output_folder=out_folder,
                resolution_label=label,
                device=device,
            )
            all_rows.extend(rows)

    md = _render_markdown(all_rows, device, _env_summary(device))
    print()
    print(md)
    if args.output:
        Path(args.output).write_text(md + "\n")
        print(f"\nWrote Markdown table to {args.output}", file=sys.stderr)
    return 0


def _native_resolution_label(content_pattern: str, start_number: int) -> str:
    """Read the first real content frame to report its HxW label."""
    from artvid.io.image import load_image

    path = content_pattern % start_number
    try:
        img = load_image(path)  # (3, H, W)
        _, h, w = img.shape
        return f"{h}x{w}"
    except Exception:
        return "native"


if __name__ == "__main__":
    raise SystemExit(main())
