#!/usr/bin/env python3
"""End-to-end smoke test for both artvid engines (run on your Apple Silicon Mac).

This is a *fast, tiny* sanity check that the full Phase 1 (``optim``) and Phase 2
(``diffusion``) style-transfer pipelines actually run on the user's hardware and
produce well-formed output frames. It is deliberately NOT a quality benchmark
(:mod:`scripts.benchmark` is) and NOT a correctness test of any single function
(the torch-free unit tests under ``tests/`` cover that). It answers one question:
"does each engine run end-to-end on this machine and write output frames of the
right shape and value range?"

What it does
------------
1. Generates (or reuses ``example/``) a tiny 2-3 frame content sequence at low
   resolution into a scratch work dir, plus a small style image.
2. Runs ``--engine optim`` single-pass for a handful of L-BFGS iterations by
   driving :func:`artvid.pipeline.singlepass.stylize_video` directly (the same
   entry point ``artvid stylize`` uses; see ``artvid/cli.py:cmd_stylize``).
3. If ``--diffusion`` is passed, runs ``--engine diffusion`` for a few denoising
   steps via :func:`artvid.diffusion.video.stylize_video_diffusion` (the entry
   point ``artvid stylize --engine diffusion`` uses). This downloads multi-GB
   model weights on first run, so it is opt-in.
4. Optionally (``--cli``) exercises the real ``artvid`` CLI via ``subprocess``
   instead of the in-process API, to also smoke-test argument wiring.
5. Asserts, per engine, that the expected number of output frames exist, decode,
   are 3-channel, match the input resolution, and have pixel values in ``[0,1]``.
6. Prints ``PASS``/``FAIL`` per engine and a per-engine timing line.

Why both an in-process and a CLI mode
-------------------------------------
The in-process mode is the default because it is faster and surfaces tracebacks
directly. ``--cli`` builds the exact ``artvid stylize`` command line (with the
tiny ``--num-iterations`` / ``--diff-steps`` overrides) and runs it as a
subprocess, validating the ``build_config`` flag wiring in ``artvid/cli.py`` end
to end.

IMPORTANT: like :mod:`scripts.benchmark`, this is meant to RUN ON THE USER'S
MACHINE (any Apple Silicon M-series Mac with a torch MPS build). Apple Silicon uses
unified memory, so the GPU shares the Mac's RAM — there is no separate VRAM budget;
the practical cap is your Mac's RAM minus what the OS/apps use. The
development/CI environment for this port has **no GPU and no torch installed**,
so do not execute it there. It imports torch/diffusers and the artvid pipeline
**lazily inside main()** so the module itself imports cleanly without torch
(``python -c "import scripts.smoke_test"`` / ``py_compile`` work torch-free).

Usage
-----
Optim engine only, synthetic frames, the default fast settings::

    python scripts/smoke_test.py

Use the bundled real example frames instead of synthetic ones::

    python scripts/smoke_test.py --content "example/marple8_%02d.ppm" \
        --style example/seated-nude.jpg --start-number 1

Also smoke-test the diffusion engine (downloads weights on first run)::

    python scripts/smoke_test.py --diffusion --diff-steps 4

Drive the real CLI instead of the in-process API::

    python scripts/smoke_test.py --cli

Exit code is ``0`` only if every selected engine passed.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

# ---------------------------------------------------------------------------
# Pure / torch-free helpers (kept importable without torch)
# ---------------------------------------------------------------------------

# Default tiny smoke settings: small enough to finish in seconds on your Apple
# Silicon Mac while still exercising the temporal warp/flow path (>=2 frames).
DEFAULT_FRAMES = 3
DEFAULT_RESOLUTION = (64, 96)  # (H, W) — small but not degenerate.
DEFAULT_OPTIM_ITERS = 4        # L-BFGS iterations per frame (first, subsequent).
DEFAULT_DIFF_STEPS = 6         # diffusion denoising steps.


def _parse_resolution(text: str) -> Tuple[int, int]:
    """Parse ``"HxW"`` (or ``"H:W"``) into an ``(H, W)`` int tuple."""
    sep = "x" if "x" in text.lower() else ":"
    parts = text.lower().split(sep)
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(
            f"resolution must be 'HxW' (e.g. 64x96), got {text!r}"
        )
    return int(parts[0]), int(parts[1])


def build_arg_parser() -> argparse.ArgumentParser:
    """Construct the smoke-test CLI parser (torch-free)."""
    p = argparse.ArgumentParser(
        prog="smoke_test",
        description=(
            "Fast end-to-end smoke test for the artvid optim (and optionally "
            "diffusion) engines. Run on your Apple Silicon Mac (any M-series with MPS)."
        ),
    )
    p.add_argument(
        "--work-dir",
        default=None,
        help="Scratch work dir for generated frames + outputs (default: a temp dir).",
    )
    p.add_argument(
        "--content",
        default=None,
        help=(
            "printf-style content frame pattern to reuse instead of generating "
            "synthetic frames, e.g. 'example/marple8_%%02d.ppm'. The first frame's "
            "native resolution is used and --resolution is ignored."
        ),
    )
    p.add_argument(
        "--style",
        default=None,
        help="Style image path (default: a generated tiny style, or example/seated-nude.jpg with --content).",
    )
    p.add_argument(
        "--frames",
        type=int,
        default=DEFAULT_FRAMES,
        help=f"Number of synthetic frames to generate (default: {DEFAULT_FRAMES}; min 2).",
    )
    p.add_argument(
        "--resolution",
        type=_parse_resolution,
        default=DEFAULT_RESOLUTION,
        help="Synthetic frame resolution 'HxW' (default: 64x96).",
    )
    p.add_argument(
        "--start-number",
        type=int,
        default=1,
        help="Index of the first frame (default: 1).",
    )
    p.add_argument(
        "--iterations",
        type=int,
        default=DEFAULT_OPTIM_ITERS,
        help=f"Optim L-BFGS iterations per frame (default: {DEFAULT_OPTIM_ITERS}).",
    )
    p.add_argument(
        "--diffusion",
        action="store_true",
        help="Also smoke-test the diffusion engine (downloads weights on first run).",
    )
    p.add_argument(
        "--diff-steps",
        type=int,
        default=DEFAULT_DIFF_STEPS,
        help=f"Diffusion denoising steps (default: {DEFAULT_DIFF_STEPS}).",
    )
    p.add_argument(
        "--device",
        choices=("mps", "cuda", "cpu"),
        default=None,
        help="Compute device; default autodetect (mps>cuda>cpu).",
    )
    p.add_argument(
        "--cli",
        action="store_true",
        help="Drive the real 'artvid' CLI via subprocess instead of the in-process API.",
    )
    p.add_argument(
        "--keep",
        action="store_true",
        help="Keep the scratch work dir instead of deleting it on success.",
    )
    return p


# ---------------------------------------------------------------------------
# Frame / style generation (torch-touching; called only from main())
# ---------------------------------------------------------------------------

def _generate_synthetic_frames(
    frames_dir: Path,
    *,
    pattern: str,
    num_frames: int,
    resolution: Tuple[int, int],
    start_number: int,
) -> str:
    """Write ``num_frames`` tiny synthetic content frames; return the abs pattern.

    Frames are a smooth color gradient with a small bright block that translates
    by a few pixels per frame, so the optical-flow / temporal-warp path is
    actually exercised (a static image would make every reliability mask trivial,
    exactly as noted in scripts/benchmark.py). Saved as ``.ppm`` (lossless, what
    the legacy example frames use) directly via Pillow, since the frames are
    already in ``[0,1]`` RGB and need no VGG de-normalization.
    """
    import torch

    h, w = resolution
    frames_dir.mkdir(parents=True, exist_ok=True)

    # Static smooth gradient base in [0,1] RGB CHW.
    yy = torch.linspace(0.0, 1.0, h).view(h, 1).expand(h, w)
    xx = torch.linspace(0.0, 1.0, w).view(1, w).expand(h, w)
    base = torch.stack([xx, yy, (1.0 - xx) * 0.5 + 0.25], dim=0)  # (3, H, W)

    block = max(4, min(h, w) // 6)
    for k in range(num_frames):
        img = base.clone()
        # Translate a bright square diagonally across frames.
        off = 3 * k
        y0 = min(h - block, 4 + off)
        x0 = min(w - block, 4 + off)
        img[:, y0 : y0 + block, x0 : x0 + block] = torch.tensor(
            [1.0, 0.2, 0.2]
        ).view(3, 1, 1)
        img = img.clamp(0.0, 1.0)
        out_path = frames_dir / (pattern % (start_number + k))
        _save_rgb01_direct(img, out_path)
    return str(frames_dir / pattern)


def _generate_style_image(path: Path, resolution: Tuple[int, int]) -> str:
    """Write a tiny synthetic style image (colorful diagonal stripes)."""
    import torch

    h, w = resolution
    yy = torch.arange(h).view(h, 1).expand(h, w)
    xx = torch.arange(w).view(1, w).expand(h, w)
    stripes = ((xx + yy) % 12 < 6).to(torch.float32)
    img = torch.stack([stripes, 1.0 - stripes, (xx.float() / max(1, w))], dim=0)
    _save_rgb01_direct(img, path)
    return str(path)


def _save_rgb01_direct(img01, path: Path) -> None:
    """Pillow-encode a ``[0,1]`` RGB CHW tensor (style image helper)."""
    from PIL import Image

    arr = (
        img01.clamp(0.0, 1.0)
        .mul(255.0)
        .round()
        .to("cpu")
        .byte()
        .permute(1, 2, 0)
        .numpy()
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr, mode="RGB").save(path)


# ---------------------------------------------------------------------------
# Output validation
# ---------------------------------------------------------------------------

def _assert_output_frame(path: Path, expected_hw: Optional[Tuple[int, int]]) -> Tuple[int, int]:
    """Assert one output frame exists, decodes, is 3-channel, in range; return (H, W).

    Validates the contract every engine must satisfy: the file exists, decodes to
    a 3-channel image of the expected resolution, with pixel values in ``[0,1]``
    after :func:`artvid.io.image.load_image` (which returns ``[0,1]`` RGB CHW).
    """
    import torch

    from artvid.io.image import load_image

    if not path.is_file():
        raise AssertionError(f"expected output frame missing: {path}")
    img = load_image(path)  # (3, H, W) float32 in [0,1]
    if img.dim() != 3 or img.shape[0] != 3:
        raise AssertionError(
            f"output frame {path} has shape {tuple(img.shape)}; expected (3, H, W)."
        )
    h, w = int(img.shape[1]), int(img.shape[2])
    if expected_hw is not None and (h, w) != expected_hw:
        raise AssertionError(
            f"output frame {path} is {h}x{w}; expected {expected_hw[0]}x{expected_hw[1]}."
        )
    mn = float(img.min())
    mx = float(img.max())
    if not (mn >= -1e-4 and mx <= 1.0 + 1e-4):
        raise AssertionError(
            f"output frame {path} pixel range [{mn:.4f}, {mx:.4f}] outside [0,1]."
        )
    if not torch.isfinite(img).all():
        raise AssertionError(f"output frame {path} contains non-finite values.")
    return h, w


def _expected_output_paths(config, num_frames: int) -> List[Path]:
    """Resolve the single-pass output frame paths for the first ``num_frames``."""
    from artvid.pipeline.singlepass import build_out_filename

    return [
        Path(build_out_filename(config, rel))
        for rel in range(1, num_frames + 1)
    ]


# ---------------------------------------------------------------------------
# Engine runners (in-process API)
# ---------------------------------------------------------------------------

def _build_config(
    *,
    content_pattern: str,
    style_image: str,
    out_dir: Path,
    start_number: int,
    num_images: int,
    device: Optional[str],
    engine: str,
    iterations: int,
    diff_steps: int,
):
    """Construct a tiny :class:`artvid.config.Config` for a smoke run.

    Mirrors how ``artvid/cli.py:build_config`` maps flags onto Config, but only
    sets the handful of fields the smoke test needs (everything else stays at the
    Config defaults). Crucially shrinks the iteration / step counts so each engine
    finishes in seconds.
    """
    from artvid.config import Config

    overrides = dict(
        content_pattern=content_pattern,
        style_image=style_image,
        start_number=start_number,
        num_images=num_images,
        output_folder=str(out_dir) + "/",
        output_image="smoke.png",
        device=device,
        # Tiny iteration budget for both the first and subsequent frames.
        num_iterations=(iterations, iterations),
        print_iter=max(1, iterations),  # avoid noisy per-iter prints.
        save_iter=0,                    # only final frame.
        seed=0,                         # deterministic-ish run.
    )
    if engine == "diffusion":
        overrides["diff_steps"] = diff_steps
    return Config(**overrides)


def _run_optim_inprocess(config, *, num_frames: int, expected_hw) -> None:
    """Run the optim single-pass engine and validate its output frames."""
    from artvid.pipeline.singlepass import stylize_video

    # flow_source="raft": synthetic frames have no precomputed .flo files. (With
    # the bundled example frames, example/deepflow/ exists, but raft still works
    # and keeps the smoke test self-contained.)
    results = stylize_video(config, flow_source="raft")
    if len(results) < num_frames:
        raise AssertionError(
            f"optim produced {len(results)} frame result(s); expected {num_frames}."
        )
    for res in results[:num_frames]:
        _assert_output_frame(Path(res.output_path), expected_hw)


def _run_diffusion_inprocess(config, *, num_frames: int, expected_hw) -> None:
    """Run the diffusion engine and validate its output frames."""
    from artvid.diffusion.video import stylize_video_diffusion

    results = stylize_video_diffusion(config, flow_source="raft")
    if len(results) < num_frames:
        raise AssertionError(
            f"diffusion produced {len(results)} frame result(s); expected {num_frames}."
        )
    for res in results[:num_frames]:
        _assert_output_frame(Path(res.output_path), expected_hw)


# ---------------------------------------------------------------------------
# Engine runners (real CLI via subprocess)
# ---------------------------------------------------------------------------

def _run_via_cli(
    *,
    engine: str,
    content_pattern: str,
    style_image: str,
    out_dir: Path,
    start_number: int,
    num_images: int,
    device: Optional[str],
    iterations: int,
    diff_steps: int,
) -> None:
    """Invoke the real ``artvid stylize`` CLI as a subprocess for ``engine``.

    Builds the same command a user would run, with tiny iteration/step overrides,
    and asserts a zero exit code. Output-frame validation is done by the caller
    after this returns. Uses ``python -m artvid`` (the package's CLI entry) so it
    works without an installed console script.
    """
    import subprocess

    cmd = [
        sys.executable,
        "-m",
        "artvid",
        "--engine",
        engine,
        "stylize",
        content_pattern,
        style_image,
        "--start-number",
        str(start_number),
        "--num-images",
        str(num_images),
        "--output-folder",
        str(out_dir) + "/",
        "--output-image",
        "smoke.png",
        "--flow-source",
        "raft",
        "--seed",
        "0",
    ]
    if engine == "optim":
        cmd += ["--num-iterations", f"{iterations},{iterations}",
                "--print-iter", str(max(1, iterations))]
    else:
        cmd += ["--diff-steps", str(diff_steps)]
    if device is not None:
        cmd += ["--device", device]

    proc = subprocess.run(cmd, check=False)
    if proc.returncode != 0:
        raise AssertionError(
            f"artvid CLI ({engine}) exited with code {proc.returncode}: "
            f"{' '.join(cmd)}"
        )


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def _run_engine(
    engine: str,
    *,
    args,
    content_pattern: str,
    style_image: str,
    out_dir: Path,
    num_frames: int,
    expected_hw,
) -> Tuple[bool, float, str]:
    """Run one engine; return ``(passed, elapsed_seconds, message)``.

    Catches *any* exception so one engine failing still lets the other report,
    and so the script exits with a clean PASS/FAIL summary rather than a raw
    traceback (the traceback text is captured into the message).
    """
    import traceback

    out_dir.mkdir(parents=True, exist_ok=True)
    config = _build_config(
        content_pattern=content_pattern,
        style_image=style_image,
        out_dir=out_dir,
        start_number=args.start_number,
        num_images=num_frames,
        device=args.device,
        engine=engine,
        iterations=args.iterations,
        diff_steps=args.diff_steps,
    )

    t0 = time.perf_counter()
    try:
        if args.cli:
            _run_via_cli(
                engine=engine,
                content_pattern=content_pattern,
                style_image=style_image,
                out_dir=out_dir,
                start_number=args.start_number,
                num_images=num_frames,
                device=args.device,
                iterations=args.iterations,
                diff_steps=args.diff_steps,
            )
            # CLI mode: validate the frames it wrote (paths come from Config).
            for path in _expected_output_paths(config, num_frames):
                _assert_output_frame(path, expected_hw)
        elif engine == "optim":
            _run_optim_inprocess(config, num_frames=num_frames, expected_hw=expected_hw)
        else:
            _run_diffusion_inprocess(config, num_frames=num_frames, expected_hw=expected_hw)
    except Exception as exc:  # noqa: BLE001 — smoke test: report, don't crash.
        elapsed = time.perf_counter() - t0
        msg = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
        return False, elapsed, msg
    elapsed = time.perf_counter() - t0
    return True, elapsed, "output frames present, 3-channel, in [0,1]"


def main(argv: Optional[List[str]] = None) -> int:
    """Smoke-test entry point. Heavy imports happen here, not at module load."""
    args = build_arg_parser().parse_args(argv)

    if args.frames < 2:
        print("error: need at least 2 frames for the temporal path.", file=sys.stderr)
        return 2

    # torch is needed for frame/style generation and device setup. Imported here
    # so the module stays importable without torch (py_compile / torch-free tests).
    from artvid import device as _device

    _device.enable_mps_fallback()

    # --- scratch work dir ---------------------------------------------------
    import shutil
    import tempfile

    if args.work_dir is not None:
        work_dir = Path(args.work_dir)
        work_dir.mkdir(parents=True, exist_ok=True)
        cleanup = False
    else:
        work_dir = Path(tempfile.mkdtemp(prefix="artvid_smoke_"))
        cleanup = not args.keep

    print(f"[smoke] work dir: {work_dir}")

    # --- content frames -----------------------------------------------------
    if args.content is not None:
        # Reuse an existing content pattern (e.g. the bundled example frames).
        content_pattern = args.content
        from artvid.io.image import load_image

        first = load_image(Path(content_pattern % args.start_number))
        expected_hw = (int(first.shape[1]), int(first.shape[2]))
        num_frames = args.frames
        print(f"[smoke] using existing content {content_pattern!r} at {expected_hw[0]}x{expected_hw[1]}")
    else:
        frames_dir = work_dir / "frames"
        content_pattern = _generate_synthetic_frames(
            frames_dir,
            pattern="frame_%04d.ppm",
            num_frames=args.frames,
            resolution=args.resolution,
            start_number=args.start_number,
        )
        expected_hw = args.resolution
        num_frames = args.frames
        print(
            f"[smoke] generated {num_frames} synthetic frame(s) at "
            f"{expected_hw[0]}x{expected_hw[1]} -> {content_pattern}"
        )

    # --- style image --------------------------------------------------------
    if args.style is not None:
        style_image = args.style
    elif args.content is not None:
        # Pair the bundled example frames with the bundled example style.
        style_image = "example/seated-nude.jpg"
    else:
        style_image = _generate_style_image(work_dir / "style.png", args.resolution)
    print(f"[smoke] style image: {style_image}")

    # --- run engines --------------------------------------------------------
    engines = ["optim"]
    if args.diffusion:
        engines.append("diffusion")

    summary: List[Tuple[str, bool, float, str]] = []
    for engine in engines:
        out_dir = work_dir / f"out_{engine}"
        print(f"\n[smoke] === engine: {engine} ===")
        passed, elapsed, msg = _run_engine(
            engine,
            args=args,
            content_pattern=content_pattern,
            style_image=style_image,
            out_dir=out_dir,
            num_frames=num_frames,
            expected_hw=expected_hw,
        )
        status = "PASS" if passed else "FAIL"
        print(f"[smoke] {engine}: {status}  ({elapsed:.2f}s)  {msg.splitlines()[0]}")
        if not passed:
            # Print the full traceback for the failing engine (after the summary
            # line) so the user can debug, while keeping the PASS/FAIL legible.
            print(msg, file=sys.stderr)
        summary.append((engine, passed, elapsed, msg))

    # --- summary ------------------------------------------------------------
    print("\n[smoke] ---- summary ----")
    all_passed = True
    for engine, passed, elapsed, _msg in summary:
        status = "PASS" if passed else "FAIL"
        print(f"[smoke] {engine:<10} {status}  {elapsed:7.2f}s")
        all_passed = all_passed and passed

    if cleanup and all_passed:
        shutil.rmtree(work_dir, ignore_errors=True)
        print(f"[smoke] cleaned up {work_dir}")
    elif not all_passed:
        print(f"[smoke] left work dir for inspection: {work_dir}")

    print(f"[smoke] overall: {'PASS' if all_passed else 'FAIL'}")
    return 0 if all_passed else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
