"""Video frame extraction and encoding via ffmpeg.

Wraps the first and last steps of the legacy ``stylizeVideo.sh`` (lines 7-14,
58-64, 97-98): ffmpeg-or-avconv detection, decoding a video into per-frame
images, and re-encoding stylized frames back into a video.

Everything is driven through ``subprocess`` against an ``ffmpeg``/``avconv``
binary on ``PATH``; no Python media decoding dependency is required. This module
has no torch dependency.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

# Candidate binaries, in the same preference order as stylizeVideo.sh: ffmpeg
# first, then avconv (the Libav fork that historically shipped on some distros).
_FFMPEG_CANDIDATES = ("ffmpeg", "avconv")


class FFmpegNotFoundError(RuntimeError):
    """Raised when neither ffmpeg nor avconv is available on ``PATH``."""


def find_ffmpeg() -> str:
    """Return the path to an ffmpeg-compatible binary.

    Ports the detection in ``stylizeVideo.sh:7-14``: try ``ffmpeg``, then
    ``avconv``.

    Returns:
        The resolved absolute path to the binary.

    Raises:
        FFmpegNotFoundError: If neither binary is found.
    """
    for name in _FFMPEG_CANDIDATES:
        path = shutil.which(name)
        if path is not None:
            return path
    raise FFmpegNotFoundError(
        "This requires either ffmpeg or avconv installed and on PATH."
    )


def _run(cmd: list[str]) -> None:
    """Run an ffmpeg command, raising with captured stderr on failure."""
    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"ffmpeg command failed ({proc.returncode}): {' '.join(cmd)}\n"
            f"{proc.stderr.strip()}"
        )


def extract_frames(
    video: str | Path,
    out_dir: str | Path,
    *,
    pattern: str = "frame_%04d.ppm",
    resolution: str | None = None,
    ffmpeg: str | None = None,
) -> str:
    """Extract a video's frames into individual image files.

    Ports ``stylizeVideo.sh:58-64``: ``ffmpeg -i video [-vf scale=w:h]
    out_dir/frame_%04d.ppm``. ``out_dir`` is created if needed.

    Args:
        video: Path to the input video.
        out_dir: Directory to write frames into (created if missing).
        pattern: ffmpeg output filename pattern (must contain a ``%0Nd`` token).
            Defaults to the legacy ``frame_%04d.ppm``.
        resolution: Optional ``"w:h"`` string; when given, frames are rescaled
            via ``-vf scale=w:h`` (legacy default keeps original resolution).
        ffmpeg: Optional explicit binary path; autodetected if ``None``.

    Returns:
        The output path pattern (``str(out_dir / pattern)``), suitable for
        feeding to the pipeline's ``content_pattern``.
    """
    binary = ffmpeg or find_ffmpeg()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_pattern = out_dir / pattern

    cmd = [binary, "-i", str(video)]
    if resolution:
        cmd += ["-vf", f"scale={resolution}"]
    cmd.append(str(out_pattern))
    _run(cmd)
    return str(out_pattern)


def encode_video(
    frames_glob: str | Path,
    out: str | Path,
    *,
    framerate: int | None = None,
    ffmpeg: str | None = None,
) -> str:
    """Encode a sequence of stylized frames into a video.

    Ports ``stylizeVideo.sh:98``: ``ffmpeg -i out-%04d.png stylized.ext``. The
    container/codec is inferred by ffmpeg from the output extension.

    Args:
        frames_glob: An ffmpeg input pattern with a numeric token, e.g.
            ``"out/out-%04d.png"``. (Despite the name, this is an ffmpeg
            ``%0Nd`` pattern, not a shell glob -- matching the legacy script.)
        out: Output video path; extension selects the format.
        framerate: Optional input framerate (``-framerate``/``-r``); if ``None``
            ffmpeg's default (25 fps) is used, as in the legacy script.
        ffmpeg: Optional explicit binary path; autodetected if ``None``.

    Returns:
        ``str(out)`` for convenience.
    """
    binary = ffmpeg or find_ffmpeg()
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [binary]
    if framerate is not None:
        cmd += ["-framerate", str(framerate)]
    cmd += ["-i", str(frames_glob), str(out_path)]
    _run(cmd)
    return str(out_path)
