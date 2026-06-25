"""Device & dtype policy for artvid.

Replaces the legacy ``-gpu`` / ``-backend`` / ``-cudnn_autotune`` options
(``artistic_video.lua:22,61,62``) with a single device abstraction, per
``docs/02-migration-map.md`` section 4 and ``docs/01-architecture.md`` 3.2.

Device selection order: ``mps`` (Apple Silicon) > ``cuda`` > ``cpu``.

dtype policy
------------
- The **image variable being optimized stays ``float32``**. L-BFGS is sensitive
  to numerical precision and the legacy implementation operates in float; we
  keep the optimized pixels in float32 on every backend for stability.
- VGG forward passes may *optionally* run in ``float16`` / autocast on MPS or
  CUDA for speed. This is an experimental knob (see architecture doc 3.2) and is
  not enabled by the device layer itself — callers opt in.
- MPS has occasional unimplemented ops; we enable ``PYTORCH_ENABLE_MPS_FALLBACK``
  so those transparently fall back to CPU rather than erroring.

torch is imported lazily *inside* functions so this module is importable (and
testable for its env-var helper) without torch installed.
"""

from __future__ import annotations

import os

# Valid string device identifiers this project understands.
VALID_DEVICES = ("mps", "cuda", "cpu")

# Env var that makes the MPS backend fall back to CPU for unimplemented ops.
MPS_FALLBACK_ENV = "PYTORCH_ENABLE_MPS_FALLBACK"


def enable_mps_fallback() -> None:
    """Set ``PYTORCH_ENABLE_MPS_FALLBACK=1`` if it is not already set.

    This must be called before torch dispatches the first MPS op to take
    effect; calling it at process/CLI startup is recommended. We do not
    overwrite a value the user has explicitly set.
    """
    os.environ.setdefault(MPS_FALLBACK_ENV, "1")


def pick_device(prefer: str | None = None) -> str:
    """Return the best available device string: ``'mps'``, ``'cuda'`` or ``'cpu'``.

    Args:
        prefer: Optional explicit device. If given and available it is returned;
            if given but unavailable, falls back to autodetection. ``None``
            autodetects.

    Returns:
        One of :data:`VALID_DEVICES`.
    """
    import torch

    if prefer is not None:
        prefer = prefer.lower()
        if prefer not in VALID_DEVICES:
            raise ValueError(
                f"Unknown device {prefer!r}; expected one of {VALID_DEVICES}."
            )
        if _device_available(prefer):
            return prefer
        # Requested device not available -> fall through to autodetect.

    if _device_available("mps"):
        return "mps"
    if _device_available("cuda"):
        return "cuda"
    return "cpu"


def _device_available(name: str) -> bool:
    """Whether the given device backend is usable in this torch build."""
    import torch

    if name == "cpu":
        return True
    if name == "cuda":
        return bool(torch.cuda.is_available())
    if name == "mps":
        # ``torch.backends.mps`` only exists on builds that include the backend.
        backend = getattr(torch.backends, "mps", None)
        return bool(backend is not None and backend.is_available())
    return False


def get_device(prefer: str | None = None):
    """Convenience wrapper returning a ``torch.device`` for :func:`pick_device`."""
    import torch

    return torch.device(pick_device(prefer))
