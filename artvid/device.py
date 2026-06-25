"""Device & dtype policy for artvid.

Replaces the legacy ``-gpu`` / ``-backend`` / ``-cudnn_autotune`` options
(``artistic_video.lua:22,61,62``) with a single device abstraction, per
``docs/02-migration-map.md`` section 4 and ``docs/01-architecture.md`` 3.2.

Device selection order: ``mps`` (Apple Silicon) > ``cuda`` > ``cpu``.

dtype policy
------------
- The **image variable being optimized stays ``float32``** (see
  :func:`image_optim_dtype`). L-BFGS is sensitive to numerical precision and the
  legacy implementation operates in float; we keep the optimized pixels in
  float32 on every backend for stability. This is the Phase 1 optim engine's
  policy and is **independent** of the diffusion-inference dtype.
- **Diffusion inference** (UNet / ControlNet / VAE weights) may run in
  ``float16`` / ``bfloat16`` for speed and memory — see :func:`autocast_dtype`,
  the single source of truth for that choice (the diffusion engine defers to it).
- VGG forward passes may *optionally* run in low precision on MPS/CUDA for speed;
  this is an experimental knob (architecture doc 3.2) callers opt into.
- MPS has occasional unimplemented ops; we enable ``PYTORCH_ENABLE_MPS_FALLBACK``
  so those transparently fall back to CPU rather than erroring.
- **MPS does not support ``float64``.** All tensors created/loaded in this
  project stay ``float32`` (or a diffusion low-precision dtype); never promote to
  ``torch.float64`` / ``.double()`` on any path that may run on MPS.

torch is imported lazily *inside* functions so this module is importable (and
testable for its env-var / dtype-name helpers) without torch installed.
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


# ---------------------------------------------------------------------------
# Centralized dtype policy
# ---------------------------------------------------------------------------

def _device_type(device) -> str:
    """Normalize a ``str`` / ``torch.device`` / ``None`` to a backend type string.

    Accepts ``'mps'`` / ``'cuda'`` / ``'cpu'`` (optionally with an index, e.g.
    ``'cuda:0'``), a ``torch.device``, or ``None`` (treated as ``'cpu'``). Returns
    the bare backend name without doing any torch import for the string path, so
    callers can reason about the policy without torch installed.
    """
    if device is None:
        return "cpu"
    # ``torch.device`` exposes ``.type``; a plain string is parsed by splitting
    # off any ``:index`` suffix. We avoid importing torch for the string case.
    dtype_attr = getattr(device, "type", None)
    if isinstance(dtype_attr, str):
        return dtype_attr
    return str(device).split(":", 1)[0].lower()


def image_optim_dtype():
    """Return the dtype the **optimized image variable** must use: ``float32``.

    Phase 1 policy (see module docstring): the L-BFGS-optimized pixel tensor stays
    ``float32`` on **every** backend (mps/cuda/cpu) for numerical stability. This
    helper centralizes that single constant so the pipelines do not hard-code
    ``torch.float32`` in scattered places; it is intentionally device-independent.
    """
    import torch

    return torch.float32


def autocast_dtype(device):
    """Compute dtype for **diffusion inference** weights on ``device``.

    This is the single source of truth for the low-precision inference dtype used
    by the diffusion engine (UNet / ControlNet / VAE). It does **not** apply to
    the Phase 1 optimized image (that stays ``float32`` — see
    :func:`image_optim_dtype`).

    Policy:

    * ``cuda`` → ``bfloat16`` (wide dynamic range, well supported on modern GPUs).
    * ``mps``  → ``float16``. bf16 op coverage on MPS has historically been
      spotty; fp16 is the broadly-supported fast path on Apple Silicon. The VAE
      can need an fp16-fix / fp32 decode — that caveat lives with the engine.
    * ``cpu``  → ``float32`` (fp16/bf16 CPU kernels are slow/unsupported).

    Args:
        device: ``'mps'|'cuda'|'cpu'`` string, a ``torch.device``, or ``None``.

    Returns:
        A ``torch.dtype``.
    """
    import torch

    dev = _device_type(device)
    if dev == "cuda":
        return torch.bfloat16
    if dev == "mps":
        # TODO(tuning): bf16-vs-fp16 on the M5 Max; default fp16 for op coverage.
        return torch.float16
    return torch.float32  # cpu / unknown


# ---------------------------------------------------------------------------
# Device info / diagnostics
# ---------------------------------------------------------------------------

def device_info(prefer: str | None = None) -> dict:
    """Return a small dict describing the selected device + dtype policy.

    Useful for a one-line startup banner and for debugging "is it actually on the
    GPU?" on the M5 Max. Lazily imports torch; safe to call once at run start.

    Returns keys: ``device`` (selected backend string), ``mps_available``,
    ``cuda_available``, ``mps_fallback`` (the env-var value, or ``None`` if
    unset), ``image_optim_dtype`` and ``inference_dtype`` (as ``str``).
    """
    import torch

    selected = pick_device(prefer)
    return {
        "device": selected,
        "mps_available": _device_available("mps"),
        "cuda_available": _device_available("cuda"),
        "mps_fallback": os.environ.get(MPS_FALLBACK_ENV),
        "image_optim_dtype": str(image_optim_dtype()),
        "inference_dtype": str(autocast_dtype(selected)),
        "torch_version": getattr(torch, "__version__", "?"),
    }


def print_device_info(prefer: str | None = None) -> None:
    """Print a one-line summary of :func:`device_info` (startup banner)."""
    info = device_info(prefer)
    print(
        "[artvid] device={device} (mps={mps_available} cuda={cuda_available} "
        "fallback={mps_fallback}) optim_dtype={image_optim_dtype} "
        "inference_dtype={inference_dtype} torch={torch_version}".format(**info)
    )
