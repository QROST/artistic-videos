"""Shared pytest fixtures for the artvid test suite.

torch is not installable in the dev/CI scaffold environment, so any test that
needs it should depend on the ``torch`` fixture below. The fixture skips the
test when torch is missing, letting the suite run (and pass non-torch tests)
until torch is installed on the target machine.
"""

from __future__ import annotations

import importlib.util

import pytest

#: True when torch can be imported in the current environment.
TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


def pytest_configure(config: "pytest.Config") -> None:
    """Register custom markers so CI can (de)select expensive test paths.

    ``pyproject.toml`` is owned by another component this run, so the markers are
    registered here instead of under ``[tool.pytest.ini_options] markers``.
    Registering them silences ``PytestUnknownMarkWarning`` and (with
    ``-W error``) keeps an unfiltered run clean.

    Markers
    -------
    * ``network`` — the test would download model weights (torchvision VGG-19 /
      RAFT, or diffusers SDXL / ControlNet / IP-Adapter checkpoints) or otherwise
      touch the network. CI runs the offline subset with ``-m 'not network'``.
    * ``mps`` — the test requires the Apple-Silicon Metal (MPS) backend, which is
      unavailable on the Linux CI runner. Deselect with ``-m 'not mps'``.
    * ``slow`` — heavier CPU end-to-end paths (e.g. multi-iteration optimization)
      that are still offline but noticeably slower; deselectable independently.
    """
    config.addinivalue_line(
        "markers",
        "network: test downloads model weights or needs network access "
        "(deselect on offline CI with -m 'not network').",
    )
    config.addinivalue_line(
        "markers",
        "mps: test requires the Apple-Silicon MPS backend "
        "(deselect on non-Mac CI with -m 'not mps').",
    )
    config.addinivalue_line(
        "markers",
        "slow: offline but slower CPU end-to-end test.",
    )


@pytest.fixture
def torch():
    """Yield the imported ``torch`` module, or skip the test if unavailable."""
    if not TORCH_AVAILABLE:
        pytest.skip("torch is not installed in this environment")
    import torch as _torch

    return _torch


@pytest.fixture
def device(torch):
    """Best available torch device for tests (mps|cuda|cpu)."""
    from artvid.device import get_device

    return get_device()


def requires_torch(func):
    """Decorator marking a test as needing torch (skips when unavailable)."""
    return pytest.mark.skipif(
        not TORCH_AVAILABLE, reason="torch is not installed in this environment"
    )(func)


@pytest.fixture
def mps(torch):
    """Skip unless the Apple-Silicon MPS backend is actually available.

    Use together with the ``@pytest.mark.mps`` marker so CI on non-Mac runners
    can deselect these with ``-m 'not mps'`` *and* they self-skip if collected.
    """
    backend = getattr(torch.backends, "mps", None)
    if backend is None or not backend.is_available():
        pytest.skip("MPS backend not available in this environment")
    return torch.device("mps")
