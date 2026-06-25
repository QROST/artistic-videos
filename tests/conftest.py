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
