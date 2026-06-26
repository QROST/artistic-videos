"""Import-graph / structure tests that run WITHOUT torch installed.

These assert the package layout matches ``docs/01-architecture.md`` and that the
public functions/classes the pipeline depends on exist with the expected
signatures, *without* importing any torch-dependent code. The strategy:

* Use :func:`importlib.util.find_spec` to assert a module is importable as a
  spec (which does not execute it) for torch-dependent modules.
* For modules that are intentionally torch-free at import time (``config``,
  ``device``, and source-level checks), import or read them directly.
* Any assertion that needs torch is guarded by the ``requires_torch`` decorator
  / ``torch`` fixture from ``conftest.py``.

This file must remain collectable and passing in the torch-less scaffold
environment.
"""

from __future__ import annotations

import ast
import importlib
import importlib.util
import inspect
from dataclasses import fields
from pathlib import Path

import pytest

from tests.conftest import TORCH_AVAILABLE, requires_torch

REPO_ROOT = Path(__file__).resolve().parent.parent
ARTVID = REPO_ROOT / "artvid"


# --------------------------------------------------------------------------- #
# 1. Package layout: every Phase-1 module from 01-architecture.md exists.
# --------------------------------------------------------------------------- #
EXPECTED_MODULES = [
    "artvid",
    "artvid.config",
    "artvid.device",
    "artvid.io.image",
    "artvid.io.video",
    "artvid.models.vgg",
    "artvid.losses.content",
    "artvid.losses.style",
    "artvid.losses.tv",
    "artvid.losses.temporal",
    "artvid.optim.lbfgs",
    "artvid.optim.runner",
    "artvid.pipeline.stylize_image",
]


@pytest.mark.parametrize("module_name", EXPECTED_MODULES)
def test_module_is_importable_spec(module_name: str) -> None:
    """The module exists and is discoverable without executing it (no torch)."""
    spec = importlib.util.find_spec(module_name)
    assert spec is not None, f"module {module_name!r} not found on the import path"


# --------------------------------------------------------------------------- #
# 2. Source files exist where the architecture says they should.
# --------------------------------------------------------------------------- #
EXPECTED_FILES = [
    ARTVID / "config.py",
    ARTVID / "device.py",
    ARTVID / "io" / "image.py",
    ARTVID / "models" / "vgg.py",
    ARTVID / "losses" / "content.py",
    ARTVID / "losses" / "style.py",
    ARTVID / "losses" / "tv.py",
    ARTVID / "losses" / "temporal.py",
    ARTVID / "optim" / "lbfgs.py",
    ARTVID / "optim" / "runner.py",
    ARTVID / "pipeline" / "stylize_image.py",
    REPO_ROOT / "scripts" / "demo_stylize_image.py",
]


@pytest.mark.parametrize("path", EXPECTED_FILES, ids=lambda p: str(p.name))
def test_expected_file_exists(path: Path) -> None:
    assert path.is_file(), f"expected source file missing: {path}"


# --------------------------------------------------------------------------- #
# 3. config.Config — fields, defaults, parsing (torch-free).
# --------------------------------------------------------------------------- #
def test_config_imports_without_torch() -> None:
    """artvid.config must not import torch (it stays cheap/torch-free)."""
    source = (ARTVID / "config.py").read_text()
    tree = ast.parse(source)
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert "torch" not in imported, "config.py must not import torch"


def test_config_has_expected_fields() -> None:
    """The Config fields the pipeline reads are present."""
    from artvid.config import Config

    field_names = {f.name for f in fields(Config)}
    required = {
        "style_image",
        "style_blend_weights",
        "content_pattern",
        "content_weight",
        "style_weight",
        "tv_weight",
        "temporal_weight",
        "num_iterations",
        "init",
        "optimizer",
        "content_layers",
        "style_layers",
        "style_scale",
        "pooling",
        "vgg_weights",
        "device",
        "normalize_gradients",
        "tol_loss_relative",
        "tol_loss_relative_interval",
        "learning_rate",
        "print_iter",
        "save_iter",
        "seed",
        "output_image",
    }
    missing = required - field_names
    assert not missing, f"Config is missing expected fields: {sorted(missing)}"


def test_config_pair_fields_parsed() -> None:
    """num_iterations / init coerce to (first, subsequent) 2-tuples."""
    from artvid.config import Config

    cfg = Config()
    assert isinstance(cfg.num_iterations, tuple) and len(cfg.num_iterations) == 2
    assert isinstance(cfg.init, tuple) and len(cfg.init) == 2
    # Strings broadcast to a pair.
    cfg2 = Config(num_iterations="500", init="image")
    assert cfg2.num_iterations == (500, 500)
    assert cfg2.init == ("image", "image")
    # Comma-separated split into (first, subsequent).
    cfg3 = Config(num_iterations="2000,1000", init="random,prevWarped")
    assert cfg3.num_iterations == (2000, 1000)
    assert cfg3.init == ("random", "prevWarped")


def test_config_layer_defaults() -> None:
    """Default content/style layers match the legacy neural-style defaults."""
    from artvid.config import Config

    cfg = Config()
    assert cfg.content_layers == ("relu4_2",)
    assert cfg.style_layers == (
        "relu1_1",
        "relu2_1",
        "relu3_1",
        "relu4_1",
        "relu5_1",
    )


# --------------------------------------------------------------------------- #
# 4. device.py — env-var helper works without torch.
# --------------------------------------------------------------------------- #
def test_enable_mps_fallback_no_torch(monkeypatch) -> None:
    """enable_mps_fallback sets the env var without importing torch."""
    from artvid import device

    monkeypatch.delenv(device.MPS_FALLBACK_ENV, raising=False)
    device.enable_mps_fallback()
    import os

    assert os.environ.get(device.MPS_FALLBACK_ENV) == "1"


def test_device_constants() -> None:
    from artvid import device

    assert device.VALID_DEVICES == ("mps", "cuda", "cpu")


# --------------------------------------------------------------------------- #
# 5. stylize_image — public API exists with the right signature (torch-free).
#
# stylize_image.py imports torch-free modules at module top (config, io.image,
# models.vgg, optim.runner). io.image / optim.runner / models.vgg import torch
# only lazily inside functions, so importing the pipeline module itself must
# succeed without torch. We verify that, plus the public signature, by source
# inspection AND a real import (the import is torch-free by construction).
# --------------------------------------------------------------------------- #
def test_stylize_image_module_imports_without_torch() -> None:
    """The pipeline module imports without torch present."""
    mod = importlib.import_module("artvid.pipeline.stylize_image")
    assert hasattr(mod, "stylize_image")
    assert callable(mod.stylize_image)


def test_stylize_image_signature() -> None:
    """stylize_image exposes the expected parameters for the demo + callers."""
    from artvid.pipeline.stylize_image import stylize_image

    sig = inspect.signature(stylize_image)
    params = sig.parameters
    assert "content" in params
    assert "style" in params
    assert "config" in params
    # device + output_path are keyword-only conveniences.
    assert "device" in params
    assert "output_path" in params
    assert params["config"].default is None


def test_stylize_helpers_exist() -> None:
    """Internal helpers the pipeline relies on are present."""
    from artvid.pipeline import stylize_image as si

    for name in (
        "_normalized_blend_weights",
        "_build_style_targets",
        "_init_image",
        "_preprocess_mode_for",
    ):
        assert hasattr(si, name), f"stylize_image is missing helper {name!r}"


def test_normalized_blend_weights_torchfree() -> None:
    """Blend-weight normalization is pure-python and matches legacy semantics."""
    from artvid.pipeline.stylize_image import _normalized_blend_weights

    # Equal weighting when unspecified.
    assert _normalized_blend_weights(None, 3) == pytest.approx([1 / 3, 1 / 3, 1 / 3])
    assert _normalized_blend_weights("nil", 2) == pytest.approx([0.5, 0.5])
    # Explicit weights are normalized to sum to 1.
    assert _normalized_blend_weights("1,3", 2) == pytest.approx([0.25, 0.75])
    # Mismatched count is an error (legacy assert).
    with pytest.raises(ValueError):
        _normalized_blend_weights("1,2,3", 2)
    with pytest.raises(ValueError):
        _normalized_blend_weights("0", 1)


def test_preprocess_mode_selection() -> None:
    """torchvision weights -> torchvision mode; a path -> caffe mode."""
    from artvid.config import Config
    from artvid.io.image import MODE_CAFFE, MODE_TORCHVISION
    from artvid.pipeline.stylize_image import _preprocess_mode_for

    assert _preprocess_mode_for(Config(vgg_weights="torchvision")) == MODE_TORCHVISION
    assert (
        _preprocess_mode_for(Config(vgg_weights="/models/vgg19_caffe.pth"))
        == MODE_CAFFE
    )


# --------------------------------------------------------------------------- #
# 6. Cross-module signature reconciliation (torch-free, source-level).
#
# stylize_image relies on specific function names/signatures from the modules
# written by other agents. Catch a drift here without needing torch.
# --------------------------------------------------------------------------- #
def test_collaborator_callables_present() -> None:
    """The torch-free functions/classes stylize_image imports exist by name.

    ``io.image`` and ``optim.runner`` import torch only lazily, so they are
    importable here without torch. ``models.vgg`` imports torch at module top
    level — its symbols are checked in the torch-gated test below.
    """
    from artvid.io import image as io_image
    from artvid.optim import runner

    for name in (
        "load_image",
        "preprocess",
        "save_image",
        "scale_style_image",
        "MODE_CAFFE",
        "MODE_TORCHVISION",
    ):
        assert hasattr(io_image, name), f"io.image is missing {name!r}"

    assert hasattr(runner, "run_optimization")
    assert hasattr(runner, "RunResult")


@requires_torch
def test_vgg_callables_present() -> None:
    """``models.vgg`` (torch module) exposes the symbols stylize_image uses."""
    from artvid.models import vgg

    assert hasattr(vgg, "build_feature_net")
    assert hasattr(vgg, "split_activations")


def test_run_optimization_signature() -> None:
    """run_optimization accepts the args stylize_image passes."""
    from artvid.optim.runner import run_optimization

    params = inspect.signature(run_optimization).parameters
    for name in (
        "image_var",
        "loss_fn",
        "max_iter",
        "optimizer",
        "tol_loss_relative",
        "tol_loss_relative_interval",
        "learning_rate",
        "print_iter",
        "save_iter",
        "save_fn",
    ):
        assert name in params, f"run_optimization is missing parameter {name!r}"


@requires_torch
def test_build_feature_net_signature() -> None:
    """build_feature_net accepts content/style layers, pooling, weights."""
    from artvid.models.vgg import build_feature_net

    params = inspect.signature(build_feature_net).parameters
    for name in ("content_layers", "style_layers", "pooling", "weights"):
        assert name in params, f"build_feature_net is missing parameter {name!r}"


# --------------------------------------------------------------------------- #
# 7. A torch-gated smoke check that the loss modules construct (skipped w/o torch).
# --------------------------------------------------------------------------- #
@requires_torch
def test_loss_modules_constructible() -> None:
    import torch

    from artvid.losses.content import ContentLoss
    from artvid.losses.style import StyleLoss
    from artvid.losses.tv import TVLoss

    feat = torch.randn(1, 4, 8, 8)
    ContentLoss(feat, strength=5.0)
    StyleLoss(feat, strength=1e2)
    TVLoss(strength=1e-3)


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not installed")
def test_init_image_random_is_float32_grad() -> None:
    import torch

    from artvid.pipeline.stylize_image import _init_image

    content_pre = torch.zeros(3, 8, 8)
    img = _init_image("random", content_pre, torch.device("cpu"), seed=0)
    assert img.dtype == torch.float32
    assert img.requires_grad is True
    assert img.shape == (3, 8, 8)
