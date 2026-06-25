"""Structure tests for the Phase 2 diffusion video stylization scaffold.

These assert the diffusion package layout, the public API surface
(``stylize_video_diffusion`` + the engine / latent-warp symbols it depends on),
and the torch-free helper behaviour of ``artvid.diffusion.video`` *without*
importing torch or diffusers.

Strategy (mirrors ``tests/test_structure.py``):

* ``importlib.util.find_spec`` to assert a module is importable as a spec
  (does NOT execute it) for the framework-touching modules.
* Source-level (``ast``) assertions that ``torch`` / ``diffusers`` are only
  *lazily* imported (inside functions), never at module top level — the
  ``--help`` / collection path must stay framework-free.
* Real imports + ``inspect.signature`` for the parts that ARE torch-free at
  import time (the ``video`` module top-level imports are all torch-free; the
  heavy imports live inside ``stylize_video_diffusion``).
* Pure-python behaviour checks for ``fuse_step_set`` (no torch needed).
* ``@requires_torch`` gates for anything that would actually run a tensor op.

This file must remain collectable and passing in the torch-less scaffold.
"""

from __future__ import annotations

import ast
import importlib
import importlib.util
import inspect
from pathlib import Path

import pytest

from tests.conftest import requires_torch

REPO_ROOT = Path(__file__).resolve().parent.parent
ARTVID = REPO_ROOT / "artvid"
DIFFUSION = ARTVID / "diffusion"


# --------------------------------------------------------------------------- #
# 1. Package layout: the Phase 2 diffusion modules from docs/07 §3 exist.
# --------------------------------------------------------------------------- #
EXPECTED_MODULES = [
    "artvid.diffusion",
    "artvid.diffusion.engine",
    "artvid.diffusion.latent_warp",
    "artvid.diffusion.video",
]


@pytest.mark.parametrize("module_name", EXPECTED_MODULES)
def test_diffusion_module_importable_spec(module_name: str) -> None:
    """The module exists and is discoverable without executing it (no torch)."""
    spec = importlib.util.find_spec(module_name)
    assert spec is not None, f"module {module_name!r} not found on the import path"


EXPECTED_FILES = [
    DIFFUSION / "__init__.py",
    DIFFUSION / "engine.py",
    DIFFUSION / "latent_warp.py",
    DIFFUSION / "video.py",
]


@pytest.mark.parametrize("path", EXPECTED_FILES, ids=lambda p: str(p.name))
def test_diffusion_file_exists(path: Path) -> None:
    assert path.is_file(), f"expected diffusion source file missing: {path}"


# --------------------------------------------------------------------------- #
# 2. video.py keeps torch / diffusers LAZY (no top-level framework imports).
# --------------------------------------------------------------------------- #
def _top_level_imports(source: str) -> set[str]:
    """Root module names imported at module top level (not inside functions)."""
    tree = ast.parse(source)
    names: set[str] = set()
    for node in tree.body:  # only module-level statements
        if isinstance(node, ast.Import):
            names.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module.split(".")[0])
    return names


def test_video_does_not_import_torch_at_module_level() -> None:
    """artvid.diffusion.video must lazy-import torch/diffusers (inside funcs)."""
    source = (DIFFUSION / "video.py").read_text()
    top = _top_level_imports(source)
    assert "torch" not in top, "video.py must not import torch at module level"
    assert "diffusers" not in top, "video.py must not import diffusers at module level"


def test_video_module_imports_without_torch() -> None:
    """The module imports cleanly without torch present (lazy framework imports)."""
    mod = importlib.import_module("artvid.diffusion.video")
    assert hasattr(mod, "stylize_video_diffusion")
    assert callable(mod.stylize_video_diffusion)
    assert hasattr(mod, "FrameResult")
    assert hasattr(mod, "fuse_step_set")


# --------------------------------------------------------------------------- #
# 3. Public API surface / signatures (torch-free; the heavy imports are lazy).
# --------------------------------------------------------------------------- #
def test_stylize_video_diffusion_signature() -> None:
    """The video entry point exposes the parameters the CLI / callers pass."""
    from artvid.diffusion.video import stylize_video_diffusion

    params = inspect.signature(stylize_video_diffusion).parameters
    for name in ("config", "engine", "device", "flow_source"):
        assert name in params, f"stylize_video_diffusion missing parameter {name!r}"
    # config defaults to None (build a default Config); flow_source defaults to auto.
    assert params["config"].default is None
    assert params["flow_source"].default == "auto"


def test_frame_result_fields() -> None:
    """FrameResult mirrors the optim engine's per-frame result shape."""
    from dataclasses import fields

    from artvid.diffusion.video import FrameResult

    names = {f.name for f in fields(FrameResult)}
    for name in ("frame_idx", "output_path", "previous_indices", "used_temporal"):
        assert name in names, f"FrameResult missing field {name!r}"


def test_video_helpers_exist() -> None:
    """Internal helpers the loop relies on are present by name."""
    from artvid.diffusion import video

    for name in (
        "fuse_step_set",
        "_output_path_for",
        "_flow_pair_for",
        "_save_rgb",
        "_get",
    ):
        assert hasattr(video, name), f"video is missing helper {name!r}"


# --------------------------------------------------------------------------- #
# 4. fuse_step_set — pure-python window math (no torch).
# --------------------------------------------------------------------------- #
def test_fuse_step_set_window() -> None:
    """The fuse window maps fractions of K to the right integer step indices."""
    from artvid.diffusion.video import fuse_step_set

    # Default window [0.0, 0.7) over 30 steps -> {0 .. 20}.
    s = fuse_step_set(30, 0.0, 0.7)
    assert s == set(range(0, 21))
    # Mid window [0.2, 0.6) over 10 steps -> {2,3,4,5}.
    assert fuse_step_set(10, 0.2, 0.6) == {2, 3, 4, 5}
    # Full window covers every step.
    assert fuse_step_set(5, 0.0, 1.0) == {0, 1, 2, 3, 4}


def test_fuse_step_set_degenerate() -> None:
    """Empty / degenerate windows yield an empty set (fusion disabled)."""
    from artvid.diffusion.video import fuse_step_set

    assert fuse_step_set(0, 0.0, 1.0) == set()
    assert fuse_step_set(10, 0.7, 0.7) == set()  # empty window
    assert fuse_step_set(10, 0.8, 0.2) == set()  # inverted window
    # Out-of-range fractions clamp into [0, K).
    assert fuse_step_set(4, -0.5, 2.0) == {0, 1, 2, 3}


# --------------------------------------------------------------------------- #
# 5. _get — defensive config reads (torch-free).
# --------------------------------------------------------------------------- #
def test_get_defensive_reads() -> None:
    """_get falls back to the default for missing OR None attributes."""
    from artvid.diffusion.video import _get

    class Cfg:
        present = 5
        none_field = None

    cfg = Cfg()
    assert _get(cfg, "present", 99) == 5
    assert _get(cfg, "missing", 99) == 99
    # None is treated as "not set" so the documented default wins.
    assert _get(cfg, "none_field", 99) == 99
    assert _get(None, "anything", 7) == 7


def test_video_default_constants_present() -> None:
    """The §4.1 temporal defaults the loop reads are defined as module constants."""
    from artvid.diffusion import video

    for name in (
        "DEFAULT_TEMPORAL_STRENGTH",
        "DEFAULT_TEMPORAL_FUSE_START",
        "DEFAULT_TEMPORAL_FUSE_END",
        "DEFAULT_TEMPORAL_INIT_STRENGTH",
        "DEFAULT_RELIABILITY_GAMMA",
        "DEFAULT_VAE_FACTOR",
        "DEFAULT_USE_ANCHOR",
    ):
        assert hasattr(video, name), f"video missing default constant {name!r}"
    # The fuse window default is the early/mid window documented in docs §2.5.
    assert video.DEFAULT_TEMPORAL_FUSE_START == 0.0
    assert 0.0 < video.DEFAULT_TEMPORAL_FUSE_END <= 1.0


# --------------------------------------------------------------------------- #
# 6. Cross-module contract: the symbols video.py imports from collaborators
#    exist by name (source-level / torch-free where possible). Catches drift in
#    the engine / latent_warp / singlepass APIs the loop depends on.
# --------------------------------------------------------------------------- #
def test_engine_callables_present() -> None:
    """DiffusionEngine exposes the methods the video loop drives.

    engine.py lazy-imports torch/diffusers, so it is importable here without the
    frameworks; we check the method names exist on the class.
    """
    from artvid.diffusion.engine import DiffusionEngine

    for name in (
        "from_config",
        "load",
        "encode_style",
        "denoise_frame",
        "decode",
        "_build_control",
    ):
        assert hasattr(DiffusionEngine, name), f"DiffusionEngine missing {name!r}"


def test_denoise_frame_accepts_temporal_kwargs() -> None:
    """denoise_frame accepts the init+fuse kwargs the video loop passes."""
    from artvid.diffusion.engine import DiffusionEngine

    params = inspect.signature(DiffusionEngine.denoise_frame).parameters
    for name in (
        "content_rgb",
        "control_image",
        "style",
        "init_latents",
        "reliability",
        "warped_latent",
        "strength",
        "steps",
        "seed",
        "fuse_steps",
        "temporal_strength",
    ):
        assert name in params, f"denoise_frame missing parameter {name!r}"


def test_latent_warp_callables_present() -> None:
    """latent_warp exposes warp_latent / latent_reliability / combine helper."""
    from artvid.diffusion import latent_warp

    for name in ("warp_latent", "latent_reliability", "combine_latent_reliability"):
        assert hasattr(latent_warp, name), f"latent_warp missing {name!r}"


def test_singlepass_reuse_helpers_present() -> None:
    """video.py reuses these (torch-free-importable) singlepass helpers.

    singlepass.py imports torch only lazily, so it is importable here; we assert
    the exact helper names video._flow_pair_for / _output_path_for depend on.
    """
    from artvid.pipeline import singlepass

    for name in (
        "discover_num_images",
        "_content_frame_path",
        "build_out_filename",
        "format_flow_filename",
        "_read_flo_tensor",
        "_use_precomputed",
    ):
        assert hasattr(singlepass, name), f"singlepass missing reused helper {name!r}"


# --------------------------------------------------------------------------- #
# 7. A torch-gated smoke check (skipped without torch): the torch-free building
#    blocks compose. We do NOT build a pipeline (needs diffusers + weights); we
#    only exercise the framework-free helpers under a real torch import.
# --------------------------------------------------------------------------- #
@requires_torch
def test_save_rgb_roundtrip(torch, tmp_path) -> None:
    """_save_rgb writes a plain RGB[0,1] CHW tensor without VGG deprocessing."""
    from artvid.diffusion.video import _save_rgb
    from artvid.io.image import load_image

    img = torch.zeros(3, 4, 6)
    img[0] = 1.0  # pure red
    out = tmp_path / "frame-1.png"
    _save_rgb(img, str(out))
    assert out.is_file()
    # Reload: red channel ~1, others ~0 (no deprocess mean-shift applied).
    back = load_image(out)
    assert back.shape == (3, 4, 6)
    assert float(back[0].mean()) > 0.9
    assert float(back[1].mean()) < 0.1
    assert float(back[2].mean()) < 0.1
