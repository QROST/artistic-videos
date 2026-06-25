"""Configuration dataclass + parser for artvid.

Ports every ``cmd:option`` from the legacy ``artistic_video.lua`` (lines 11-70)
and ``artistic_video_multiPass.lua`` (lines 14-75), per the parameter mapping in
``docs/02-migration-map.md`` section 3. Field names and defaults are kept the
same as the Lua originals so existing users can migrate easily.

Two legacy options accepted comma-separated "first, subsequent" values
(``-num_iterations '2000,1000'`` and ``-init 'random,prevWarped'``); these are
represented as 2-tuples ``(first, subsequent)``.

The legacy ``-gpu`` / ``-backend`` / ``-cudnn_autotune`` options are dropped in
favor of :mod:`artvid.device`; a single ``device`` field replaces them.

No torch import here — this module stays cheap and torch-free.
"""

from __future__ import annotations

import shlex
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

# VGG mean pixel (caffe BGR convention); also used to fill warp out-of-bounds
# regions. Kept here so config consumers don't have to import torch/io. See
# artistic_video_core.lua:475-492.
CAFFE_BGR_MEAN = (103.939, 116.779, 123.68)


def _parse_int_pair(value: str | int | tuple[int, int]) -> tuple[int, int]:
    """Parse ``"2000,1000"`` / ``"2000"`` / ``2000`` into ``(first, subsequent)``.

    A single value is broadcast to both entries (legacy semantics: one value
    applies to all frames).
    """
    if isinstance(value, tuple):
        return (int(value[0]), int(value[1]))
    if isinstance(value, int):
        return (value, value)
    parts = [p.strip() for p in str(value).split(",") if p.strip()]
    if len(parts) == 1:
        n = int(parts[0])
        return (n, n)
    if len(parts) == 2:
        return (int(parts[0]), int(parts[1]))
    raise ValueError(f"Expected one or two comma-separated ints, got {value!r}.")


def _parse_str_pair(value: str | tuple[str, str]) -> tuple[str, str]:
    """Parse ``"random,prevWarped"`` / ``"random"`` into ``(first, subsequent)``.

    A single value is broadcast to both entries.
    """
    if isinstance(value, tuple):
        return (str(value[0]), str(value[1]))
    parts = [p.strip() for p in str(value).split(",") if p.strip()]
    if len(parts) == 1:
        return (parts[0], parts[0])
    if len(parts) == 2:
        return (parts[0], parts[1])
    raise ValueError(f"Expected one or two comma-separated strings, got {value!r}.")


def _parse_str_list(value: str | list[str] | tuple[str, ...]) -> tuple[str, ...]:
    """Parse comma-separated layer names into a tuple."""
    if isinstance(value, (list, tuple)):
        return tuple(str(v) for v in value)
    return tuple(p.strip() for p in str(value).split(",") if p.strip())


@dataclass
class Config:
    """All artvid run parameters.

    Defaults match the single-pass legacy ``artistic_video.lua`` unless a field
    only exists in the multi-pass tool, in which case its multi-pass default is
    used. ``temporal_weight`` differs between modes (1e3 single / 5e2 multi);
    the single-pass default is kept here and the multi-pass pipeline overrides
    it (see docs/02-migration-map.md section 3).
    """

    # --- Basic options (artistic_video.lua:14-23) ---
    style_image: str = "example/seated-nude.jpg"
    style_blend_weights: str | None = None
    content_pattern: str = "example/marple8_%02d.ppm"
    num_images: int = 0  # 0 = autodetect
    start_number: int = 1
    continue_with: int = 1
    number_format: str = "%d"

    # --- Flow options (single-pass: artistic_video.lua:26-32) ---
    flow_pattern: str = "example/deepflow/backward_[%d]_{%d}.flo"
    flow_weight_pattern: str = "example/deepflow/reliable_[%d]_{%d}.pgm"
    flow_relative_indices: tuple[int, ...] = (1,)
    use_flow_every: int = -1  # -1 disables
    invert_flow_weights: bool = False

    # --- Flow options (multi-pass: artistic_video_multiPass.lua:26-33) ---
    forward_flow_pattern: str = "example/deepflow/forward_[%d]_{%d}.flo"
    backward_flow_pattern: str = "example/deepflow/backward_[%d]_{%d}.flo"
    forward_flow_weight_pattern: str = "example/deepflow/reliable_[%d]_{%d}.pgm"
    backward_flow_weight_pattern: str = "example/deepflow/reliable_[%d]_{%d}.pgm"

    # --- Multi-pass options (artistic_video_multiPass.lua:36-40) ---
    blend_weight: float = 1.0
    blend_weight_last_pass: float = 0.0
    use_temporal_loss_after: int = 8
    num_passes: int = 15
    continue_with_pass: int = 1

    # --- Optimization options (artistic_video.lua:35-47) ---
    content_weight: float = 5e0
    style_weight: float = 1e2
    temporal_weight: float = 1e3  # multi-pass overrides to 5e2
    tv_weight: float = 1e-3
    temporal_criterion: str = "mse"  # mse|smoothl1
    num_iterations: tuple[int, int] = (2000, 1000)  # (first, subsequent)
    tol_loss_relative: float = 1e-4
    tol_loss_relative_interval: int = 50
    normalize_gradients: bool = False
    init: tuple[str, str] = ("random", "prevWarped")  # (first, subsequent)
    optimizer: str = "lbfgs"  # lbfgs|adam
    learning_rate: float = 1e1

    # --- Output options (artistic_video.lua:50-54) ---
    print_iter: int = 100
    save_iter: int = 0
    output_image: str = "out.png"
    output_folder: str = ""
    save_init: bool = False

    # --- Other / model options (artistic_video.lua:57-69) ---
    style_scale: float = 1.0
    pooling: str = "max"  # max|avg
    seed: int = -1
    content_layers: tuple[str, ...] = ("relu4_2",)
    style_layers: tuple[str, ...] = (
        "relu1_1",
        "relu2_1",
        "relu3_1",
        "relu4_1",
        "relu5_1",
    )
    combine_flow_weights_method: str = "closestFirst"  # normalize|closestFirst

    # --- Diffusion engine (Phase 2; see docs/07-phase2-design.md §4.1) ---
    # Consumed by the (lazy, torch-touching) artvid.diffusion package when the
    # CLI runs with --engine diffusion. All fields are torch-free plain values;
    # nothing here imports torch/diffusers. The flow_* patterns, content_pattern,
    # style_image, output_*, start_number, num_images and number_format above are
    # REUSED unchanged by the diffusion path -- these only add the model/sampling
    # and latent-consistency knobs.

    # Model stack (HF ids). TODO(tuning): confirm exact ids/revisions are the
    # mid-2026 SOTA + available + license-clean before download; pin revision=.
    diff_base_model: str = "stabilityai/stable-diffusion-xl-base-1.0"
    controlnet_model: str = "diffusers/controlnet-depth-sdxl-1.0"
    controlnet_kind: str = "depth"  # depth|canny|lineart|hed|tile (structure preproc)
    controlnet_scale: float = 0.7  # ControlNet conditioning_scale; TODO(tuning)
    ip_adapter_repo: str = "h94/IP-Adapter"
    ip_adapter_subfolder: str = "sdxl_models"
    ip_adapter_weight: str = "ip-adapter_sdxl.bin"
    ip_adapter_scale: float = 0.7  # style-from-reference strength; TODO(tuning)

    # Sampling / prompting.
    diff_prompt: str = ""  # optional text prompt (style/content hints)
    diff_negative_prompt: str = ""
    diff_steps: int = 30  # K denoising steps; TODO(tuning) per scheduler
    guidance_scale: float = 6.0  # classifier-free guidance; TODO(tuning)
    denoise_strength: float = 1.0  # img2img denoise fraction (1.0 = full); TODO(tuning)
    temporal_init_strength: float = 0.6  # img2img strength for warped-latent init
    # on subsequent frames (lower = stay closer to the warped previous frame =
    # more temporally stable; higher = more fresh synthesis). TODO(tuning).
    diff_scheduler: str = "euler"  # euler|ddim|dpm

    # Temporal latent-flow consistency (the Phase 2 differentiator; §2).
    temporal_strength: float = 0.6  # per-step fusion blend cap in reliable regions; TODO(tuning)
    temporal_fuse_start: float = 0.0  # fraction of steps where fusion begins; TODO(tuning)
    temporal_fuse_end: float = 0.7  # fraction where it ends (fuse early/mid only); TODO(tuning)
    latent_consistency_weight: float = 0.6  # alias-friendly weight for the latent
    # consistency mechanism; mirrors temporal_strength as the "how hard to pull
    # toward the warped previous latent" knob. TODO(tuning): decide whether to
    # collapse with temporal_strength once tuned on hardware.
    latent_reliability_gamma: float = 2.0  # erosion of downsampled reliability mask; TODO(tuning)
    warp_space: str = "latent"  # latent|pixel (latent warp vs VAE-decode->warp->encode)
    use_anchor: bool = False  # long-term anchor warp (combine_longterm_weights); §2.6
    vae_factor: int = 8  # VAE spatial downsample (SDXL/SD1.5 = 8); change only if model differs

    # --- Modernization: replaces -proto_file/-model_file (loadcaffe) ---
    # 'torchvision' uses torchvision VGG-19 weights (RGB, ImageNet norm);
    # any other value is treated as a path to caffe VGG-19 weights (BGR mean).
    vgg_weights: str = "torchvision"

    # --- Modernization: replaces -gpu/-backend/-cudnn_autotune ---
    # None => autodetect via artvid.device.pick_device(); else 'mps'|'cuda'|'cpu'.
    device: str | None = None

    def __post_init__(self) -> None:
        # Coerce tuple/list-shaped fields so callers may pass raw strings.
        self.num_iterations = _parse_int_pair(self.num_iterations)
        self.init = _parse_str_pair(self.init)
        self.content_layers = _parse_str_list(self.content_layers)
        self.style_layers = _parse_str_list(self.style_layers)
        if not isinstance(self.flow_relative_indices, tuple):
            if isinstance(self.flow_relative_indices, (list, tuple)):
                self.flow_relative_indices = tuple(
                    int(i) for i in self.flow_relative_indices
                )
            else:
                self.flow_relative_indices = tuple(
                    int(p.strip())
                    for p in str(self.flow_relative_indices).split(",")
                    if p.strip()
                )


# Mapping of legacy ``-option`` names to Config field names, used by the
# ``-args`` file parser to accept legacy-style argument files unchanged.
_LEGACY_ALIASES: dict[str, str] = {
    "flowWeight_pattern": "flow_weight_pattern",
    "forwardFlow_pattern": "forward_flow_pattern",
    "backwardFlow_pattern": "backward_flow_pattern",
    "forwardFlow_weight_pattern": "forward_flow_weight_pattern",
    "backwardFlow_weight_pattern": "backward_flow_weight_pattern",
    "blendWeight": "blend_weight",
    "blendWeight_lastPass": "blend_weight_last_pass",
    "use_temporalLoss_after": "use_temporal_loss_after",
    "temporal_loss_criterion": "temporal_criterion",
    "invert_flowWeights": "invert_flow_weights",
    "combine_flowWeights_method": "combine_flow_weights_method",
}


def _coerce_to_field(name: str, raw: str) -> Any:
    """Coerce a raw string token to the type of ``Config`` field ``name``."""
    type_map = {f.name: f.type for f in fields(Config)}
    if name not in type_map:
        raise KeyError(f"Unknown config field: {name!r}")
    if name in ("num_iterations", "init"):
        # Leave as the raw string; __post_init__ parses the pair.
        return raw
    if name == "flow_relative_indices":
        return raw  # __post_init__ parses comma list
    if name in ("content_layers", "style_layers"):
        return _parse_str_list(raw)

    current = getattr(Config(), name)
    if isinstance(current, bool):
        return str(raw).strip().lower() in ("1", "true", "yes", "on")
    if isinstance(current, int):
        return int(raw)
    if isinstance(current, float):
        return float(raw)
    return raw


def load_args_file(path: str | Path) -> dict[str, Any]:
    """Parse a legacy ``-args`` file (one ``-option value`` per line).

    Replicates the legacy behavior (artistic_video.lua:331-357): each line is a
    single argument; a leading ``-`` on the option name is optional; blank lines
    and ``#`` comments are ignored. Returns a kwargs dict for :class:`Config`.
    """
    overrides: dict[str, Any] = {}
    text = Path(path).read_text()
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        tokens = shlex.split(line)
        if not tokens:
            continue
        key = tokens[0].lstrip("-")
        value = " ".join(tokens[1:]) if len(tokens) > 1 else "true"
        key = _LEGACY_ALIASES.get(key, key)
        overrides[key] = _coerce_to_field(key, value)
    return overrides


def parse_config(argv: list[str] | None = None) -> Config:
    """Build a :class:`Config` from CLI args using tyro.

    tyro derives a typed CLI directly from the dataclass, so every field is an
    exposed flag (e.g. ``--content-weight 10``). An ``-args`` legacy file can be
    layered first via :func:`load_args_file` by the caller if desired.

    Falls back to ``Config()`` defaults when tyro is unavailable is *not* done
    silently here — import errors propagate so the CLI fails loudly.
    """
    import tyro

    return tyro.cli(Config, args=argv)
