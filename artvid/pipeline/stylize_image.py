"""Single-image neural style transfer.

Wires the M0 pieces (``models/vgg``, ``losses/*``, ``optim/runner``,
``io/image``, ``device``) into the classic Gatys-style single-image optimization
that the video frames build on. It is the Python equivalent of the *core* of
``buildNet`` + ``runOptimization`` (``artistic_video_core.lua:137-250`` and
``10-134``) specialized to one frame with **no** temporal term -- exactly what
``artistic_video.lua`` does for the very first frame when ``init=random`` /
``image``.

Modernization (see ``docs/01-architecture.md`` 3.4/3.5,
``docs/02-migration-map.md`` lines 28, 48): the loss modules are *not* inserted
into the network. We run the VGG feature net once per optimizer step, route the
named activations to externally-held :class:`~artvid.losses.content.ContentLoss`
/ :class:`~artvid.losses.style.StyleLoss` / :class:`~artvid.losses.tv.TVLoss`
modules, sum their scalar outputs and let autograd back-propagate to the image
tensor (the only ``requires_grad=True`` object).

Style target construction follows ``getStyleImages``
(``artistic_video_core.lua:589-610``): each style image is scaled so its area
equals the content area times ``style_scale`` (via
:func:`artvid.io.image.scale_style_image`), preprocessed, run through VGG, and
its per-layer Gram matrices are blended (``buildNet`` lines 138-160, 188-200)
into a single target Gram per style layer using normalized
``style_blend_weights``.

Framework-agnostic core: only torch tensor ops. Device selection goes through
:mod:`artvid.device`; nothing here hardcodes a backend.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence

from artvid.config import Config

# ``io.image`` imports torch only lazily inside its functions, so importing
# these names keeps this module importable without torch. ``models.vgg`` and
# ``optim.runner`` are imported lazily *inside* :func:`stylize_image` because
# ``models.vgg`` imports torch at module top level (it is a torch ``nn.Module``);
# deferring it keeps ``import artvid.pipeline.stylize_image`` torch-free.
from artvid.io.image import (
    MODE_CAFFE,
    MODE_TORCHVISION,
    load_image,
    preprocess,
    save_image,
    scale_style_image,
)


def _preprocess_mode_for(config: Config) -> str:
    """Pick the preprocessing convention matching the chosen VGG weights.

    ``vgg_weights == "torchvision"`` (option A) uses ImageNet RGB normalization;
    any other value is a caffe weights path (option B) and uses the BGR/caffe
    convention. See ``docs/01-architecture.md`` section 5 and ``io/image.py``.
    """
    return MODE_TORCHVISION if config.vgg_weights == "torchvision" else MODE_CAFFE


def _normalized_blend_weights(
    raw: Optional[str], n_styles: int
) -> List[float]:
    """Parse + normalize ``style_blend_weights`` to sum to 1.

    Ports ``buildNet`` (``artistic_video_core.lua:138-160``): if unspecified
    every style image gets equal weight; otherwise the comma-separated weights
    must match the number of style images. The result is normalized to sum to 1.
    """
    if raw is None or str(raw).strip().lower() in ("", "nil", "none"):
        weights = [1.0] * n_styles
    else:
        weights = [float(p.strip()) for p in str(raw).split(",") if p.strip()]
        if len(weights) != n_styles:
            raise ValueError(
                "style_blend_weights and style_image must have the same number "
                f"of elements (got {len(weights)} weights for {n_styles} style "
                "images)."
            )
    total = sum(weights)
    if total == 0:
        raise ValueError("style_blend_weights must not sum to zero.")
    return [w / total for w in weights]


def _build_style_targets(
    style_paths: Sequence[str],
    blend_weights: Sequence[float],
    content_hw: tuple[int, int],
    config: Config,
    net,
    device,
):
    """Compute one blended target Gram matrix per style layer.

    For each style image: scale to the content area (``scale_style_image`` ->
    ``getStyleImages``), preprocess, forward through ``net`` to get the style
    layers' activations, and accumulate ``blend_weight * gram(activation)`` into
    the per-layer target. Mirrors ``buildNet`` lines 188-200 where the Gram
    targets of multiple style images are blended.

    Returns:
        Dict ``layer_name -> target_gram`` (detached, on ``device``).
    """
    import torch

    from artvid.losses.style import gram_matrix

    mode = _preprocess_mode_for(config)
    targets: Dict[str, torch.Tensor] = {}
    for path, weight in zip(style_paths, blend_weights):
        style_rgb = load_image(path)  # (3, H, W) in [0,1]
        scaled = scale_style_image(style_rgb, content_hw, config.style_scale)
        style_in = preprocess(scaled, mode=mode).unsqueeze(0).to(device)
        with torch.no_grad():
            acts = net(style_in)
        for layer in config.style_layers:
            g = gram_matrix(acts[layer]).detach()
            if layer in targets:
                targets[layer] = targets[layer] + g * weight
            else:
                targets[layer] = g * weight
    return targets


def _init_image(
    init_mode: str,
    content_pre,
    device,
    seed: int = -1,
):
    """Build the initial optimized image tensor (float32, ``requires_grad``).

    Ports the relevant branches of the legacy init block
    (``artistic_video.lua:231-254``):

    * ``"random"``: Gaussian noise scaled by ``0.001`` (legacy ``randn*0.001``).
    * ``"image"``: a clone of the preprocessed content image.

    Temporal inits (``prevWarped`` / ``prev`` / ``first``) are video-pipeline
    concerns handled in ``pipeline/singlepass.py``; for a single image only
    ``random`` and ``image`` are valid here.

    The image variable stays ``float32`` regardless of backend (see
    ``device.py`` dtype policy) for L-BFGS numerical stability.
    """
    import torch

    if seed is not None and seed >= 0:
        torch.manual_seed(seed)

    if init_mode == "random":
        img = torch.randn_like(content_pre, dtype=torch.float32).mul_(0.001)
    elif init_mode == "image":
        img = content_pre.detach().clone().to(torch.float32)
    else:
        raise ValueError(
            f"Unsupported init mode {init_mode!r} for single-image stylization; "
            "expected 'random' or 'image'. Temporal inits live in the video "
            "pipeline (pipeline/singlepass.py)."
        )

    img = img.to(device=device, dtype=torch.float32).contiguous()
    img.requires_grad_(True)
    return img


def stylize_image(
    content: str | Path,
    style: str | Path | Sequence[str | Path],
    config: Optional[Config] = None,
    *,
    device: Optional[object] = None,
    output_path: Optional[str | Path] = None,
):
    """Run single-image style transfer and return the stylized image tensor.

    This is the neural-style core that the video frames build on. It builds the
    VGG feature net, computes the (blended) style Gram targets and the content
    target, sets up the content/style/TV losses, initializes the image
    (``random`` | ``image``, from ``config.init[0]``), and runs
    :func:`artvid.optim.runner.run_optimization`.

    Args:
        content: Path to the content image.
        style: One style image path, or a sequence of paths to blend (their
            Gram targets are blended via ``config.style_blend_weights``).
        config: Run parameters. Defaults to :class:`~artvid.config.Config`'s
            defaults. Only the *first-frame* entries of the ``(first,
            subsequent)`` tuples (``init[0]``, ``num_iterations[0]``) are used,
            since this stylizes a single (first) image.
        device: Optional torch device or device string to run on. ``None``
            autodetects via :func:`artvid.device.get_device` (mps > cuda > cpu).
        output_path: If given, the stylized result is saved there (deprocessed +
            clamped). Defaults to ``config.output_image`` only when explicitly
            passed; ``None`` skips saving and just returns the tensor.

    Returns:
        A tuple ``(image_pre, result)`` where ``image_pre`` is the optimized
        *preprocessed* image tensor ``(1, 3, H, W)`` float32 (use
        :func:`artvid.io.image.deprocess` / ``save_image`` to view it) and
        ``result`` is the :class:`~artvid.optim.runner.RunResult`.
    """
    import torch

    from artvid.device import enable_mps_fallback, get_device
    from artvid.losses.content import ContentLoss
    from artvid.losses.style import StyleLoss
    from artvid.losses.tv import TVLoss
    from artvid.models.vgg import build_feature_net, split_activations
    from artvid.optim.runner import run_optimization

    config = config or Config()

    enable_mps_fallback()
    if device is None:
        device = get_device(config.device)
    elif isinstance(device, str):
        device = torch.device(device)

    mode = _preprocess_mode_for(config)

    # --- Content image + preprocessing ---------------------------------------
    content_rgb = load_image(content)  # (3, H, W) in [0,1] RGB
    _, ch, cw = content_rgb.shape
    content_pre = preprocess(content_rgb, mode=mode).unsqueeze(0).to(device)

    # --- Feature network (content + style layers, single forward via hooks) ---
    net = build_feature_net(
        content_layers=config.content_layers,
        style_layers=config.style_layers,
        pooling=config.pooling,
        weights=config.vgg_weights,
    ).to(device)
    net.eval()

    # --- Content target ------------------------------------------------------
    with torch.no_grad():
        content_acts = net(content_pre)
    content_targets = {
        layer: content_acts[layer].detach() for layer in config.content_layers
    }

    # --- Style targets (blended Gram per style layer) ------------------------
    style_paths = [str(s) for s in ([style] if isinstance(style, (str, Path)) else list(style))]
    blend_weights = _normalized_blend_weights(config.style_blend_weights, len(style_paths))
    style_targets = _build_style_targets(
        style_paths, blend_weights, (ch, cw), config, net, device
    )

    # --- Loss modules (held externally, not inserted into the net) -----------
    content_losses = {
        layer: ContentLoss(
            content_targets[layer],
            strength=config.content_weight,
            normalize=config.normalize_gradients,
        ).to(device)
        for layer in config.content_layers
    }
    style_losses = {
        layer: StyleLoss(
            style_targets[layer],
            strength=config.style_weight,
            normalize=config.normalize_gradients,
            target_is_gram=True,
        ).to(device)
        for layer in config.style_layers
    }
    tv_loss = (
        TVLoss(strength=config.tv_weight).to(device)
        if config.tv_weight > 0
        else None
    )

    # --- Initial image -------------------------------------------------------
    init_mode = config.init[0]
    image_var = _init_image(init_mode, content_pre.squeeze(0), device, seed=config.seed)

    # --- Loss closure (re-runs forward each optimizer step) ------------------
    def loss_fn() -> Dict[str, object]:
        acts = net(image_var.unsqueeze(0) if image_var.dim() == 3 else image_var)
        content_acts_i, style_acts_i = split_activations(
            acts, config.content_layers, config.style_layers
        )
        terms: Dict[str, object] = {}
        for layer in config.content_layers:
            terms[f"content[{layer}]"] = content_losses[layer](content_acts_i[layer])
        for layer in config.style_layers:
            terms[f"style[{layer}]"] = style_losses[layer](style_acts_i[layer])
        if tv_loss is not None:
            terms["tv"] = tv_loss(image_var)
        return terms

    # --- Optional save callback ----------------------------------------------
    save_fn = None
    if output_path is not None:
        out = Path(output_path)

        def save_fn(iteration: int, is_end: bool) -> None:  # noqa: ARG001
            save_image(image_var.detach(), out, mode=mode)

    # --- Optimize ------------------------------------------------------------
    result = run_optimization(
        image_var,
        loss_fn,
        max_iter=config.num_iterations[0],
        optimizer=config.optimizer,
        tol_loss_relative=config.tol_loss_relative,
        tol_loss_relative_interval=config.tol_loss_relative_interval,
        learning_rate=config.learning_rate,
        print_iter=config.print_iter,
        save_iter=config.save_iter,
        save_fn=save_fn,
    )

    return image_var.detach(), result
