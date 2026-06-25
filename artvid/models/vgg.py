"""VGG-19 feature extractor for style/content/temporal losses.

Ports the CNN-handling part of ``buildNet`` from
``artistic_video_core.lua:137-250`` (specifically the layer iteration,
the ``-pooling avg`` max->avg replacement at lines 197-209, and the
named layer selection at lines 211-246).

Modernization vs. the legacy code
---------------------------------
The Torch7 implementation inserted loss ``nn.Module`` objects directly into
an ``nn.Sequential`` and read activations as a side effect of the forward
pass. With PyTorch autograd we keep the network pure: this module only
*extracts* named activations (in a single forward pass via forward hooks),
and the loss modules (``losses/style.py``, ``losses/content.py``,
``losses/temporal.py``) consume those activations externally. See
``docs/02-migration-map.md`` section 2 / 5 and ``docs/01-architecture.md``
section 3.4.

Layer naming
------------
The legacy caffe model exposes layers named ``relu1_1 .. relu5_4`` etc.
torchvision's ``vgg19().features`` is an ``nn.Sequential`` indexed by integer.
We map the relu names we care about to their torchvision feature indices
(see :data:`RELU_NAME_TO_INDEX`).

Parity (VGG weights) — option A vs. B
-------------------------------------
- **Option A (default):** torchvision VGG-19 ImageNet weights. RGB input,
  ImageNet mean/std normalization (handled in ``io/image.py``).
- **Option B (stub):** original caffe VGG-19 (``VGG_ILSVRC_19_layers``)
  weights for bit-closer reproduction of the 2016 paper. BGR input with
  caffe mean subtraction. See :func:`load_caffe_vgg19_features` for the
  documented loading hook.

torch is imported at module level (importable without execution); no device
is hardcoded — the caller is responsible for ``.to(device)``.
"""

from __future__ import annotations

from typing import Iterable, Mapping

import torch
from torch import nn

# ---------------------------------------------------------------------------
# Layer name <-> torchvision feature index mapping.
#
# torchvision ``vgg19().features`` (an ``nn.Sequential``) has this layout. Each
# conv is immediately followed by a ReLU; the relu*_* name refers to the ReLU
# output *after* the n-th conv of block m. MaxPool sits between blocks.
#
#   idx  module          name
#   ---  --------------  --------
#    0   Conv2d          conv1_1
#    1   ReLU            relu1_1   <-- style
#    2   Conv2d          conv1_2
#    3   ReLU            relu1_2
#    4   MaxPool2d       pool1
#    5   Conv2d          conv2_1
#    6   ReLU            relu2_1   <-- style
#    7   Conv2d          conv2_2
#    8   ReLU            relu2_2
#    9   MaxPool2d       pool2
#   10   Conv2d          conv3_1
#   11   ReLU            relu3_1   <-- style
#   12   Conv2d          conv3_2
#   13   ReLU            relu3_2
#   14   Conv2d          conv3_3
#   15   ReLU            relu3_3
#   16   Conv2d          conv3_4
#   17   ReLU            relu3_4
#   18   MaxPool2d       pool3
#   19   Conv2d          conv4_1
#   20   ReLU            relu4_1   <-- style
#   21   Conv2d          conv4_2
#   22   ReLU            relu4_2   <-- content (default)
#   23   Conv2d          conv4_3
#   24   ReLU            relu4_3
#   25   Conv2d          conv4_4
#   26   ReLU            relu4_4
#   27   MaxPool2d       pool4
#   28   Conv2d          conv5_1
#   29   ReLU            relu5_1   <-- style
#   ... (relu5_2..5_4, pool5 follow; unused by default config)
# ---------------------------------------------------------------------------
RELU_NAME_TO_INDEX: dict[str, int] = {
    "relu1_1": 1,
    "relu1_2": 3,
    "relu2_1": 6,
    "relu2_2": 8,
    "relu3_1": 11,
    "relu3_2": 13,
    "relu3_3": 15,
    "relu3_4": 17,
    "relu4_1": 20,
    "relu4_2": 22,
    "relu4_3": 24,
    "relu4_4": 26,
    "relu5_1": 29,
    "relu5_2": 31,
    "relu5_3": 33,
    "relu5_4": 35,
}

# Default layer sets, matching the legacy ``cmd:option`` defaults
# (``artistic_video.lua``): content ``relu4_2``; style
# ``relu1_1,relu2_1,relu3_1,relu4_1,relu5_1``.
DEFAULT_CONTENT_LAYERS: tuple[str, ...] = ("relu4_2",)
DEFAULT_STYLE_LAYERS: tuple[str, ...] = (
    "relu1_1",
    "relu2_1",
    "relu3_1",
    "relu4_1",
    "relu5_1",
)


class VGGFeatures(nn.Module):
    """VGG-19 feature extractor returning named activations in one forward pass.

    The wrapped ``features`` sub-network is run once; forward hooks capture the
    activations of every requested ``relu*_*`` layer. Truncation: layers beyond
    the deepest requested relu are dropped so we never compute unused work.

    Args:
        layers: Iterable of relu layer names to expose (e.g.
            ``("relu1_1", "relu4_2")``). Order is irrelevant; duplicates are
            collapsed. Must be keys of :data:`RELU_NAME_TO_INDEX`.
        pooling: ``"max"`` (default, matching torchvision / legacy default) or
            ``"avg"`` to replace every ``MaxPool2d`` with an ``AvgPool2d`` of
            the same geometry (legacy ``-pooling avg``,
            ``artistic_video_core.lua:197-209``).
        weights: ``"torchvision"`` (option A, ImageNet pretrained) or a
            filesystem path string to caffe VGG-19 weights (option B, see
            :func:`load_caffe_vgg19_features`). ``None`` builds an untrained
            network (useful for tests without a weight download).
        requires_grad: If ``False`` (default) the VGG parameters are frozen
            (``requires_grad_(False)``); only the input image carries gradient.

    The module is created on CPU; call ``.to(device)`` afterwards. No device is
    selected here (see ``artvid/device.py``).
    """

    def __init__(
        self,
        layers: Iterable[str] = (*DEFAULT_STYLE_LAYERS, *DEFAULT_CONTENT_LAYERS),
        pooling: str = "max",
        weights: str | None = "torchvision",
        requires_grad: bool = False,
    ) -> None:
        super().__init__()

        if pooling not in ("max", "avg"):
            raise ValueError(f"pooling must be 'max' or 'avg', got {pooling!r}")

        # Deduplicate + validate requested layer names.
        requested = list(dict.fromkeys(layers))
        unknown = [name for name in requested if name not in RELU_NAME_TO_INDEX]
        if unknown:
            raise ValueError(
                f"Unknown VGG layer name(s) {unknown}; "
                f"valid names: {sorted(RELU_NAME_TO_INDEX)}"
            )
        if not requested:
            raise ValueError("At least one layer must be requested.")

        # Map names -> indices and remember the deepest one for truncation.
        self._name_to_index: dict[str, int] = {
            name: RELU_NAME_TO_INDEX[name] for name in requested
        }
        self._index_to_name: dict[int, str] = {
            idx: name for name, idx in self._name_to_index.items()
        }
        self.layer_names: tuple[str, ...] = tuple(requested)
        last_index = max(self._name_to_index.values())

        features = _build_features(weights)
        if pooling == "avg":
            features = _replace_maxpool_with_avgpool(features)

        # Truncate: keep modules [0 .. last_index] only — everything deeper is
        # never read, so we skip the compute (legacy buildNet stopped iterating
        # once all requested layers were found, line 193).
        self.features = nn.Sequential(*list(features.children())[: last_index + 1])

        if not requires_grad:
            for param in self.features.parameters():
                param.requires_grad_(False)

        # Buffer for hook outputs of the current forward pass.
        self._captured: dict[str, torch.Tensor] = {}
        self._register_hooks()

    # -- hook plumbing ------------------------------------------------------

    def _register_hooks(self) -> None:
        """Attach a forward hook to each requested relu module."""

        def make_hook(name: str):
            def hook(_module: nn.Module, _inp, output: torch.Tensor) -> None:
                self._captured[name] = output

            return hook

        for idx, name in self._index_to_name.items():
            self.features[idx].register_forward_hook(make_hook(name))

    # -- forward ------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Run one forward pass and return the requested named activations.

        Args:
            x: Preprocessed image batch, shape ``(N, 3, H, W)``, float32, on the
                same device as this module. Preprocessing (BGR/caffe or
                ImageNet) is the caller's responsibility (``io/image.py``).

        Returns:
            Dict mapping each requested relu name to its activation tensor.
            Activations keep the input's autograd graph, so callers can
            ``backward()`` through them to the input image.
        """
        self._captured = {}
        # Single forward pass; hooks populate ``self._captured``. We do not need
        # the final tensor, only the hooked intermediates.
        self.features(x)
        # Return in the user-requested order, copying out of the buffer so a
        # subsequent forward cannot mutate a dict the caller still holds.
        out = {name: self._captured[name] for name in self.layer_names}
        self._captured = {}
        return out


# ---------------------------------------------------------------------------
# Construction helpers
# ---------------------------------------------------------------------------


def _build_features(weights: str | None) -> nn.Sequential:
    """Return the ``vgg19().features`` sub-network for the chosen weights.

    - ``"torchvision"``: ImageNet pretrained (option A).
    - ``None``: untrained (random init) — avoids any download; for tests.
    - any other str: treated as a path to caffe VGG-19 weights (option B);
      delegated to :func:`load_caffe_vgg19_features`.
    """
    from torchvision.models import VGG19_Weights, vgg19

    if weights == "torchvision":
        model = vgg19(weights=VGG19_Weights.IMAGENET1K_V1)
        return model.features
    if weights is None:
        model = vgg19(weights=None)
        return model.features
    # Anything else is interpreted as a caffe weights path (parity option B).
    return load_caffe_vgg19_features(weights)


def _replace_maxpool_with_avgpool(features: nn.Sequential) -> nn.Sequential:
    """Return a copy of ``features`` with every ``MaxPool2d`` -> ``AvgPool2d``.

    Geometry (kernel/stride/padding) is preserved, mirroring the legacy
    replacement in ``artistic_video_core.lua:197-209`` which asserts zero
    padding and reuses ``kW,kH,dW,dH``. Empirically average pooling can yield
    smoother stylizations (Gatys et al.).
    """
    new_children: list[nn.Module] = []
    for module in features.children():
        if isinstance(module, nn.MaxPool2d):
            new_children.append(
                nn.AvgPool2d(
                    kernel_size=module.kernel_size,
                    stride=module.stride,
                    padding=module.padding,
                    ceil_mode=module.ceil_mode,
                )
            )
        else:
            new_children.append(module)
    return nn.Sequential(*new_children)


def load_caffe_vgg19_features(weights_path: str) -> nn.Sequential:
    """Load original caffe VGG-19 ``features`` weights (parity option B).

    DOCUMENTED STUB — not yet implemented. The original implementation used
    ``loadcaffe`` to load ``models/VGG_ILSVRC_19_layers.caffemodel`` +
    ``VGG_ILSVRC_19_layers_deploy.prototxt``. To honor bit-closer parity with
    the 2016 paper we want the same convolution weights here.

    Intended implementation outline (to be filled in on the target machine):

    1. Build a torchvision ``vgg19(weights=None).features`` skeleton so the
       module ordering matches :data:`RELU_NAME_TO_INDEX`.
    2. Obtain caffe weights as a ``state_dict`` of conv weight/bias tensors.
       Options: (a) a pre-converted ``.pth`` (e.g. the community
       ``vgg_conv.pth`` from Gatys' PyTorch port, which is already in
       torchvision layer order); or (b) parse the ``.caffemodel`` protobuf and
       map ``conv{m}_{n}`` -> the corresponding ``features.{idx}`` Conv2d.
       NOTE caffe conv weights are stored in the same (out, in, kH, kW) layout
       as PyTorch for VGG (cross-correlation), so no flip is required, but this
       must be verified against the chosen converted file.
    3. ``model.load_state_dict(state_dict)`` (conv layers only; VGG has no
       BatchNorm) and return ``model.features``.

    Crucially, caffe VGG-19 expects **BGR** input with caffe mean subtraction
    ``[103.939, 116.779, 123.68]`` and 0-255 scale (see ``io/image.py`` and
    ``docs/01-architecture.md`` section 5) — the preprocessing must match the
    weights, otherwise outputs are garbage.

    Args:
        weights_path: Path to a caffe-derived VGG-19 weight file (``.pth`` or
            ``.caffemodel``).

    Raises:
        NotImplementedError: always, until the converter is wired up.
    """
    raise NotImplementedError(
        "caffe VGG-19 weight loading (parity option B) is not implemented yet. "
        f"Requested weights path: {weights_path!r}. "
        "See load_caffe_vgg19_features docstring for the intended approach; "
        "default to weights='torchvision' (option A) for now."
    )


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------


def build_feature_net(
    content_layers: Iterable[str] = DEFAULT_CONTENT_LAYERS,
    style_layers: Iterable[str] = DEFAULT_STYLE_LAYERS,
    *,
    temporal: bool = False,
    pooling: str = "max",
    weights: str | None = "torchvision",
) -> VGGFeatures:
    """Build a :class:`VGGFeatures` covering content + style layers.

    This is the ``models/vgg.py:build_feature_net`` entry referenced in
    ``docs/02-migration-map.md`` (the CNN half of legacy ``buildNet``). Loss
    construction (Gram targets, content/temporal targets) is intentionally NOT
    done here — it lives in ``losses/*`` and ``optim/runner.py``, unlike the
    legacy monolithic ``buildNet`` which built losses inline.

    Args:
        content_layers: relu names used for the content loss (default
            ``relu4_2``).
        style_layers: relu names used for the style loss (default
            ``relu1_1..relu5_1``).
        temporal: Present for signature parity with the legacy temporal-layer
            handling. In this modernized design the temporal loss is a
            pixel-space loss (legacy ``initWeighted``) and needs no VGG layer,
            so this flag does not add any feature layer. Kept so callers can be
            explicit and so the parameter can grow if a feature-space temporal
            loss is ever added.
        pooling: ``"max"`` | ``"avg"``.
        weights: see :class:`VGGFeatures`.

    Returns:
        A :class:`VGGFeatures` exposing the union of content and style layers.
    """
    layers = (*style_layers, *content_layers)
    return VGGFeatures(
        layers=layers,
        pooling=pooling,
        weights=weights,
    )


def split_activations(
    activations: Mapping[str, torch.Tensor],
    content_layers: Iterable[str],
    style_layers: Iterable[str],
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Split a forward-pass activation dict into content and style sub-dicts.

    Convenience for callers that requested a combined feature net and now want
    to route activations to the content vs. style losses.

    Args:
        activations: Output of :meth:`VGGFeatures.forward`.
        content_layers: relu names for content.
        style_layers: relu names for style.

    Returns:
        ``(content_acts, style_acts)``.
    """
    content_acts = {name: activations[name] for name in content_layers}
    style_acts = {name: activations[name] for name in style_layers}
    return content_acts, style_acts
