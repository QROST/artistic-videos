"""Image I/O and VGG preprocessing.

Ports ``preprocess`` / ``deprocess`` / ``save_image`` from
``artistic_video_core.lua:475-498`` and the style/content image loading from
``getContentImage`` / ``getStyleImages`` (``artistic_video_core.lua:580-610``).

Two preprocessing conventions are supported, selected by the ``mode`` argument,
per the parity decision in ``docs/01-architecture.md`` section 5:

* ``"caffe"`` (default) -- the original 2016 convention used by the caffe
  VGG-19 weights: convert RGB->BGR, scale ``[0,1] -> [0,256]`` and subtract the
  per-channel mean pixel ``[103.939, 116.779, 123.68]``. This is a faithful
  port of the Lua ``preprocess`` and is required when loading caffe weights
  (``Config.vgg_weights`` set to a path).
* ``"torchvision"`` -- the modern long-term default: keep RGB, normalize with
  the ImageNet ``mean=[0.485,0.456,0.406]`` / ``std=[0.229,0.224,0.225]`` used
  by torchvision's pretrained VGG-19 (``Config.vgg_weights == "torchvision"``).

Images are represented throughout as CHW ``float32`` tensors. The optimized
image variable stays ``float32`` regardless of backend (see ``device.py``).

This module is framework-agnostic: it uses only ``torch`` tensor ops plus
Pillow for file decode/encode, with no MPS-specific calls.
"""

from __future__ import annotations

from pathlib import Path

# Caffe BGR convention constants. Mirrors ``artistic_video_core.lua:476`` and
# ``config.CAFFE_BGR_MEAN``. The legacy code scales by 256.0 (not 255.0); we
# preserve that exact factor for bit-faithful caffe parity.
CAFFE_BGR_MEAN = (103.939, 116.779, 123.68)
CAFFE_SCALE = 256.0

# torchvision / ImageNet normalization (RGB, values already in [0,1]).
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Preprocessing modes.
MODE_CAFFE = "caffe"
MODE_TORCHVISION = "torchvision"
_VALID_MODES = (MODE_CAFFE, MODE_TORCHVISION)


def load_image(path: str | Path):
    """Load an image file as a CHW ``float32`` tensor in ``[0, 1]`` RGB.

    Equivalent to the legacy ``image.load(path, 3)`` (3 forces RGB). No model
    preprocessing is applied here -- call :func:`preprocess` afterwards.

    Args:
        path: Path to an image file (any format Pillow can decode).

    Returns:
        A ``(3, H, W)`` float32 tensor with values in ``[0, 1]``.
    """
    import numpy as np
    import torch
    from PIL import Image

    with Image.open(path) as im:
        im = im.convert("RGB")
        arr = np.asarray(im, dtype=np.uint8)  # (H, W, 3)
    # HWC uint8 [0,255] -> CHW float32 [0,1].
    tensor = torch.from_numpy(arr.copy()).to(torch.float32).div_(255.0)
    return tensor.permute(2, 0, 1).contiguous()


def _mode_tensors(mode: str, ref):
    """Return ``(mean, std_or_none, bgr)`` broadcast tensors for ``mode``.

    ``ref`` provides the target dtype/device. For caffe mode ``std`` is ``None``
    (the scaling is handled separately) and ``bgr`` is ``True``.
    """
    import torch

    if mode not in _VALID_MODES:
        raise ValueError(f"Unknown preprocess mode {mode!r}; expected one of {_VALID_MODES}.")
    if mode == MODE_CAFFE:
        mean = torch.tensor(CAFFE_BGR_MEAN, dtype=ref.dtype, device=ref.device).view(3, 1, 1)
        return mean, None, True
    mean = torch.tensor(IMAGENET_MEAN, dtype=ref.dtype, device=ref.device).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD, dtype=ref.dtype, device=ref.device).view(3, 1, 1)
    return mean, std, False


def preprocess(img, mode: str = MODE_CAFFE):
    """Preprocess a ``[0,1]`` RGB CHW image for a VGG model.

    Caffe mode ports ``artistic_video_core.lua:475-482``: RGB->BGR, ``*256``,
    subtract the BGR mean pixel. Torchvision mode keeps RGB and applies ImageNet
    ``(x - mean) / std``.

    Args:
        img: ``(3, H, W)`` float tensor in ``[0, 1]`` RGB (as from
            :func:`load_image`).
        mode: ``"caffe"`` or ``"torchvision"``.

    Returns:
        A ``(3, H, W)`` float32 tensor ready for the feature network. Not
        in-place: the input is left unmodified.
    """
    import torch

    out = img.to(torch.float32)
    mean, std, bgr = _mode_tensors(mode, out)
    if bgr:
        # RGB -> BGR (reverse channel order), like ``img:index(1, {3,2,1})``.
        out = out.flip(0).mul(CAFFE_SCALE).sub(mean)
    else:
        out = (out - mean) / std
    return out.contiguous()


def deprocess(img, mode: str = MODE_CAFFE):
    """Invert :func:`preprocess`, returning a ``[0,1]``-ish RGB CHW tensor.

    Ports ``artistic_video_core.lua:484-492`` for caffe mode (add mean,
    ``/256``, BGR->RGB). Torchvision mode applies ``x * std + mean``.

    The result is *not* clamped here; :func:`save_image` does the clamp.

    Args:
        img: A preprocessed ``(3, H, W)`` tensor.
        mode: Must match the mode used by :func:`preprocess`.

    Returns:
        A ``(3, H, W)`` float32 RGB tensor (values may slightly exceed
        ``[0, 1]``).
    """
    import torch

    out = img.to(torch.float32)
    mean, std, bgr = _mode_tensors(mode, out)
    if bgr:
        # Add mean, divide scale, then BGR -> RGB.
        out = out.add(mean).div(CAFFE_SCALE).flip(0)
    else:
        out = out * std + mean
    return out.contiguous()


def save_image(img, path: str | Path, mode: str = MODE_CAFFE) -> None:
    """Deprocess, clamp to ``[0, 1]`` and write an image to ``path``.

    Ports ``save_image`` (``artistic_video_core.lua:494-498``). The legacy code
    uses ``image.minmax{min=0, max=1}`` which simply clamps to the range; we do
    a plain ``clamp(0, 1)`` for the same effect.

    Args:
        img: A preprocessed ``(3, H, W)`` tensor (the optimized variable).
        path: Output file path; the extension selects the encoder.
        mode: The preprocessing mode the image was created with.
    """
    import torch
    from PIL import Image

    disp = deprocess(img.detach(), mode=mode).clamp(0.0, 1.0)
    # CHW [0,1] float -> HWC uint8.
    arr = disp.mul(255.0).round().to("cpu").permute(1, 2, 0).to(dtype=torch.uint8)
    np_arr = arr.numpy()
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np_arr, mode="RGB").save(out_path)


def scale_style_image(style_img, content_hw: tuple[int, int], style_scale: float = 1.0):
    """Scale a style image so its area equals the content area times ``style_scale``.

    Ports the scaling in ``getStyleImages`` (``artistic_video_core.lua:596-599``):
    ``img_scale = sqrt(content_area / style_area) * style_scale`` applied to both
    dimensions, with bilinear interpolation.

    Args:
        style_img: ``(3, H, W)`` float tensor in ``[0, 1]`` RGB.
        content_hw: ``(H, W)`` of the content image.
        style_scale: Multiplier on the matched area (legacy ``-style_scale``).

    Returns:
        The resized ``(3, H', W')`` style tensor.
    """
    import math

    import torch.nn.functional as F

    _, sh, sw = style_img.shape
    ch, cw = content_hw
    img_scale = math.sqrt((ch * cw) / (sh * sw)) * style_scale
    new_h = max(1, int(round(sh * img_scale)))
    new_w = max(1, int(round(sw * img_scale)))
    resized = F.interpolate(
        style_img.unsqueeze(0),
        size=(new_h, new_w),
        mode="bilinear",
        align_corners=False,
    )
    return resized.squeeze(0)
