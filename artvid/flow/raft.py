"""RAFT optical-flow wrapper.

Replaces the legacy DeepFlow / DeepMatching CPU binaries and the
``run-deepflow.sh`` + ``makeOptFlow.sh`` forward/backward flow generation
(see ``docs/02-migration-map.md`` section 1: ``makeOptFlow.sh`` →
``artvid/flow/raft.py``). The old pipeline shelled out to two external C++
binaries (DeepFlow + DeepMatching) and wrote ``.flo`` files; this module
computes dense optical flow directly on-device with
:func:`torchvision.models.optical_flow.raft_large`.

Output convention
-----------------
Flow is returned as a ``(2, H, W)`` float32 tensor in **(u, v)** order, i.e.
channel 0 is the horizontal displacement ``u`` (along x / width) and channel 1
is the vertical displacement ``v`` (along y / height), measured in pixels.
``flow[:, y, x]`` is the displacement that maps pixel ``(x, y)`` in ``img1`` to
its corresponding location in ``img2``. This matches the standard Middlebury
``.flo`` / torchvision RAFT convention and is what ``flow/warp.py`` consumes
(it converts to the ``grid_sample`` normalized convention internally).

Input normalization RAFT expects
---------------------------------
The torchvision ``Raft_Large_Weights.DEFAULT`` transforms map RGB images into
``float32`` tensors normalized to **[-1, 1]** (``x*2 - 1`` on [0, 1] inputs)
and resize each spatial dimension so it is divisible by 8 (RAFT requires
dimensions that are multiples of 8). We use the weights' own ``transforms()``
so preprocessing always tracks the checkpoint, then up-sample the predicted
flow back to the *original* resolution and rescale the flow vectors by the
resize factor so displacements stay in original-resolution pixels.

.. note::
   This is **not** the same normalization as the VGG / caffe preprocessing used
   by the style losses (BGR + caffe mean, see ``io/image.py``). RAFT takes plain
   RGB in [0, 1] (or [0, 255], which the transform rescales). Callers should pass
   RGB images in [0, 1]; this module does not assume caffe-preprocessed tensors.

.. note::
   ``raft_large(weights=Raft_Large_Weights.DEFAULT)`` **downloads the
   checkpoint on first use** to the torch hub cache. The user's machine needs
   network access the first time; subsequent runs load from cache offline.

Ports the role of ``makeOptFlow.sh`` (forward_<i>_<j> / backward_<j>_<i> flow
generation) onto ``artvid.device``; the consistency / reliability step lives in
``flow/consistency.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only, torch is optional at import
    import torch

from artvid import device as _device


@dataclass
class FlowPair:
    """Forward and backward optical flow for an ordered frame pair.

    Attributes:
        forward: ``(2, H, W)`` flow mapping ``img1`` pixels into ``img2``
            (legacy ``forward_<i>_<j>.flo``).
        backward: ``(2, H, W)`` flow mapping ``img2`` pixels into ``img1``
            (legacy ``backward_<j>_<i>.flo``).
    """

    forward: "torch.Tensor"
    backward: "torch.Tensor"


@lru_cache(maxsize=None)
def _load_model(device_str: str):
    """Load (and cache) a frozen RAFT-large model on the given device.

    Cached per device string so repeated calls in a pipeline reuse the same
    weights instead of re-instantiating / re-downloading. Downloads the
    checkpoint on first use (network required once).
    """
    import torch
    from torchvision.models.optical_flow import Raft_Large_Weights, raft_large

    weights = Raft_Large_Weights.DEFAULT
    model = raft_large(weights=weights, progress=False)
    model = model.eval().to(torch.device(device_str))
    for param in model.parameters():
        param.requires_grad_(False)
    return model


def _get_transforms():
    """Return the torchvision RAFT preprocessing transforms (RGB → [-1, 1])."""
    from torchvision.models.optical_flow import Raft_Large_Weights

    return Raft_Large_Weights.DEFAULT.transforms()


def _round_up_to_multiple(value: int, multiple: int = 8) -> int:
    """Smallest multiple of ``multiple`` that is ``>= value`` (RAFT needs /8)."""
    return ((value + multiple - 1) // multiple) * multiple


def _as_batch(img: "torch.Tensor") -> "torch.Tensor":
    """Coerce a ``(3, H, W)`` or ``(N, 3, H, W)`` tensor to a 4-D batch."""
    if img.dim() == 3:
        return img.unsqueeze(0)
    if img.dim() == 4:
        return img
    raise ValueError(
        f"Expected image tensor of shape (3, H, W) or (N, 3, H, W); got {tuple(img.shape)}."
    )


def _prepare(
    img1: "torch.Tensor",
    img2: "torch.Tensor",
    device: "torch.device",
):
    """Preprocess a frame pair for RAFT.

    Resizes each image so H and W are multiples of 8 (RAFT requirement),
    applies the weights' normalization to [-1, 1], and returns the batched
    tensors plus the original (H, W) and the per-axis resize scale factors used
    to rescale the predicted flow back to original pixels.
    """
    import torch
    import torch.nn.functional as F

    b1 = _as_batch(img1).to(device=device, dtype=torch.float32)
    b2 = _as_batch(img2).to(device=device, dtype=torch.float32)
    if b1.shape != b2.shape:
        raise ValueError(
            f"img1 and img2 must have the same shape; got {tuple(b1.shape)} vs {tuple(b2.shape)}."
        )

    _, _, h, w = b1.shape
    h8 = _round_up_to_multiple(h)
    w8 = _round_up_to_multiple(w)
    if (h8, w8) != (h, w):
        b1 = F.interpolate(b1, size=(h8, w8), mode="bilinear", align_corners=False)
        b2 = F.interpolate(b2, size=(h8, w8), mode="bilinear", align_corners=False)

    transforms = _get_transforms()
    b1, b2 = transforms(b1, b2)
    return b1, b2, (h, w), (h8, w8)


def _postprocess_flow(
    flow: "torch.Tensor",
    orig_hw: tuple[int, int],
    proc_hw: tuple[int, int],
) -> "torch.Tensor":
    """Resize a ``(N, 2, h8, w8)`` flow back to original res and rescale vectors.

    When the input was up/down-sampled to satisfy RAFT's /8 constraint, the
    predicted displacements are in the processed-resolution pixel scale; we
    bilinearly resize back to ``orig_hw`` and multiply each component by the
    corresponding axis scale so vectors stay in original-resolution pixels.
    """
    import torch.nn.functional as F

    h, w = orig_hw
    h8, w8 = proc_hw
    if (h8, w8) == (h, w):
        return flow

    flow = F.interpolate(flow, size=(h, w), mode="bilinear", align_corners=False)
    # channel 0 = u (x / width), channel 1 = v (y / height)
    flow[:, 0, :, :] *= float(w) / float(w8)
    flow[:, 1, :, :] *= float(h) / float(h8)
    return flow


def compute_flow(
    img1: "torch.Tensor",
    img2: "torch.Tensor",
    *,
    device: "torch.device | str | None" = None,
) -> "torch.Tensor":
    """Compute dense optical flow from ``img1`` to ``img2`` with RAFT-large.

    Args:
        img1: Source frame, RGB in ``[0, 1]``, shape ``(3, H, W)`` (a batched
            ``(N, 3, H, W)`` tensor is also accepted, in which case an
            ``(N, 2, H, W)`` flow is returned).
        img2: Target frame, same shape / range as ``img1``.
        device: Optional device override (``torch.device`` or ``'mps'|'cuda'|'cpu'``).
            Defaults to :func:`artvid.device.get_device` autodetection.

    Returns:
        ``(2, H, W)`` float32 flow in (u, v) pixel-displacement order mapping
        ``img1`` → ``img2`` (or ``(N, 2, H, W)`` for batched input). On the same
        device as the computation.

    Notes:
        RAFT is an iterative refinement network; we return its final (most
        refined) flow estimate. Runs under ``torch.no_grad()`` — optical flow is
        a fixed preprocessing step, not part of the style-transfer autograd graph.
    """
    import torch

    dev = (
        torch.device(device)
        if isinstance(device, str)
        else (device if device is not None else _device.get_device())
    )

    was_3d = img1.dim() == 3
    model = _load_model(str(dev))
    b1, b2, orig_hw, proc_hw = _prepare(img1, img2, dev)

    with torch.no_grad():
        # raft_large returns a list of successively refined flow predictions;
        # the last element is the final estimate.
        predictions = model(b1, b2)
        flow = predictions[-1]

    flow = _postprocess_flow(flow.float(), orig_hw, proc_hw)
    return flow[0] if was_3d else flow


def compute_flow_pair(
    img1: "torch.Tensor",
    img2: "torch.Tensor",
    *,
    device: "torch.device | str | None" = None,
) -> FlowPair:
    """Compute both forward (img1→img2) and backward (img2→img1) flow.

    Equivalent to the two ``eval $flowCommandLine`` calls per frame pair in
    ``makeOptFlow.sh`` (forward_<i>_<j> and backward_<j>_<i>), but on-device in
    a single call. The backward flow is consumed by ``flow/warp.py`` to warp the
    previous frame/output into the current frame's coordinates, and together
    with the forward flow by ``flow/consistency.py`` to build the reliability
    (occlusion) mask.

    Args:
        img1: First frame, RGB in ``[0, 1]``, shape ``(3, H, W)``.
        img2: Second frame, same shape / range.
        device: Optional device override; defaults to autodetection.

    Returns:
        A :class:`FlowPair` with ``forward`` and ``backward`` ``(2, H, W)`` flows.
    """
    forward = compute_flow(img1, img2, device=device)
    backward = compute_flow(img2, img1, device=device)
    return FlowPair(forward=forward, backward=backward)
