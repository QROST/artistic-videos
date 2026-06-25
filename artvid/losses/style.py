"""Style loss and Gram matrix.

Ports ``GramMatrix`` (``artistic_video_core.lua:351-360``) and ``nn.StyleLoss``
(``artistic_video_core.lua:364-397``).

Legacy behaviour
----------------
``GramMatrix`` reshaped a ``C x H x W`` feature map to ``C x (H*W)`` and
computed ``F · Fᵀ`` (a ``C x C`` matrix). ``StyleLoss`` (``:378-385``) then
divided that Gram by ``input:nElement()`` (the *total* element count
``C*H*W``) and took the MSE against a cached target Gram, scaled by
``strength``. The backward (``:387-397``) optionally L1-normalized the
gradient.

Modernization
-------------
A plain :class:`torch.nn.Module` whose :meth:`forward` returns a scalar; the
backward is left to autograd. The division by ``nElement`` is preserved so the
loss magnitude matches the legacy implementation for a given ``strength``.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def gram_matrix(feat: torch.Tensor) -> torch.Tensor:
    """Compute the (normalized) Gram matrix of a feature map.

    Reproduces ``GramMatrix`` followed by the ``:div(input:nElement())`` step
    in ``StyleLoss:updateOutput`` (``artistic_video_core.lua:351-360,380``):
    the raw ``F·Fᵀ`` is divided by the number of elements in the feature map.

    Supports an unbatched ``(C, H, W)`` tensor (legacy layout) or a batched
    ``(N, C, H, W)`` tensor. The normalization uses ``C*H*W`` (the legacy
    ``nElement`` of a single ``C x H x W`` map) so batched results match the
    per-sample legacy values.

    Args:
        feat: Feature activations, ``(C, H, W)`` or ``(N, C, H, W)``.

    Returns:
        ``(C, C)`` for unbatched input, or ``(N, C, C)`` for batched input.
    """
    if feat.dim() == 3:
        c, h, w = feat.shape
        flat = feat.reshape(c, h * w)
        gram = flat @ flat.t()
        return gram / feat.numel()
    if feat.dim() == 4:
        n, c, h, w = feat.shape
        flat = feat.reshape(n, c, h * w)
        gram = flat @ flat.transpose(1, 2)
        # Divide by per-sample element count (C*H*W), matching legacy nElement.
        return gram / (c * h * w)
    raise ValueError(
        f"gram_matrix expects a 3D (C,H,W) or 4D (N,C,H,W) tensor, got "
        f"{feat.dim()}D shape {tuple(feat.shape)}."
    )


class StyleLoss(nn.Module):
    """MSE between the input's Gram matrix and a fixed target Gram.

    Args:
        target: Either a raw target *feature map* (``(C,H,W)`` / ``(N,C,H,W)``)
            or a precomputed target *Gram matrix*. Set ``target_is_gram`` to
            indicate which. The Gram is cached and not optimized.
        strength: Scalar weight multiplied into the loss.
        normalize: If ``True``, reproduce the legacy L1 gradient normalization
            (``artistic_video_core.lua:391-392``). Defaults to ``False``.
        target_is_gram: If ``True``, ``target`` is already a Gram matrix and is
            cached as-is. If ``False`` (default), ``target`` is a feature map
            and :func:`gram_matrix` is applied first.
    """

    def __init__(
        self,
        target: torch.Tensor,
        strength: float = 1.0,
        normalize: bool = False,
        target_is_gram: bool = False,
    ) -> None:
        super().__init__()
        self.strength = float(strength)
        self.normalize = bool(normalize)
        target_gram = target if target_is_gram else gram_matrix(target)
        self.register_buffer("target", target_gram.detach().clone())

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Return ``strength * MSE(gram(input), target_gram)`` as a scalar."""
        g = gram_matrix(input)
        if self.normalize:
            g = _L1NormalizeGrad.apply(g)
        return F.mse_loss(g, self.target) * self.strength


class _L1NormalizeGrad(torch.autograd.Function):
    """Identity forward; L1-normalizes the gradient in backward.

    Reproduces the legacy ``gradInput:div(torch.norm(gradInput,1)+1e-8)`` step
    (``artistic_video_core.lua:392``).
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:  # noqa: D102
        return x

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):  # noqa: D102
        norm = grad_output.abs().sum() + 1e-8
        return grad_output / norm
