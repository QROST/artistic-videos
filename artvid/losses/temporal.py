"""Temporal (weighted content) loss.

Ports ``nn.WeightedContentLoss`` from ``artistic_video_core.lua:291-347``.

Purpose
-------
Encourages the current stylized frame to match the *warped previous output*
(``target``) in regions the optical flow deems reliable (``weights``). This is
the temporal-consistency term of the video style-transfer pipeline.

Legacy weighting trick (parity-critical)
----------------------------------------
The original criterion is a per-element MSE/SmoothL1. We want a *reliability-
weighted* error ``w * err^2``, but a criterion can only be fed a single tensor,
so the legacy code (``:296-301, 321-322``) exploits::

    (sqrt(w) * err)^2 = w * err^2

by (a) taking ``sqrt`` of the weights once at construction, (b) pre-multiplying
the cached target by ``sqrt(w)``, and (c) multiplying the *input* by ``sqrt(w)``
inside ``updateOutput`` before the criterion. We preserve this exact semantics:
``forward`` computes ``crit(input * sqrt(w), target * sqrt(w)) * strength``.

For the ``smoothl1`` criterion the algebra is not a clean square, but we match
the legacy behaviour faithfully — it applies ``sqrt(w)`` to both operands the
same way regardless of criterion (``:308-315, 333-339``).

Modernization
-------------
A plain :class:`torch.nn.Module`; backward handled by autograd instead of the
legacy hand-written ``updateGradInput``. The L1 gradient ``normalize`` option
(``:341-342``) is reproduced when requested.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class _L1NormalizeGrad(torch.autograd.Function):
    """Identity forward; L1-normalizes the gradient in backward (legacy :342)."""

    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:  # noqa: D102
        return x

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):  # noqa: D102
        norm = grad_output.abs().sum() + 1e-8
        return grad_output / norm


class WeightedContentLoss(nn.Module):
    """Per-pixel reliability-weighted content loss against a warped target.

    Args:
        target: The target tensor, typically the warped previous stylized frame
            (in feature or pixel space). Cached, not optimized.
        weights: Optional per-pixel reliability weights (occlusion mask),
            broadcastable to ``target``. If ``None``, this reduces to a plain
            content loss. Internally ``sqrt``'d to realize ``w * err^2``
            semantics (see module docstring). Values are expected to be
            non-negative.
        strength: Scalar weight multiplied into the loss.
        criterion: ``"mse"`` (default) or ``"smoothl1"``.
        normalize: If ``True``, reproduce the legacy L1 gradient normalization
            (``artistic_video_core.lua:341-342``). Defaults to ``False``.
    """

    def __init__(
        self,
        target: torch.Tensor,
        weights: torch.Tensor | None = None,
        strength: float = 1.0,
        criterion: str = "mse",
        normalize: bool = False,
    ) -> None:
        super().__init__()
        self.strength = float(strength)
        self.normalize = bool(normalize)

        crit = criterion.lower()
        if crit not in ("mse", "smoothl1"):
            # Legacy warns and falls back to MSE (:312-314).
            crit = "mse"
        self.criterion = crit

        target = target.detach().clone()
        if weights is not None:
            # sqrt the weights once, pre-multiply the target (legacy :300-301).
            sqrt_w = weights.detach().clone().clamp_min(0).sqrt()
            self.register_buffer("sqrt_weights", sqrt_w)
            self.register_buffer("target", target * sqrt_w)
        else:
            self.register_buffer("sqrt_weights", None)
            self.register_buffer("target", target)

    def _apply_criterion(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        if self.criterion == "smoothl1":
            return F.smooth_l1_loss(x, y)
        return F.mse_loss(x, y)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Return ``strength * crit(input', target')`` as a scalar tensor.

        ``input'`` is ``input * sqrt(weights)`` when weights were given, else
        ``input``; ``target'`` is the (pre-weighted) cached target. Mismatched
        element counts (legacy guard at ``:319``) return a zero scalar.
        """
        if input.numel() != self.target.numel():
            return input.new_zeros(())

        x = input
        if self.sqrt_weights is not None:
            x = x * self.sqrt_weights
        if self.normalize:
            x = _L1NormalizeGrad.apply(x)
        return self._apply_criterion(x, self.target) * self.strength
