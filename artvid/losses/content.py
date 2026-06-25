"""Content loss.

Ports ``nn.ContentLoss`` from ``artistic_video_core.lua:257-288``.

Legacy behaviour
----------------
The Lua module computed an MSE between the input feature map and a cached
target feature map, scaled by ``strength`` (``:268-276``). Its backward pass
(``:278-288``) optionally L1-normalized the gradient (divide by
``||grad||_1 + 1e-8``) before scaling by ``strength`` — the ``normalize``
option of the original Neural-Style codebase.

Modernization
-------------
Instead of inserting this module into the network and hand-writing
``updateGradInput``, this is a plain :class:`torch.nn.Module` whose
:meth:`forward` returns a scalar loss; autograd produces the backward pass.

The legacy ``normalize`` gradient trick is reproduced *exactly* via a custom
autograd function only when ``normalize=True``; the common ``normalize=False``
path is a straightforward differentiable MSE.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class _L1NormalizeGrad(torch.autograd.Function):
    """Identity in forward; L1-normalizes the incoming gradient in backward.

    Reproduces the legacy ``gradInput:div(torch.norm(gradInput, 1) + 1e-8)``
    step (``artistic_video_core.lua:283``). Placing this just before the loss
    means the gradient flowing *into* the feature map is divided by its own L1
    norm, matching the legacy semantics where normalization happened on the
    criterion's gradient.
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:  # noqa: D102
        return x

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):  # noqa: D102
        norm = grad_output.abs().sum() + 1e-8
        return grad_output / norm


class ContentLoss(nn.Module):
    """MSE between input features and a fixed target feature map.

    Args:
        target: Target feature map (e.g. the content image's ``relu4_2``
            activations). Detached and cached; not optimized.
        strength: Scalar weight multiplied into the loss.
        normalize: If ``True``, reproduce the legacy L1 gradient normalization
            (``artistic_video_core.lua:282-283``). Defaults to ``False``.
    """

    def __init__(
        self,
        target: torch.Tensor,
        strength: float = 1.0,
        normalize: bool = False,
    ) -> None:
        super().__init__()
        self.strength = float(strength)
        self.normalize = bool(normalize)
        # Cache the target as a non-trainable buffer so it moves with .to(device)
        # and is excluded from gradients (matches the legacy fixed target).
        self.register_buffer("target", target.detach().clone())

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Return ``strength * MSE(input, target)`` as a scalar tensor.

        If the element counts mismatch (legacy guard at ``:269``), returns a
        zero scalar on the input's device/dtype rather than crashing.
        """
        if input.numel() != self.target.numel():
            return input.new_zeros(())
        x = input
        if self.normalize:
            x = _L1NormalizeGrad.apply(x)
        return F.mse_loss(x, self.target) * self.strength
