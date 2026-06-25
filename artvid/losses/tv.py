"""Total variation (TV) loss.

Ports ``nn.TVLoss`` from ``artistic_video_core.lua:400-430``.

Legacy behaviour
----------------
The Lua module was a pass-through in forward and only defined a backward
(``:415-429``, "inspired by kaishengtai/neuralart"). Reading that backward, the
gradient it produces is exactly the derivative of the *squared* difference
energy::

    E = strength * 0.5 * sum_{i,j} ( (x[i,j] - x[i,j+1])^2
                                     + (x[i,j] - x[i+1,j])^2 )

restricted to the top-left ``(H-1) x (W-1)`` region (the legacy slices
``{1,-2}`` over both spatial dims). Concretely, for ``x_diff = x[:, :-1, :-1] -
x[:, :-1, 1:]`` and ``y_diff = x[:, :-1, :-1] - x[:, 1:, :-1]``, the legacy
``gradInput`` accumulates ``+x_diff+y_diff`` at the anchor pixel, ``-x_diff``
at the right neighbour and ``-y_diff`` at the bottom neighbour — which is
precisely ``dE/dx`` for the energy above (the ``0.5`` cancels the factor 2).

Modernization
-------------
We express the energy directly and let autograd differentiate it, which
reproduces the legacy gradient bit-for-bit (same finite-difference stencil and
same ``(H-1)x(W-1)`` support). No hand-written backward.

.. note::
   Despite the docstring naming in some neural-style forks, this is the
   *squared* (L2) difference form, matching the legacy stencil exactly. A true
   absolute-value (L1, "anisotropic") variant is available via ``form="l1"``
   for callers who prefer it, but ``form="l2"`` (default) is the parity match.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def _diffs(input: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Horizontal and vertical forward differences over the legacy support.

    Matches the legacy slicing (``artistic_video_core.lua:420-423``): both
    differences are taken over the top-left ``(H-1) x (W-1)`` block so the
    energy support is identical to the original implementation. Works for
    ``(C,H,W)`` and ``(N,C,H,W)`` tensors.
    """
    # x_diff: anchor minus right neighbour;  y_diff: anchor minus bottom neighbour.
    x_diff = input[..., :-1, :-1] - input[..., :-1, 1:]
    y_diff = input[..., :-1, :-1] - input[..., 1:, :-1]
    return x_diff, y_diff


class TVLoss(nn.Module):
    """Total-variation regularizer on the optimized image.

    Args:
        strength: Scalar weight on the TV energy (legacy ``tv_weight``).
        form: ``"l2"`` (default, parity with the legacy squared-difference
            stencil) or ``"l1"`` for true anisotropic absolute-difference TV.
    """

    def __init__(self, strength: float = 1.0, form: str = "l2") -> None:
        super().__init__()
        if form not in ("l2", "l1"):
            raise ValueError(f"form must be 'l2' or 'l1', got {form!r}.")
        self.strength = float(strength)
        self.form = form

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Return the (scaled) TV energy as a scalar tensor.

        For ``form='l2'`` the returned energy is ``strength * 0.5 * sum(x_diff^2
        + y_diff^2)`` whose gradient equals the legacy ``updateGradInput``
        (``artistic_video_core.lua:415-429``). For ``form='l1'`` it is
        ``strength * sum(|x_diff| + |y_diff|)``.
        """
        x_diff, y_diff = _diffs(input)
        if self.form == "l2":
            energy = 0.5 * (x_diff.pow(2).sum() + y_diff.pow(2).sum())
        else:  # "l1"
            energy = x_diff.abs().sum() + y_diff.abs().sum()
        return energy * self.strength
