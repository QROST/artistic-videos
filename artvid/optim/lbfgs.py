"""Optimizers for the pixel-optimization engine.

Ports the optimizer driving logic of the legacy ``lbfgs.lua`` (the custom
``optim.lbfgs`` with a *relative loss change* stopping criterion) and the
ADAM branch of ``runOptimization`` (``artistic_video_core.lua:121-126``).

The legacy ``lbfgs.lua`` is a fork of Torch's ``optim.lbfgs`` (minFunc-style)
that adds one feature on top of the standard algorithm: it stops early when the
relative change of the loss over a fixed interval of iterations falls below a
threshold (``tolFunRelative`` / ``tolFunRelativeInterval``; see
``lbfgs.lua:43-44,259-265``)::

    if nIter % tolFunRelativeInterval == 0 then
      if f_past ~= nil and (abs(f - f_past) / f_past) < tolFunRelative then
        break  -- relative change in function value below threshold
      end
      f_past = f
    end

Rather than re-implement the full L-BFGS recursion in pure Python, we delegate
the numerical core to :class:`torch.optim.LBFGS` (with strong-Wolfe line search,
the modern equivalent of the legacy line search) and re-implement *only* the
relative-loss stopping criterion as an outer driving loop. An Adam path is also
exposed for parity with the legacy ``-optimizer adam`` option.

The object being optimized is the image tensor (``requires_grad=True``,
float32); see :mod:`artvid.optim.runner` for how the closure is built.

Framework-agnostic: only ``torch`` tensor / optimizer ops, no MPS-specific
calls.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List

# A closure compatible with ``torch.optim.Optimizer.step``: it must zero grads,
# run forward, ``backward()`` and return the scalar loss (as a 0-dim tensor or
# python float). This mirrors the legacy ``feval`` in ``runOptimization``.
Closure = Callable[[], "object"]


@dataclass
class StopConfig:
    """Relative-loss-change stopping criterion (ports ``lbfgs.lua:43-44``).

    Attributes:
        tol_loss_relative: Threshold on the relative loss change. ``0`` (the
            legacy default) disables the criterion entirely. Stop when
            ``abs(f - f_past) / f_past < tol_loss_relative``.
        tol_loss_relative_interval: Number of iterations between comparisons
            (legacy ``tolFunRelativeInterval``, default 100; artvid default 50).
    """

    tol_loss_relative: float = 0.0
    tol_loss_relative_interval: int = 50


def _as_float(loss: object) -> float:
    """Coerce a closure return (0-dim tensor or number) to a python float."""
    item = getattr(loss, "item", None)
    if callable(item):
        return float(item())
    return float(loss)  # type: ignore[arg-type]


def run_lbfgs(
    closure: Closure,
    params,
    max_iter: int,
    stop: StopConfig | None = None,
    *,
    lr: float = 1.0,
    history_size: int = 100,
    on_step: Callable[[int, float], None] | None = None,
) -> List[float]:
    """Optimize ``params`` with L-BFGS + the relative-loss stopping criterion.

    Wraps :class:`torch.optim.LBFGS` with ``line_search_fn='strong_wolfe'`` and
    drives it one iteration at a time so the legacy early-stop on relative loss
    change (``lbfgs.lua:259-265``) can be checked.

    To keep the same *total* number of inner function evaluations as a single
    ``LBFGS(max_iter=max_iter).step()`` call would have, the underlying
    optimizer is configured with ``max_iter=1`` and we loop ``max_iter`` times.
    L-BFGS history (``history_size`` correction pairs, legacy ``nCorrection``)
    persists across these single-iteration steps because the same optimizer
    instance is reused.

    Args:
        closure: Evaluates loss + grads and returns the scalar loss. Called
            (possibly multiple times per step, due to line search) by the
            optimizer; see :class:`torch.optim.LBFGS`.
        params: Iterable of tensors to optimize (typically ``[image_var]``).
        max_iter: Maximum number of outer iterations (legacy ``max_iter``).
        stop: Relative-loss stopping criterion; ``None`` disables it.
        lr: Step size / learning rate (legacy ``learningRate``, default 1).
        history_size: Number of L-BFGS correction pairs to keep (legacy
            ``nCorrection``, default 100).
        on_step: Optional callback ``(iteration, loss)`` invoked once per outer
            iteration *after* the step, with the post-step loss. Used by the
            runner for print/save side effects keyed on the iteration count.

    Returns:
        The history of loss values, one per completed outer iteration (the
        legacy ``f_hist``, minus the pre-optimization value).
    """
    import torch

    stop = stop or StopConfig()
    params = list(params)

    optimizer = torch.optim.LBFGS(
        params,
        lr=lr,
        max_iter=1,
        max_eval=None,
        history_size=history_size,
        tolerance_grad=1e-9,
        tolerance_change=1e-12,
        line_search_fn="strong_wolfe",
    )

    losses: List[float] = []
    f_past: float | None = None
    interval = max(1, int(stop.tol_loss_relative_interval))

    for t in range(1, max_iter + 1):
        loss = optimizer.step(closure)
        f = _as_float(loss)
        losses.append(f)
        if on_step is not None:
            on_step(t, f)

        # Relative-loss-change stopping criterion (lbfgs.lua:259-265).
        if stop.tol_loss_relative > 0 and t % interval == 0:
            if f_past is not None and f_past != 0.0:
                if abs(f - f_past) / abs(f_past) < stop.tol_loss_relative:
                    break
            f_past = f

    return losses


def run_adam(
    closure: Closure,
    params,
    max_iter: int,
    *,
    lr: float = 1e1,
    on_step: Callable[[int, float], None] | None = None,
) -> List[float]:
    """Optimize ``params`` with Adam.

    Ports the legacy ADAM branch (``artistic_video_core.lua:121-126``), which
    loops ``max_iter`` times calling ``optim.adam`` once per iteration. The
    relative-loss criterion is L-BFGS-only in the legacy code, so it is not
    applied here.

    Args:
        closure: Evaluates loss + grads and returns the scalar loss.
        params: Iterable of tensors to optimize.
        max_iter: Number of Adam steps (legacy ``max_iter``).
        lr: Learning rate (legacy ``-learning_rate``, default 10).
        on_step: Optional ``(iteration, loss)`` callback invoked after each step.

    Returns:
        The history of loss values, one per iteration.
    """
    import torch

    params = list(params)
    optimizer = torch.optim.Adam(params, lr=lr)

    losses: List[float] = []
    for t in range(1, max_iter + 1):
        loss = optimizer.step(closure)
        f = _as_float(loss)
        losses.append(f)
        if on_step is not None:
            on_step(t, f)
    return losses
