"""Single-frame optimization driver.

Ports ``runOptimization`` (``artistic_video_core.lua:10-134``): the function
that optimizes one image by running the feature network forward, summing the
content / style / TV / temporal losses, back-propagating to the image, and
handling ``print_iter`` / ``save_iter`` side effects.

Key modernization (see ``docs/01-architecture.md`` 3.5 and
``docs/02-migration-map.md`` lines 27, 48): the legacy code inserts loss
``nn.Module``s into the network and back-propagates a zero gradient from the top
(``net:backward(x, dy)``) so the gradients flow only from the embedded loss
modules. PyTorch autograd makes this unnecessary -- **the optimized object is
the image tensor** (``requires_grad=True``, float32), and we simply call
``total_loss.backward()``. We therefore do not take a network here; instead the
caller passes a ``loss_fn`` that, given nothing (it closes over the image
variable and the network), returns the individual named loss terms as scalar
tensors. The runner sums them, exactly mirroring the legacy loss summation
(``artistic_video_core.lua:95-103``).

Framework-agnostic: only ``torch`` tensor ops, no MPS-specific calls. Device is
chosen upstream (``artvid.device``); this driver runs on whatever device the
image variable and network already live on.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from artvid.optim.lbfgs import StopConfig, run_adam, run_lbfgs

# A loss function: evaluates all loss terms against the current (mutated in
# place) image variable and returns them as a dict of name -> scalar tensor.
# It must build a fresh autograd graph each call (i.e. run the forward pass),
# because L-BFGS / Adam call it once per optimizer step. Mirrors the legacy
# ``feval`` body that re-runs ``net:forward(x)`` every call.
LossFn = Callable[[], Dict[str, "object"]]


@dataclass
class RunResult:
    """Outcome of :func:`run_optimization`.

    Attributes:
        loss_history: Total loss after each completed iteration.
        last_losses: The final per-term loss values (name -> float).
        num_iterations: Number of optimizer iterations actually run (may be less
            than requested if the relative-loss criterion stopped early).
        elapsed_seconds: Wall-clock optimization time.
    """

    loss_history: List[float] = field(default_factory=list)
    last_losses: Dict[str, float] = field(default_factory=dict)
    num_iterations: int = 0
    elapsed_seconds: float = 0.0


def _detach_floats(losses: Dict[str, object]) -> Dict[str, float]:
    """Convert a name->scalar-tensor dict to name->float for printing/return."""
    out: Dict[str, float] = {}
    for name, value in losses.items():
        item = getattr(value, "item", None)
        out[name] = float(item()) if callable(item) else float(value)  # type: ignore[arg-type]
    return out


def run_optimization(
    image_var,
    loss_fn: LossFn,
    *,
    max_iter: int,
    optimizer: str = "lbfgs",
    tol_loss_relative: float = 0.0,
    tol_loss_relative_interval: int = 50,
    learning_rate: float = 1e1,
    print_iter: int = 0,
    save_iter: int = 0,
    save_fn: Optional[Callable[[int, bool], None]] = None,
    history_size: int = 100,
) -> RunResult:
    """Optimize a single image variable to minimize the summed losses.

    Equivalent to the legacy ``runOptimization`` (``artistic_video_core.lua:10-134``).
    The closure evaluates ``loss_fn`` (which runs the feature network forward and
    returns each loss term), sums the terms into a scalar, back-propagates it to
    ``image_var`` and returns the total. Printing and saving are driven by the
    iteration counter, as in the legacy ``maybe_print`` / ``maybe_save``.

    Args:
        image_var: The image tensor being optimized. Must be float32 and have
            ``requires_grad=True`` (see ``device.py`` dtype policy). It is
            mutated in place by the optimizer.
        loss_fn: Callable returning a dict of named scalar loss tensors for the
            *current* ``image_var``. Must rebuild the autograd graph on each
            call (run the forward pass), since the optimizer invokes it once per
            step (and possibly more, under line search).
        max_iter: Maximum optimizer iterations (legacy ``max_iter``;
            single-pass: ``num_iterations`` first vs subsequent frame).
        optimizer: ``"lbfgs"`` or ``"adam"`` (legacy ``-optimizer``).
        tol_loss_relative: Relative-loss early-stop threshold (L-BFGS only;
            ``0`` disables). Legacy ``-tol_loss_relative``.
        tol_loss_relative_interval: Iterations between relative-loss checks.
            Legacy ``-tol_loss_relative_interval``.
        learning_rate: Adam learning rate (legacy ``-learning_rate``).
        print_iter: Print all loss terms every ``print_iter`` iterations
            (``<= 0`` disables periodic printing). Legacy ``-print_iter``.
        save_iter: Hint passed through to ``save_fn`` so it can save every
            ``save_iter`` iterations. Legacy ``-save_iter``.
        save_fn: Optional callback ``(iteration, is_end)`` performing the actual
            save. Keeping it injectable avoids coupling the runner to
            filename-building / image-saving owned elsewhere. The runner calls
            it on intermediate iterations (when ``save_iter > 0`` and
            ``iteration % save_iter == 0``) and once at the end with
            ``is_end=True``. Mirrors the legacy ``maybe_save``.
        history_size: L-BFGS correction history (legacy ``nCorrection``).

    Returns:
        A :class:`RunResult` with the loss history and final per-term losses.
    """
    import torch

    if optimizer not in ("lbfgs", "adam"):
        raise ValueError(f'Unrecognized optimizer "{optimizer}"')

    # Holds the most recent per-term loss values so print/save callbacks (which
    # only receive the iteration index) can report them, matching the legacy
    # access to ``loss_module.loss`` after the forward pass.
    last_terms: Dict[str, float] = {}

    def closure():
        """The optimization closure (legacy ``feval``)."""
        # Zero grads on every optimizer-managed param. ``set_to_none`` keeps the
        # graph clean and is the modern default.
        if image_var.grad is not None:
            image_var.grad.detach_()
            image_var.grad.zero_()
        terms = loss_fn()
        total = None
        for value in terms.values():
            total = value if total is None else total + value
        if total is None:
            total = torch.zeros((), dtype=image_var.dtype, device=image_var.device)
        total.backward()
        # Snapshot per-term values for printing/saving without holding the graph.
        last_terms.clear()
        last_terms.update(_detach_floats(terms))
        last_terms["total"] = float(total.detach().item())
        return total

    def maybe_print(iteration: int) -> None:
        """Print all loss terms (legacy ``maybe_print``)."""
        if print_iter > 0 and iteration % print_iter == 0:
            _print_losses(iteration, max_iter, last_terms)

    def on_step(iteration: int, _loss: float) -> None:
        maybe_print(iteration)
        if save_fn is not None and save_iter > 0 and iteration % save_iter == 0:
            save_fn(iteration, False)

    start = time.time()
    if optimizer == "lbfgs":
        print("Running optimization with L-BFGS")
        stop = StopConfig(
            tol_loss_relative=tol_loss_relative,
            tol_loss_relative_interval=tol_loss_relative_interval,
        )
        history = run_lbfgs(
            closure,
            [image_var],
            max_iter=max_iter,
            stop=stop,
            history_size=history_size,
            on_step=on_step,
        )
    else:  # adam
        print("Running optimization with ADAM")
        history = run_adam(
            closure,
            [image_var],
            max_iter=max_iter,
            lr=learning_rate,
            on_step=on_step,
        )
    elapsed = time.time() - start
    print(f"Running time: {elapsed:.0f}s")

    n = len(history)
    # Always print the final losses (legacy ``print_end`` -> ``maybe_print`` with
    # ``alwaysPrint=true``).
    _print_losses(n, max_iter, last_terms)
    if save_fn is not None:
        save_fn(n, True)

    return RunResult(
        loss_history=history,
        last_losses=dict(last_terms),
        num_iterations=n,
        elapsed_seconds=elapsed,
    )


def _print_losses(iteration: int, max_iter: int, terms: Dict[str, float]) -> None:
    """Print the iteration header and each loss term (legacy ``maybe_print``)."""
    print(f"Iteration {iteration} / {max_iter}")
    for name, value in terms.items():
        if name == "total":
            continue
        print(f"  {name} loss: {value:f}")
    total = terms.get("total")
    if total is not None:
        print(f"  Total loss: {total:f}")
