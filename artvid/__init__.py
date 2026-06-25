"""artvid — artistic video style transfer (PyTorch/MPS).

Modernized port of the Torch7/Lua reference implementation (Ruder, Dosovitskiy
& Brox, 2016) to Python 3.11 + PyTorch with an Apple-Silicon (MPS) target.

Submodules are intentionally *not* imported here so that importing
``artvid`` (and inexpensive modules such as :mod:`artvid.config`) does not pull
in torch. Import the submodule you need explicitly, e.g.
``from artvid.losses.style import StyleLoss``.
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
