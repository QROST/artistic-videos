"""Phase 2 diffusion video-stylization package.

This package grafts the 2016 optical-flow temporal-consistency idea (ported in
Phase 1) onto a modern diffusion stylizer, reusing the Phase 1 flow stack
(:mod:`artvid.flow.raft`, :mod:`artvid.flow.warp`, :mod:`artvid.flow.consistency`)
but moving the warp + reliability masking from pixel space into VAE *latent*
space (see ``docs/07-phase2-design.md``).

Modules:
    latent_warp:  latent-space optical-flow warp + reliability masking
                  (``warp_latent`` / ``latent_reliability``); the §2.3/§2.4
                  reuse layer. Framework-agnostic torch ops, no diffusers.
    engine:       ``DiffusionEngine`` (SDXL + ControlNet + IP-Adapter
                  single-frame stylizer; alias ``DiffusionStylizer``).
    video:        ``stylize_video_diffusion`` per-frame loop.               [other agent]
    preprocess:   structure-signal extractors for ControlNet.              [other agent]

Heavy framework imports (``torch`` / ``diffusers``) are kept lazy inside the
modules that need them so importing this package and querying ``--help`` stays
torch-free, matching the Phase 1 convention in :mod:`artvid.cli`.
"""

from __future__ import annotations

__all__ = [
    "warp_latent",
    "latent_reliability",
    "LatentWarpResult",
    "DiffusionEngine",
    "DiffusionStylizer",
    "StyleReference",
]


def __getattr__(name: str):
    # Lazy re-export so importing the package does not eagerly import torch via
    # latent_warp's function-level imports. (latent_warp itself is torch-free at
    # import time, but we keep the pattern uniform with the engine/video modules
    # that genuinely require lazy framework imports.)
    if name in ("warp_latent", "latent_reliability", "LatentWarpResult"):
        from artvid.diffusion import latent_warp

        return getattr(latent_warp, name)
    if name in ("DiffusionEngine", "DiffusionStylizer", "StyleReference"):
        from artvid.diffusion import engine

        return getattr(engine, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
