#!/usr/bin/env python3
"""Pre-download all model weights artvid needs, so the first stylization is fast.

Run this ONCE on the user's machine (an Apple Silicon M5 Max with a torch MPS
build) after ``pip install -e ".[all]"`` and before the first ``artvid stylize``
/ ``artvid run``. Both engines lazily download their weights on first use, which
otherwise stalls the first invocation for several minutes (and several GB for the
diffusion stack). Fetching them ahead of time means the real run starts warm.

What it fetches
---------------
* **Always** — the RAFT optical-flow checkpoint used by ``artvid flow`` and the
  on-the-fly flow path (``torchvision`` ``Raft_Large_Weights.DEFAULT``; see
  ``artvid/flow/raft.py``). This is what *both* the optim and diffusion engines
  need for temporal consistency.
* **With ``--diffusion``** — the Phase 2 SDXL stack named in the
  :class:`artvid.config.Config` defaults: the base text-to-image model
  (``diff_base_model``), the structure ControlNet (``controlnet_model``) and the
  IP-Adapter repo (``ip_adapter_repo`` / ``ip_adapter_subfolder`` /
  ``ip_adapter_weight``). The IP-Adapter repo also carries the CLIP image encoder
  that ``pipe.load_ip_adapter`` pulls, so we snapshot the whole repo.

The flag/field names are read live from :class:`artvid.config.Config` (the
single source of truth) so this script tracks any default changes automatically.

Cache locations
---------------
* RAFT weights land in the **torch hub cache** (``TORCH_HOME``; default
  ``~/.cache/torch``).
* The diffusion repos land in the **Hugging Face cache** (``HF_HOME`` /
  ``HUGGINGFACE_HUB_CACHE``; default ``~/.cache/huggingface``).
Set those env vars before running to relocate (e.g. to an external SSD).

Torch-free import
-----------------
This module imports cleanly without torch / torchvision / huggingface_hub
present (e.g. on the CI box): every heavy import lives inside :func:`main` (or
the helpers it calls), so ``python -c "import scripts.prefetch_models"`` and
``py_compile`` work with only the standard library available. The downloads of
course require the real dependencies installed on the M5 Max.

Usage
-----
::

    # RAFT only (enough for --engine optim):
    python scripts/prefetch_models.py

    # RAFT + the full SDXL + ControlNet + IP-Adapter stack (--engine diffusion):
    python scripts/prefetch_models.py --diffusion

    # Override the diffusion model ids to match a custom config:
    python scripts/prefetch_models.py --diffusion \
        --controlnet-model diffusers/controlnet-canny-sdxl-1.0
"""

from __future__ import annotations

import argparse
import sys


def _build_parser() -> argparse.ArgumentParser:
    """Construct the CLI parser. Defaults come from ``artvid.config.Config``.

    Reading the defaults lazily inside :func:`main` keeps this module torch-free;
    ``argparse`` itself is stdlib. We default the diffusion-model overrides to
    ``None`` and fill them from Config at fetch time so the help text does not
    require importing anything heavy.
    """
    p = argparse.ArgumentParser(
        prog="prefetch_models.py",
        description=(
            "Pre-download artvid model weights (RAFT always; SDXL + ControlNet "
            "+ IP-Adapter with --diffusion) so the first stylization is not "
            "blocked on downloads."
        ),
    )
    p.add_argument(
        "--diffusion",
        action="store_true",
        help="Also fetch the Phase 2 diffusion stack (SDXL base + ControlNet + IP-Adapter).",
    )
    p.add_argument(
        "--no-raft",
        action="store_true",
        help="Skip the RAFT checkpoint (only useful with --diffusion to fetch diffusion-only).",
    )
    # Optional overrides; None => use the matching Config default.
    p.add_argument("--diff-base-model", default=None,
                   help="Override Config.diff_base_model (SDXL base HF id).")
    p.add_argument("--controlnet-model", default=None,
                   help="Override Config.controlnet_model (structure ControlNet HF id).")
    p.add_argument("--ip-adapter-repo", default=None,
                   help="Override Config.ip_adapter_repo (IP-Adapter HF repo).")
    p.add_argument(
        "--ip-adapter-full-repo",
        action="store_true",
        help=(
            "Snapshot the ENTIRE IP-Adapter repo (every subfolder/variant). By "
            "default only the configured subfolder + image encoder are fetched."
        ),
    )
    return p


def _fetch_raft() -> str:
    """Download the RAFT-Large checkpoint into the torch hub cache.

    Mirrors exactly what ``artvid/flow/raft.py`` uses
    (``torchvision.models.optical_flow.raft_large(weights=Raft_Large_Weights.DEFAULT)``).
    Instantiating the model triggers the checkpoint download; we drop the model
    immediately since we only want the cached weights on disk.

    Returns:
        A human-readable description of where the weights were cached.
    """
    import torch  # noqa: F401  (ensures a torch build is present before torchvision)
    from torchvision.models.optical_flow import Raft_Large_Weights, raft_large

    weights = Raft_Large_Weights.DEFAULT
    print(f"[prefetch] RAFT: downloading {weights} (torchvision raft_large) ...")
    # Constructing with weights pulls the checkpoint via torch.hub into the
    # torch hub cache; we discard the model afterwards.
    _ = raft_large(weights=weights)
    del _

    try:
        hub_dir = torch.hub.get_dir()
    except Exception:  # pragma: no cover - defensive; get_dir is stable
        hub_dir = "<TORCH_HOME>/hub (default ~/.cache/torch/hub)"
    print(f"[prefetch] RAFT: done. Cached under torch hub dir: {hub_dir}")
    return hub_dir


def _fetch_diffusion(
    *,
    base_model: str,
    controlnet_model: str,
    ip_adapter_repo: str,
    ip_adapter_subfolder: str,
    ip_adapter_weight: str,
    ip_adapter_full_repo: bool,
) -> str:
    """Snapshot the SDXL base, ControlNet and IP-Adapter repos into the HF cache.

    Uses ``huggingface_hub.snapshot_download`` (a lazy import) so the repos are
    materialised in the standard HF cache that ``diffusers`` reads on first run
    (``from_pretrained`` / ``load_ip_adapter`` then load from cache, offline).

    For the IP-Adapter repo we fetch only the configured ``subfolder`` weight
    plus the CLIP image encoder folders by default (the h94 repo is large and
    contains many SD1.5 / SDXL variants); pass ``ip_adapter_full_repo`` to grab
    everything.

    Returns:
        The HF cache directory the repos were written to.
    """
    from huggingface_hub import snapshot_download

    # SDXL base (UNet + VAE + text encoders + tokenizers + scheduler config).
    print(f"[prefetch] diffusion: SDXL base {base_model!r} ...")
    snapshot_download(repo_id=base_model)

    # Structure ControlNet.
    print(f"[prefetch] diffusion: ControlNet {controlnet_model!r} ...")
    snapshot_download(repo_id=controlnet_model)

    # IP-Adapter: the configured weight in its subfolder, plus the CLIP image
    # encoder ``pipe.load_ip_adapter`` loads (lives in image_encoder subfolders).
    if ip_adapter_full_repo:
        print(f"[prefetch] diffusion: IP-Adapter {ip_adapter_repo!r} (full repo) ...")
        cache_dir = snapshot_download(repo_id=ip_adapter_repo)
    else:
        weight_path = (
            f"{ip_adapter_subfolder}/{ip_adapter_weight}"
            if ip_adapter_subfolder
            else ip_adapter_weight
        )
        allow = [
            weight_path,
            # CLIP/OpenCLIP image encoder folders the loader resolves; the h94
            # repo nests these under both the SDXL models dir and an explicit
            # image_encoder/ tree. Globs cover the known layouts.
            "**/image_encoder/**",
            f"{ip_adapter_subfolder}/image_encoder/**" if ip_adapter_subfolder else "image_encoder/**",
            "models/image_encoder/**",
            "sdxl_models/image_encoder/**",
        ]
        print(
            f"[prefetch] diffusion: IP-Adapter {ip_adapter_repo!r} "
            f"(weight {weight_path!r} + image encoder) ..."
        )
        cache_dir = snapshot_download(repo_id=ip_adapter_repo, allow_patterns=allow)

    # cache_dir is the resolved snapshot path; report the cache root above it.
    try:
        from huggingface_hub.constants import HF_HUB_CACHE

        cache_root = HF_HUB_CACHE
    except Exception:  # pragma: no cover - constant name is stable across recent hub
        cache_root = cache_dir
    print(f"[prefetch] diffusion: done. Cached under HF cache: {cache_root}")
    return str(cache_root)


def main(argv: "list[str] | None" = None) -> int:
    """Parse args, fetch the requested model weights, print a summary.

    All heavy imports (torch, torchvision, huggingface_hub) and the
    :class:`artvid.config.Config` import happen here, keeping the module
    importable without those dependencies.

    Returns:
        Process exit code (``0`` on success, ``1`` if a fetch failed).
    """
    parser = _build_parser()
    args = parser.parse_args(argv)

    # Config defaults are the single source of truth for the diffusion model ids.
    from artvid.config import Config

    cfg = Config()

    fetched: list[str] = []
    locations: set[str] = set()

    if not args.no_raft:
        try:
            loc = _fetch_raft()
            fetched.append("RAFT (torchvision Raft_Large_Weights.DEFAULT)")
            locations.add(f"torch hub cache: {loc}")
        except Exception as exc:  # pragma: no cover - network/runtime only
            print(f"[prefetch] ERROR fetching RAFT: {exc}", file=sys.stderr)
            return 1

    if args.diffusion:
        base_model = args.diff_base_model or cfg.diff_base_model
        controlnet_model = args.controlnet_model or cfg.controlnet_model
        ip_adapter_repo = args.ip_adapter_repo or cfg.ip_adapter_repo
        try:
            loc = _fetch_diffusion(
                base_model=base_model,
                controlnet_model=controlnet_model,
                ip_adapter_repo=ip_adapter_repo,
                ip_adapter_subfolder=cfg.ip_adapter_subfolder,
                ip_adapter_weight=cfg.ip_adapter_weight,
                ip_adapter_full_repo=args.ip_adapter_full_repo,
            )
            fetched.append(f"SDXL base: {base_model}")
            fetched.append(f"ControlNet: {controlnet_model}")
            fetched.append(
                f"IP-Adapter: {ip_adapter_repo} "
                f"({cfg.ip_adapter_subfolder}/{cfg.ip_adapter_weight})"
            )
            locations.add(f"Hugging Face cache: {loc}")
        except Exception as exc:  # pragma: no cover - network/runtime only
            print(f"[prefetch] ERROR fetching diffusion stack: {exc}", file=sys.stderr)
            return 1
    else:
        print(
            "[prefetch] Skipping diffusion stack (pass --diffusion to fetch "
            "SDXL + ControlNet + IP-Adapter for --engine diffusion)."
        )

    # --- Summary -----------------------------------------------------------
    print("\n[prefetch] Fetched:")
    for item in fetched:
        print(f"  - {item}")
    print("[prefetch] Cache location(s):")
    for loc in sorted(locations):
        print(f"  - {loc}")
    print(
        "[prefetch] All set. The first 'artvid stylize' / 'artvid run' will now "
        "load these from cache instead of downloading."
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
