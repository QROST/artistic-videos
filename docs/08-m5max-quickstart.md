# artvid — Apple M5 Max Quickstart

A concise, copy-paste path to a clean first run on an **Apple M5 Max**
(Metal / MPS, 128 GB unified memory). It covers system prep, install, weight
prefetch, and the three commands (`flow` / `stylize` / `run`) for both engines.

For the **full flag reference**, see [`docs/usage.md`](usage.md). For tuning the
diffusion engine, see [`docs/07-phase2-design.md`](07-phase2-design.md) and the
`TODO(tuning)` markers in `artvid/config.py` / `artvid/cli.py`.

---

## 1. System prep

Requires **Python 3.11+** and **ffmpeg** (used by `artvid run` for frame
extraction / re-encode via `imageio-ffmpeg`).

```bash
# Python 3.11 (skip if you already have it; pyenv also works)
brew install python@3.11

# ffmpeg
brew install ffmpeg

# A fresh virtual environment keeps the torch/diffusers stack isolated
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

> macOS arm64 wheels for `torch` / `torchvision` already ship the **MPS**
> (Metal) backend — no special index URL is needed.

---

## 2. Install

From the repo root, install artvid plus everything (dev + the Phase 2 diffusion
stack):

```bash
# everything: pytest + diffusers/transformers/accelerate/safetensors/controlnet_aux/peft
pip install -e ".[all]"
```

If you only want the **optim** engine (L-BFGS / Adam, no diffusion), the base
deps are enough:

```bash
pip install -e .          # optim engine only
# or: pip install -e ".[dev]"   # + pytest
```

> The diffusion extras (`.[all]` / `.[diffusion]`) pull several large packages.
> The pinned lower bounds are conservative floors — if `diffusers` /
> `transformers` / `peft` complain about version interlock, pin a known-good set
> on this machine (see the note in `pyproject.toml`).

---

## 3. Enable the MPS CPU fallback

A few ops are not yet implemented on MPS. artvid calls
`artvid.device.enable_mps_fallback()` at startup (it sets this for you), but
exporting it in your shell guarantees it is in effect before the very first op:

```bash
export PYTORCH_ENABLE_MPS_FALLBACK=1
```

This transparently runs any unimplemented op on CPU instead of erroring. Leave
it set for both engines.

---

## 4. Prefetch model weights (recommended)

Both engines download their weights on first use. Fetch them ahead of time so
the first real stylization starts warm instead of stalling on multi-GB
downloads:

```bash
# RAFT optical-flow checkpoint only (enough for --engine optim):
python scripts/prefetch_models.py

# RAFT + the full SDXL + ControlNet + IP-Adapter stack (for --engine diffusion):
python scripts/prefetch_models.py --diffusion
```

The script prints exactly what it fetched and the cache locations:

- **RAFT** → torch hub cache (`TORCH_HOME`, default `~/.cache/torch`).
- **SDXL base + ControlNet + IP-Adapter** → Hugging Face cache (`HF_HOME` /
  `HUGGINGFACE_HUB_CACHE`, default `~/.cache/huggingface`).

Set those env vars before running to relocate the caches (e.g. to an external
SSD). The diffusion model ids come straight from the `artvid/config.py`
defaults; override them with `--diff-base-model` / `--controlnet-model` /
`--ip-adapter-repo` if your config differs.

---

## 5. The three commands

`artvid` has a top-level `--engine {optim,diffusion}` flag (default `optim`) and
three subcommands. `flow` is engine-agnostic; `stylize` and `run` honour
`--engine`. Pick a device with `--device {mps,cuda,cpu}` (default autodetects
`mps > cuda > cpu`).

### `--engine optim` (Phase 1, default — L-BFGS / Adam)

```bash
# 1. Optical flow + reliability for a frame sequence (replaces makeOptFlow.sh)
artvid flow "frames/frame_%04d.ppm" --out flow/ --steps 1

# 2. Per-frame style transfer over the sequence (single-pass)
artvid stylize "frames/frame_%04d.ppm" style.jpg \
    --flow-pattern "flow/backward_[%d]_{%d}.flo" \
    --flow-weight-pattern "flow/reliable_[%d]_{%d}.pgm" \
    --output-folder out/ --output-image stylized.png

# 3. End-to-end: video -> stylized video (replaces stylizeVideo.sh)
artvid run input.mp4 style.jpg -o output.mp4
```

`run` extracts frames, computes RAFT flow, stylizes and re-encodes in one shot.
Add `--multipass` (or `--passes N`) to `stylize` / `run` for the forward/backward
multi-pass pipeline.

### `--engine diffusion` (Phase 2 — SDXL + ControlNet + IP-Adapter)

The diffusion engine is **single-pass** (temporal coherence comes from latent
optical-flow consistency, not forward/backward passes), so `--multipass` /
`--passes` are rejected. `flow` is shared and unchanged; the diffusion stylize
reuses the same `.flo` / reliability files. The positional `style` image is used
as the IP-Adapter style reference.

```bash
# 1. Flow is identical (engine-agnostic)
artvid flow "frames/frame_%04d.ppm" --out flow/ --steps 1

# 2. Diffusion per-frame stylize
artvid --engine diffusion stylize "frames/frame_%04d.ppm" style.jpg \
    --flow-pattern "flow/backward_[%d]_{%d}.flo" \
    --flow-weight-pattern "flow/reliable_[%d]_{%d}.pgm" \
    --output-folder out/ --output-image stylized.png \
    --diff-prompt "oil painting, vivid brushstrokes"

# 3. End-to-end diffusion run
artvid --engine diffusion run input.mp4 style.jpg -o output.mp4 \
    --diff-prompt "oil painting, vivid brushstrokes"
```

The first diffusion invocation prints a one-line banner about the weight
download / model licenses (SDXL: CreativeML Open RAIL++-M). If you ran
`prefetch_models.py --diffusion`, the weights load from cache instead.

---

## 6. Where to tune

The diffusion knobs (ControlNet / IP-Adapter strength, guidance, denoise, the
latent-flow temporal consistency window and reliability gamma) are surfaced as
CLI flags but their **defaults are `TODO(tuning)`** on this hardware:

- Defaults live in `artvid/config.py` (the diffusion field group).
- CLI flags that override them are in `artvid/cli.py` (`_add_diffusion_flags`).
- Design rationale + recommended starting points: `docs/07-phase2-design.md`.

Key flags to sweep first on the M5 Max:
`--controlnet-scale`, `--ip-adapter-scale`, `--guidance-scale`, `--diff-steps`,
`--denoise-strength`, and the temporal trio `--temporal-strength` /
`--temporal-fuse-start` / `--temporal-fuse-end` (plus
`--latent-reliability-gamma`).

For the complete, authoritative flag list across all subcommands and both
engines, see [`docs/usage.md`](usage.md).

---

## MPS caveats

- **Keep `PYTORCH_ENABLE_MPS_FALLBACK=1` set.** Some ops still fall back to CPU;
  without the fallback they error. Fallbacks are transparent but slower for the
  affected op.
- **No float64 on MPS.** artvid stays in float32 (optim) / a low-precision
  diffusion dtype on purpose; do not force `.double()` anywhere in the path.
- **Diffusion dtype / memory.** With 128 GB unified memory the full SDXL stack
  fits resident, so CPU offload (slow on MPS) is intentionally **not** enabled.
  At 1024²+ resolutions, if peak memory is a concern, the design doc notes
  `enable_attention_slicing()` / `enable_vae_tiling()` as `TODO(tuning)` levers.
- **First run needs network.** RAFT and the diffusion repos download on first
  use; after `prefetch_models.py` (or one warm run) everything loads from cache
  offline.
