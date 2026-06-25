# artvid — Usage (Phase 1 CLI)

End-user guide for the Phase 1 PyTorch port of *artistic-videos* (Ruder et al.,
2016). It covers installation on Apple Silicon, the three CLI subcommands
(`flow`, `stylize`, `run`), and the mapping from the old `stylizeVideo.sh` /
`artistic_video.lua` parameters to the new flags.

> Scope: Phase 1 ships the **optimization** engine (L-BFGS / Adam), which is a
> faithful port of the Lua pipeline. The `--engine diffusion` engine is a
> **Phase 2** deliverable and is not implemented yet (see
> [Diffusion is Phase 2](#diffusion-is-phase-2)).

---

## Installation (Apple Silicon / M-series)

artvid targets the Metal (MPS) backend on Apple Silicon. You need the arm64
build of `torch` / `torchvision` that ships the MPS backend; the default
macOS arm64 wheels from PyPI already include it.

```bash
# from the repo root, ideally inside a fresh venv (Python >= 3.11)
python3 -m venv .venv
source .venv/bin/activate

# install artvid plus its dev extras (pytest) in editable mode
pip install -e ".[dev]"
```

`pip install -e ".[dev]"` pulls in `torch>=2.2`, `torchvision>=0.17`, `numpy`,
`pillow`, `imageio` (+ `imageio-ffmpeg`), `tyro`, and `pytest`. The
`imageio-ffmpeg` dependency bundles an ffmpeg binary used by the `run`
subcommand for frame extraction / re-encoding.

After installation the `artvid` console script is on your `PATH`:

```bash
artvid --help
artvid flow --help
artvid stylize --help
artvid run --help
```

### Device selection

Device handling replaces the legacy `-gpu` / `-backend` / `-cudnn_autotune`
options. Every subcommand accepts `--device {mps,cuda,cpu}`; when omitted the
device is autodetected in the order **mps > cuda > cpu**. On an M-series Mac you
normally do not pass `--device` at all — it picks `mps` automatically. The
`PYTORCH_ENABLE_MPS_FALLBACK` workaround for unsupported ops is enabled
automatically by the CLI.

---

## The three subcommands

```
artvid [--engine {optim,diffusion}] <flow|stylize|run> ...
```

`--engine` is a top-level flag (default `optim`). `--engine diffusion` is
rejected with a clear "not implemented in Phase 1" error by `stylize` and `run`.

### `flow` — optical flow + reliability (replaces `makeOptFlow.sh`)

Computes forward + backward RAFT optical flow and forward/backward consistency
**reliability masks** for adjacent (and optional long-term) frame pairs. This
replaces the old `makeOptFlow.sh` + `run-deepflow.sh` + C++ `consistencyChecker`
toolchain (RAFT replaces DeepFlow/DeepMatching).

```bash
artvid flow 'frames/frame_%04d.ppm' --out frames/flow
```

Long-term flow (multiple step sizes), matching the legacy practice of running
`makeOptFlow.sh` once per step:

```bash
artvid flow 'frames/frame_%04d.ppm' --out frames/flow --steps 1 2 4
```

Key options:

| Flag | Default | Meaning |
|---|---|---|
| `frames` (positional) | — | `printf`-style frame pattern, one integer conversion (e.g. `frame_%04d.ppm`). |
| `-o, --out` | *(required)* | Output folder for `.flo` / reliability files. |
| `--start-number` | `1` | Index of the first frame. |
| `--num-images` | `0` | Frame count; `0` = autodetect by file existence. |
| `--steps` | `[1]` | Long-term step sizes (frame index deltas); `1` = adjacent frames. |
| `--smooth-sigma` | `0.8` | Gaussian sigma for reliability-mask smoothing (`0` disables). |
| `--mask-ext` | `.pgm` | Reliability mask extension (`.pgm` or `.png`). |
| `--device` | autodetect | `mps` / `cuda` / `cpu`. |
| `--overwrite` | off | Recompute even if outputs exist (default: skip existing). |
| `-v, --verbose` | off | Print each frame pair as it is written. |

Output filenames follow the legacy `[from]`/`{to}` placeholder convention so the
files feed the `stylize` patterns unchanged:
`forward_<i>_<j>.flo`, `backward_<j>_<i>.flo`, `reliable_<j>_<i>.pgm`,
`reliable_<i>_<j>.pgm`.

### `stylize` — per-frame style transfer (`artistic_video[_multiPass].lua`)

Stylizes an existing frame sequence. **Single-pass by default**; add
`--multipass` (or `--passes N`) for the forward/backward multi-pass pipeline.
You provide the content frame pattern and a style image, plus already-computed
flow (or let it use RAFT on the fly via `--flow-source raft`).

Single-pass, using precomputed flow from `artvid flow`:

```bash
artvid stylize 'frames/frame_%04d.ppm' example/seated-nude.jpg \
  --flow-pattern 'frames/flow/backward_[%d]_{%d}.flo' \
  --flow-weight-pattern 'frames/flow/reliable_[%d]_{%d}.pgm' \
  --style-weight 1e2 --temporal-weight 1e3 \
  --output-folder out/ --output-image out.png --number-format '%04d'
```

Multi-pass (15 passes by default):

```bash
artvid stylize 'frames/frame_%04d.ppm' example/seated-nude.jpg --multipass --passes 15 \
  --forward-flow-pattern  'frames/flow/forward_[%d]_{%d}.flo' \
  --backward-flow-pattern 'frames/flow/backward_[%d]_{%d}.flo' \
  --forward-flow-weight-pattern  'frames/flow/reliable_[%d]_{%d}.pgm' \
  --backward-flow-weight-pattern 'frames/flow/reliable_[%d]_{%d}.pgm' \
  --output-folder out/
```

No precomputed flow — compute RAFT flow on the fly:

```bash
artvid stylize 'frames/frame_%04d.ppm' example/seated-nude.jpg --flow-source raft
```

`--flow-source` is one of `auto` (default), `precomputed`, or `raft`.

`stylize` also accepts a legacy `-args` parameter file via `--args FILE`
(repeatable, applied at lowest priority so explicit flags win). The file format
is one `-option value` per line, the same format the Lua tool consumed; legacy
camelCase option names (e.g. `-flowWeight_pattern`, `-blendWeight`) are accepted
and mapped to the new field names automatically.

The full set of optimization / model flags is listed in the
[parameter mapping](#parameter-mapping-old--new) below and via
`artvid stylize --help`.

### `run` — end-to-end video → stylized video (replaces `stylizeVideo.sh`)

One command: extract frames, compute flow + reliability, stylize, re-encode.
This is the Python port of `stylizeVideo.sh`; the old interactive prompts for
backend / GPU / resolution / style weight are now plain flags.

```bash
artvid run input.mp4 example/seated-nude.jpg
```

With explicit knobs:

```bash
artvid run input.mp4 example/seated-nude.jpg \
  --resolution 960:540 --style-weight 1e2 --temporal-weight 1e3 \
  --framerate 25 --output input-stylized.mp4
```

`run`-specific options:

| Flag | Default | Meaning |
|---|---|---|
| `video` (positional) | — | Input video file. |
| `style` (positional) | — | Style image (comma-separate several). |
| `--work-dir` | from video basename | Folder for `frames/`, `flow/`, `out/`. |
| `-o, --output` | `<work_dir>-stylized.<ext>` | Output video path. |
| `--frame-pattern` | `frame_%04d.ppm` | ffmpeg extraction filename pattern. |
| `--resolution` | original | `w:h` rescale during extraction (replaces the interactive prompt). |
| `--framerate` | ffmpeg default (25) | Output video framerate. |
| `--no-flow` | off | Skip flow precompute; stylize with on-the-fly RAFT flow. |
| `--steps` | `[1]` | Long-term step sizes for the flow precompute. |
| `--smooth-sigma` | `0.8` | Reliability-mask smoothing sigma. |
| `--mask-ext` | `.pgm` | Reliability mask extension. |
| `--overwrite-flow` | off | Recompute flow even if it already exists. |

`run` also accepts every `stylize` optimization/model flag (it shares the same
flag set), plus `--multipass` / `--passes` to choose the multi-pass pipeline.
When no flow flags are given, `run` wires the precomputed flow patterns to its
`flow/` folder automatically.

---

## Single-pass vs multi-pass

| | Single-pass | Multi-pass |
|---|---|---|
| Legacy source | `artistic_video.lua` | `artistic_video_multiPass.lua` |
| Selected by | *(default)* | `--multipass` or `--passes N` |
| Direction | one forward sweep over frames | alternating forward/backward sweeps |
| Flow patterns | `--flow-pattern` + `--flow-weight-pattern` (backward only) | `--forward-flow-pattern` / `--backward-flow-pattern` (+ their `-weight-` variants) |
| Default `temporal_weight` | `1e3` | `5e2` (applied automatically when you don't set `--temporal-weight`) |
| Extra knobs | — | `--passes`, `--blend-weight`, `--blend-weight-last-pass`, `--use-temporal-loss-after`, `--continue-with-pass` |

Guidance:

- **Single-pass** is the faster, lower-memory default and reproduces the
  baseline `stylizeVideo.sh` behaviour. Use it first.
- **Multi-pass** generally yields smoother long-range temporal consistency at
  the cost of `num_passes`× the work. It needs **both** forward and backward
  flow (run `artvid flow`, which writes both directions; or let `run` compute
  them). Start from `--passes 15`, `--use-temporal-loss-after 8` (the
  defaults), and tune `--blend-weight` if you see ghosting.
- Note the **temporal weight default differs by mode** (1e3 single, 5e2 multi).
  If you set `--temporal-weight` explicitly, your value is used for both.

---

## Parameter mapping (old → new)

The new flags keep the legacy names, defaults, and semantics (see
[`docs/02-migration-map.md`](02-migration-map.md) §3). Old long-form CLI flags
used a single dash and underscores (`-style_weight`); the new ones use a double
dash and hyphens (`--style-weight`). Below, the **most common** mappings.

### `stylizeVideo.sh` (interactive) → `artvid run`

| `stylizeVideo.sh` prompt / arg | `artvid run` flag |
|---|---|
| `<path_to_video>` (arg 1) | `video` positional |
| `<path_to_style_image>` (arg 2) | `style` positional |
| "Which backend?" (`nn` / `cudnn` / `clnn`) + GPU id prompt | `--device {mps,cuda,cpu}` (autodetect by default) |
| "resolution w:h" prompt | `--resolution w:h` |
| "style reconstruction weight" prompt | `--style-weight` |
| `temporal_weight=1e3` (hardcoded) | `--temporal-weight` (default 1e3) |
| `makeOptFlow.sh ...` step | built in (`--no-flow` to skip; `--steps` for long-term) |
| `-number_format %04d` | `--number-format` (and `--frame-pattern`) |
| final `ffmpeg -i out-%04d.png ...` re-encode | built in (`--framerate`, `-o/--output`) |

### `artistic_video.lua` `-option` → `artvid` `--flag`

| Legacy `-option` | Default | New `--flag` (→ `Config` field) |
|---|---|---|
| `-content_pattern` | `example/marple8_%02d.ppm` | `frames` positional |
| `-style_image` | `example/seated-nude.jpg` | `style` positional |
| `-content_weight` | `5e0` | `--content-weight` |
| `-style_weight` | `1e2` | `--style-weight` |
| `-temporal_weight` | `1e3` / `5e2` | `--temporal-weight` |
| `-tv_weight` | `1e-3` | `--tv-weight` |
| `-num_iterations` | `2000,1000` | `--num-iterations '2000,1000'` (first,subsequent) |
| `-init` | `random,prevWarped` | `--init 'random,prevWarped'` (first,subsequent) |
| `-optimizer` | `lbfgs` | `--optimizer {lbfgs,adam}` |
| `-learning_rate` | `1e1` | `--learning-rate` |
| `-content_layers` | `relu4_2` | `--content-layers` |
| `-style_layers` | `relu1_1,…,relu5_1` | `--style-layers` |
| `-temporal_loss_criterion` | `mse` | `--temporal-criterion {mse,smoothl1}` |
| `-flow_pattern` | `…/backward_[%d]_{%d}.flo` | `--flow-pattern` |
| `-flowWeight_pattern` | `…/reliable_[%d]_{%d}.pgm` | `--flow-weight-pattern` |
| `-flow_relative_indices` | `1` | `--flow-relative-indices` (and `flow --steps`) |
| `-use_flow_every` | `-1` | `--use-flow-every` |
| `-invert_flowWeights` | off | `--invert-flow-weights` |
| `-combine_flowWeights_method` | `closestFirst` | `--combine-flow-weights-method {normalize,closestFirst}` |
| `-num_images` | `0` (autodetect) | `--num-images` |
| `-start_number` | `1` | `--start-number` |
| `-continue_with` | `1` | `--continue-with` |
| `-number_format` | `%d` | `--number-format` |
| `-style_blend_weights` | none | `--style-blend-weights` |
| `-style_scale` | `1.0` | `--style-scale` |
| `-pooling` | `max` | `--pooling {max,avg}` |
| `-seed` | `-1` | `--seed` |
| `-tol_loss_relative` | `1e-4` | `--tol-loss-relative` |
| `-tol_loss_relative_interval` | `50` | `--tol-loss-relative-interval` |
| `-normalize_gradients` | off | `--normalize-gradients` |
| `-print_iter` | `100` | `--print-iter` |
| `-save_iter` | `0` | `--save-iter` |
| `-output_image` | `out.png` | `--output-image` |
| `-output_folder` | `""` | `--output-folder` |
| `-save_init` | off | `--save-init` |
| `-args <file>` | — | `--args <file>` (repeatable) |
| `-gpu` / `-backend` / `-cudnn_autotune` | `0` / `nn` | **removed** → `--device {mps,cuda,cpu}` |
| `-proto_file` / `-model_file` | caffe VGG-19 | `--vgg-weights {torchvision,<caffe-path>}` |

### `artistic_video_multiPass.lua` extras

| Legacy `-option` | Default | New `--flag` |
|---|---|---|
| `-num_passes` | `15` | `--passes` (implies `--multipass`) |
| `-blendWeight` | `1.0` | `--blend-weight` |
| `-blendWeight_lastPass` | `0.0` | `--blend-weight-last-pass` |
| `-use_temporalLoss_after` | `8` | `--use-temporal-loss-after` |
| `-continue_with_pass` | `1` | `--continue-with-pass` |
| `-forwardFlow_pattern` | `…/forward_[%d]_{%d}.flo` | `--forward-flow-pattern` |
| `-backwardFlow_pattern` | `…/backward_[%d]_{%d}.flo` | `--backward-flow-pattern` |
| `-forwardFlow_weight_pattern` | `…/reliable_[%d]_{%d}.pgm` | `--forward-flow-weight-pattern` |
| `-backwardFlow_weight_pattern` | `…/reliable_[%d]_{%d}.pgm` | `--backward-flow-weight-pattern` |

> **Removed in the port:** `run-deepflow.sh`, the C++ `consistencyChecker/`, and
> the DeepFlow/DeepMatching binaries (all replaced by `artvid flow` / RAFT), and
> the `loadcaffe` model loading (replaced by torchvision weights, with an
> optional caffe-weights path via `--vgg-weights`).

---

## Diffusion is Phase 2

The top-level `--engine` flag exposes a `diffusion` value, but it is **not
implemented in Phase 1**. The only working engine is `--engine optim` (the
default) — the ported L-BFGS / Adam optimization method. Selecting
`--engine diffusion` for `stylize` or `run` exits with:

```
artvid stylize: --engine diffusion is not implemented in Phase 1. Use
--engine optim (the default), the ported L-BFGS/Adam optimization method.
The diffusion engine is a Phase 2 deliverable.
```

The diffusion-based engine is planned for Phase 2; see
[`docs/04-phase2-plan.md`](04-phase2-plan.md).
