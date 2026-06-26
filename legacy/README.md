# `legacy/` — original Torch7 / Lua implementation (2016)

This directory holds the **original** `artistic-videos` implementation by Ruder,
Dosovitskiy & Brox (2016) — the Torch7/Lua code, shell scripts, and the C++
optical-flow consistency checker — kept **unchanged** as a historical and parity
reference. It is **not** maintained and does **not** run on Apple Silicon (it
needs Torch7 + CUDA, both unavailable on modern macOS).

For the maintained, runnable version, use the PyTorch port (`artvid`) at the repo
root — see [../README.md](../README.md) / [../README.zh.md](../README.zh.md).

## Contents

| File / dir | Role |
| --- | --- |
| `artistic_video.lua` | single-pass stylization main loop |
| `artistic_video_multiPass.lua` | multi-pass stylization |
| `artistic_video_core.lua` | losses, network build, optimization, I/O |
| `lbfgs.lua` | L-BFGS with relative-loss stopping |
| `flowFileLoader.lua` | Middlebury `.flo` reader |
| `makeOptFlow.sh` / `run-deepflow.sh` | DeepFlow/DeepMatching optical-flow driver |
| `stylizeVideo.sh` | end-to-end `video → stylized video` wrapper |
| `consistencyChecker/` | C++ forward/backward flow consistency checker |
| `models/download_models.sh` | downloads the caffe VGG-19 weights |

## Notes

- The shell scripts reference each other **by relative path** (`th artistic_video.lua`,
  `bash run-deepflow.sh`, `./consistencyChecker/...`), so run them **from inside
  `legacy/`**.
- The design docs (`../docs/02-migration-map.md`, `../docs/06-phase1-known-deviations.md`)
  cite these files by line number (e.g. `artistic_video_core.lua:364-397`); those
  citations refer to the sources **in this directory**.
- Sample inputs (`marple8_*.ppm`, `seated-nude.jpg`, precomputed DeepFlow under
  `deepflow/`) live in the shared `../example/` directory, used by both the legacy
  scripts and the PyTorch port.

## Citation

```
@inproceedings{RuderDB2016,
  author = {Manuel Ruder and Alexey Dosovitskiy and Thomas Brox},
  title  = {Artistic Style Transfer for Videos},
  booktitle = {German Conference on Pattern Recognition},
  pages  = {26--36},
  year   = {2016},
}
```
