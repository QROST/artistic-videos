# artistic-videos（中文说明）

> English: see **[README.md](README.md)**. 本文件是 2026 PyTorch 重写版的中文说明；原始 Torch7/Lua 实现的完整文档见英文 README 下半部分。

## 2026 现代化 —— PyTorch 移植版（`artvid`）

本仓库现在提供一个 **PyTorch + Metal (MPS)** 重写版，打包为 `artvid`，面向
**Apple Silicon / M5 Max**（128GB 统一内存，MPS 后端，带 CPU 回退）。它忠实复刻了
Ruder 等人 2016 论文《Artistic style transfer for videos》的优化式视频风格迁移，
并新增了一个可选的现代**扩散引擎**。

### 两套引擎，一个 CLI

| 引擎 | 方法 | 说明 |
| --- | --- | --- |
| `optim`（默认） | 逐帧 L-BFGS 像素优化 + 冻结 VGG-19 + Gram 风格损失 + 光流时序一致性 | 2016 论文的忠实移植。任意风格、零训练。分钟/帧。 |
| `diffusion` | SDXL + depth ControlNet + IP-Adapter，并把 2016 的光流时序思想嫁接到 **latent** 空间 | 参考图零样本风格、零训练。秒/帧。属基础实现，画质需在硬件上调参。 |

### 相比原项目变了什么（完整复盘：[docs/09-retrospective.md](docs/09-retrospective.md)）

- **语言/框架**：Torch7/Lua → PyTorch + MPS
- **硬件**：只能 CUDA → 可跑在 Apple Silicon（128GB 统一内存消除原版显存瓶颈）
- **光流**：DeepFlow/DeepMatching + C++ consistencyChecker 三件套 → **RAFT** 一条命令
- **质量保障**：每个 PR 由 **CI** 在 GitHub runner 上装 CPU torch 真跑完整测试套件
- **训练/数据**：两套引擎都**不训练网络、不收集数据**（详见复盘里的范式对比表）

## 快速开始（M5 Max）

详见 [docs/08-m5max-quickstart.md](docs/08-m5max-quickstart.md)。概要：

```bash
# 1. 系统准备：Python 3.11、ffmpeg（brew install ffmpeg）
# 2. 安装（含扩散依赖）
pip install -e ".[all]"

# 3.（可选）预拉取模型权重，避免首跑卡在下载
python scripts/prefetch_models.py --diffusion

# 4.（可选）端到端冒烟测试
python scripts/smoke_test.py --diffusion

# 5. 出片
artvid run input.mp4 example/seated-nude.jpg                      # 优化引擎（默认）
artvid --engine diffusion run input.mp4 example/seated-nude.jpg   # 扩散引擎
```

> 注意：`--engine` 是全局开关，要放在子命令**之前**（`artvid --engine diffusion run ...`）。
> 建议保留环境变量 `PYTORCH_ENABLE_MPS_FALLBACK=1`，让个别未支持的算子回退 CPU。

## 三个子命令

- `artvid flow <帧目录/pattern>` —— 算前/后向 RAFT 光流 + 可靠性掩码（取代 `makeOptFlow.sh`）
- `artvid stylize <帧目录/pattern> <风格图>` —— 逐帧风格化（`--multipass` / `--passes` 切多遍）
- `artvid run <视频> <风格图>` —— 端到端：抽帧 → 光流 → 风格化 → 合成（取代 `stylizeVideo.sh`）

完整参数与旧 Lua 参数的对照见 [docs/usage.md](docs/usage.md)。

## 文档导航

| 文档 | 内容 |
| --- | --- |
| [docs/README.md](docs/README.md) | 设计与文档索引 |
| [docs/00-overview.md](docs/00-overview.md) | 背景、目标、关键决策 |
| [docs/01-architecture.md](docs/01-architecture.md) | 目标架构、模块职责、parity |
| [docs/02-migration-map.md](docs/02-migration-map.md) | Lua → PyTorch 逐函数迁移映射 |
| [docs/03-phase1-plan.md](docs/03-phase1-plan.md) · [04](docs/04-phase2-plan.md) | Phase 1 / Phase 2 计划 |
| [docs/06-phase1-known-deviations.md](docs/06-phase1-known-deviations.md) | 已知 parity 偏差与修复记录 |
| [docs/07-phase2-design.md](docs/07-phase2-design.md) | 扩散方案具体设计 |
| [docs/08-m5max-quickstart.md](docs/08-m5max-quickstart.md) | M5 Max 上手 |
| [docs/09-retrospective.md](docs/09-retrospective.md) | 复盘：做了什么、成果、经验教训 |

## 现状与局限（诚实声明）

- **画质/性能尚未在真 M5 Max 上实测**：扩散画质、时序稳定性、每帧耗时、显存、CPU 回退点都待硬件反馈。
- **扩散引擎是"地基"**：代码里大量 `TODO(tuning)` 待调（strength/steps/scale/scheduler 等）。
- 离线（无 GPU）能高置信度完成的工作已到顶；下一步需要 M5 Max 实跑数据来做有依据的调优与基准。

## 关于"是否训练 / 收集数据"的澄清

- 原项目（及本仓库 `optim` 引擎）用 **L-BFGS 优化每帧像素**，VGG 权重全程冻结——这是**优化**，不是 ML 意义上的**训练**（训练指在数据集上拟合权重得到可复用模型）。
- `diffusion` 引擎用的是**预训练好的** SDXL/ControlNet/IP-Adapter，下载即用、**零训练零数据收集**。
- 真正"为每种风格训练一个前馈网络"的是旁系项目 `fast-artistic-videos`，与本仓库不同。详见 [docs/09-retrospective.md](docs/09-retrospective.md) §6。

---

原始 Torch7/Lua 实现的引用、致谢与详细参数说明，请见英文 [README.md](README.md)。
