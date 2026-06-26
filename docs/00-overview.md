# 00 · 概览、目标与关键决策

> 本文档是 2026 年现代化重写工作的起点。它解释**现状是什么**、**我们要做什么**、以及**为什么这样选**。
> 后续文档（架构、迁移映射、分阶段计划、workflow 执行计划）都建立在这里定下的决策之上。

## 1. 现状：这份代码是什么

本仓库是 Ruder, Dosovitskiy, Brox (2016) 论文 *Artistic style transfer for videos*（[arXiv:1604.08610](https://arxiv.org/abs/1604.08610)）的**原始 Torch7 / Lua 实现**，由 Justin Johnson 的 `neural-style` 派生而来。

核心方法：

- **逐帧优化**：在像素上跑 L-BFGS，最小化 Gatys 式的「内容 + 风格 + 全变差」损失。
- **风格损失**：VGG-19 多层特征的 Gram 矩阵 MSE。
- **时序一致性**（本论文相对静态 neural-style 的核心贡献）：用光流把前一帧 warp 过来，做一个被光流置信度 mask 的像素级损失（代码里的 `initWeighted` / `WeightedContentLoss`）。
- **长时一致性**：可同时约束多个历史帧（`flow_relative_indices`、`closestFirst` 加权）。
- 有单遍（single-pass）和多遍（multi-pass）两种流程。

关键依赖（也是它今天跑不起来的原因）：

| 依赖 | 问题 |
|---|---|
| Torch7 / Lua | 自 ~2017 起停止维护，现代 macOS / Apple Silicon 难以编译 |
| CUDA / cuDNN | Apple Silicon 无 CUDA |
| clnn / cltorch (OpenCL) | 早已弃坑，Apple 已废弃 OpenCL |
| loadcaffe + caffe 版 VGG-19 | caffe 生态停滞 |
| DeepFlow + DeepMatching（CPU 静态二进制） | 老旧、仅 CPU、质量落后于学习式光流 |

结论：**现有 Lua/Torch7 代码在 Apple Silicon Mac 上基本无法有意义地运行**，必须换框架重写。

## 2. 目标

1. **能在 Apple Silicon Mac 上跑通**，吃满 Apple GPU（Metal/MPS）。
2. **方法学上忠实复刻** 2016 论文（单遍 + 多遍 + 长时一致性），结果质量不低于原版。
3. **用现代学习式光流（RAFT）替换 DeepFlow/DeepMatching**，提质提速。
4. **打磨成一条命令出片的 CLI 工具**（替代现在的 `stylizeVideo.sh`）。
5. 在复刻稳固之后，**演进到扩散式（diffusion）视频风格化**这一当代 SOTA 方向。

## 3. 关键决策（Decision Record）

这些决策是在与用户确认后定下的，构成后续所有设计的前提。

### D1 — 实现栈：Python + PyTorch + MPS

- **选择**：Python，PyTorch，`device="mps"`。
- **理由**：迁移成本最低（本质就是 `neural-style`）；生态最全——RAFT 光流（torchvision）、VGG-19 权重、以及 Phase 2 要用的整个扩散生态（diffusers / ControlNet / IP-Adapter / AnimateDiff / SVD）**几乎全是 PyTorch**。
- **被否方案**：
  - *Swift + MLX*：仅在「交付物=原生 Mac App」时划算；本项目交付 CLI，且 Phase 2 扩散生态在 MLX 上要手工移植，过度限制。
  - *Rust + candle*：autograd / 光流 / 损失生态薄，工作量最大。
  - *Python + MLX*：性能略优（部分负载 ~1.2–2×），但生态比 PyTorch 薄，且无法平滑过渡到 Phase 2 扩散。
- **性能取舍**：MPS 与 MLX 差距非数量级；本负载瓶颈是 VGG 前向/反向的 conv+matmul，两边都 GPU-bound。把光流和损失模块写得**框架无关**，将来若出现热点可单点替换为 MLX/Metal kernel，但不预先支付该成本。

### D2 — 范围：分阶段（先复刻，后 SOTA）

- **Phase 1**：忠实现代化 2016 优化法 + RAFT 光流，先在你的 Apple Silicon Mac（任意 M 系列 / MPS）上跑通并验证质量。
- **Phase 2**：扩散式视频风格化；复用 Phase 1 的光流/warp/一致性模块，把光流一致性思想当作 latent 正则项。
- **理由**：Phase 1 风险低、里程碑清晰，且产出（光流栈、I/O、CLI 骨架）能被 Phase 2 复用。

### D3 — 交付形态：命令行工具

- 一条命令完成「抽帧 → 算光流 → 风格化 → 合成视频」，对标并取代 `stylizeVideo.sh`。
- 同时保留可编程的 Python API，便于研究迭代。

### D4 — 与旧代码共存

- 新代码放在独立 Python 包（暂定 `artvid/`）与 `docs/` 中，**不删除**原始 Lua 文件，保留为参考实现与 parity 对照基准。

## 4. 非目标（Non-goals）

- 不追求与 2016 输出**逐比特**一致（VGG 权重来源不同会有数值差异；见架构文档 parity 小节）。我们追求**视觉质量等价或更好**。
- Phase 1 不做实时；逐帧优化在你的 Apple Silicon Mac 上预期每帧数十秒到一两分钟级别，可接受。
- 不维护 CUDA/OpenCL 老后端；但代码不硬编码 MPS，`mps|cuda|cpu` 由设备层自动选择。

## 5. 成功标准

- **Phase 1 验收**：用 `example/` 帧序列，单遍与多遍均能在你的 Apple Silicon Mac 上产出时序稳定（无明显闪烁）的风格化视频；RAFT 光流路径替代 DeepFlow；一条 CLI 命令端到端跑通；附在你的 Apple Silicon Mac（任意 M 系列 / MPS）上测得的性能基准。
- **Phase 2 验收**：扩散引擎产出质量明显优于 Phase 1 的风格化视频，且时序一致性不差于 Phase 1；同一 CLI 通过 `--engine diffusion` 切换。

## 6. 统一内存为什么是优势

原版最大的工程约束是**显存**（README 反复强调 450×350 要 4GB、Titan X 12GB 最多 960×540）。Apple Silicon 是统一内存——GPU 与 CPU 共享 Mac 的 RAM，没有独立的「显存」预算；上限≈你这台 Mac 的 RAM − 系统/应用占用，于是：

- RAM 越多，可处理的分辨率越高、headroom 越大，不必为 OOM 降分辨率（原版最大痛点消失）；RAM 较小则用更低分辨率/更轻设置，但仍能跑。
- 可一次性 batch 多帧、把光流与风格优化放在一起，提高吞吐。
- Phase 2 的扩散大模型在 RAM 足够时也能容纳在统一内存里。

## 7. 文档地图

| 文档 | 内容 |
|---|---|
| `00-overview.md`（本文） | 背景、目标、决策 |
| `01-architecture.md` | 目标架构、模块职责、数据流、parity 决策 |
| `02-migration-map.md` | Lua → PyTorch 逐文件/逐函数映射 |
| `03-phase1-plan.md` | Phase 1 里程碑、验证、风险 |
| `04-phase2-plan.md` | Phase 2 扩散式方案 |
| `05-workflow-plan.md` | 用 Workflow 工具执行 Phase 1 的编排计划 |
