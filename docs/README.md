# artistic-videos · 2026 现代化设计文档

本目录是把本仓库（Ruder 等 2016 *Artistic style transfer for videos* 的 Torch7/Lua 原始实现）
现代化重写、并推进研究方向的**设计与规划**。先有文档与规划，再用 workflow 执行。

## 决策摘要

| 维度 | 决定 |
|---|---|
| **实现栈** | Python + PyTorch + MPS（吃满 M5 Max Apple GPU；Phase 2 扩散生态也在 PyTorch） |
| **范围** | 分阶段：Phase 1 复刻优化法 + RAFT 光流；Phase 2 扩散式 SOTA |
| **交付** | 命令行工具（取代 `stylizeVideo.sh`），并保留 Python API |
| **光流** | RAFT（torchvision）取代 DeepFlow/DeepMatching CPU 二进制 |
| **旧代码** | 保留为参考实现与 parity 基准，不删除 |

## 文档导航

| # | 文档 | 内容 |
|---|---|---|
| 00 | [overview](./00-overview.md) | 背景、目标、关键决策（Decision Record）、非目标、成功标准 |
| 01 | [architecture](./01-architecture.md) | 目标 Python 包结构、模块职责、数据流、MPS/128GB、parity 决策 |
| 02 | [migration-map](./02-migration-map.md) | Lua → PyTorch 逐文件/逐函数映射、参数对照、现代化简化 |
| 03 | [phase1-plan](./03-phase1-plan.md) | M0–M4 里程碑、验证策略、风险、排序 |
| 04 | [phase2-plan](./04-phase2-plan.md) | 扩散式视频风格化方向、复用 Phase 1、风险 |
| 05 | [workflow-plan](./05-workflow-plan.md) | 用 Workflow 工具执行 Phase 1 的 agent 拆分、DAG、脚本骨架 |
| 06 | [phase1-known-deviations](./06-phase1-known-deviations.md) | 已知 parity 偏差与修复记录（含 M2–M4、Phase 2） |
| 07 | [phase2-design](./07-phase2-design.md) | 扩散方案的具体可实现设计（模型栈、latent 一致性算法） |
| 08 | [m5max-quickstart](./08-m5max-quickstart.md) | M5 Max 上手：安装 → 预拉取 → smoke → 出片 |
| 09 | [retrospective](./09-retrospective.md) | 复盘：起点 vs 现状、做了什么、成果、经验教训、范式对比 |

## 当前状态

- [x] 方向确认：PyTorch + MPS，分阶段，CLI
- [x] 设计文档与规划（本目录）
- [x] Phase 1（M0–M4）实现并合并：optim 引擎 + RAFT 光流栈 + CLI
- [x] Phase 2 基础 + 深化并合并：扩散引擎 + latent 一致性 + masked init/anchor/cross-attn/pixel-warp
- [x] CI（GitHub runner，CPU torch 全套测试）守护每个 PR
- [ ] **在 M5 Max 上的真机验证**：扩散画质、时序稳定性、性能基准（需用户侧执行）
- [ ] 据真机反馈做有依据的扩散调参（各 `TODO(tuning)`）

## 如何往下走

1. Review 本目录文档，对范围/里程碑提意见。
2. 确认后，以显式授权（"use a workflow" / ultracode）启动 `05-workflow-plan.md` 描述的 Phase 1 workflow。
3. 实现与单测可在任意环境完成；**真机性能基准需在 M5 Max 上执行**（本 CI 环境无 GPU）。
