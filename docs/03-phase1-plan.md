# 03 · Phase 1 计划：复刻优化法 + RAFT，跑在 Apple Silicon Mac

> 目标：把 2016 优化法忠实搬到 PyTorch+MPS，用 RAFT 取代 DeepFlow，产出一条命令出片的 CLI，并在你的 Apple Silicon Mac（任意 M 系列 / MPS）上验证质量与性能。

## 1. 里程碑

每个里程碑都有明确的**可验证产出**（demo + 测试），便于 workflow 分阶段验收。

### M0 — 骨架与单图风格迁移（neural-style parity）
**产出**：`artvid` 包骨架；`config.py`/`device.py`/`io/image.py`/`models/vgg.py`/`losses/{content,style,tv}.py`/`optim/{lbfgs,runner}.py`；能对**单张图**做风格迁移。
**验收**：
- 用 `example/` 中一帧 + 一张风格图，产出合理的风格化静图。
- 单测：`gram_matrix` 形状/数值、TV 损失方向、L-BFGS 收敛、preprocess↔deprocess 往返。
- 在 MPS 上跑通（CPU 回退点记录在案）。

### M1 — 光流子系统（RAFT 取代 DeepFlow）
**产出**：`flow/raft.py`（前/后向光流）、`flow/warp.py`（grid_sample + 遮挡填充）、`flow/consistency.py`（一致性 mask + 长时权重）、`io/flow_io.py`（`.flo` 互通）、`artvid flow` 子命令。
**验收**：
- 对 `example/` 相邻帧算光流，warp 前一帧到当前帧，残差明显小于不 warp。
- 一致性 mask 在遮挡/越界处为低权重（可视化检查）。
- 单测：`.flo` 读写往返、warp 的 (y,x)/grid_sample 约定、一致性对称性。
- **与旧 DeepFlow 对照**（若能跑旧二进制则定量对比 EPE；否则定性 + 下游质量对比）。

### M2 — 单遍视频管线（时序一致性）
**产出**：`pipeline/singlepass.py`（逐帧顺序、首帧 init、后续 prevWarped、时序损失、长时多历史帧）。
**验收**：
- 对 `example/` 整段产出风格化序列；**相邻帧闪烁明显低于无时序损失版本**（用 warp 后帧间差异度量）。
- 长时一致性（`flow_relative_indices`）可用。
- 端到端不 OOM（受益于统一内存）。

### M3 — 多遍管线
**产出**：`pipeline/multipass.py`（前后向多趟 + blend + `use_temporalLoss_after`）。
**验收**：
- 强相机运动片段上，多遍结果比单遍更稳定（定性 + 帧间差异度量）。

### M4 — CLI 一条龙 + 基准 + 文档
**产出**：`artvid run <video> <style>`（抽帧→光流→风格化→合成）；在你的 Apple Silicon Mac（任意 M 系列 / MPS）上测得的性能基准；用户文档（README 更新或 `docs/usage.md`）。
**验收**：
- 一条命令从 mp4 到 stylized.mp4 跑通。
- 基准表：分辨率 × 每帧迭代数 × 每帧墙钟 × 峰值内存 × 是否 CPU 回退。
- 与旧 `stylizeVideo.sh` 的参数对应关系在文档中说明。

## 2. 依赖与环境

```
python >= 3.11
torch (含 mps 后端)            # Apple Silicon 构建
torchvision                    # VGG-19 权重 + RAFT 光流
imageio / imageio-ffmpeg       # 抽帧/合成（或直接 subprocess 调 ffmpeg）
numpy, pillow
tyro 或 argparse               # CLI
pytest                         # 测试
ffmpeg                         # 系统依赖
```
- `PYTORCH_ENABLE_MPS_FALLBACK=1` 默认开启。
- 提供 `pyproject.toml`，`artvid` 作为可安装包，暴露 `artvid` 命令。

## 3. 验证策略

| 层级 | 方法 |
|---|---|
| **单元** | 损失数值/形状、gram、TV、warp 约定、`.flo` 往返、一致性 mask、L-BFGS 停止准则 |
| **集成** | 单图风格化、单帧时序损失下降、整段管线产出 |
| **回归** | 固定 `example/` 帧 + 固定 seed，存基准输出图，后续改动比对（SSIM/感知差异阈值） |
| **质量** | 帧间稳定性度量：`mean |warp(out_{t-1}) - out_t|·mask`，越低越稳 |
| **parity** | 与 2016 输出视觉对照；可选 caffe 权重路径做更接近的复现 |
| **性能** | 在你的 Apple Silicon Mac（任意 M 系列 / MPS）上测得的基准（见 M4） |

## 4. 风险与缓解

| 风险 | 影响 | 缓解 |
|---|---|---|
| MPS 个别算子未实现/回退 CPU | 变慢或报错 | `MPS_FALLBACK=1`；基准记录回退点；必要时换等价算子 |
| torchvision VGG 与 caffe VGG 结果有差异 | 与 2016 不完全一致 | 已在非目标声明；提供可选 caffe 权重路径 |
| RAFT 在极端运动/遮挡下的边界质量 | 时序伪影 | 一致性 mask 抑制不可靠区；可选 tile 处理高分辨率 |
| L-BFGS 在 MPS 上的数值稳定性 | 收敛差 | 图像变量保持 float32；必要时回退 Adam |
| `.flo` (y,x) 与 grid_sample 约定搞反 | warp 错位 | 单独单测 + 可视化金标准 |
| 高分辨率单帧显存/算力 | 慢 | 统一内存可容纳；提供分辨率参数与可选 tiling |

## 5. 工作量与排序

建议顺序：M0 → M1 →（M2 与「回归基准」并行）→ M3 → M4。
M0/M1 是基础，必须先稳。M2 是本论文核心价值（时序一致性），是 Phase 1 的重心。M3 增量、M4 收尾。
具体的 agent 拆分与并行编排见 `05-workflow-plan.md`。
