# 04 · Phase 2 计划：扩散式视频风格化（SOTA）

> 在 Phase 1 稳固之后启动。目标：把质量推到 2026 年的当代水平，同时复用 Phase 1 的光流/warp/一致性栈做时序约束。
> 本文是方向性规划，细节待 Phase 1 完成后再细化。

## 1. 为什么要做这一阶段

逐帧优化法（Phase 1）的天花板由 Gram 矩阵风格表示决定：可控、任意风格、但纹理化、语义弱、慢。2026 年的 SOTA 是**扩散模型**——语义理解强、风格表达丰富、可由参考图/LoRA 灵活注入。Phase 2 把 2016 论文的**光流时序一致性思想**嫁接到扩散框架上，这正是这条研究线在今天最有价值的推进点。

## 2. 技术路线（候选，择一或组合）

### 结构保持（content 约束）
- **ControlNet**（depth / lineart / HED / Canny）锁定每帧结构，替代 Phase 1 的内容损失。

### 风格注入（style 约束）
- **IP-Adapter**（参考风格图，零训练）——对应 Phase 1「任意风格」的优势。
- **风格 LoRA**（需少量训练，质量更高，风格固定）——对应 `fast-artistic-videos` 的「固定风格换速度」路线。

### 时序一致性（本项目的差异化）
三种手段，可叠加：
1. **Latent 光流 warp**：用 Phase 1 的 RAFT 光流在**latent 空间**warp 前一帧，约束当前帧去噪轨迹（latent-space 版的 2016 时序损失）。
2. **Cross-frame attention**：让每帧 attention 看到锚定帧/前一帧，天然抑制闪烁（AnimateDiff / 视频扩散思路）。
3. **2016 一致性损失当 guidance**：把 `flow/consistency.py` 的可靠性 mask + warp 残差作为采样期 guidance/正则项。

### 骨干模型候选
- 图像扩散 + 逐帧 + 上述时序约束（SDXL / SD3 类 + ControlNet + IP-Adapter）。
- 原生视频扩散（AnimateDiff、Stable Video Diffusion、DiT 类视频模型）——时序一致性更天然，但风格控制接口不同。

**初步倾向**：先做「图像扩散 + ControlNet + IP-Adapter + latent 光流一致性」，因为它能最大化复用 Phase 1 的光流栈，且控制力强、可解释。视频扩散作为后续对比项。

## 3. 复用 Phase 1 的部分

| Phase 1 模块 | Phase 2 用途 |
|---|---|
| `flow/raft.py` | latent/pixel warp 的光流来源 |
| `flow/warp.py` | latent 空间 warp（采样到 latent 网格） |
| `flow/consistency.py` | 可靠性 mask → guidance 权重 |
| `io/video.py` | 抽帧/合成完全复用 |
| `cli.py` | 同一 CLI，`--engine diffusion` 切换 |
| `config.py` | 扩展扩散相关字段 |

## 4. 新增依赖（预估）

```
diffusers, transformers, accelerate, safetensors
controlnet_aux            # 结构预处理器（depth/lineart/hed）
# 模型权重通过 huggingface 下载；注意许可证
```
- 在 M5 Max 上：扩散推理走 MPS；128GB 统一内存足以容纳 SDXL + ControlNet + IP-Adapter 同时驻留。
- 速度预期：每帧秒级（远快于 Phase 1 的逐帧优化）。

## 5. 里程碑（草案）

- **P2-M0**：单图扩散风格化（ControlNet + IP-Adapter）在 MPS 跑通。
- **P2-M1**：逐帧 + latent 光流一致性，产出时序稳定的短片。
- **P2-M2**：与 cross-frame attention / 视频扩散对比，挑稳定性最好的。
- **P2-M3**：并入 CLI（`--engine diffusion`），基准与文档。

## 6. 风险

| 风险 | 缓解 |
|---|---|
| 模型许可证（商用/再分发） | 选明确许可的权重；文档标注 |
| 扩散在 MPS 上的算子覆盖/速度 | 基准先行；必要时降精度/换调度器 |
| 时序一致性调参复杂 | 复用 Phase 1 的稳定性度量做客观评估 |
| 质量主观 | 固定评测片段 + 并排对比 + 稳定性指标 |
| 范围膨胀 | 严格在 Phase 1 验收后才启动；P2 里程碑独立交付 |

## 7. 与 Phase 1 的关系

Phase 2 **不替换** Phase 1：优化法保留为「任意风格 + 精确控制 + 无需模型下载」的轻量路径；扩散法作为「高质量」路径。两者共用 CLI 与光流栈，由 `--engine` 选择。
