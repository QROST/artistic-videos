# 09 · 复盘：从 2016 原项目到当前 fork

> 这份文档记录我们把原始 `artistic-videos`（Ruder 等 2016，Torch7/Lua）现代化为 PyTorch+MPS 双引擎代码库（`artvid`）的全过程：起点、做了什么、成果、经验教训与局限。

## 1. 起点：原 artistic-videos 是什么

- 2016 年 Ruder/Dosovitskiy/Brox 论文 *Artistic style transfer for videos* 的 **Torch7 / Lua** 实现，跑在 **CUDA** 上。
- 方法：**逐帧优化**——冻结一个预训练 VGG-19 当"打分器"，用 L-BFGS **直接优化每帧像素**（内容 + Gram 风格 + TV + 光流时序损失）。**不训练任何网络、不收集数据**（VGG 是别人在 ImageNet 上预训练好、冻结使用的）。
- 光流靠外部 **DeepFlow/DeepMatching** CPU 二进制 + C++ `consistencyChecker`。
- 现状问题：Torch7 停维、Apple Silicon 无 CUDA、OpenCL 已弃——**在 Apple Silicon Mac 上基本跑不起来**。

> 容易混淆的一点：真正"在数据集上训练一个前馈网络"的是**另一个项目** `fast-artistic-videos`（快但风格固定）。本仓库的原始版本不训练网络。详见下方「范式对比」。

## 2. 我们做了什么（7 个 PR）

| PR | 内容 |
|---|---|
| #1 | 设计文档：决策（PyTorch+MPS / 分阶段 / CLI）、架构、Lua→PyTorch 逐函数迁移映射、workflow 编排计划 |
| #2 | Phase 1 M0+M1：基础层（config/device/io/vgg/losses/optim）+ RAFT 光流栈取代 DeepFlow |
| #3 | Phase 1 M2–M4：单遍/多遍管线 + 端到端 CLI（`flow`/`stylize`/`run`） |
| #4 | Phase 2 基础：扩散引擎（SDXL+ControlNet+IP-Adapter）+ latent 光流一致性 |
| #5 | Apple Silicon 启用：打包 extras、dtype/MPS 硬化、模型预拉取、smoke 脚本 |
| #6 | 加 CI（GitHub runner 装 CPU torch 真跑全套）+ 修 Phase 1 偏差 |
| #7 | Phase 2 深化：masked init / 长时 anchor / cross-frame attention / pixel-warp |

## 3. 取得的成果

- **一个完整的 PyTorch+MPS 重写**：31 个模块、10 个测试文件、11+ 篇文档，全部合并进 master。
- **两套引擎共用一个 CLI**：`optim`（忠实复刻 2016）+ `diffusion`（当代 SOTA 方向）。
- **光流现代化**：RAFT 一条命令搞定，取代 DeepFlow/DeepMatching/consistencyChecker 三件套。
- **CI 在每个 PR 上真跑 ~199 个 torch 测试**（CPU），并全绿。
- **可在你的 Apple Silicon Mac（任意 M 系列 / MPS）上"装好即跑"**：`pip install -e ".[all]"` → prefetch → smoke → run；统一内存让 GPU 共享 Mac 的 RAM（没有独立显存预算），原版最大的显存瓶颈消失——上限改由你这台 Mac 的 RAM 决定。
- **把 2016 的核心思想传承到扩散时代**：光流时序一致性从像素空间搬到 latent 空间。

## 4. 经验教训

### 技术层面
1. **"迁移语言"其实是"换框架"**：Lua 只是胶水，真正的决策是选 PyTorch 生态（RAFT、扩散全在这）。Swift/Rust 在「交付 CLI + 走扩散」这条线下会过度限制。
2. **PyTorch autograd 砍掉了原版近一半复杂度**：旧版手动把损失模块插进网络、手写反向、管理插入索引——现代框架里全部消失。
3. **老论文里真正"耐放"的是思想，不是实现**：Gram 风格表示被扩散取代了，但**光流时序一致性**这个 idea 一路传承下来，是两个时代之间真正的连接点。
4. **parity 是个工程取舍**：默认 torchvision VGG（视觉等价、好维护），保留 caffe 权重路径给需要逐比特复刻的人。

### 过程 / 工程层面
5. **CI 是单点杠杆最高的一步**。它抓出了静态审查抓不到的真问题：先是 16 个 import 排序，再是 4 个写错的测试期望（而源码全对）。"**能跑的证据**"远胜"审查通过"。
6. **没法运行的环境里，验证要分层**：本开发环境装不了 torch → 改用 (a) 对抗式代码审查 + (b) GitHub runner 上的 CI。诚实地讲，这里产出的代码是**单测级 + 审查级**可信，**扩散画质/真机性能仍未验证**。
7. **盲写的测试会写错期望**：那 4 个 CI 失败全是测试自身的错（源码对）——印证了"没法跑就别假装验证过"，CI 就是兜底网。
8. **纪律的形成**：头两轮 CI 连红（lint、测试期望）后，本地装了 ruff、每次推送前先跑 ruff+pytest，后面就一次过。
9. **多 agent workflow + 审查闸门确实拦住了真 blocking bug**：`artvid run` 默认崩溃、扩散前向光流文件名顺序错……每轮都抓到 1 个。
10. **范围要诚实**：到"离线高置信度的天花板"就停——再往下的扩散调参/基准必须靠 Apple Silicon Mac 实跑反馈，盲写只是低置信度代码。

## 5. 局限 / 尚未完成（同样要诚实）

- **从未在真实 Apple Silicon Mac 上跑过**：画质、时序稳定性、每帧耗时、内存占用、CPU 回退点——全部未实测。
- **扩散引擎是"地基"**：大量 `TODO(tuning)`（各 strength/steps/scale/scheduler、`noise_seed_mode='warped'` 暂回退 random、cross-attn 层选择）待硬件调。
- 几处 parity 偏差已文档化（`docs/06`，非逐比特一致）；未实现风格 LoRA 训练（刻意）。

## 6. 范式对比（澄清"训练/数据"的常见误解）

| | 训练网络 | 逐帧像素优化(L-BFGS) | 扩散采样 |
|---|---|---|---|
| 原项目 / 我们的 optim 引擎 | ❌ 不训练 | ✅ | ❌ |
| `fast-artistic-videos`（旁系） | ✅ 每风格训练一个前馈网 | ❌ | ❌ |
| 我们的 diffusion 引擎 | ❌（用预训练大模型） | ❌ | ✅ |

- **L-BFGS 是"优化"但不是"训练"**：它是基于梯度的优化，与训练同源；但它优化的是**输出图像的像素**，VGG 权重**全程冻结**（原 `artistic_video.lua` 甚至显式 `gradWeight=nil` 丢掉权重梯度，正说明权重不更新）。"训练"在 ML 里专指"在数据集上拟合**权重**、得到可复用模型"。
- **三端都不训练网络、不收集数据**；区别在"怎么生成像素"：原项目/optim 引擎对像素做梯度优化，diffusion 引擎走扩散采样。

## 7. 一句话总结

我们把一个跑不起来的 2016 Lua 项目，变成了一个结构清晰、CI 守护、能在 Apple Silicon Mac（任意 M 系列 / MPS）上跑的**双引擎 PyTorch 代码库**，并把它的核心思想推进到了扩散时代；最大的收获不是某段代码，而是**"用 CI 和对抗式审查替代跑不了的运行时"这套在受限环境里仍能保证质量的工作方式**——以及知道在哪儿该停下来等真机反馈。
