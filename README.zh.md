# artistic-videos（中文说明）

> English: see **[README.md](README.md)**. 本文件是 2026 PyTorch 重写版的中文说明，并在下半部分翻译了原始 Torch7/Lua 实现的文档。

## 2026 现代化 —— PyTorch 移植版（`artvid`）

本仓库现在提供一个 **PyTorch + Metal (MPS)** 重写版，打包为 `artvid`，面向
**任意 Apple Silicon / M 系列 Mac**（Metal/MPS 后端，带 CPU 回退，任意统一内存大小均可）。它忠实复刻了
Ruder 等人 2016 论文《Artistic style transfer for videos》的优化式视频风格迁移，
并新增了一个可选的现代**扩散引擎**。

### 两套引擎，一个 CLI

| 引擎 | 方法 | 说明 |
| --- | --- | --- |
| `optim`（默认） | 逐帧 L-BFGS 像素优化 + 冻结 VGG-19 + Gram 风格损失 + 光流时序一致性 | 2016 论文的忠实移植。任意风格、零训练。分钟/帧。 |
| `diffusion` | SDXL + depth ControlNet + IP-Adapter，并把 2016 的光流时序思想嫁接到 **latent** 空间 | 参考图零样本风格、零训练。秒/帧。属基础实现，画质需在你的 Apple Silicon Mac 上实测调参。 |

### 相比原项目变了什么（完整复盘：[docs/09-retrospective.md](docs/09-retrospective.md)）

- **语言/框架**：Torch7/Lua → PyTorch + MPS
- **硬件**：只能 CUDA → 可跑在任意 Apple Silicon / M 系列 Mac（Metal/MPS）。统一内存意味着 GPU 与整机 RAM 共享，没有独立显存预算：上限是**你 Mac 的 RAM**（减去系统/应用占用），而非单独的显存额度——RAM 越大可上更高分辨率、扩散模型余量越足。配置参考见 quickstart 的"内存与 RAM"小节。
- **光流**：DeepFlow/DeepMatching + C++ consistencyChecker 三件套 → **RAFT** 一条命令
- **质量保障**：每个 PR 由 **CI** 在 GitHub runner 上装 CPU torch 真跑完整测试套件

## 快速开始

详见 [docs/08-quickstart.md](docs/08-quickstart.md)（内含按内存大小配置的"内存与 RAM"小节）。概要：

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
| [docs/08-quickstart.md](docs/08-quickstart.md) | Apple Silicon Mac 上手 |
| [docs/09-retrospective.md](docs/09-retrospective.md) | 复盘：做了什么、成果、经验教训 |

## 现状与局限（诚实声明）

- **画质/性能尚未在真机上实测**：扩散画质、时序稳定性、每帧耗时、内存占用、CPU 回退点都待在你的 Apple Silicon Mac（任意带 MPS 的 M 系列）上反馈。
- **扩散引擎是"地基"**：代码里大量 `TODO(tuning)` 待调（strength/steps/scale/scheduler 等）。
- 离线（无 GPU）能高置信度完成的工作已到顶；下一步需要在 Apple Silicon Mac（任意带 MPS 的 M 系列）上的实跑数据来做有依据的调优与基准。

---

# 原始实现（Torch7 / Lua，2016）

> 以下是本仓库**原始** Torch7/Lua 实现（论文作者 Ruder 等）的中文翻译。新的 PyTorch 版（`artvid`）见上文；这部分作为原始参考保留。

这是论文《[Artistic style transfer for videos](http://arxiv.org/abs/1604.08610)》的 Torch 实现，基于 Justin Johnson 的 neural-style 代码：<https://github.com/jcjohnson/neural-style>。

我们的算法能把一张图片（例如一幅画）的风格迁移到整段视频，并生成时序一致、稳定的风格化视频。

**更新**：一个快得多、每帧不到一秒的版本见 [fast-artistic-videos](https://github.com/manuelruder/fast-artistic-videos)，但它只支持预先计算好的固定风格模板。本仓库支持任意风格，但每帧需要几分钟。

**示例视频**：

[![Artistic style transfer for videos](http://img.youtube.com/vi/Khuj4ASldmU/0.jpg)](https://www.youtube.com/watch?v=Khuj4ASldmU "Artistic style transfer for videos")

## 联系方式

与本实现相关的问题，请使用 [issue tracker](https://github.com/manuelruder/artistic-videos/issues)。其他事宜（包括授权问题）请邮件联系，联系方式见[论文](http://arxiv.org/pdf/1604.08610.pdf)。

## 安装（Setup）

在 Ubuntu 14.04 上测试通过。

* 按 jcjohnson 的 [neural-style#setup](https://github.com/jcjohnson/neural-style#setup) 安装 torch7、loadcaffe 和 CUDA 后端（否则只能用慢到不可用的 CPU 模式），并下载 VGG 模型。可选：安装 cuDNN（需在 NVIDIA 注册开发者，但能显著降低显存占用）。非 NVIDIA GPU 也可用 OpenCL 后端。
* 要使用时序一致性约束，需要一个估计两帧之间[光流](https://en.wikipedia.org/wiki/Optical_flow)的工具。可用论文里用的 [DeepFlow](http://lear.inrialpes.fr/src/deepflow/)：从其网站下载 DeepFlow 和 DeepMatching（CPU 版），把静态二进制 `deepmatching-static` 和 `deepflow2-static` 放到仓库主目录。然后用本仓库的脚本为所有帧生成光流及其置信度。若想用别的光流算法，在 `makeOptFlow.sh` 第一行指定其路径；光流文件需为 [middlebury 格式](http://vision.middlebury.edu/flow/code/flow-code/README.txt)。

## 运行要求（Requirements）

推荐使用显存大的快速 GPU。CPU 模式因耗时巨大而不实用。

450×350 分辨率至少需 4GB 显存（约 3.5GB 占用）；若用 cuDNN 则 2GB 显存即可（约 1.7GB）。显存占用随分辨率线性增长，遇到 OOM 时请降低分辨率。

其他降低显存的办法：用 ADAM 替代 L-BFGS，和/或用 NIN Imagenet 模型替代 VGG-19——但作者未测试这两者，效果可能更差。

## 简单风格迁移

用大致默认的参数做风格迁移：执行 `stylizeVideo.sh <视频路径> <风格图路径>`，该脚本会完成创建风格化视频所需的全部步骤。注意：需已安装 ffmpeg（Ubuntu 14.10 及更早为 libav-tools）。

NameRX 的 fork 提供了一个更高级的版本，把光流计算与视频风格化并行以提速：[NameRX/artistic-videos](https://github.com/NameRX/artistic-videos)。

## FAQ

常见问题列表见[这里](https://github.com/manuelruder/artistic-videos/issues?q=label%3Aquestion)。

## 高级用法（Advanced Usage）

请阅读脚本 `stylizeVideo.sh` 了解需要预先完成哪些步骤。基本上：把视频各帧存成单独的图片文件，并计算所有相邻帧之间的光流及其置信度（两者都可用 `makeOptFlow.sh` 完成）。

算法有单遍和多遍两个版本。多遍版在强相机运动下效果更好，但每帧迭代更多。

基本用法：

```
th artistic_video.lua <参数> [-args <文件名>]
```

```
th artistic_video_multiPass.lua <参数> [-args <文件名>]
```

参数可通过命令行给出，也可写在文件里（一行一个），用 `-args` 指定该文件路径。命令行参数会覆盖文件里的同名参数。

**基本参数**：
* `-style_image`：风格图。
* `-content_pattern`：视频各帧的文件路径模式，例如 `frame_%04d.png`。
* `-num_images`：帧数。设为 `0` 处理全部可用帧。
* `-start_number`：首帧索引。默认 1。
* `-gpu`：要使用的 GPU 的零基索引；CPU 模式设为 -1。

**单遍算法参数**（仅 `artistic_video.lua` 有）：
* `-flow_pattern`：存储帧间后向光流的文件路径模式。方括号占位符指光流的起始帧位置，花括号占位符指光流指向的帧索引。例如 `flow_[%02d]_{%02d}.flo` 表示文件名形如 *flow_02_01.flo*、*flow_03_02.flo* 等。用本仓库脚本（makeOptFlow.sh）时模式为 `backward_[%d]_{%d}.flo`。
* `-flowWeight_pattern`：光流场权重/置信度的文件路径模式。这些文件应为灰度图，白像素表示高权重、黑像素表示低权重。格式同上。用脚本时模式为 `reliable_[%d]_{%d}.pgm`。
* `-flow_relative_indices`：长时一致性约束的索引，逗号分隔，相对当前帧。例如 `1,2,4` 表示对位置 *i* 的当前帧，使用 warp 后的 *i-1*、*i-2*、*i-4* 帧作为一致性约束。默认值 1，即只用短时一致性。使用非默认值时需相应地计算长时光流。

**多遍算法参数**（仅 `artistic_video_multiPass.lua` 有）：
* `-forwardFlow_pattern`：前向光流的文件路径模式。格式同 `-flow_pattern`。
* `-backwardFlow_pattern`：后向光流的文件路径模式。格式同上。
* `-forwardFlow_weight_pattern`：前向光流权重的文件路径模式。格式同上。
* `-backwardFlow_weight_pattern`：后向光流权重的文件路径模式。格式同上。
* `-num_passes`：遍数。默认 15。
* `-use_temporalLoss_after`：从指定遍开始（含该遍）启用时序一致性损失。默认 `8`。
* `-blendWeight`：上一风格化帧的混合系数。该值越大，时序一致性越强。默认值 `1`，即上一风格化帧与当前帧等权混合。

**优化选项**：
* `-content_weight`：内容重建项的权重。默认 5e0。
* `-style_weight`：风格重建项的权重。默认 1e2。
* `-temporal_weight`：时序一致性损失的权重。默认 1e3。设为 0 可禁用时序一致性损失。
* `-temporal_loss_criterion`：时序一致性损失使用的误差函数。可为 `mse`（均方误差）或 `smoothl1`（[smooth L1 criterion](https://github.com/torch/nn/blob/master/doc/criterion.md#nn.SmoothL1Criterion)）。
* `-tv_weight`：全变差（TV）正则的权重，有助于平滑图像。默认 1e-3。设为 0 可禁用 TV 正则。
* `-num_iterations`：
  * 单遍：两个逗号分隔的值，分别为首帧和后续帧的最大迭代数。默认 2000,1000。
  * 多遍：单个值，表示*每遍*的迭代数。
* `-tol_loss_relative`：若损失函数在 `tol_loss_relative_interval` 次迭代区间内的相对变化低于此阈值则停止。默认 `0.0001`，即损失在该区间变化小于 0.01% 时优化器停止。在默认区间下有意义的取值在 `0.001` 到 `0.0001` 之间。
* `-tol_loss_relative_interval`：见上。默认值 `50`。
* `-init`：
  * 单遍：两个逗号分隔的值，分别为首帧和后续帧的初始化方法；取 `random`、`image`、`prev` 或 `prevWarped` 之一。默认 `random,prevWarped`：首帧用噪声初始化、后续帧用 warp 后的上一风格化帧。`image` 用内容帧初始化；`prev` 用未 warp 的上一风格化帧初始化。
  * 多遍：单个值，`random` 或 `image`。
* `-optimizer`：优化算法，`lbfgs` 或 `adam`，默认 `lbfgs`。L-BFGS 效果通常更好但更占内存；换成 ADAM 可降低内存，但通常需调整其他参数（尤其风格权重、内容权重和学习率）才能得到好结果，使用 ADAM 时可能还需归一化梯度。
* `-learning_rate`：ADAM 优化器的学习率。默认 1e1。
* `-normalize_gradients`：加上此开关，则各层的风格与内容梯度做 L1 归一化。思路来自 [andersbll/neural_artistic_style](https://github.com/andersbll/neural_artistic_style)。

**输出选项**：
* `-output_image`：输出图名。默认 `out.png`，单遍产出形如 *out-\<帧号\>.png*、多遍产出形如 *out-\<帧号\>_\<遍号\>.png*。
* `-number_format`：输出图使用的数字格式。例如 `%04d` 最多补三个前导零。有用户反映 ffmpeg 在某些情况下按字典序排序，因此无前导零会导致输出帧合成顺序错误。默认 `%d`。
* `-output_folder`：保存输出图的目录，必须以斜杠结尾。
* `-print_iter`：每 `print_iter` 次迭代打印一次进度。设为 0 可禁用打印。
* `-save_iter`：每 `save_iter` 次迭代保存一次图像。设为 0 可禁用中间结果保存。
* `-save_init`：加上此选项则保存初始化图像。

**其他参数**：
* `-content_layers`：用于内容重建的层名列表，逗号分隔。默认 `relu4_2`。
* `-style_layers`：用于风格重建的层名列表，逗号分隔。默认 `relu1_1,relu2_1,relu3_1,relu4_1,relu5_1`。
* `-style_blend_weights`：多张风格图的风格混合权重，逗号分隔，如 `-style_blend_weights 3,7`。默认各风格图等权。
* `-style_scale`：相对于内容视频尺寸，从风格图提取特征的缩放比例。默认 `1.0`。
* `-proto_file`：VGG Caffe 模型的 `deploy.txt` 路径。
* `-model_file`：VGG Caffe 模型的 `.caffemodel` 路径。默认为原始 VGG-19 模型；也可试试论文中用的归一化 VGG-19 模型。
* `-pooling`：池化层类型，`max` 或 `avg`。默认 `max`。VGG-19 使用最大池化，但 Gatys 等人指出把这些层换成平均池化可能改善效果；作者用平均池化未能得到好结果，但仍保留此选项。
* `-backend`：`nn`、`cudnn` 或 `clnn`。默认 `nn`。`cudnn` 需 [cudnn.torch](https://github.com/soumith/cudnn.torch)，可降低显存。`clnn` 需 [cltorch](https://github.com/hughperkins/cltorch) 和 [clnn](https://github.com/hughperkins/clnn)。
* `-cudnn_autotune`：使用 cuDNN 后端时加上此开关，启用内置 autotuner 为你的架构选择最优卷积算法。会让首次迭代稍慢、占用内存稍多，但可能显著加速 cuDNN 后端。

## 致谢（Acknowledgement）

* 本工作受 Leon A. Gatys、Alexander S. Ecker 和 Matthias Bethge 的论文《[A Neural Algorithm of Artistic Style](http://arxiv.org/abs/1508.06576)》启发，该文提出了静态图像的风格迁移方法。
* 我们的实现基于 Justin Johnson 的实现 [neural-style](https://github.com/jcjohnson/neural-style)。

## 引用（Citation）

若在研究中使用本代码或其部分，请引用以下论文：

```
@inproceedings{RuderDB2016,
  author = {Manuel Ruder and Alexey Dosovitskiy and Thomas Brox},
  title = {Artistic Style Transfer for Videos},
  booktitle = {German Conference on Pattern Recognition},
  pages     = {26--36},
  year      = {2016},
}
```
