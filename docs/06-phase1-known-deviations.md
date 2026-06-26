# 06 · Phase 1 已知偏差与待办（M0+M1 实现）

> M0/M1 两道代码审查闸门均**通过**，仅有以下**非阻塞**的 parity 偏差，记录于此，待在你的 Apple Silicon Mac（任意 M 系列 / MPS）上做运行时验证时确认/处理。
> 这些不是 bug，多为「刻意的现代化取舍」或「数值幅度差异（可被 style_weight/learning_rate 吸收）」，符合 `00-overview.md` 中「追求视觉等价而非逐比特一致」的非目标。
>
> **注意**：文中按文件名+行号引用的 Lua/C++ 源（`artistic_video*.lua`、`consistencyChecker.cpp` 等）现位于仓库的 **`legacy/`** 目录。

## 验证现状

- 全部 Python 文件通过 `py_compile`。
- **48 个不依赖 torch 的单测通过**（`.flo` 往返与轴约定、config、结构）；41 个依赖 torch 的单测在本环境按 conftest 跳过，需在装好 torch 的 Apple Silicon Mac 上执行。

## M0（损失/IO/VGG/优化器）

1. **StyleLoss 反向的二次 `/nElement`（style.py，minor）← 已文档化（FIXED-doc）**
   旧 `StyleLoss:updateGradInput`（`artistic_video_core.lua:389`）在反向里对 `dG` 再除一次 `input:nElement()`，叠加在前向除法之上。现代 autograd 路径不复现这个「仅反向」的额外因子。仅影响梯度幅度（可被 `style_weight`/优化器吸收）。若将来需要严格数值复刻，加一行缩放即可。
   **本轮处理**：在 `style.py` 模块 docstring 加了「Deferred parity note (#1)」说明此「仅反向」额外因子被刻意省略、其影响（仅梯度幅度）、以及如何用自定义 `autograd.Function` 复刻。**无行为改动。**

2. **`normalize_gradients=True` 的 L1 归一化位置（style/content/temporal.py，minor）**
   旧版对「gram 反向得到的、关于输入特征图的 gradInput」做 L1 归一化；新版对 gram 自身的梯度做归一化。三个损失彼此一致，且该开关**默认 False**。仅在显式开启时幅度不同。

3. **加权 SmoothL1 的语义（temporal.py，minor）**
   `(sqrt(w)·err)² = w·err²` 仅对 MSE 精确；对 SmoothL1 不成立。这**与旧代码行为完全一致**（旧版同样无差别地对两操作数乘 `sqrt(w)`，`lua:308-339`），属「继承而来的怪异语义」，已在 docstring 标注。

## M1（光流栈）

4. **越界像素的可靠性 seed（consistency.py，major 但非阻塞）← 已修复（FIXED）**
   `consistencyChecker.cpp:64-65` 把「前向落点越界」的像素 seed 为 `reliable=0.0`（同运动边界 cpp:84），**而非**遮挡负 seed（−255 cpp:80）。旧实现把越界并入负 seed（`occluded = occluded | (~in_bounds)`），高斯平滑后负 seed 会沿帧边界向**内**侵蚀邻近像素的可靠性，产生与 C++ 不同的边界 halo。
   **本轮修复**：在 `consistency.py:consistency_mask` 把两类 seed 分开：`inconsistent`（仅前后向往返失败 cpp:77）保留 `_OCCLUDED_SEED`（−1）负 seed；越界像素用 `torch.where(~in_bounds, ...)` seed 为 `_MOTION_BOUNDARY_SEED`（0.0），且在 fb seed **之后**应用以匹配 C++ 中 OOB 守卫先 `continue` 的优先级。运动边界检测也相应排除 `~in_bounds`。加了引用 cpp:64-65,78-81,84 的注释；模块/函数 docstring 同步订正。

5. **RAFT transforms 的 docstring 不准确（raft.py，minor）**
   注释称 `Raft_Large_Weights.DEFAULT.transforms()` 会把空间尺寸对齐到 8 的倍数；实际该 transform 只做 dtype 转换与归一化到 [−1,1]，/8 对齐是代码里用 `F.interpolate` 手动完成的。**行为正确，仅注释误导**，需订正。

6. **一致性梯度算子为近似（consistency.py，minor）← 已修复（FIXED-doc）**
   用「中心差分 + replicate 边界」近似 C++ 的 Brox `CDerivative(3)` 核与 `NFilter` 边界处理。因梯度被平方+阈值用于运动边界检测（属内部特征），影响可忽略；但 docstring 写「same operator」略过强。
   **本轮修复**：`_flow_gradient_sq` docstring 已把「same operator」改为明确的「close *approximation*」，说明内部 tap 权重一致而 `CDerivative(3)`/`NFilter` 的边界处理不同、且因平方+阈值用作内部特征故差异可忽略。**无行为改动。**

7. **`consistency_mask` 参数命名（consistency.py，minor）← 已修复（FIXED）**
   旧形参名为 `(forward_flow, backward_flow)`，而 `cli.py` 按 `consistency_mask(backward, forward)` 调用（镜像 consistencyChecker 的 `flow1=backward, flow2=forward`）。数学在两种顺序下都正确，纯属命名易误导。
   **本轮修复**：形参改名为 `(flow1, flow2)`，与 C++ `checkConsistency(flow1, flow2, ...)` 一致，并在 docstring 加注：mask 定义在 `flow1` 帧上，`flow1` 为被验证流、`flow2` 为反方向交叉校验流。全部调用点（`cli.py:279,285`、`singlepass.py:665`、`multipass.py:675`）均按位置传参，**行为不变**，无需改调用点。

## 处理优先级

- **运行时验证前必做**：在你的 Apple Silicon Mac 上 `pip install -e ".[dev]"` 后跑 `pytest`（确认 torch 单测通过）。
- **建议在进入 M2 前修**：~~#4（越界 seed）、#5/#7（注释与命名，低成本）~~ → **已修复**（见各条 FIXED 标注；本轮另修 #1/#6 文档化）。
- **可延后**：#2/#3（仅在需要严格数值复刻时）。

### 本轮 lint 清理（无行为改动）

- `multipass.py`：删除未使用的 `from dataclasses import ... field` 导入（该文件未用 `field()`）。
- `singlepass.py`：删除已失效的 `is_first` 形参 —— 自 #9 修复起 `_init_frame_image` 改按 `frame_idx > start` 门控，`is_first` 已不再被读取。同步删除调用点的 `is_first=is_first` kwarg 与循环体内未使用的 `is_first = frame_idx == first_idx` 局部变量（`first_idx` 仍用于循环范围，保留）。第 ~744 行解释性注释保留（仍正确对比「按 frame_idx 而非 is_first 门控」）。

## M2/M3/M4（管线与 CLI）

M2、M3、M4 三道代码审查闸门均通过（M4 初次发现 1 个 blocking，已修复，见下）。

8. **`artvid run` 默认调用崩溃（cli.py，blocking → 已修复）**
   `run` 复用 `stylize` 的 `--start-number`（默认 `None`），光流预计算分支把 `None` 直接传入 `_compute_flow_for_run` → `TypeError`。已改为 `start_number=args.start_number or 1`。

9. **`continue_with>1` 续跑首帧初始化错误（singlepass.py，major → 已修复）**
   旧版按 `frameIdx > start_number` 门控 prevWarped/prev 初始化；原实现按 `is_first` 门控，导致续跑（`continue_with>1`）的首个迭代帧本应 warp 上一输出却回退到 random。已改为按 `frame_idx > start_number` 门控（默认 `continue_with=1` 路径本就正确）。

10. **多遍默认值 parity（multipass.py / config.py，major，未改 — 见说明）**
    旧 `artistic_video_multiPass.lua` 的 `temporal_weight` 默认 `5e2`、`num_iterations` 默认 `100`（每遍）；共享 `Config` 沿用单遍默认 `1e3` 与 `(2000,1000)`。因 `Config` 为单/多遍共享，**未改默认值以免影响单遍**。用法文档与 docstring 已提示：跑多遍时显式传 `--temporal-weight 5e2 --num-iterations 100`。若后续决定为多遍提供独立默认，应在 CLI `--multipass` 分支按「用户未显式覆盖」时注入。

11. **时序 mask 额外 AND warp 有效性（singlepass.py，minor，刻意保留）**
    新版把可靠性 mask 再乘以 warp 的 `valid`（遮挡区），旧 `processFlowWeights` 未这样做。理由正当（被遮挡像素确实不可靠），已在代码注释；属刻意的合理偏差。

12. **`docs/usage.md` 示例风格图路径（minor → 已修复）**
    示例命令把风格图写成 `style/seated-nude.jpg`，仓库实际位于 `example/seated-nude.jpg`。已订正。

### 本轮已直接修复

#8（blocking）、#9（major）、#12（minor）已在提交中修复。#10/#11 为刻意保留/文档化的 parity 取舍。

## Phase 2 基础（扩散引擎脚手架）

> 这是**可审查的扩散基础 + 具体设计**，不是已验证可跑的管线。画质/数值必须在你的 Apple Silicon Mac（任意 M 系列 / MPS）上实跑调参（首次运行才下载模型权重）。审查闸门初次 `passed=false`，blocking 已修复。具体设计见 `docs/07-phase2-design.md`。

13. **`run --engine diffusion` 前向光流文件名顺序错误（diffusion/video.py，blocking → 已修复）**
    `_flow_pair_for` 解析预计算前向光流文件时 from/to 传反，得到 `forward_<cur>_<prev>.flo`，而 `cmd_flow` 写的是 `forward_<prev>_<cur>.flo`。在 `flow_source='precomputed'`（`run` 强制此模式）下读不存在的文件 → `FileNotFoundError`。后向方向本就正确。已把前向调用的两个 index 参数对调修复。

14. **`temporal_init_strength` 不可调（config.py / diffusion/video.py，minor → 已修复）**
    `video.py` 防御式读取 `temporal_init_strength`，但 `Config` 无此字段，永远回退默认 0.6，用户无法调 warped-init 的 img2img 强度。已在 `Config` 新增该字段（与 `denoise_strength` 区分开）。

15. **mechanism-1 init 未用 reliability（diffusion/engine.py，minor，刻意保留为 TODO）**
    `denoise_frame` 拿到 `init_latents` 时直接 `add_noise`，未做设计文档 §2.5 的「reliability 掩码 init 混合」（`reliability` 仅用于逐步融合 mechanism 2）。属合理的更简单默认（完全从 warped latent 起步），与 spec 的 mechanism 1 有出入；保留为 **TODO(tuning)**，待在硬件上看遮挡区渗漏再决定是否实现掩码 init。

### 本轮已直接修复（Phase 2）

#13（blocking）、#14（minor）已修复。#15 保留为待硬件调参的 TODO。其余大量 `TODO(tuning)`（fp16/bf16、各 scale/strength/steps、scheduler 选择等）均为**预期内**的硬件调参项，不是代码缺陷。
