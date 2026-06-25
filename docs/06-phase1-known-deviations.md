# 06 · Phase 1 已知偏差与待办（M0+M1 实现）

> M0/M1 两道代码审查闸门均**通过**，仅有以下**非阻塞**的 parity 偏差，记录于此，待在 M5 Max 上做运行时验证时确认/处理。
> 这些不是 bug，多为「刻意的现代化取舍」或「数值幅度差异（可被 style_weight/learning_rate 吸收）」，符合 `00-overview.md` 中「追求视觉等价而非逐比特一致」的非目标。

## 验证现状

- 全部 Python 文件通过 `py_compile`。
- **48 个不依赖 torch 的单测通过**（`.flo` 往返与轴约定、config、结构）；41 个依赖 torch 的单测在本环境按 conftest 跳过，需在装好 torch 的 M5 Max 上执行。

## M0（损失/IO/VGG/优化器）

1. **StyleLoss 反向的二次 `/nElement`（style.py，minor）**
   旧 `StyleLoss:updateGradInput`（`artistic_video_core.lua:389`）在反向里对 `dG` 再除一次 `input:nElement()`，叠加在前向除法之上。现代 autograd 路径不复现这个「仅反向」的额外因子。仅影响梯度幅度（可被 `style_weight`/优化器吸收）。若将来需要严格数值复刻，加一行缩放即可。

2. **`normalize_gradients=True` 的 L1 归一化位置（style/content/temporal.py，minor）**
   旧版对「gram 反向得到的、关于输入特征图的 gradInput」做 L1 归一化；新版对 gram 自身的梯度做归一化。三个损失彼此一致，且该开关**默认 False**。仅在显式开启时幅度不同。

3. **加权 SmoothL1 的语义（temporal.py，minor）**
   `(sqrt(w)·err)² = w·err²` 仅对 MSE 精确；对 SmoothL1 不成立。这**与旧代码行为完全一致**（旧版同样无差别地对两操作数乘 `sqrt(w)`，`lua:308-339`），属「继承而来的怪异语义」，已在 docstring 标注。

## M1（光流栈）

4. **越界像素的可靠性 seed（consistency.py，major 但非阻塞）** ← 最值得关注
   `consistencyChecker.cpp:64-65` 把「前向落点越界」的像素 seed 为 `reliable=0.0`（同运动边界），**而非**遮挡负 seed（−255）。当前实现把越界并入负 seed（`_OCCLUDED_SEED`），高斯平滑后负 seed 会沿帧边界向**内**侵蚀邻近像素的可靠性，产生与 C++ 不同的边界 halo（越界像素本身裁剪后仍为 0，差异在邻域）。
   **建议修复**：把越界像素按运动边界一样 seed 为 0，与「前后向不一致」的负 seed 分开处理。影响有界（边缘略偏保守），可在 M5 Max 验证可视化后决定是否改。

5. **RAFT transforms 的 docstring 不准确（raft.py，minor）**
   注释称 `Raft_Large_Weights.DEFAULT.transforms()` 会把空间尺寸对齐到 8 的倍数；实际该 transform 只做 dtype 转换与归一化到 [−1,1]，/8 对齐是代码里用 `F.interpolate` 手动完成的。**行为正确，仅注释误导**，需订正。

6. **一致性梯度算子为近似（consistency.py，minor）**
   用「中心差分 + replicate 边界」近似 C++ 的 Brox `CDerivative(3)` 核与 `NFilter` 边界处理。因梯度被平方+阈值用于运动边界检测（属内部特征），影响可忽略；但 docstring 写「same operator」略过强，应改为「近似」。

7. **`consistency_mask` 参数命名（consistency.py，minor）**
   形参名为 `(forward_flow, backward_flow)`，而 `cli.py` 按 `consistency_mask(backward, forward)` 调用（镜像 consistencyChecker 的 `flow1=backward, flow2=forward`）。数学在两种顺序下都正确，纯属命名易误导，建议改名为 `(flow1, flow2)` 并加注释。

## 处理优先级

- **运行时验证前必做**：在 M5 Max 上 `pip install -e ".[dev]"` 后跑 `pytest`（确认 torch 单测通过）。
- **建议在进入 M2 前修**：#4（越界 seed）、#5/#7（注释与命名，低成本）。
- **可延后**：#1/#2/#3（仅在需要严格数值复刻时）。

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
