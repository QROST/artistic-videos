# 01 · 目标架构

> 基于 `00-overview.md` 的决策：Python + PyTorch + MPS，分阶段，CLI 交付。
> 本文定义 Phase 1 的目标架构；Phase 2 在此之上扩展（见 `04-phase2-plan.md`）。

## 1. 设计原则

1. **框架无关的核心**：损失、光流、warp、一致性等核心算法只依赖 PyTorch 张量接口，不散落 MPS 专属代码，便于将来单点替换后端。
2. **设备抽象**：`mps | cuda | cpu` 由设备层统一选择，算法代码不写死设备。
3. **与参考实现可对照**：每个模块都对应 `02-migration-map.md` 中的一段 Lua 源，便于 parity 验证。
4. **配置即数据**：用 dataclass 描述全部参数，CLI/配置文件/Python API 三种入口共享同一份配置定义。
5. **Phase 1 / Phase 2 同壳**：CLI 与 I/O 层共用，引擎通过 `--engine optim|diffusion` 切换。

## 2. 目标目录结构

```
artvid/                      # 新的 Python 包（与旧 Lua 文件共存）
  __init__.py
  config.py                  # dataclass 配置 + 解析（argparse / tyro）
  device.py                  # 设备与 dtype 选择（mps/cuda/cpu），autocast 策略
  io/
    video.py                 # ffmpeg 抽帧 / 合成（imageio-ffmpeg 或 subprocess）
    image.py                 # 读写 + preprocess/deprocess（caffe BGR 均值方案）
    flow_io.py               # Middlebury .flo 读写（与旧格式互通）
  models/
    vgg.py                   # VGG-19 特征提取器；relu*_* 层名 → 索引映射；avg/max pool 切换
  losses/
    content.py               # ContentLoss
    style.py                 # StyleLoss + gram_matrix
    tv.py                    # TVLoss
    temporal.py              # WeightedContentLoss（光流 warp 时序损失）
  flow/
    raft.py                  # RAFT 封装（torchvision.models.optical_flow）
    warp.py                  # 反向 warp（grid_sample）+ 遮挡区填充
    consistency.py           # 前/后向一致性 → 可靠性 mask（取代 consistencyChecker）
  optim/
    lbfgs.py                 # torch.optim.LBFGS + 相对损失变化停止准则
    runner.py                # runOptimization 等价物（驱动一帧的优化）
  pipeline/
    singlepass.py            # 单遍：逐帧顺序风格化（含时序 init + loss）
    multipass.py             # 多遍：前后向多趟 + blend
  cli.py                     # `artvid` / `stylize-video` 入口
tests/
  test_losses.py  test_flow.py  test_warp.py  test_parity.py ...
docs/
example/                     # 复用旧仓库的示例帧做回归
```

## 3. 模块职责与关键实现点

### 3.1 `config.py`
- 一个 `@dataclass Config`，字段对齐旧 `cmd:option`（见迁移映射表）。
- 支持「首帧 vs 后续帧」分别配置（旧版 `num_iterations`、`init` 用逗号分隔），用 `tuple` 或 `(first, subsequent)` 表达。
- CLI、`-args` 文件、Python 调用共享同一 dataclass。

### 3.2 `device.py`
- `pick_device()`：优先 `mps`，否则 `cuda`，否则 `cpu`。
- 建议默认设置 `PYTORCH_ENABLE_MPS_FALLBACK=1`，对个别未实现的算子回退 CPU。
- dtype 策略：**被优化的图像变量保持 `float32`**（L-BFGS 数值稳定性）；VGG 前向可选 `float16`/autocast 提速，标注为实验项。

### 3.3 `io/`
- `image.py`：`preprocess`/`deprocess` 复刻 caffe 约定——RGB↔BGR、均值 `[103.939,116.779,123.68]`、缩放 0–255（见旧 `artistic_video_core.lua:475-492`）。**这是 parity 的关键**，必须与所选 VGG 权重的期望输入一致。
- `video.py`：抽帧（`ffmpeg -i in.mp4 frame_%04d.ppm`）与合成（`ffmpeg -i out-%04d.png out.mp4`），封装旧 `stylizeVideo.sh` 的首尾两步。
- `flow_io.py`：保留 `.flo` 读写以兼容外部光流；注意旧 `flowFileLoader.lua` 为配合 `image.warp` 做了 (y,x) 维度交换，新实现自定义约定并在 I/O 边界转换。

### 3.4 `models/vgg.py`
- 提供 `relu1_1, relu2_1, relu3_1, relu4_1, relu5_1`（风格）与 `relu4_2`（内容）的命名访问。
- 把 torchvision VGG-19 的 `features` 索引映射到这些 relu 名。
- 支持 `pooling=avg|max`：把 `MaxPool2d` 替换为 `AvgPool2d`（旧版同样做了此替换，`buildNet` 中）。
- **一次前向、多点取特征**：注册 forward hook 或切片子网络，避免重复前向。

### 3.5 `losses/`
- `style.py`：`gram_matrix(feat)` = `F·Fᵀ`，除以元素数；`StyleLoss` 缓存目标 Gram，前向算 MSE。复刻旧 `GramMatrix` / `StyleLoss`。
- `content.py`：目标特征的 MSE。
- `tv.py`：各向异性全变差，复刻旧 `TVLoss:updateGradInput` 的差分形式（可直接用 autograd 表达，无需手写反向）。
- `temporal.py`：`WeightedContentLoss(target=warped_prev, weights=reliability)`。注意旧版对权重取 `sqrt` 后乘进 MSE（因为 `(w·err)² = w²·err²`，见 `artistic_video_core.lua:300`），新实现保持等价语义。
- **现代化简化**：旧版用 in-place nn.Module 把损失插进网络再 backward；PyTorch 直接对图像变量 `loss.backward()` 即可，**不需要把损失模块插入网络**。这会大幅简化 `artistic_video.lua` 里那段「逐帧 insert/remove 损失层」的复杂逻辑。

### 3.6 `flow/`
- `raft.py`：用 `torchvision.models.optical_flow.raft_large` 直接在 GPU/MPS 上算前向与后向光流，**取代 DeepFlow + DeepMatching 两个 CPU 二进制**。
- `warp.py`：用 `F.grid_sample` 做反向 warp（把前一帧/前一输出按后向光流采样到当前帧坐标）。遮挡/越界区域用 padding + 有效性 mask 处理，复刻旧 `warpImage` 用 VGG 均值像素填充的行为。
- `consistency.py`：前后向光流一致性检查生成可靠性权重（occlusion mask），**取代 `consistencyChecker/`**。复刻 `processFlowWeights` 的 `closestFirst` / `normalize` 长时加权方案。

### 3.7 `optim/`
- `lbfgs.py`：用 `torch.optim.LBFGS`（`line_search_fn="strong_wolfe"`），并实现旧 `lbfgs.lua` 增加的**相对损失变化停止准则**（`tol_loss_relative`、`tol_loss_relative_interval`）。也支持 `adam`。
- `runner.py`：等价于旧 `runOptimization` 的 `feval` 闭包——前向取各损失、求和、`backward`、按 `print_iter`/`save_iter` 打印与存盘。

### 3.8 `pipeline/`
- `singlepass.py`：复刻 `artistic_video.lua` 主循环——逐帧；首帧用 `random|image` 初始化，后续帧用 `prevWarped`；按 `flow_relative_indices` 取多个历史帧加时序损失；产出 `out-%04d.png`。
- `multipass.py`：复刻 `artistic_video_multiPass.lua`——整段序列前后向多趟（`num_passes`），用前/后向光流与 `blendWeight` 混合，`use_temporalLoss_after` 之后启用时序损失。

### 3.9 `cli.py`
- 子命令：
  - `artvid flow <frames_dir>`：算光流+可靠性（取代 `makeOptFlow.sh`）。
  - `artvid stylize <frames_dir> <style.jpg>`：核心风格化（单遍/多遍由 `--passes` 或 `--multipass` 切换）。
  - `artvid run <video> <style.jpg>`：端到端一条龙（取代 `stylizeVideo.sh`）。
- `--engine optim`（Phase 1 默认）/ `--engine diffusion`（Phase 2）。

## 4. 数据流（Phase 1，单遍）

```
video.mp4
  └─(ffmpeg 抽帧)→ frame_0001.ppm … frame_NNNN.ppm
                      │
                      ├─(RAFT)→ 前向/后向光流  ─┐
                      │                         ├─(consistency)→ reliability mask
                      │                         │
   style.jpg ─(VGG Gram)→ 风格目标             │
                      │                         │
   逐帧:  init(首帧 random/image; 后续 prevWarped)
            │
            └─ L-BFGS 优化像素:  内容损失 + 风格损失 + TV损失 + 时序损失(warped_prev, mask)
                 │
                 └→ out_0001.png … out_NNNN.png ─(ffmpeg 合成)→ stylized.mp4
```

## 5. parity（与 2016 输出的一致性）决策

> 这是最重要的一个工程权衡，单列说明。

风格迁移结果对**用哪个 VGG-19 权重**很敏感。两个选项：

| 方案 | 优点 | 缺点 |
|---|---|---|
| **A. torchvision VGG-19（RGB, ImageNet 归一化）** | 一行下载、纯 PyTorch、长期可维护 | 与 2016 caffe 权重数值不同，结果会有可见差异 |
| **B. 原始 caffe VGG-19 权重（BGR, 均值减法）** | 可逼近逐比特复刻 | 需转换/加载 caffe 权重，维护负担 |

**决策**：默认用 **A**（torchvision），把它作为长期主线；同时提供一个**可选的 caffe 权重加载路径（B）**，给需要忠实复现 2016 论文数值的用户。`00-overview.md` 的非目标已声明我们追求**视觉等价/更好**而非逐比特一致。两套权重对应的 `preprocess` 必须分别正确（A 用 ImageNet mean/std；B 用 BGR + caffe 均值）。

## 6. MPS / 统一内存相关注意点

- `grid_sample`、`conv2d` 等在近版 PyTorch 的 MPS 后端均支持；个别算子可能回退 CPU，用 `PYTORCH_ENABLE_MPS_FALLBACK=1` 兜底，并在基准中记录回退点。
- L-BFGS 的历史向量在 host 上维护，张量运算在 MPS 上，正常。
- **统一内存（无独立显存）**：Apple Silicon 的 GPU 与 Mac 共享同一份 RAM，没有单独的 VRAM 预算。旧 CUDA 版受 4–12 GB 显存限制；这里的上限只是你这台 Mac 的 RAM 减去系统/应用占用——RAM 越大可用的分辨率与余量越高，RAM 越小则用更低分辨率/更轻的设置，但仍能跑。把分辨率作为 CLI 参数，默认原分辨率，不再像旧版那样强制降分辨率。各内存档位的具体取舍见 quickstart 的 [Memory & RAM considerations](08-quickstart.md#memory--ram-considerations)。
- 性能基准应在你的 Apple Silicon Mac（任意支持 MPS 的 M 系列芯片）上记录：每帧迭代数、每帧墙钟、峰值内存、是否触发 CPU 回退。这些数字均为估计、从未在真实硬件上验证过。
