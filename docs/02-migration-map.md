# 02 · 迁移映射：Lua → PyTorch

> 逐文件、逐函数把现有 Torch7/Lua 实现映射到 `01-architecture.md` 定义的 PyTorch 模块。
> 这是实现和 parity 验证的对照表：每个新模块都能指回它复刻的那段旧源。

## 1. 文件级映射

| 旧文件（Lua/Torch7） | 新模块（Python/PyTorch） | 说明 |
|---|---|---|
| `artistic_video.lua` | `artvid/pipeline/singlepass.py` + `artvid/config.py` + `artvid/cli.py` | CLI 解析、主循环、warp、长时权重 |
| `artistic_video_multiPass.lua` | `artvid/pipeline/multipass.py` | 多遍前后向流程 |
| `artistic_video_core.lua` | `artvid/optim/runner.py` + `artvid/losses/*` + `artvid/models/vgg.py` + `artvid/io/image.py` | 优化循环、损失模块、建网、预处理、I/O |
| `lbfgs.lua` | `artvid/optim/lbfgs.py` | L-BFGS + 相对损失停止准则 |
| `flowFileLoader.lua` | `artvid/io/flow_io.py` | `.flo` 读取（含 y,x 维度交换语义） |
| `makeOptFlow.sh` | `artvid/flow/raft.py` + `artvid/flow/consistency.py` + `artvid cli flow` | 光流生成（RAFT 取代 DeepFlow） |
| `run-deepflow.sh` | — | 删除：被 RAFT 取代 |
| `consistencyChecker/` (C++) | `artvid/flow/consistency.py` | 前后向一致性 → 可靠性 mask |
| `stylizeVideo.sh` | `artvid cli run` | 端到端一条龙 |
| `models/VGG_ILSVRC_19_layers*` | `artvid/models/vgg.py`（torchvision 权重，可选 caffe） | 见架构文档 parity 小节 |

## 2. 函数级映射（核心逻辑）

### `artistic_video_core.lua`

| 旧函数 / 类 | 行号 | 新归属 | parity 备注 |
|---|---|---|---|
| `runOptimization` | 10–134 | `optim/runner.py:run_optimization` | `feval` 闭包→PyTorch closure；按 `print_iter`/`save_iter` 打印存盘 |
| `buildNet` | 137–250 | `models/vgg.py:build_feature_net` + `losses/style.py` | 现代化：**不再把损失模块插进网络**，改为 hook 取特征后在外部算损失 |
| `ContentLoss` | 257–288 | `losses/content.py` | 直接 MSE，可选 L1 归一化梯度 |
| `WeightedContentLoss` | 291–347 | `losses/temporal.py` | 权重先 `sqrt` 再乘进 MSE（保持 `w²·err²` 语义，见旧 300 行注释） |
| `GramMatrix` | 351–360 | `losses/style.py:gram_matrix` | `F·Fᵀ`，除以元素数 |
| `StyleLoss` | 364–397 | `losses/style.py:StyleLoss` | 缓存目标 Gram；MSE×strength |
| `TVLoss` | 400–430 | `losses/tv.py` | 旧版手写差分反向；新版可用 autograd 表达各向异性 TV |
| `preprocess`/`deprocess` | 475–492 | `io/image.py` | caffe 约定：RGB↔BGR、均值 `[103.939,116.779,123.68]`、×256 |
| `save_image` | 494–498 | `io/image.py:save_image` | deprocess + minmax 裁剪到 [0,1] |
| `getStyleImages` | 589–610 | `io/image.py` + `pipeline` | 风格图按内容图面积×`style_scale` 缩放 |
| `getContentImage` | 580–587 | `io/image.py` | |
| `build_OutFilename` / `getFormatedFlowFileName` | 558–578 | `io/`（文件名工具） | 保留 `[from]`/`{to}` 占位约定以兼容外部光流 |
| `getContentLossModuleForLayer` / `getWeightedContentLossModuleForLayer` | 432–455 | 由 `losses/*` + `runner` 内联 | 现代化后不需要「按层切子网」 |

### `artistic_video.lua`

| 旧逻辑 | 行号 | 新归属 | 备注 |
|---|---|---|---|
| `cmd:option` 全部参数 | 11–70 | `config.py` | 见下「参数映射」 |
| 主帧循环 | 137–286 | `pipeline/singlepass.py` | |
| 历史帧集合 `J` / `flow_relative_indices` / `use_flow_every` | 159–187 | `pipeline/singlepass.py` | 长时一致性：选取并排序历史帧 |
| 逐帧 insert/remove 损失层 | 190–229, 268–281 | **删除**（现代化） | PyTorch 直接对图像变量 backward，无需插层 |
| 初始化 `random/image/prev/prevWarped/first` | 231–254 | `pipeline/singlepass.py:init_image` | `prevWarped` 用光流 warp 前一输出 |
| `warpImage`（含遮挡填充） | 291–304 | `flow/warp.py` | `grid_sample` + 越界→VGG 均值像素 |
| `processFlowWeights`（`normalize`/`closestFirst`） | 307–329 | `flow/consistency.py:combine_longterm_weights` | 长时权重合并 |
| `-args` 文件解析 | 331–357 | `config.py` | 保留「参数文件，一行一参数」 |

### `artistic_video_multiPass.lua`

| 旧逻辑 | 行号 | 新归属 |
|---|---|---|
| 多遍参数 `blendWeight`/`num_passes`/`use_temporalLoss_after` | 35–40 | `config.py` |
| 前/后向光流双 pattern | 25–33 | `flow/raft.py`（直接生成两向） |
| 前后向交替遍 + blend | 主循环 | `pipeline/multipass.py` |

### `lbfgs.lua`
- 旧版在标准 optim.lbfgs 上加了「相对损失变化低于阈值则停止」。
- 新版：`torch.optim.LBFGS` 外层循环里实现同一停止准则（每 `tol_loss_relative_interval` 步比较相对变化）。

### `flowFileLoader.lua`
- Middlebury `.flo`：4字节 tag `PIEH`(202021.25) + W(int) + H(int) + W*H*2 float。
- 旧版为配合 `image.warp` 把通道存成 (y,x)。新 `flow_io.py` 按标准 (u,v)=(x,y) 读，warp 层内部再转 `grid_sample` 约定。**这是易错点，单独写单测**。

## 3. 参数映射（`cmd:option` → `Config` 字段）

保持同名、同默认值，便于老用户迁移。重点项：

| 旧参数 | 默认 | 新字段 | 备注 |
|---|---|---|---|
| `-content_weight` | 5e0 | `content_weight` | |
| `-style_weight` | 1e2 | `style_weight` | |
| `-temporal_weight` | 1e3 (单遍) / 5e2 (多遍) | `temporal_weight` | |
| `-tv_weight` | 1e-3 | `tv_weight` | |
| `-num_iterations` | `2000,1000` | `num_iterations: (first, subseq)` | 逗号分隔→元组 |
| `-init` | `random,prevWarped` | `init: (first, subseq)` | |
| `-optimizer` | `lbfgs` | `optimizer` | `lbfgs\|adam` |
| `-content_layers` | `relu4_2` | `content_layers` | |
| `-style_layers` | `relu1_1,…,relu5_1` | `style_layers` | |
| `-temporal_loss_criterion` | `mse` | `temporal_criterion` | `mse\|smoothl1` |
| `-flow_relative_indices` | `1` | `flow_relative_indices` | 长时一致性 |
| `-pooling` | `max` | `pooling` | `max\|avg` |
| `-gpu` / `-backend` | 0 / nn | **删除**，由 `device.py` 取代 | `--device mps\|cuda\|cpu` |
| `-proto_file`/`-model_file` | caffe | `vgg_weights` | `torchvision\|<caffe路径>` |
| `-tol_loss_relative` / `-..._interval` | 1e-4 / 50 | 同名 | 停止准则 |

## 4. 被删除/取代的东西

- `run-deepflow.sh`、`consistencyChecker/`（C++）、DeepFlow/DeepMatching 二进制 → 全部由 `flow/`（RAFT + grid_sample + 一致性）取代。
- `-gpu`/`-backend`/`-cudnn_autotune`/loadcaffe → 由 `device.py` + torchvision 取代。
- 「逐帧把损失 nn.Module 插入/移除网络」的整套机制 → PyTorch autograd 直接 backward，删除。

## 5. 现代化带来的简化（重要）

旧代码约一半复杂度来自 Torch7 的限制：手动把损失模块插进 `nn.Sequential`、手动管理插入索引（`additional_layers`）、手写各损失的 `updateGradInput`。PyTorch 的 autograd 让这些全部消失——

- 损失只需 `forward` 返回标量，反向自动。
- 取多层特征用 forward hook，一次前向拿全部。
- 被优化对象就是一个 `requires_grad=True` 的图像张量。

预期新核心代码量显著小于旧版，且更易读、易测。
