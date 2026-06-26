# 05 · 执行计划：用 Workflow 编排 Phase 1

> 设计与文档定稿后，用多 agent workflow 执行 Phase 1 实现。本文定义 agent 拆分、依赖 DAG、并行/流水策略、验收闸门，以及 workflow 脚本骨架。
> 这样 workflow 真正运行时，脚本是照着这份计划写出来的，而不是临场拍脑袋。

## 1. 编排原则

- **里程碑即闸门**：每个里程碑（M0–M4）结束都有一个 verify agent 跑测试/demo，未过不进入下一里程碑。
- **里程碑内并行，里程碑间串行**：M0 内部若干独立模块可并行写；但 M1 依赖 M0 的设备/IO/VGG，必须等 M0 验收。
- **写文件的并行 agent 用 worktree 隔离**：同一里程碑内多个 agent 同时改文件时用 `isolation:'worktree'`，避免冲突；产物由集成 agent 合并。
- **对抗式验收**：关键正确性（warp 约定、parity、时序稳定性）用独立 verify agent 复核，而非自证。

## 2. Agent 拆分（Phase 1）

### M0 — 骨架与单图风格迁移
并行可拆为：
- `scaffold`：`pyproject.toml`、包结构、`config.py`、`device.py`。（其余依赖它，先单独跑）
- `io-image`：`io/image.py`（preprocess/deprocess/save）。
- `vgg`：`models/vgg.py`（特征提取 + 层名映射 + pool 切换）。
- `losses`：`losses/{content,style,tv}.py`。
- `optim`：`optim/{lbfgs,runner}.py`。
- `integrate-m0`：把单图风格迁移串起来跑通。
- `verify-m0`：单测 + 单图 demo，产出验收报告。

### M1 — 光流子系统
- `flow-raft`：`flow/raft.py`。
- `flow-warp`：`flow/warp.py`（grid_sample + 遮挡填充）。
- `flow-consistency`：`flow/consistency.py` + 长时权重。
- `flow-io`：`io/flow_io.py`（`.flo` 互通）。
- `cli-flow`：`artvid flow` 子命令。
- `verify-m1`：warp 残差、mask 可视化、`.flo` 往返单测、对照报告。

### M2 — 单遍管线（重心）
- `pipeline-singlepass`：逐帧顺序 + init + 时序损失 + 长时多历史帧。
- `verify-m2`：整段产出 + 帧间稳定性度量（有/无时序损失对比）。

### M3 — 多遍管线
- `pipeline-multipass`：前后向多趟 + blend。
- `verify-m3`：强运动片段稳定性对比。

### M4 — CLI 一条龙 + 基准
- `cli-run`：`artvid run` 端到端。
- `bench`：在你的 Apple Silicon Mac（任意 M 系列 / MPS）上测得的基准表（注：基准需真实硬件；workflow 在本环境只能产出脚本与占位，真实数字由用户在自己的 Apple Silicon Mac 上回填）。
- `docs-usage`：用法文档 + README 更新。
- `verify-m4`：端到端 smoke。

## 3. 依赖 DAG

```
scaffold
  ├─> io-image ─┐
  ├─> vgg ──────┤
  ├─> losses ───┼─> integrate-m0 ─> verify-m0(闸门)
  └─> optim ────┘                        │
                                         v
        flow-io ─┐                   (M1 开始)
        flow-raft ┼─> flow-warp ─> flow-consistency ─> cli-flow ─> verify-m1(闸门)
                                                                        │
                                                                        v
                                              pipeline-singlepass ─> verify-m2(闸门)
                                                                        │
                                                                        v
                                              pipeline-multipass ─> verify-m3(闸门)
                                                                        │
                                                                        v
                                  cli-run + bench + docs-usage ─> verify-m4(闸门)
```

## 4. Workflow 脚本骨架（示意）

> 实际运行时按此结构用 Workflow 工具提交。里程碑间用「verify 闸门」串联，里程碑内用 `parallel`（worktree 隔离）写文件。

```js
export const meta = {
  name: 'artvid-phase1',
  description: 'Implement Phase 1: PyTorch+MPS port of artistic-videos with RAFT',
  phases: [
    { title: 'M0-scaffold-singleimage' },
    { title: 'M1-flow' },
    { title: 'M2-singlepass' },
    { title: 'M3-multipass' },
    { title: 'M4-cli-bench' },
  ],
}

const VERIFY = { /* JSON schema: {passed:bool, failures:[], notes} */ }

// ---- M0 ----
phase('M0-scaffold-singleimage')
await agent('实现 scaffold：pyproject、config.py、device.py …', { label: 'scaffold', isolation: 'worktree' })
const m0 = await parallel([
  () => agent('实现 io/image.py …',  { label: 'io-image',  phase: 'M0-scaffold-singleimage', isolation: 'worktree' }),
  () => agent('实现 models/vgg.py …', { label: 'vgg',       phase: 'M0-scaffold-singleimage', isolation: 'worktree' }),
  () => agent('实现 losses/* …',      { label: 'losses',    phase: 'M0-scaffold-singleimage', isolation: 'worktree' }),
  () => agent('实现 optim/* …',       { label: 'optim',     phase: 'M0-scaffold-singleimage', isolation: 'worktree' }),
])
await agent('集成 M0：单图风格迁移跑通 …', { label: 'integrate-m0' })
const v0 = await agent('验收 M0：跑 pytest + 单图 demo，报告', { label: 'verify-m0', schema: VERIFY })
if (!v0.passed) return { stoppedAt: 'M0', v0 }   // 闸门

// ---- M1 ----  （结构同上：flow-io/raft/warp/consistency → cli-flow → verify-m1 闸门）
// ---- M2/M3/M4 ---- 依次推进，每个里程碑末尾 verify 闸门
```

要点：
- **schema 化验收**：verify agent 用 `schema` 强制返回 `{passed, failures, notes}`，主循环据此决定是否过闸。
- **worktree 隔离**：仅在同一 `parallel` 内多 agent 写文件时使用；集成 agent 在主工作树合并。
- **闸门即 `return`**：某里程碑未过则停下并报告，由人决定是否继续/修正，符合「人在环」。
- **基准的现实约束**：真实硬件数字无法在本 CI 环境产出；`bench` agent 只产出可复跑脚本与表格模板，数字由用户在自己的 Apple Silicon Mac（任意 M 系列 / MPS）上回填。

## 5. 启动条件（什么时候真正跑 workflow）

1. 本批设计文档已 review、合入分支。✅（本 PR）
2. 用户确认 Phase 1 范围与里程碑无异议。
3. 明确 workflow 运行的算力预期（本环境无 Apple Silicon Mac / 无 GPU，代码实现与单测可做，真机性能基准需用户侧执行）。

满足后，以「ultracode / use a workflow」显式授权方式启动 `artvid-phase1` workflow。

## 6. 范围与成本提示

- Phase 1 workflow 预计十余个 agent、多轮 verify，属中等规模编排。
- Phase 2 待 Phase 1 验收后单独规划 workflow，不与 Phase 1 混跑。
