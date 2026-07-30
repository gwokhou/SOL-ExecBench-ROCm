# gfx1200 微架构诊断纵向闭环实施计划

## Summary

首版实现从 SOLAR 语义工作量、编译后 ISA、rocprofv3 counter 和 canonical runtime 到 `T_pred(IR/HW)`、`C/R`、结构化 Agent 反馈的完整闭环。

保持以下边界：

- `T_SOL`、正式计分和 canonical runtime 语义不变。
- 性能预测全部标记为 diagnostic-only，不进入 SOL Score。
- `L` 仅在调用者提供可信 frontier 时计算，否则显式 `unavailable`。
- 首版不依赖 TraceLens；直接使用 rocprofv3、rocprofv3-avail、AMD SMI 和现有 AMD ISA 工具。
- 首版支持 elementwise、transpose、reduction/norm、matmul 四类 workload；CrossEntropy、复杂图和 TraceLens adapter 后置。

## Key Changes

### 1. 建立统一证据契约

新增 `core/bench/performance_model/`，定义以下严格、冻结、`extra="forbid"` 的类型：

- `SemanticCharacterization`：从经验证的 SOLAR request manifest 及其
  `solar-analysis.yaml` 提取 workload UUID、typed workload descriptor、
  resource work、融合区域、语义 FLOP/Byte 和 `T_SOL`。
- `CompiledCharacterization`：code-object hash、kernel symbol、ISA functional-group counts、WMMA/VALU 类型、VGPR/LDS/scratch footprint。
- `DispatchEvidence`：dispatch/correlation ID、kernel、grid/workgroup、iteration、counter pass、动态指令/流量/cache/LDS/wave 计数。
- `DiagnosticCalibrationProfile`：GPU/ROCm/compiler/clock 身份、参数值、置信区间、适用区间，以及 probe、tuning 和独立 parameter-estimation 证据 hash。
- `PerformancePrediction`：状态、预测时间、置信区间、各资源/dispatch component、模型版本和限制。
- `PerformanceTimingEvidenceSidecar`：保存 canonical timing 的 trial/iteration
  原始样本和固定 seed、10,000 replicate 的 hierarchical-bootstrap 区间。
- `PerformanceReplayEvidenceSidecar`：绑定 canonical input hash、独立 replay
  进程、ROCTx marker、cache policy、跨 pass dispatch 序列及 pre/post AMD SMI
  环境快照。
- `PerformanceEvidenceManifest`：以 SHA256 绑定 definition、workload、
  solution、编译命令/compiler、code object、GPU/ROCm/clock、Trace、timing、
  static ISA、counter CSV、ROCPD 和 counter provenance。
- `PerformanceDiagnosticSidecar`，schema
  `sol_execbench.performance_diagnostic.v3`：保存 `T_SOL`、`T_pred(IR)`、
  `T_pred(HW)`、`T_measured` 区间、可选 `T_frontier`、`L/C/R`、归因、
  行动建议及全部证据引用。

通过 `solar_bridge` 读取并验证 SOLAR artifact；其他外层模块不得直接 import `solar`。

所有身份不一致、缺失 counter、unsupported op 或跨 pass 对齐失败均产生 `partial/unavailable + reason_codes`，不得猜测或静默回退。

### 2. 扩展 rocprofv3 为显式 counter 模式

保留现有 `--profile rocprofv3` 行为，新增显式的：

```text
--profile rocprofv3-counters
```

该模式：

- 要求 `--workload-uuid` 精确选择一个 workload、`--output` 和
  `--static-evidence auto`；
- 先完成唯一 canonical run，再进行 diagnostic-only counter replay；
- 使用 `rocprofv3-avail` 检查当前 gfx1200/ROCm 实际支持的 counter。
- 从版本化 gfx1200 manifest 选择四个经真实 `pmc-check` 验证的 counter
  groups：`SQ_WAVES_sum`、完整 memory traffic、cache hit/miss、LDS conflict
  percentage；首版不建立 occupancy/residency 模型。
- 生成受控 YAML job，使用 CSV 作为规范化输入，同时保留 rocpd 作为审计引用。
- 记录 profiler、counter definition、配置文件和可执行文件 SHA256。
- 将多 pass 结果按 workload、candidate hash、kernel、grid/workgroup 和 iteration ordinal 对齐；任何不一致使相关 dispatch 失效。
- canonical Trace 时间仍是 `T_measured`；profiler duration 和由 duration 推导的 achieved rate禁止进入 `T_pred(HW)`。
- dispatch 缺 queue/stream identity、多 queue/stream 或 timestamp overlap
  均返回 `partial/unavailable`，首版不建 overlap 模型。

扩展现有 bounded parser，支持官方 `Counter_Name`、`Counter_Value`、`Dispatch_Id`、grid/workgroup、timestamp 等字段及 ROCm 版本差异。

### 3. 扩展 gfx1200 校准参数包

复用现有 `hardware_calibration_probes`、锁频、ISA 验证、tuning/parameter-estimation、bootstrap 和 content-addressed audit 流程，增加：

- 空 kernel/device dispatch floor；
- L2/L3/VRAM working-set 带宽转折；
- contiguous 和真实 2D transpose 访问效率；
- LDS 正常访问与 bank conflict；
- reduction/barrier 宽度 sweep；
- irregular WMMA tile、边缘 tile 和 active-wave sweep。

冻结配置后重新采集独立 parameter-estimation batches。验收 held-out 样本
与 tuning/parameter-estimation 严格分离。校准 artifact 必须绑定 GPU UUID、
BDF、ROCm、hipcc、code object、时钟和功耗状态；不匹配时预测不可用。

首版不实现 gather/scatter/atomic、完整 cache latency、跨 kernel overlap 和任意图模型。

### 4. 实现预测和归因

`T_pred(IR)`：

- 使用 SOLAR 融合区域作为逻辑 dispatch。
- 使用语义工作量与相同的 gfx1200 诊断参数。
- 每个区域计算固定 dispatch cost 与 compute/memory/LDS/reduction component。
- 仅支持首版四类 workload；其他类型返回 partial。

`T_pred(HW)`：

- 以实际 dispatch 分解、ISA、footprint 和动态 counter 为输入。
- 单 dispatch 采用校准后的固定成本与资源瓶颈组合；多 dispatch 仅在
  同一已验证 queue/stream 且无 overlap 时按时间顺序求和。
- 首版不支持 interval union；任何 overlap 证据均 fail closed。
- 禁止读取 profiler duration、canonical runtime 或同一 candidate 的 achieved throughput。

归因规则：

```text
C = T_pred(HW) / T_pred(IR)
R = T_measured / T_pred(HW)
L = T_frontier / T_SOL  # 仅可信 frontier 可用时
```

- `C` 高：输出 fusion、padding、额外流量、spill、barrier、未生成 WMMA 等 implementation/codegen 原因。
- `R` 高：输出 cache、访问效率、LDS、调度或模型缺项原因。
- `L` 无 frontier：状态为 unavailable，不得用 scoring baseline 替代。
- 任一比值显著小于 1：优先输出 identity/model contradiction，不产生“超越模型”的优化建议。
- 阈值必须结合 prediction interval 和 canonical timing noise；差异未超过联合不确定度时输出 inconclusive。

### 5. CLI与 Agent 闭环

新增顶层命令：

```text
sol-execbench diagnostics performance \
  --evidence-manifest TRACE.jsonl.performance-evidence.json \
  --solar-manifest SOLAR_REQUEST/manifest.yaml \
  [--frontier-trace TRACE.jsonl] \
  [--calibration-profile CALIBRATION.json] \
  [--inference-profile INFERENCE.json] \
  --output TRACE.performance-diagnostic.json
```

命令只支持 evidence manifest 中的单 workload，并要求每个 artifact 通过
hash、run、candidate、GPU 和 workload identity 检查。GPU/compiler/power
身份不得由 CLI 手工覆盖。

新增独立命令：

```text
sol-execbench diagnostics agent-feedback \
  --performance-diagnostic DIAGNOSTIC.json \
  --evidence-manifest EVIDENCE.json \
  --acceptance ACCEPTANCE.json \
  --output FEEDBACK.json
```

开发集和 held-out 集各至少包含四类 workload 每类 20 个 case（各至少 80
个），两者按 workload/candidate pair 严格不相交。开发集冻结按 family 的
95% split-conformal 区间和动作阈值；held-out 验收后 Agent Feedback 才可
输出以下代码修改行动码：

- `stop_launch_bound_search`
- `reduce_dispatch_count`
- `restore_wmma_path`
- `remove_extra_traffic`
- `improve_coalescing`
- `reduce_lds_barriers`
- `reprofile_missing_counters`
- `model_gap_no_kernel_action`

未经验证的 `R` 只能生成重新采证或模型缺项建议，不能要求 Agent 改代码。首版不修改 score、leaderboard 或 reward 数值。

## Test Plan

### CPU与契约测试

- 严格 schema、未知字段、NaN/Inf、负时间和非法 ratio。
- SOLAR artifact hash、workload UUID、candidate/code-object 和 calibration 身份不匹配。
- rocprofv3 官方 CSV 字段、不同单位、缺 counter、unsupported counter 和多 pass 错位。
- 确认 profiler duration 与 achieved bandwidth 无法进入预测输入。
- elementwise、transpose、reduction/norm、matmul 的确定性 component 计算。
- `L` 在无 frontier 时 unavailable；scoring baseline 不得被自动采用。
- static/runtime 冲突时 runtime evidence 优先，但保留冲突和限制。
- Agent Feedback 只消费 current、usable、足够置信的诊断。
- 现有 `T_SOL`、Trace JSONL 和 scoring 保持兼容；performance diagnostic、
  calibration 及其证据契约只接受各自的 current schema，不提供旧版本
  compatibility loader。

### gfx1200硬件验收

使用 `requires_rocm` 和 `requires_rdna4` 标记，在锁频主机上验证：

| Workload | 预期归因 |
| --- | --- |
| 小 Sigmoid | dispatch-floor/launch-bound，建议停止局部 tile 搜索 |
| 两种 Transpose | 相同最低流量但访问效率不同，定位 coalescing/cache-line |
| 两种 RMSNorm | reduction 宽度、LDS/barrier 差异 |
| Irregular Matmul | WMMA 路径、padding 和边缘 tile 差异 |

在冻结、与 calibration 和开发集完全独立的 held-out 样本上，每个 workload
family 至少 20 个 case（总计至少 80 个），并要求：

- `T_pred(HW)` median absolute percentage error ≤ 15%；
- P90 absolute percentage error ≤ 30%；
- 每个 family 的区间 empirical coverage ≥ 90%；
- 每个启用的代码行动 precision ≥ 90%、recall ≥ 70%，允许 abstain；
- tuning 样本不得计入验收统计；
- profiler 或 GPU 资源在 sandbox 中不可见时，按仓库规则申请精确 host 命令重试，不得误判为硬件不支持。

最终运行相关 Pytest、完整静态类型检查、Ruff、质量门禁及 GPU 集成测试；不提高任何质量基线。

## Assumptions and Deferred Work

- 首版唯一正式目标为 RX 9060 XT/gfx1200、ROCm 7.2 兼容工具链。
- TraceLens、ROCm Compute Profiler和 PerfXpert 不成为核心依赖；预留 provider-neutral artifact 接口，后续以 `tools/` adapter 接入。
- `T_frontier` 首版由调用者显式提供可信 trace，不建设全局 frontier registry。
- CrossEntropy、MiniGPT、gather/scatter/atomic、复杂 overlap、训练 reward 更新和跨架构参数包进入后续里程碑。
- `T_SOL` 与经验校准严格隔离，任何持续吞吐、launch floor、counter 或 candidate 信息均不得进入 `solar`。
