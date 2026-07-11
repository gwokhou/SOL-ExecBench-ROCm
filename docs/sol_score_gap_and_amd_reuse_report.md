# SOL Score 与 NVIDIA 实现差异及可复用基础设施评估（ROCm 视角）

更新时间：2026-07-10

本报告用于回答三个问题：

1) 当前仓库实现的 SOL Score 与 NVIDIA 实现差距在哪；
2) 当前 AMD SOL bound 与正式 baseline 实际如何计算、是否已有权威工件；
3) 在 `github.com/AMD-AGI` 或 `github.com/ROCm` 下有哪些基础设施可以复用。

## 结论先行

最关键结论是：**SOL Score 的数学公式本体与 NVIDIA 实现对齐**，但“可验证性/可复用评分流程”与 NVIDIA 公开语义尚不一致，主要差异集中在：

- t_SOL 的来源模型（`SOLAR` 可比但非等价）；
- 基线（`t_b`）来源与可追溯性；
- 聚合策略与失败记账；
- 评分产出是否有官方化/可发表的 `official_score` 载荷。

换句话说：**不是公式错了，而是“评分语义闭环”还没完成 NVIDIA 公共语义等价闭环。**

进一步按当前代码和工作区现存工件审计，结论是：**当前不存在已经具备
official score authority 的 AMD SOL bound 与正式 scoring baseline 组合。**
仓库已经具备可审计的 AMD-native bound 推导、baseline schema、实测 trace
和 score gate，但默认 `gfx1200` 硬件模型仍是 provisional/inexact，正式
`scoring_baseline.v1` 也没有自动生成或已发布的完整工件。

## 当前 AMD SOL bound 与正式 baseline 的计算现状

### 1. 三个时间量及完整评分链

真正的 SOL Score 需要三个相互独立且可追溯的输入：

```text
Definition + Workload
        ↓
BoundGraph → FLOPs/Bytes → AMD hardware model
        ↓
      T_SOL

优化基准实现 → 同机实测 → T_b
候选实现     → 同机实测 → T_k

S(T_k) = 1 / [1 + (T_k - T_SOL) / (T_b - T_SOL)]
```

- `T_SOL`：目标 AMD 硬件上的理论下界；
- `T_b`：release 固定的优化 baseline 在同一 workload 上的实测延迟；
- `T_k`：候选实现的实测延迟。

`src/sol_execbench/sol_score.py` 只实现最后一步公式。它不会生成
`T_SOL` 或 `T_b`，也不会自动为输入赋予权威性。

### 2. AMD SOL bound 的当前推导方法

入口是
`src/sol_execbench/core/scoring/amd_sol/v2_builder.py::build_amd_sol_bound_v2_artifact()`。
当前链路按以下步骤生成 `sol_execbench.amd_sol_bound.v2` sidecar：

1. 将 `Definition` 与具体 `Workload` 绑定，构建包含 operator、tensor、shape、
   dtype 和数据流证据的 `BoundGraph`；
2. 将节点分类为 GEMM、attention、convolution、elementwise、reduction、
   normalization、softmax、MoE、SSM/Mamba、data movement 或 unsupported；
3. 为每个节点估算 FLOPs、读取字节、写入字节和总移动字节；
4. 将估算值代入 AMD 硬件模型，分别计算 compute bound 与 memory bound；
5. 取二者较大值作为单个 operator 的 SOL bound，再对所有 operator 求和。

单个 operator 的公式位于
`src/sol_execbench/core/scoring/amd_sol/v2_math.py`：

```text
T_compute_ms = FLOPs / (peak_tflops × 10^12) × 1000
T_memory_ms  = Bytes / (memory_bandwidth_gbps × 10^9) × 1000
T_SOL_op     = max(T_compute_ms, T_memory_ms)
T_SOL        = Σ T_SOL_op
```

例如 GEMM 使用 `2*M*N*K`，batched GEMM 使用 `2*B*M*N*K`。部分复杂
operator 仍是保守经验模型，例如 reduction 约按输入元素数计、normalization
按 `4*input_elements`、softmax 按 `5*input_elements`；这些估算会标成
`inexact`，而不是无条件视为精确证据。

这条链路是 AMD-local roofline 近似，并非 upstream SOLAR 的完整等价实现。
当前聚合是逐 operator 下界求和，不能自动表达全图融合、跨 operator 数据复用、
on-chip buffer capacity 或所有 value-dependent 优化。

### 3. 默认 gfx1200 hardware model 的权威边界

默认模型位于
`src/sol_execbench/data/amd_hardware_models/gfx1200.json`：

```json
{
  "architecture": "gfx1200",
  "dtype_or_path": "bf16/fp32 mixed benchmark path",
  "peak_tflops": 48.0,
  "memory_bandwidth_gbps": 640.0,
  "confidence": "inexact",
  "hardware_validation_status": "validated",
  "model_validation_status": "provisional"
}
```

含义是实际 RDNA4 执行路径已有验证证据，但 48 TFLOPS/640 GB/s 这组统一
BF16/FP32 roofline 输入仍是 provisional。`v2_math.py::_aggregate_for_bounds()`
只有在以下条件全部满足时才返回无降级的 `status="scored"`：

- 每个 operator estimate 都是 `supported`；
- hardware validation 为 `validated`；
- model validation 为 `validated`；
- hardware model confidence 为 `supported`。

只要存在 `inexact` operator 或 provisional hardware model，结果就是
`status="degraded"`；存在 unsupported operator 则为 `status="unscored"`。
需要特别区分：当前 `degraded` 工件中的 `scored=true` 只表示数值可供
AMD-native 派生报告使用，不等于 official score authority。

因此默认 gfx1200 模型当前最多产生带警告的 derived bound，不能单独称为
“权威 AMD SOL bound”。

### 4. 两种 baseline 不是同一个概念

| Baseline | 工件/入口 | 保存的核心量 | 当前用途 |
| --- | --- | --- | --- |
| Release scoring baseline | `sol_execbench.scoring_baseline.v1` / `--scoring-baseline` | 每个 definition + workload UUID 的优化实现实测 `latency_ms`，即 `T_b` | SOL Score 公式输入 |
| HIP measured baseline registry | `baseline_registry.v1` / `hip-agent run baseline` | 当前 target solution 的实测 latency、reference latency、speedup 与环境 provenance | 覆盖率、同环境比较和 release gate |

#### Release scoring baseline

正式 `T_b` 的 schema 在
`src/sol_execbench/core/scoring/baseline_artifact.py`。当前代码只负责读取和
校验 caller 提供的 JSON，没有自动生成正式 optimized baseline 的发布流程：

```json
{
  "schema_version": "sol_execbench.scoring_baseline.v1",
  "release": "v2.14-gfx1200-rocm7.1",
  "source": "optimized-baseline-release",
  "entries": [
    {
      "definition": "example_problem",
      "workload_uuid": "workload-uuid",
      "latency_ms": 0.123,
      "solution": "optimized_gfx1200_baseline"
    }
  ]
}
```

一个可发表的 baseline artifact 应由 release 流程在固定 GPU、ROCm、时钟、
编译选项和 timing policy 下，对选定的强实现逐 workload 实测后生成。强实现
可以来自 hipBLASLt、rocBLAS、CK、AITER、Triton 或专家 HIP kernel，但必须
固定 solution identity 与完整 provenance。

没有 matching scoring-baseline entry 时，AMD-native report 可以退回
`trace.evaluation.performance.reference_latency_ms`。这时
`baseline_source="reference_latency"`，并带 provisional warning；它不是正式
release baseline。

#### HIP measured baseline registry

`hip-agent` 的 baseline capture 会运行 target 当前配置的 solution，通过
SOL CLI 生成 trace，再由 `baseline export` 保存所有通过 workload 的：

- `latency_ms`；
- `reference_latency_ms`；
- `speedup_factor = reference_latency_ms / latency_ms`；
- hardware、ROCm version、SOL version、target id、timing policy；
- trace 和 trace line 引用。

它在 workload 全覆盖且 provenance 完整时可以得到
`coverage_status="confirmed"`，但 registry 的 `score` 当前仍是相对 reference
的 speedup，不是公式计算后的 SOL Score。`hip_agent.baselines.capture::_baseline_score()`
又会对这些 speedup 求均值并截断到 `[0,1]`。因此该 registry 可以证明
“baseline 确实在同一环境测过”，却不能直接替代 `scoring_baseline.v1` 中的
正式 `T_b`。

本地 v2.14/v1.43 refresh 工件显示：GEMM 三个 workload 的 baseline latency
约为 0.5140、1.9833、58.9968 ms，平均相对 reference speedup 约 0.9878；
Attention 三个 workload 的平均相对 reference speedup 约 0.99925。它们证明
了完整实测覆盖，但性能基本处于 reference parity，不是已发布的强优化
scoring baseline。

### 5. AMD-native score 与 official score gate

`src/sol_execbench/core/scoring/amd_score/workload.py` 在以下数值输入完整时调用
`sol_score()`：

- 正数的 candidate measured latency；
- 正数的 baseline latency；
- 正数的 AMD SOL bound。

输出首先是 `sol_execbench.amd_native_score.v1`，claim level 为
`amd-native-derived`。如果 bound 为 `unscored`、SOLAR derivation 为
`unscored`，或任一数值缺失，则 score 保持 `null`。

`src/sol_execbench/core/scoring/official_score.py` 进一步要求：

- 明确的 suite aggregation policy；
- 有限且为正的 candidate latency 和 SOL bound；
- `baseline_source` 必须属于正式来源，默认只接受 `scoring_baseline`；
- 有限且为正的正式 baseline latency；
- 每项没有 blocker，suite 才能宣称完整 score authority。

但该模块当前仍明确标记为 `STAGING`，尚未接入 CLI、dataset runner 或 sidecar
写出路径。并且在真正接线前，还应让 official gate 显式重新检查
aggregate status、hardware/model validation 与 derived warnings，避免仅凭完整
数值把 degraded bound 升格为 official evidence。

### 6. 当前工件审计结果

工作区现存 `out/rdna4-v135-rerun-20260611/derived/amd-score-report.json`
记录：

```text
mean_score: null
scored_count: 0
unscored_count: 958
baseline_source: missing
```

该结果说明 AMD bound sidecar 与执行证据已经可以批量生成，但当时没有完整
正式 baseline 输入，因此没有任何 workload 获得可用 AMD-native score。
仓库中的其他历史报告曾产生部分 derived score，也仍受 provisional model、
不完整 profiler coverage 和非官方 baseline 等边界限制。

综上，当前状态应表述为：

- AMD SOL bound：已有公式、graph/work estimate 与 sidecar，但 gfx1200 默认模型
  仍是 provisional derived evidence；
- measured baseline registry：已有真实测量与 provenance，但不是正式 `T_b`
  artifact；
- scoring baseline：schema 和消费路径已存在，但没有自动发布器或完整正式工件；
- official AMD SOL Score：当前不存在，缺失证据时应保持 unscored，而不能用
  截断 speedup 代替。

### 7. 升级为权威评分所需闭环

至少需要完成：

1. 按 dtype/path 拆分并实测校准 gfx1200 peak compute 与有效 HBM/GDDR 带宽，
   将 hardware model 的 confidence 与 model validation 升为受支持状态；
2. 补齐目标 op family 的 FLOP/byte 与融合建模，使正式集合没有 unsupported，
   并为 inexact 路径定义不可升格策略；
3. 建立 release baseline 生成器，对固定强实现重复测量并输出带 solution hash、
   环境、时钟和 timing policy 的 `scoring_baseline.v1`；
4. 固定失败/未评分的 suite 分母与聚合策略；
5. 将 official score evidence 接入 runner/CLI/sidecar，并要求 trace、timing、
   bound、baseline 和 hardware model 引用完整；
6. 对 official gate 增加 degraded/unsupported、provisional model、环境不匹配和
   stale artifact blocker；
7. 在独立 rerun 中验证 candidate、baseline 与 bound 的一致性，再发布版本化
   AMD score artifact。

### 7.3 / 7.7 实施状态（2026-07-11）

Release baseline 发布器现可从固定优化解的 trace 与 suite manifest 生成
`scoring_baseline.v1` 和 `release_baseline_bundle.v1`，记录 solution hash、
环境/时钟、编译标识、timing policy、trace、bound/model checksum 与完整分母。
独立 rerun 会对这些不变量和相对延迟容差逐 workload 验证，并发布
`release_baseline_verification.v1`。发布包和 readiness gate 仅接受互相校验的
工件对；每个 workload 明确为 `official`、`derived` 或 `blocked`。这关闭了
baseline 生成与独立复跑的工程发布闭环，但不改变本报告中硬件模型、bound
精确度或 NVIDIA leaderboard 可比性的未解决边界。

## 与 NVIDIA SOL ExecBench 实现的差距

### 1. 公式实现基本一致（gap 最小）

- 公式实现：`1 / (1 + (T_k - T_SOL) / (T_b - T_SOL))` 与 NVIDIA 的公开定义一致。
- 本项目位置：`src/sol_execbench/sol_score.py`。
- 实际边界处理：当 `T_b <= T_SOL` 时返回 `1.0`（若 `T_k <= T_SOL`）否则 `0.0`，这与现网实现模式一致。

### 2. “公式一致”不等于“指标口径一致”

1) `t_SOL` 口径不同  
- NVIDIA 路线是基于 SOLAR 图/算子分析链条产出可复现实验中的 SOlar 下界（当前官方也有已知局限，如融合模型采用 whole-graph 近似等）；  
- 当前仓库是 AMD-native 推导链路（`amd_sol/v2_math.py`），以每算子 FLOP/byte 与硬件模型的 roofline/归约规则近似下界，且存在 `unsupported` 降级。

2) `t_b` 的可追溯基线差异  
- NVIDIA 侧依赖“可复现实验 baseline（预先建立）”；  
- 当前仓库有基线工件与 `scoring_baseline` 概念，但官方化基线发布与完整覆盖尚未完全闭环。

3) 失败与未评分的处理不同  
- NVIDIA 公开公式语义下，未打分通常仍按明确分母策略计入（如失败记为 0）并保持闭环可解释；  
- 当前仓库的 AMD-native 套件侧重“有完整输入才计入 mean”，因此缺失/失败会被排除，导致结果更接近“可用样本均值”，而非统一可比较论文/榜单口径。

4) 公式层不等于官方评分产出  
- 现在有 `official_score` 模块（`src/sol_execbench/core/scoring/official_score.py`）以及完整的 gate 设计，但文档与代码都指出**尚未接入 CLI/runner/sidecar 写出路径**；目前主输出仍是 `amd_native_score` 派生侧链。  
- 官方闭环一旦接入，关键新增项是统一的聚合策略（aggregation policy）、可复核基线源枚举、以及与执行迹一致的 provenance。

### 3. 架构范围差距

ROCm 端目前更偏向“RDNA4/CDNA3 可复现实验范围”，并保留了对 NVIDIA 路径语义的兼容形态，但未主张与 NVIDIA leaderboard 结果直接可比。  
`AMD SOL` 当前文档说明“可复用/derived”与“论文级口径”是分层的，不能直接等同于官方 leaderboard semantics。

## 可以复用的基础设施（AMD-AGI / ROCm）

建议按“可直接复用 / 有条件复用 / 先抽象再复用”分级。

### 可直接复用（价值高）

- **ROCm/rocm-libraries/shared/origami**  
  - 价值：GEMM 的确定性解析式最优配置/解析时延可直接用于局部 `t_SOL` 近似。  
  - 复用点：把当前 `amd_sol` 的某些 op 家族与 Origami 输出对齐，用于高置信 op 的 bound 计算。  

- **TraceLens（AMD-AGI）**  
  - 价值：trace 采集与算子图解析（roofline / bound）体系可复用；已有与 Origami 对接接口；可复用 MI 架构参数/基准抽象。  
  - 复用点：作为 `t_SOL` 与覆盖率审计输入，替代当前部分内建 trace 处理链路。  

- **rocprofiler-compute（ROCm/rocm-systems/projects/rocprofiler-compute）**  
  - 价值：硬件计数、roofline 微基准、系统级指标和基准对比；能校准硬件模型（尤其是 peak/带宽/功耗相关）。  
  - 复用点：把软件预测模型（Origami/TraceLens/amd_sol）与实测 profile 结果闭环。

- **Magpie / APEX（AMD-AGI）**  
  - 价值：评测流程编排、结果持久化、正确性与时延对比、反作弊与 trace 管线经验。  
  - 复用点：无需从零搭环境与数据追踪，直接复用 isolation、sidecar、审计思路。

### 有条件复用（需要接口对齐）

- **ROCm libraries 与 AITER/MIOpen/CK/rocBLAS/hipBLASLt**  
  - 价值：用于稳定 baseline source（`t_b`）与对照解实现。  
  - 条件：需固化版本、编译选项、时钟和输入形状，使每次 `scoring_baseline` 可复现。

- **hipBench（ROCm）**  
  - 价值：原生 HIP kernel 基准工具链，适合做小规模对照实验。  
  - 条件：项目状态不同于成熟生产基准，当前需验证成熟度与覆盖。

### 不建议直接复用（适配成本高）

- **NVIDIA-specific 的全流程 benchmark 依赖栈**（例如原生 NCCL/CUPTI/CUDA timing）  
  - 本质上与 ROCm 运行时不同，仍可借鉴策略，但实现层不能直接复用。

## 建议的最小迁移路径

1) 保留当前公式实现不变；把“分歧点”集中在可追溯的入口上：  
   - 统一采集 `trace`、`timing`、`baseline`、`sol_bound` 的来源与 evidence id；
   - 对每个 workload 产出官方化证据（official evidence）并记录 block reason。

2) 将 `t_SOL` 变成分层生成链：  
   - 基线默认 `NVIDIA SOLAR` 公开链路 + AMD-specific fallback；  
   - 对 GEMM 先用 Origami，其他 op 可先走 TraceLens/amd_sol，明确声明覆盖与降级。

3) 在官方聚合前先统一分母策略：  
   - 明确失败/未评分是 0 计还是剔除；  
   - 与 NVIDIA 口径一致时在文档中固定该策略并上架到官方 Score evidence。

4) 先小步复用：先把 `baseline artifact` 与 `官方评分 sidecar` 接入；再逐步扩展到更多 `op_family`。

## 与现有本地文档映射

- `docs/original_parity.md`：记录当前对上游兼容边界；
- `docs/EVALUATOR-CONTRACT.md`：定义官方评分证据门槛；
- `docs/internal/RDNA4-DENOMINATOR-POLICY.md`：当前分母/覆盖策略现状；
- `docs/original_parity.md` 与 `docs/analysis.md` 可作为后续报告挂接点。

## 备注

本文档是**工程落地评估报告**，不是 leaderboard 等价声明。后续若要主张与 NVIDIA 公开结果可比，需要在证据链、分母策略与可复现基线发布节奏上达到可复核闭环。
# Calibrated hardware-model authority boundary

Calibration JSON and external hardware models are diagnostic evidence.  An official
score additionally requires serialized bound eligibility: scored AMD SOL and SOLAR
aggregates, measured exact profiles, validated hardware and model states, and no
degraded, inexact, or unsupported bound warnings.  Missing legacy eligibility evidence
is explicitly official-blocked.
