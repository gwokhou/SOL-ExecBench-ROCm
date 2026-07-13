# SOL Score 与 NVIDIA 实现差异及可复用基础设施评估（ROCm 视角）

更新时间：2026-07-13

已发布 25-workload 测量证据的代码 revision 是
`f958556fad0a5ce6740267c91f51956c378bf6fc`。本轮 365-workload closure 的确切源码
revision 由同名 evidence publication manifest 固定；发布前必须在最终提交后重建该
manifest。本地 v4 source evidence tree 是
`out/authority-release-20260712-v3-closure/`（目录名保留历史命名，内容为 v4）。该 source
tree 还保留生成诊断和 legacy 副本，不能作为 upload root；必须使用
`baseline publication stage` 生成只含 manifest 文件的独立目录并验证。`out/` 被 Git
忽略；本文对其的描述只能说明本工作区审计结果，不能替代干净机器上的外部发布验证。

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

进一步按当前代码和工作区现存工件审计，结论是：仓库已具备一个**本地验证的受限
25-workload `004_gemm_n128_k2048` gfx1200 authority slice**。v4 closure 中 baseline、
独立 rerun、校准 hardware model、matching fusion-validation、25 个 v4 bound、
AMD-native score 与 official-score evidence 已相互绑定；官方分数为 `0.5`，
`scored_count=25`、`blocked_count=0`。它不是 235 problem RDNA4 分母，更不是 NVIDIA
leaderboard 等价结果。

当前代码只接受 `sol_execbench.amd_sol_bound.v4`，并要求 fusion-validation 引用，因此
旧 v3 工件会被 current official gate 阻断，不能作为待发布 authority 工件。source tree
中的 legacy 副本仅保留为本地重建/诊断输入，不属于 manifest upload set。Git-tracked v4
manifest 位于 `docs/releases/gfx1200-20260712-hipblaslt-v4.evidence.json`，已
通过 `baseline publication stage` 生成严格、无额外文件的 upload directory，并由
`baseline publication verify` 对该目录复核。manifest 已 Git 跟踪，工件已上传到
不可变 GitHub Release；2026-07-12 又从该 Release 的全新下载目录解压并通过同一校验。
因此它可以称为 *published authority slice*，但仍不是完整 RDNA4 分母、NVIDIA/SOLAR
parity 或 leaderboard authority。

2026-07-13 又完成并发布了一个更大的**365-workload authority slice**：它来自固定的
235 problem / 3957 workload canonical 分母，覆盖 19 个 problem；primary 与独立 rerun
均在同一 gfx1200 上以锁频、10 warmup、50 iteration 完成，365/365 通过。发布 bundle、
逐 workload v4 bound/model checksum、official score 与 399-file upload inventory 已通过
`baseline publication stage` 和 `verify`。manifest 已由 Git revision
`6ea509a7bd6e582f96fcd67f9f23e8f9249a2a39` 跟踪，归档已发布至
[`gfx1200-20260713-authority365-v4`](https://github.com/gwokhou/SOL-ExecBench-ROCm/releases/tag/gfx1200-20260713-authority365-v4)，
并于 2026-07-13 下载到全新临时目录后通过同一严格 verifier；归档 SHA-256 为
`be0a91d1eaa2348b834c5073153c1059f85ba434c225fd266f067e3b2c622c0e`。因此它是
*published authority slice*，但仍不是完整 RDNA4 分母、NVIDIA/SOLAR parity 或 leaderboard authority。

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
`src/sol_execbench/core/scoring/amd_sol/artifact.py::build_amd_sol_bound_artifact()`。
当前链路按以下步骤生成 `sol_execbench.amd_sol_bound.v4` sidecar：

1. 将 `Definition` 与具体 `Workload` 绑定，构建包含 operator、tensor、shape、
   dtype 和数据流证据的 `BoundGraph`；
2. 将节点分类为 GEMM、attention、convolution、elementwise、reduction、
   normalization、softmax、MoE、SSM/Mamba、data movement 或 unsupported；
3. 为每个节点估算 FLOPs、读取字节、写入字节和总移动字节；
4. 将节点按 versioned fusion group 分区，并记录 group 外部输入/输出、内部
   tensor 与消除的中间流量；
5. 将 group 的 FLOP 与外部字节代入 AMD 硬件模型，分别计算 compute bound 与
   memory bound；
6. 取二者较大值作为 group 的 SOL bound，再对所有 group 求和。

单个 operator 的公式位于
`src/sol_execbench/core/scoring/amd_sol/builder.py`：

```text
T_compute_ms = FLOPs / (peak_tflops × 10^12) × 1000
T_memory_ms  = Bytes / (memory_bandwidth_gbps × 10^9) × 1000
T_SOL_group  = max(T_compute_ms, T_memory_ms)
T_SOL        = Σ T_SOL_group
```

例如 GEMM 使用 `2*M*N*K`，batched GEMM 使用 `2*B*M*N*K`。部分复杂
operator 仍是保守经验模型，例如 reduction 约按输入元素数计、normalization
按 `4*input_elements`、softmax 按 `5*input_elements`；这些估算会标成
`inexact`，而不是无条件视为精确证据。

这条链路是 AMD-local roofline 近似，并非 upstream SOLAR 的完整等价实现。
当前聚合按已证明的 fusion group 求和，能表达 group 内数据复用和 capability
budget 检查；它仍不能自动证明全图融合、未建模的 tile 策略或 value-dependent
优化。

### 3. gfx1200 hardware model 的权威边界

仓库不再内置或默认加载 `gfx1200.json`；authority path 必须显式提供外部
hardware model。2026-07-13 的完整分母静态审计已从 235 problems / 3957 workloads
冻结出 543 个静态 authority-eligible workloads，并收紧为以下 12 个必需
profile：

```text
compute.matrix.bf16.bf16.wmma
compute.matrix.fp16.fp16.wmma
compute.matrix.fp32.fp32.gfx12
compute.reduction.fp32.fp32.gfx12
compute.transcendental.fp32.fp32.gfx12
compute.vector.bf16.bf16.gfx12
compute.vector.fp32.fp32.gfx12
memory.stream_copy.bf16.bf16.gfx12
memory.stream_copy.bf16.fp32.gfx12
memory.stream_copy.fp16.fp16.gfx12
memory.stream_copy.fp32.bf16.gfx12
memory.stream_copy.fp32.fp32.gfx12
```

`out/gfx1200-full-suite-closure/` 中的 primary 与独立 verification calibration
均已实际测量这 12 项，且 clock lock 在采样期间被观察到、结束后被重置。由二者生成的
external model 为 `confidence=supported`，hardware/model 状态均为 `validated`。
它是当前本机的本地校准输入，还未形成 Git-tracked、不可变发布的完整-suite evidence。
FP8 matrix 当前没有 validated gfx1200 WMMA probe，因此显式保留为 `inexact`，而不是
借用 FP16/BF16 数值提升 authority。

`amd_sol/builder.py::_aggregate_for_groups()`
只有在以下条件全部满足时才返回无降级的 `status="scored"`：

- 每个 operator estimate 都是 `supported`；
- hardware validation 为 `validated`；
- model validation 为 `validated`；
- hardware model confidence 为 `supported`。

只要存在 `inexact` operator 或 provisional hardware model，结果就是
`status="degraded"`；存在 unsupported operator 则为 `status="unscored"`。
需要特别区分：当前 `degraded` 工件中的 `scored=true` 只表示数值可供
AMD-native 派生报告使用，不等于 official score authority。

因此，未带这套外部 model 的调用仍只能产生 derived evidence；即使使用该 model，
其余 inexact/unsupported operator 或 fusion evidence 仍不能获得 official authority。

### 4. 两种 baseline 不是同一个概念

| Baseline | 工件/入口 | 保存的核心量 | 当前用途 |
| --- | --- | --- | --- |
| Release scoring baseline | `sol_execbench.scoring_baseline.v1` / `--scoring-baseline` | 每个 definition + workload UUID 的优化实现实测 `latency_ms`，即 `T_b` | SOL Score 公式输入 |
| HIP measured baseline registry | `baseline_registry.v1` / `hip-agent run baseline` | 当前 target solution 的实测 latency、reference latency、speedup 与环境 provenance | 覆盖率、同环境比较和 release gate |

#### Release scoring baseline

正式 `T_b` 的 schema 在
`src/sol_execbench/core/scoring/baseline_artifact.py`。发布器现在可从固定优化解的
trace 和 frozen suite manifest 生成 compact baseline、完整 release bundle 与独立
rerun verification；调用方仍必须提供明确 scope、优化解身份和环境 provenance：

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

official CLI 还要求 compact baseline、matching release bundle、独立 rerun
verification 和 canonical suite manifest。只有 bundle 与 verification 都将该
`(definition, workload_uuid)` 列为 `official` 且摘要一致时，baseline 才能成为
`T_b`。缺失 derived score 的 manifest workload 会显式 blocked 并计入固定分母。
dataset runner 同样可消费这些 release 工件；未提供时仍可写诊断 sidecar，但会被
`release_baseline_not_verified` 阻断，不能宣称 authority。

该 gate 也拒绝缺失 trace/timing/bound/model 引用、`T_b <= T_SOL` 或
`T_k < T_SOL` 的不一致证据，避免仅凭完整数值把 degraded 或矛盾 evidence 升格。

### 6. 当前工件审计结果

已清理的历史 `rdna4-v135-rerun-20260611` 派生报告曾记录：

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

旧 25-workload GEMM slice 的 v2/v3 审计目录已清理。v3 bound 不能由当前的
`amd_sol_bound_from_dict()` 解析，25 条会被 `release_bound_not_verified` 阻断；故其
输出只能描述为**历史 v3 管线记录**，不能替代现行 v4 authority evidence，更不升级
完整 RDNA4 分母或 NVIDIA 可比性。

综上，当前状态应表述为：

- AMD SOL bound：当前 authority path 需要 v4 bound 加 matching fusion-validation
  evidence；完整静态审计生成 543 份 v4 bound，其中 442 份为 scored，101 份因缺少
  matching fusion validation 显式 blocked。现有 v3 graph/work sidecar 仅可作为重建
  输入。默认 package gfx1200 模型仍是 provisional derived evidence；
- measured baseline registry：仍是同环境比较/覆盖率证据，不能替代正式 `T_b`；
- scoring baseline：441 个可执行 workload 已在全新目录完成 primary 和独立 rerun，
  两轮均为 441/441 通过。official gate 随后识别出 76 个 `T_b <= T_SOL` sanity blocker；
  另有 1 个 workload 因 16 GiB gfx1200 上 reference execution OOM 被显式阻断。最终
  authority 为 365，使用同一两次原始测量安全过滤后，bundle verification 为 365/365；
- official AMD SOL Score：完整 3957 分母工件为 365 scored、3592 zero/blocked，固定分母
  score 为 `0.04612079858478645`，因此不宣称完整分母 score authority。单独的
  365-workload authority slice 为 365 scored、0 blocked、score `0.5` 且
  `score_authority=true`。缺失或 legacy evidence 仍保持 blocked/unscored，不能用截断
  speedup 代替，也不能泛化为完整 AMD、NVIDIA 或 leaderboard score。

### 7. 升级为权威评分所需闭环

下表区分“代码/受限 slice 已具备”与“完整 RDNA4 suite 和已发布证据已具备”；不能将
前者外推为后者。

| 事项 | 当前状态 | 仍需完成的可验收产物 |
| --- | --- | --- |
| dtype/path 硬件校准 | 543 个 static authority-eligible workloads 所需的 12 个 profile 已完成双轮实测并构建 validated 外部 model；新增了 gfx12 FP32 matrix profile，相关工件已随 365-slice 不可变发布 | 新增 profile 必须先加入正确性可验证的 probe |
| 算子与融合覆盖 | canonical worst-confidence 为 559 supported、3346 inexact、52 unsupported；543 份 v4 bound 中 442 scored、101 因 fusion evidence blocked；SOL sanity 再阻断 76 项 | 继续按 workload 级 blocker ledger 处理 residual inexact/unsupported/fusion 路径；新增 authority 项必须重新完成实测、sanity 与独立复跑 |
| release baseline 与 rerun | 441/441 primary 与 441/441 独立 rerun 已完成；sanity 收敛后的最终 bundle 为 365 official、0 derived、0 blocked，verification 为 365/365；最终 manifest 与不可变上传物的全新下载验证已完成 | 本地与发布链路已闭合；扩展 scope 时须重建 manifest 并独立复跑 |
| 聚合与分母 | `fixed_suite_denominator_zero_for_blocked` 已应用于 235 problem / 3957 workload canonical 分母；结果为 365 scored、3592 zero/blocked | AMD-native 派生报告必须继续明确标注“可用样本均值”；不得把 365-slice 的 `0.5` 外推为完整分母分数 |
| official evidence gate | 完整分母工件无残余 SOL 矛盾；365-workload slice 为 `score_authority=true`，并保留 3592 条完整 blocker ledger | 后续 candidate 必须继续绑定 solution/trace/timing 内容哈希、release bundle、rerun、v4 bound 与 model |
| 证据发布 | 25-workload v4 与 365-slice 均已外部发布；365-slice 的 399-file manifest 已 Git 跟踪，Release 归档已在无本地旁路文件的全新下载目录再次 verify | 后续 release 必须保持同样的 immutable upload 与 clean-download verification |

25-workload slice 的 candidate 与 baseline 相同，25 个 score 均为 `0.5`。它只证明
管线连通，不能证明评分对候选性能有区分能力。新的本地区分性审计将三份同机、锁频、
10 warmup、50 iteration 的真实 candidate trace 重新绑定到最终 v4 model/baseline；
`FlashInfer-Bench/004_gemm_n128_k2048` 中有 16 个 workload 属于最终 authority。其
AMD-native 可用样本均值依次为：slow extra-work `0.24751324566927846`、
equal matmul `0.5016611298292225`、fast `F.linear` `0.5017108041870116`。
排序没有退化，但 fast 与 equal 的差异极小，不构成吞吐改进声明。三份 candidate 均通过
现行 official evidence gate；在完整 3957 分母中，未提供 candidate trace 的其余项仍按
zero/blocked 记账。`tests/sol_execbench/core/evidence/test_official_score_evidence.py`
同时覆盖 `T_k < T_SOL` 必须得到 `candidate_below_sol_bound` blocker 的负例。

这满足了本地 slow/equal/fast/负例的区分性验收，但已发布的 365-slice manifest
仍使用 baseline portfolio 作为 candidate，score 为 `0.5`。若要对外发布区分性结论，
应把独立 candidate solution/trace/timing、重复次数、统计量、warmup/iteration、
离群值规则和接受容差纳入单独的不可变 manifest。上述 candidate 工件仍在 `out/`，
因此还不属于已发布 evidence。

本次 10 warmup + 50 iteration 是冻结的 v4 可比策略，不是建议长期保留的全套默认值。
下一版 timing policy 应采用可审计的自适应停止：warmup 最少 3、最多 20，并以 rolling
median 漂移不超过 2% 判断稳态；正式采样至少 10 个样本且累计至少 200 ms，在
`IQR / median <= 3%` 时提前停止，最多 50 个样本或 2 秒；极短 kernel 应批量执行以摊薄
event/synchronization 开销。报告继续保存 median、分位数、CV/IQR 与原始样本。该方向与
PyTorch `adaptive_autorange`、Triton 按 warmup/rep 时间预算和 Google Benchmark 的
minimum-time/repetition 模式一致；切换时必须升级 timing policy version，并重新生成
primary 与 rerun，不能混用本次 fixed-50 工件。

### 7.3 / 7.7 实施状态（2026-07-11）

Release baseline 发布器现可从固定优化解的 trace 与 suite manifest 生成
`scoring_baseline.v1` 和 `release_baseline_bundle.v1`，记录 solution hash、
环境/时钟、编译标识、timing policy、trace、bound/model checksum、完整分母与
显式 scope。
独立 rerun 会对这些不变量和相对延迟容差逐 workload 验证，并发布
`release_baseline_verification.v1`。发布包和 readiness gate 仅接受互相校验的
工件对；每个 workload 明确为 `official`、`derived` 或 `blocked`。这关闭了
baseline 生成与独立复跑的子链。authority mode 还默认要求 release baseline 和
content-addressed candidate solution、candidate trace、timing evidence；它会重读
candidate trace，核对 definition/workload/`PASSED` 状态与 measured latency，并要求
candidate environment、clock 与 timing policy 与 baseline 一致。baseline trace 与
rerun trace 也会在 authority load 时重新校验 checksum。它仍不改变本报告中硬件模型、
bound 精确度或 NVIDIA leaderboard 可比性的未解决边界。

### 7.2 融合建模实施状态（2026-07-11；v3 已被 v4 取代）

历史 `sol_execbench.amd_sol_bound.v3` 保留逐节点 estimate，并以
versioned fusion group 记录 group 边界、外部输入/输出、内部 tensor、消除的
中间流量和 group roofline bound。当前 `sol_execbench.amd_sol_bound.v4` 在此基础上
要求每个 bound 绑定 checksum-verified fusion-validation evidence；derived score 与
official gate 不再接受 v3。v2/v3 或 scope 未声明的 legacy 工件必须被 blocked 后重建。

fusion registry 识别 GEMM、linear、convolution 和 embedding producer 加单消费者
elementwise/activation epilogue。每个 group 都保存外部流量与
`required_lds_bytes`，并嵌入架构 capability budget 及其来源；当 budget 是
`supported`、LDS 足够且所有成员都有静态 supported estimate 时，group 才能是
`supported`。缺失、暂定、架构不匹配或容量不足的 budget 保持 `inexact`，不会
授予 official score authority。registry 同样覆盖 attention、reduction、
normalization、softmax、MoE 与 SSM/Mamba 的单消费者 epilogue；只有成员已有
完整静态合同才会升级。当前 `sum` reduction 以
`input_elements-output_elements` 精确计数；广播、activation、generic reduction、
normalization、generic softmax，以及缺少静态 subrole 证据的 attention、MoE 和
SSM 仍保持 `inexact`。

### 7.4 证据发布与候选身份缺口（2026-07-11）

checksum 只能证明当前文件没有变化，不能让被 `.gitignore` 排除的 `out/` 工件在
另一台机器上可取得。发布时不得把 trace、profiler output 或下载数据直接提交到
Git；而应将它们上传到不可变 HTTPS release bundle，并在 Git 中提交
`sol_execbench.evidence_publication_manifest.v1`。manifest 固定：release/scope、
源码 repository 与 full revision、container digest、artifact base URI、每个工件的
相对路径与 SHA-256，以及 candidate solution 和 candidate trace 的内容哈希绑定。

`sol-execbench baseline publication stage` 从 source evidence tree 复制仅由 manifest
声明的文件，再运行 `baseline publication verify`。后者会拒绝验证目录中的未声明文件，
并核对 bundle、score 和 v4 bound 的实际引用都解析到 manifest 中对应的文件；由此证明
upload directory 真正自包含。source tree 不等同 upload directory；外部发布前仍须在
干净下载目录重复 stage 后的 verify。

该命令还会验证 scoring baseline、suite manifest、bundle、rerun、official score 与 candidate
identity 的 release/scope/checksum 关系。清单必须列出 candidate solution/trace/timing、
baseline、bundle、verification、suite、official score，以及每条 official workload 引用的
trace、bound 和 hardware-model digest；因此“清单列出了一些文件”不再等同于发布验证。
没有通过该验证的结果仍可作为 locally verified score evidence，但不应使用
“published authority”措辞。详细发布流程见 `docs/EVIDENCE-PUBLICATION.md`。

当前 v4 25-workload closure 的 score 仍全部为 `0.5`，因为 candidate 的 measured
latency 与 baseline latency 相同；它只验证 score pipeline 连通性，不证明 baseline
对不同候选的区分能力。365-workload published authority slice 也以 baseline portfolio
作为 publication candidate，因此同样得到 `0.5`；独立 slow/equal/fast 候选只完成了
本地区分性 gate，尚未进入该 publication manifest。对外发布独立 candidate 时仍需记录
candidate solution/trace identity、编译产物、warmup/iteration、统计量、功耗/时钟观察
与离群值策略。`timing_policy="latency_ms"` 本身不是足够的可复现实验说明。

## 与 NVIDIA SOL ExecBench 实现的差距

### 1. 公式实现基本一致（gap 最小，但尚未形成上游可复核声明）

- 公式实现：本仓库使用 `1 / (1 + (T_k - T_SOL) / (T_b - T_SOL))`；该表达式与
  报告所依据的 NVIDIA 公开定义相同。
- 本项目位置：`src/sol_execbench/sol_score.py`。
- 实际边界处理：当 `T_b <= T_SOL` 时返回 `1.0`（若 `T_k <= T_SOL`）否则 `0.0`，这与现网实现模式一致。

截至本报告 revision，尚未把 upstream repository、commit/tag、源文件位置和边界
测试向量记录为可复核工件。因此“与 NVIDIA 对齐”仅是待补证的公式层判断，不构成
NVIDIA implementation parity 声明。特别是原始 `sol_score()` 在 `T_k < T_SOL` 时可返回大于 `1`，
而 official gate 会将候选低于 SOL bound 的证据 blocked；报告必须将这个本地的
numeric result 与发布语义分开说明。

### 2. “公式一致”不等于“指标口径一致”

1) `t_SOL` 口径不同  
- NVIDIA 路线是基于 SOLAR 图/算子分析链条产出可复现实验中的 SOlar 下界（当前官方也有已知局限，如融合模型采用 whole-graph 近似等）；  
- 当前仓库是 AMD-native 推导链路（`amd_sol/artifact.py` 与 `amd_sol/builder.py`），以 fusion group
  的 FLOP/byte 与硬件模型的 roofline/归约规则近似下界，且存在 `unsupported` 降级。

2) `t_b` 的可追溯基线差异  
- NVIDIA 侧依赖“可复现实验 baseline（预先建立）”；  
- 当前仓库已有 365-workload release baseline、独立 rerun 与 published authority
  closure；完整 3957 分母也已固定 365 scored / 3592 zero-blocked 的记账。365-slice
  已上传不可变存储并完成干净下载目录复核；剩余边界是扩大 authority coverage，而不是
  baseline 或发布链路本身。

3) 失败与未评分的处理不同  
- NVIDIA 公开公式语义下，未打分通常仍按明确分母策略计入（如失败记为 0）并保持闭环可解释；  
- 当前仓库的 AMD-native 套件侧重“有完整输入才计入 mean”，因此缺失/失败会被排除，导致结果更接近“可用样本均值”，而非统一可比较论文/榜单口径。

4) 公式层不等于官方评分产出  
- `official_score` 已接入 standalone CLI 和 dataset sidecar 写出路径；authority mode
  默认要求 release bundle、independent verification、canonical suite manifest 以及
  可校验的 candidate solution/trace/timing evidence。普通 `amd_native_score` 仍只是派生侧链。
- 聚合 policy 已固定为 `fixed_suite_denominator_zero_for_blocked`，并将 release
  baseline、bound 与 trace provenance 纳入阻断条件。

### 3. 架构范围差距

ROCm 端目前更偏向“RDNA4/CDNA3 可复现实验范围”，并保留了对 NVIDIA 路径语义的兼容形态，但未主张与 NVIDIA leaderboard 结果直接可比。  
`AMD SOL` 当前文档说明“可复用/derived”与“论文级口径”是分层的，不能直接等同于官方 leaderboard semantics。

## 可以复用的基础设施（AMD-AGI / ROCm）

建议按“PoC 候选 / 有条件复用 / 不建议直接复用”分级。以下是调研候选，不是已经
接入的依赖；除 `rocprofiler-compute` 的本地 calibration adapter 外，当前仓库没有
Origami、TraceLens、Magpie 或 Apex 的生产适配器。

### PoC 候选（价值高，但尚未验证接口）

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

上述候选均不等于可直接成为 official-score 的 authority producer。若未来接入，
输出仍必须经过本仓库的 trace、candidate identity、bound 和 publication gate，不能
绕过它们提升 claim level。

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

### 复用决策的最低记录

每个外部候选在进入依赖或评分链前，必须记录：repository URL、commit/tag、license、
支持的 ROCm/GPU、输入/输出 schema、数据处理边界、maintainer、PoC 验收指标与失败
fallback。`rocprofiler-compute` 等快速演进项目必须 pin 到经过验证的 revision，不能
以默认开发分支作为发布依赖。

当前该记录表尚未创建；在它完成前，上述分级不得作为依赖选型结论。最小 PoC 验收应
包括：固定 revision 可安装；在 gfx1200 与本项目输入上可运行；输出可转换为现有
BoundGraph/baseline/profiler schema；与现有链路的差异被量化；失败时退回当前链路。

## 建议的最小迁移路径

1) 保留当前公式实现不变；把“分歧点”集中在可追溯的入口上：  
   - 统一采集 `trace`、`timing`、`baseline`、`sol_bound` 的来源与 evidence id；
   - 对每个 workload 产出官方化证据（official evidence）并记录 block reason。

2) 先为外部 `t_SOL` 候选写出接口 PoC，而不是将其设为默认链路：
   - 固定上游 revision，记录其输入/输出 schema、可运行架构与 license；
   - 对 GEMM 先验证 Origami 输出能否映射为现有 bound evidence；其他 op 分别验证
     TraceLens/amd_sol 的覆盖、数值差异和失败退回策略；
   - 只有通过该 PoC 后，才决定是否将某条外部链路纳入默认生成路径。

3) 在官方聚合前先统一分母策略：  
   - official output 固定为失败/blocked 计 0；AMD-native derived output 若剔除未评分
     项，必须在名称和报告字段中明确“可用样本均值”；
   - 在 canonical suite manifest 中固定 problem/workload 映射、失败分类和完整分母。

4) 先小步复用：先把 `baseline artifact` 与 `官方评分 sidecar` 接入；再逐步扩展到更多 `op_family`。

## 与现有本地文档映射

- `docs/original_parity.md`：记录当前对上游兼容边界；
- `docs/EVALUATOR-CONTRACT.md`：定义官方评分证据门槛；
- `docs/internal/RDNA4-DENOMINATOR-POLICY.md`：当前分母/覆盖策略现状；
- `docs/original_parity.md` 与 `docs/analysis.md` 可作为后续报告挂接点。

## 备注

本文档是**工程落地评估报告**，不是 leaderboard 等价声明。本文中 `official` 指本仓库
official-score gate 的内部分类；`locally verified authority slice` 指本工作区可核对的
受限证据；`published authority slice` 还要求 publication manifest 和干净下载验证；
上述任一术语都不代表 NVIDIA/SOLAR parity、full-suite authority 或 leaderboard authority。

## 校准硬件模型的权威边界

Calibration JSON 和外部 hardware model 只是诊断或 release 输入证据。official score
还要求序列化的 bound eligibility：AMD SOL 与 SOLAR aggregate 均为 scored、exact
profile 已测量、hardware/model 状态均 validated，且不存在 degraded、inexact 或
unsupported warning。缺少 legacy eligibility evidence 必须被 official gate 阻断。
