# AMD SOL Bound 非 P0 缺口闭环与社区组件复用策略

状态：提案
调研日期：2026-07-13
适用范围：`sol_execbench.amd_sol_bound.v5` 两个 P0 正确性缺陷修复后的后续演进

## 摘要

本文承接 [`amd_sol_bound_accuracy_gap.md`](amd_sol_bound_accuracy_gap.md) 的审计结论，
讨论除 memory profile 单位错误和静态 AST shape 传播错误以外的融合、shape/layout、
launch、occupancy、cache 与验证缺口。

建议不在“整套替换”和“全部自行维护”之间二选一，而是采用分层复用：

- 本仓库继续维护 SOL authority 的最小可信内核，包括下界语义、量纲、强制流量、
  confidence/provenance、聚合、sanity gate 和 publication evidence；
- 使用 PyTorch `torch.export`/FakeTensor 替换主要的 Python/ATen 图语义与 shape
  传播实现；
- 使用 Origami、TorchInductor、rocMLIR/MIGraphX 生成性能与融合候选，但不直接把
  它们的预测或实测 latency 当作理论下界；
- 使用 rocprofv3、TraceLens，以及在目标硬件明确支持时使用 rocprofiler-compute，
  完成 trace、计数器和实测验证闭环；
- 将“理论下界”和“现实最优性能预测”拆成两个数值产品，避免用经验模型提升
  `T_SOL` 后破坏 lower-bound claim。

当前没有一个社区组件同时覆盖本项目所需的 PyTorch reference 语义、任意算子图融合、
AMD 硬件下界、证据绑定和发布 gate。整套替换不可行；继续自行维护 AST shape engine、
融合搜索器和完整性能预测器也不是合理的长期方向。

## 问题分解

剩余缺口包含两个不同目标：

1. **理论下界**：数值必须对允许范围内的任何正确实现成立，即使未来出现更好的
   compiler、tile 或融合实现也不应被突破。
2. **现实最优预测**：希望解释固定软件栈和硬件上的 launch、shape、layout、occupancy、
   cache、tile 与编译器效率，尽量接近可实现 latency。

二者不能共用同一套 authority 语义。Origami、TorchInductor、rocMLIR/MIGraphX 或
实测最快 kernel 给出的是有限搜索空间内的预测或可达实现。它们能提供最优 latency
的上界或经验估计，不能单独证明对所有实现成立的理论下界。

如果继续把两类信息都写入一个 `T_SOL`，模型越 shape-aware，越可能通过降低“可达
吞吐”来提高 bound latency，并在新 kernel 出现后产生 `T_SOL > T_fastest_known`。

## 决策：保留自有 authority reducer，复用外部 provider

建议将链路拆为四层：

```text
PyTorch reference + concrete workload
              |
              v
     semantic graph provider
  torch.export -> guarded AST fallback
              |
              v
      repository BoundGraph IR
        /                  \
       v                    v
authority lower-bound    external candidate providers
reducer                  Origami / Inductor / rocMLIR
       |                    |
       v                    v
T_SOL_floor             compiled kernels / predictions
       |                    |
       +---------+----------+
                 v
       held-out validation and evidence
     rocprofv3 / TraceLens / profiler data
```

### 本仓库继续维护的部分

- `BoundGraph` 与发布 schema 的稳定合同；
- profile 单位、量纲类型和换算；
- 哪些 FLOP、读写流量和依赖是语义上强制发生的；
- lower-bound reducer 的数学规则；
- `supported`、`inexact`、`unsupported` 的授予规则；
- 每项 shape、layout、traffic、profile 和 fusion 结论的 provenance；
- contradiction、held-out、authority 和 publication gate；
- 外部 provider 的版本绑定、失败降级和 checksum。

### 应尽量复用的部分

- Python/PyTorch 程序的完整图捕获；
- ATen 算子的 shape、dtype、stride 和 symbolic constraint 传播；
- GEMM tile/config 搜索与性能预测；
- 现实编译器的融合候选、kernel partition 和代码生成；
- trace 树、API 到 kernel 关联、计数器采集和 replay；
- 经过目标硬件验证的 kernel tuning database。

## 两个数值产品

### `T_SOL_floor`

`T_SOL_floor` 是 official score authority 可以消费的理论下界。基础形式仍为：

```text
T_SOL_floor = max(
    mandatory_flops / architectural_max_compute_rate,
    mandatory_external_bytes / architectural_max_bandwidth,
    proved_dependency_or_reduction_floor
)
```

跨多个不能重叠的语义阶段时才允许求和。任何“不能重叠”“必须 materialize”或“必须
launch 多个 kernel”的判断都需要证明；不能因为当前 compiler 恰好这样生成就视为
理论约束。

该数值允许很松，但必须满足：

- 不使用当前 compiler 的效率损失作为未来实现的限制；
- 不把当前最快实测 latency 当作理论 floor；
- 不把当前 fusion registry 的边界当作不可跨越的语义边界；
- shape、layout 或强制流量不能证明时降为 `inexact`；
- 新的更快实现不应使已发布的 `supported` floor 产生 contradiction。

### `T_predicted_best`

`T_predicted_best` 是非 authority 的现实性能预测，可综合：

- Origami 的 GEMM config latency prediction；
- TorchInductor 编译与实测候选；
- rocMLIR/MIGraphX 调优候选；
- hipBLASLt、rocBLAS、MIOpen、CK、AITER 等 library baseline；
- launch、grid/wave、tail、occupancy、alignment、cache 和当前编译器 partition；
- 固定版本下的 historical fastest-known。

它用于性能诊断、候选排序和衡量 `T_SOL_floor` 的松紧度，不得以
`theoretical SOL authority` 名义进入 official score。若未来产品确实希望使用该数值
评分，必须修改 schema、名称和 claim，而不是静默替换 `T_SOL` 的含义。

### 已落地的 v5 合同（2026-07-13）

`sol_execbench.amd_sol_bound.v5` 将这一边界写入 sidecar：

- `theoretical_lower_bound.t_sol_floor_ms` 是唯一可传给 AMD-native/official score
  的数值；状态、authority eligibility 与理由也在该对象中。
- `performance_diagnostics` 单独记录 `t_predicted_best_ms`、`fastest_known_ms`、
  ratio 和 `floor_contradicts_fastest_known`，不会参与 reducer 或 score 输入。
- provider 必须声明身份、目标架构、结果类型和原始 evidence；
  `is_theoretical_lower_bound=true` 会被拒绝，外部 provider 无法提升 authority。
- 仅接受并序列化 v5 artifact；所有旧 schema 都会被明确拒绝，不迁移或重写旧
  `sol_bound_ms` 的语义。

最快已知实测值因此是 floor contradiction 的证据，而不是新的 `T_SOL`；现实
prediction 是模型误差和候选排序的输入，而不是物理约束。

## PyTorch 复用方案

### 使用 `torch.export` 作为主要 semantic graph provider

PyTorch 官方将 `torch.export` 描述为 sound、normalized、full-graph 的 AOT IR；它会
保留输入约束和节点输出的 shape/dtype，完整图不可捕获时会失败，而不是无声产生部分
图。其 shape 传播通过 ATen 算子的 FakeTensor/meta implementation 完成，并能区分
static、backed dynamic 和 unbacked data-dependent shape。

这比本仓库继续扩展 AST/FX 特例更符合 fail-closed authority：

1. 将 reference `run` 包装为最小 `torch.nn.Module`；
2. 对 concrete workload 使用 `torch.export.export(..., strict=True)`；
3. 读取规范化 ATen graph 及其 tensor metadata；
4. 校验所有声明输出的 shape、dtype 和数量；
5. 将 view、mutation、alias 和 stride provenance 映射到 `BoundGraph`；
6. export 失败、缺少 fake kernel 或产生不可证明 shape 时降为 `inexact`；
7. AST 仅保留为源码结构证据和降级路径，不再单独授予 `supported`。

建议先使用 `run_decompositions(decomp_table={})` 只做 functionalization。不要在第一阶段
直接全部降低到 Core ATen IR：过度 decomposition 会拆散高层算子族，改变 FLOP/traffic
归属并增加与现有 estimator 对齐的成本。需要 decomposition 的 op 应使用版本化白名单。

当前项目已经 pin PyTorch 版本，因此可以在 provider adapter 中使用固定版本行为；仍应
避免把 `torch._inductor`、`torch._dynamo` 或 FakeTensor 私有对象布局直接写入发布
schema。

参考资料：

- [PyTorch `torch.export` 文档](https://docs.pytorch.org/docs/main/user_guide/torch_compiler/export.html)
- [PyTorch export programming model](https://docs.pytorch.org/docs/main/user_guide/torch_compiler/export/programming_model.html)
- [PyTorch custom compiler backend contract](https://docs.pytorch.org/docs/2.9/torch.compiler_custom_backends.html)

### PyTorch FLOP counter 的边界

`torch.utils.flop_counter` 可用于与本地 estimator 做 differential check，但不建议直接
成为 authority producer：其 op coverage、decomposition 和 FLOP 定义未必与本仓库的
SOL 口径一致，也不能推导强制 memory traffic。任何复用都应保留本仓库的 op-family
合同和覆盖报告。

### TorchInductor 的边界

TorchInductor 可用于：

- 发现长 pointwise、reduction、prologue 和 epilogue fusion；
- 生成当前 PyTorch/ROCm 版本下的实际 kernel partition；
- 记录 generated code、launch 数、compile config 和实测 latency；
- 为 fusion registry 提供候选和反例。

不应依赖 `torch._inductor` 私有 scheduler 或内部 cost model 作为 authority。更稳定的
接入边界是：固定 PyTorch revision，编译实际 workload，记录生成工件和 trace，再将
结果转换为本仓库的 diagnostic/fusion-validation schema。

## MLIR 复用方案

### 通用 MLIR 原语适合借鉴，不构成完整替代

MLIR 已提供多个有价值的结构化原语：

- Shape dialect：known、unknown、invalid shape 及约束 witness；
- Linalg dialect：iteration space、indexing map、layout、tile 和 producer-consumer
  fusion；
- One-Shot Bufferize：tensor 到 buffer、alias、in-place 和 copy/materialization；
- Transform dialect：tile、fuse、vectorize 等 transformation strategy。

这些原语能指导 `BoundGraph` 的长期表达，但 MLIR 本身没有一个可直接接管本项目的
AMD SOL lower-bound model。仅为 shape 正确性引入 PyTorch -> torch-mlir -> Linalg ->
BoundGraph 的第二套转换链，复杂度高于直接使用 `torch.export`。

`torch-mlir` 仍处于 LLVM incubator，且需跟随特定 PyTorch revision。当前阶段不建议
把它设为主 frontend；只有在未来决定将整个分析 IR 和 compiler flow 迁移到 MLIR 时，
才值得重新评估。

参考资料：

- [MLIR Shape dialect](https://mlir.llvm.org/docs/Dialects/ShapeDialect/)
- [MLIR Linalg dialect](https://mlir.llvm.org/docs/Dialects/Linalg/)
- [MLIR Bufferization](https://mlir.llvm.org/docs/Bufferization/)
- [MLIR Transform dialect](https://mlir.llvm.org/docs/Dialects/Transform/)
- [torch-mlir](https://github.com/llvm/torch-mlir)

### 优先评估 rocMLIR/MIGraphX

rocMLIR 是更贴近本项目目标的 MLIR 实现。其公开范围包括 AMD gfx9xx/gfx10xx/
gfx11xx/gfx12xx，能够生成 GEMM、convolution、attention、GEMM+GEMM 和 CONV+GEMM
kernel，并通过 TOSA/Linalg/Rock/AMDGPU/ROCDL lowering 到 HSACO。它也被 MIGraphX
作为实际 kernel generator 使用。

MIGraphX 提供 exhaustive tuning、MLIR tuning database、dot/convolution/attention
选择，以及 input/reduction/fused-dot/fused-convolution 等融合控制。建议将其用于：

- 生成 GEMM+epilogue、GEMM+GEMM、attention 和 convolution fusion 候选；
- 替换部分手写 HIP fusion probe；
- 记录实际 tile、perf config、kernel boundary 和 tuning database；
- 验证某项中间 tensor 确实可以在一个 kernel 内消除；
- 提供 `T_predicted_best` 和 fastest-known 的候选实现。

rocMLIR/MIGraphX 能证明“某种融合可行”，不能证明“更大的融合不可能”，因此其
partition 不得直接成为 `T_SOL_floor` 中的强制 materialization 边界。

参考资料：

- [ROCm rocMLIR](https://github.com/ROCm/rocMLIR)
- [MIGraphX MLIR 与 tuning 配置](https://rocm.docs.amd.com/projects/AMDMIGraphX/en/develop/reference/MIGraphX-dev-env-vars.html)
- [MIGraphX Python compile API](https://rocm.docs.amd.com/projects/AMDMIGraphX/en/latest/reference/MIGraphX-py.html)

## AMD 社区性能组件的复用边界

### Origami

Origami 是 GEMM 的解析式 solution selection 组件。它根据 M/N/K、batch、transpose、
dtype、tile 和 occupancy，在给定候选 config 中预测 latency 并选择最优项。

适合：

- GEMM shape-aware prediction；
- tile/config 候选排序；
- 与 hipBLASLt/rocBLAS 实测结果做差异分析；
- 审计当前全局 TFLOP/s roofline 在不同 shape 上的松紧度。

不适合：

- 直接替代任意算子和融合 bound；
- 将有限候选集合中的预测最优值解释为理论下界。

截至调研日期，Origami 的支持表将 gfx1200 标为 functional，而 optimized 标记仅见于
gfx942/gfx950。因此在本项目 gfx1200 目标上应先作为 diagnostic provider，不能直接
升级为 authority producer。

参考：[Origami README](https://github.com/ROCm/rocm-libraries/blob/develop/shared/origami/README.md)

### TraceLens 与 rocprofv3

TraceLens 能消费 PyTorch profiler、rocprofv3 JSON 和 pftrace，提供 Python op、CPU
dispatch、GPU kernel 的层次关联、shape 聚合、roofline、trace diff 和 event replay。
它适合减少本项目自行维护的 trace 分析与 replay 代码。

建议固定上游 commit，并把转换结果写入本仓库 schema；上游原始 trace、provider
revision 和转换器版本都必须进入 checksum/provenance。

参考：[AMD-AGI TraceLens](https://github.com/AMD-AGI/TraceLens)

### rocprofiler-compute

rocprofiler-compute 能提供 empirical roofline、system SOL、cache、wave、instruction 和
stall 指标，是 CDNA/Instinct 上成熟的性能诊断工具。但官方目前仍将其主要定位为
MI100/MI200/MI300/MI350 等 Instinct accelerator，并说明 Radeon/RDNA 支持仍在开发。

本项目当前目标是 gfx1200，因此：

- 不应把 rocprofiler-compute 设为 gfx1200 authority 的唯一依赖；
- 可以继续保留 unknown-only parser 和 capability detection；
- gfx1200 的主 trace 路径优先使用实际可用的 rocprofv3 与 TraceLens；
- 只有目标 revision 明确列出并实测支持 gfx1200 后，才允许提升其 evidence confidence。

参考：

- [ROCm Compute Profiler 定位与支持范围](https://rocm.docs.amd.com/projects/rocprofiler-compute/en/develop/what-is-rocprof-compute.html)
- [ROCm Compute Profiler compatible accelerators](https://rocm.docs.amd.com/projects/rocprofiler-compute/en/develop/reference/compatible-accelerators.html)

## Fusion 闭环原则

当前两节点 registry 的问题不仅是覆盖率低，还可能影响 lower-bound 正确性：如果把
当前未识别的可融合边界当作强制 materialization，对 singleton bound 求和可能比未来
长链融合实现更慢，从而产生 contradiction。

建议为两个数值产品采用不同策略。

### `T_SOL_floor` 的 fusion 策略

- 只在有语义或硬件证明时声明 materialization barrier；
- 对不能证明不可融合的中间 tensor，允许理想化消除其流量；
- 可以使用比现实 compiler 更大的 optimistic fusion region；
- 允许因此得到更松但更安全的 lower bound；
- reduction、同步、多输出、跨 device 或显式 host interaction 等 barrier 必须有版本化
  规则和测试。

### `T_predicted_best` 的 fusion 策略

- 汇总 TorchInductor、rocMLIR/MIGraphX、library kernel 和自有 HIP probe 的实际
  partition；
- 编译并验证 fused/unfused correctness；
- 记录绝对 latency、ratio、launch 数、workspace、LDS/VGPR/SGPR 和 spill；
- 允许多个 provider 同时为同一 pattern 提供候选，取最快已验证实现作为
  fastest-known；
- provider 失败只影响 diagnostic coverage，不得让 authority graph 静默换用猜测结果。

## Provider 合同

每个外部 provider 至少应返回：

```text
provider_name
provider_revision
provider_schema_version
target_architecture
rocm_version
input_identity_sha256
status: supported | inexact | unavailable | failed
result_kind: semantic_graph | prediction | compiled_candidate | measurement
is_theoretical_lower_bound: true | false
output_payload
warnings
raw_evidence_ref
raw_evidence_sha256
```

默认 `is_theoretical_lower_bound=false`。只有本仓库 authority reducer 根据已审核的数学
规则生成的结果才允许设为 `true`。外部 provider 不可通过返回 `supported` 自行提升
official claim level。

## 分阶段落地

### 阶段 1：semantic graph PoC

实现 `TorchExportGraphProvider`，先覆盖代表性 workload：

- GEMM 与 `A @ B.T`；
- transpose、permute、reshape、view、slice 和 broadcast；
- pointwise 长链与分支；
- reduction、normalization、attention；
- data-dependent shape/control-flow 负例；
- 多输入、多输出和 mutation/alias 负例。

产出 export/AST/现有 FX 三路差异报告，不立即替换 publication path。

验收条件：

- 所有授予 `supported` 的输出 shape/dtype 与声明输出一致；
- 所有授予 `supported` 的中间 shape/stride 均来自可追溯 metadata；
- export 失败不会通过 AST fallback 恢复为 `supported`；
- provider 有超时、cache key 和确定性序列化；
- cache key 至少包含 reference SHA-256、input metadata、PyTorch version 和 provider
  schema version。

### 阶段 2：外部候选 provider PoC

分别实现独立 adapter：

1. Origami：GEMM prediction 和 config selection；
2. TorchInductor：pointwise/reduction/epilogue compiled candidate；
3. rocMLIR/MIGraphX：GEMM+epilogue、GEMM+GEMM、attention 和 convolution fusion。

每个 PoC 必须记录 repository URL、固定 commit/tag、license、支持的 ROCm/GPU、输入/
输出 schema、安装方式、失败 fallback 和维护责任人。

### 阶段 3：双轨 artifact（已完成基础合同）

已升级 sidecar，使其同时记录：

- authority `T_SOL_floor`；
- diagnostic `T_predicted_best`；
- fastest-known measured latency；
- provider predictions 和 compiled candidates；
- 三者的 ratio 与 contradiction 状态。

v5 是唯一支持的 schema version；旧 schema 不会被读取或重新序列化。下一步只是在这个
合同上接入真实 provider，而不是把 provider 的输出接进 authority reducer。

### 阶段 4：held-out 数值验证

按 operation family、dtype、shape、layout、transpose、规模和 fusion pattern 分层。
held-out split 必须按 shape family 或 problem family 划分，不能随机拆同一 shape 的重复
workload，以免 calibration leakage。

最低报告：

- `T_SOL_floor > T_fastest_known` violation 数量和明细；
- `T_SOL_floor / T_fastest_known` 的 min、p10、median、p90、max；
- `T_predicted_best / T_fastest_known` 的误差分布；
- provider coverage、timeout、failed 和 unavailable 数量；
- graph shape/stride/dtype mismatch 数量；
- 各 family 的 `supported`、`inexact`、`unsupported` 数量。

authority 最低要求：

- held-out 与 calibration 集均为零已知 contradiction；
- 所有 contradiction 都阻断 publication，而不是从 authority slice 中移除后继续发布；
- 外部 provider 缺失或升级不会改变已发布 floor，除非显式重建和升级 evidence；
- 全部 model、bound、baseline、provider 和 publication checksum 可重建。

## 依赖和升级治理

外部项目均处于快速演进状态。生产接入必须：

- pin commit 或 ROCm release tag，禁止依赖默认开发分支；
- 与项目当前 PyTorch/ROCm/toolchain version 一起进入 lock 和 evidence；
- adapter 使用独立 schema，不把上游内部 Python/C++ 对象直接序列化；
- 上游升级先跑 compatibility、differential 和 held-out suite；
- provider 不可安装或不支持目标 GPU 时 fail closed；
- 保留 raw output 和转换后 output 的双重 checksum；
- 不允许外部 provider 绕过现有 candidate identity、trace、baseline、bound 和
  publication gate。

## 不采用的方案

### 整套替换为某个社区项目

不采用。Origami 仅聚焦 GEMM solution selection；TraceLens 和 profiler 分析已执行的
kernel；TorchInductor 与 rocMLIR/MIGraphX 生成有限 compiler search space 中的可达
实现；通用 MLIR 不提供现成 AMD SOL authority。它们都不能独立完成本项目的完整证据
与下界合同。

### 继续扩展静态 AST shape engine

不采用为主路径。Python、ATen、view、broadcast、data-dependent shape 和 alias 语义
会持续增长，维护成本和 silent-wrong 风险高于复用 PyTorch 自身的 export/meta
语义。AST 只保留为 inexact fallback 和审计证据。

### 直接把实测最快 latency 当作 `T_SOL`

不采用。实测最快实现是当前最优 latency 的上界，未来实现可以更快。它适合作为
baseline、fastest-known 和 prediction target，不是理论 lower bound。

### 用 shape bucket 吞吐直接提高 authority bound

不采用为默认 authority。经验吞吐 envelope 可以提高预测准确度，但除非能证明它是
硬件可达吞吐的上界，否则会把历史效率损失错误当作物理约束。shape bucket 应先进入
`T_predicted_best`，通过独立数学证明的项才可逐项进入 `T_SOL_floor`。

## 最终维护边界

长期目标不是减少本仓库对 SOL 结果的责任，而是减少重复实现成熟基础设施的责任：

- **必须自维护**：下界定义、证据合同、单位、强制 work/traffic、confidence、聚合和
  发布 gate；
- **通过薄 adapter 维护**：PyTorch export、Origami、Inductor、rocMLIR/MIGraphX、
  TraceLens 和 profiler；
- **不再自行扩张**：通用 AST shape 规则、完整 compiler fusion search 和覆盖所有
  shape/cache/occupancy 的单体预测模型。

该边界能在保持 authority 可审计性的同时，把社区投入最大的图语义、kernel 编译、
调优和性能观测能力纳入本项目，并避免把“社区成熟实现”误解为“社区已有完整理论
SOL 实现”。
