# AMD SOL 算子与融合 Bound 精确度缺口

更新时间：2026-07-13

审计 revision：`b3cc94e76d70297bdae9cf6cdfe31c08d83bdfc2`

## 结论

当前 `sol_execbench.amd_sol_bound.v5` 已能绑定算子图、硬件校准、融合验证、
baseline 与发布 checksum，但证据链完整不等于 `T_SOL` 数值准确。本次已在 main
工作树中修复两个 P0 数值正确性问题：memory profile 单位换算错误，以及静态 AST
图在转置/输出 shape 无法证明时仍可能用回退 shape 生成 `supported` GEMM estimate。
已完成全量静态重建验证，但已退役的 pre-v5 365-workload publication 是修复前生成的工件；它可
证明来源和完整性，不能再被解释为数值上可靠的 SOL bound authority。

除此之外，当前 roofline 仍缺少 shape、layout、launch、occupancy 与 cache 感知，
fusion registry 只覆盖有限的两节点模式。它们使通过 sanity gate 的 bound 仍可能
非常松，尤其是小算子、低并行度 shape 和 memory-bound workload。

旧 schema 中泛称的 `T_SOL` 是历史字段语义。v5 artifact 已明确将可评分的理论
下界命名为 `T_SOL_floor`（序列化字段为
`theoretical_lower_bound.t_sol_floor_ms`）；现实最优预测、最快已知实测 latency 与
它们的 ratio 只位于 `performance_diagnostics`。后者不能提高或替换 `T_SOL_floor`，也
不能作为 official score 输入。完整合同与迁移边界见
[`amd_sol_bound_closure_strategy.md`](amd_sol_bound_closure_strategy.md)。

## 2026-07-13 后续门禁状态

下列实现门禁已经落地，既有 pre-v5 产物不因此获得追认：

- 多节点图若仅被拆为 `singleton.v1`，aggregate 一律 `unscored`；不能把当前
  compiler partition 当成不可融合的语义边界。
- authority reducer 现将未被两节点物理 fusion pattern 完整覆盖的连通子图建模为
  `semantic_component.v1`：仅保留组件的外部输入/输出流量，并以各组件约束的最大值而非
  当前 partition 的串行求和生成 floor。该优化融合假设只会令 floor 更松，不声称当前
  compiler 可生成单一 kernel；只有物理 fusion pattern 才仍要求 matching capacity
  validation。
- AMD SOL authority graph 只接受 `torch.export` 产生、且具有完整 FakeTensor
  shape/dtype 元数据的 ATen 图。FX/AST 仍可作诊断，但会以
  `semantic_graph_provider_required` 阻断评分。
- `build_semantic_graph_coverage_report()` 会逐 workload 对比 authority export 图与
  诊断 FX/AST 图，报告 capture fallback、输出 metadata 完整性和 op-family 序列差异；
  诊断差异只产生审计证据，不能提升 authority。
- 标量 TFLOP/s、GB/s profile 被显式标为 prediction-only。硬件模型必须携带状态为
  `validated`、并覆盖 `shape`、`layout`、`launch`、`occupancy` 的
  `shape_aware_roofline` evidence，aggregate 才可能进入评分；旧 v3 model 缺少该
  字段时自动 `unscored`，而非默认信任。
- sidecar 若含有更快的、已校验身份的 provider 实测，`T_SOL_floor > T_fastest_known`
  会直接转为 `unscored`。`scripts/internal/report_amd_sol_heldout.py` 对独立
  held-out provider JSONL 执行同一反例检查，并可用 `--fail-on-contradiction`
  阻止 publication。

在上述语义门禁下，对本地 gfx1200 **历史 static coverage** 的重建得到 **0 个可发布 bound / 543
个 blocked**（index checksum
`9b97f7517f61dbe1bf2674650463427921600fa7aebe85723684042e2e17ce80`）。其中 396 项
含未证明的 singleton 分区，101 项缺少 fusion validation；所有静态 AST 输入还会被
semantic provider 门禁拒绝。因此旧的 426/442 等统计只可作为历史诊断，绝非当前
authority 数量。

`scripts/internal/run_torch_inductor_provider.py` 已在本机 gfx1200/ROCm 7.1 实际运行
TorchInductor，对 pointwise/reduction、conv2d 与 GEMM/transpose 三类 workload 生成
可校验 evidence。重新生成 sidecar 后，held-out gate 匹配其中 2 项：
`061_tanh_gated_residual_add_backward` 的实测 `0.049044 ms` 快于 floor
`0.093843 ms`（ratio `1.913`），因此 publication 被拒绝；另一个 GEMM workload 未在
static coverage 中，作为 `missing_v5_bound_artifact` 一并阻断。报告 checksum 为
`bf43a18c4e94935977ebc077cae3558bc0901450a629a1aa160b15902ada5918`。最终重建的
v5 index checksum 为
`b7d46b50ecd7e5df479b1b547e48fed0ed9d57612a5f7ac634c20574ec4bc9b2`。

shape/layout-aware roofline 的**authority 安全缺口已关闭**：没有 envelope 就不可评分。
剩余工作是收集可覆盖全部 authority slice 的 envelope calibration，以及扩大
Origami/rocMLIR provider 覆盖。当前 provider contract、实际 TorchInductor adapter 和
反例 gate 已可接收这些结果，但没有把任何有限的编译搜索结果提升为理论下界。

### 2026-07-13：envelope 工件合同已收紧，采样仍待执行

`shape_aware_roofline` 不再接受只有四个维度名称和任意文件名的声明。新增
`sol_execbench.shape_aware_roofline.v1` 工件要求：

- 两次独立 scalar calibration 的 payload SHA-256；
- 当前 `hardware-profile-requirements.json` 和 `authority-coverage.json` 的
  SHA-256；
- 每个采样 case 的实际 shape、layout、launch、occupancy、warmup、至少七次稳定
  latency 样本以及原始 evidence checksum；
- `shape-aware-roofline-plan.json` 中每个 `(profile, definition, workload_uuid,
  problem_id)` assignment 都必须恰好被一个 case 覆盖；不得用一个 profile 的代表样本
  代替整个 authority slice。

`sol-execbench hardware model build` 只有在同时传入 `--shape-aware-evidence`、
`--authority-coverage`、`--requirements` 和独立 verification calibration，并且上述
checksum/profile 覆盖全部匹配时，才会把 `validated shape_aware_roofline` 写进 v3
model。模型内 reference 也必须带 evidence payload checksum。因此手工添加一个
`validated` 字段、复制旧 evidence 或把其他 authority slice 的采样混入，均不能通过
正常 build 路径。

这项实现关闭的是 evidence-contract 漏洞，**不代表已经获得 shape-aware model**。
当前 gfx1200 的旧模型没有该工件，仍然正确地阻断评分。当前已由
`collect_shape_aware_roofline.py` 将 authority slice 拆为单 GPU、可恢复的 workload
任务：每项独立进程、超时、CSV rocprofv3 trace 与 provider tensor layout 都被 checksum
绑定；`build_shape_aware_roofline_evidence.py` 只接受完整 report 才能生成 artifact。

2026-07-15 的两次宿主机直连 smoke 在同一份 checksummed gfx1200 sampling plan 上，以
`gfx1200-sq-wave-cycles-stable-peak-v1` wrapper 分别采到正的 `SQ_WAVE_CYCLES` 与
`SQ_BUSY_CYCLES`，并得到有效 occupancy。两份 raw envelope 都绑定 wrapper、真实
`rocprofv3`、`amd-smi`、ROCm/HIP/PyTorch/GPU/driver identity、pre/during/post clock
state、lease/reset result、provider 和原始 CSV checksum。结论因此收窄为：该本机
gfx1200/ROCm 7.2.0/driver 环境上的 `STABLE_PEAK` workaround 已通过记录的 smoke
验证；它不推广到其他 RDNA4、驱动、固件或未来 ROCm 版本。任何这些身份变化后必须复验。

该验证仅打开完整收集的前置门槛，不替代新的 full-suite coverage、hardware requirements
和完整 authority sampling plan，也不产生 V5 bound 或 published authority。raw validator
仍只接受派生 occupancy，或同时非零的 `SQ_WAVE_CYCLES` 与 `SQ_BUSY_CYCLES`；
`SQ_WAVES`、`GRBM_GUI_ACTIVE`、`SQ_ACCUM_PREV` 等单独为正的计数只能作为 trace
provenance，不能使 envelope validated。

TorchInductor provider 也可按需写出 PyTorch profiler Chrome trace（含 concrete
shape），并把 trace checksum 写入 raw provider evidence。held-out gate 的
`--require-profiler-trace` 会递归校验 provider raw evidence 与该 trace；这只增强反例
可观测性，不会把 provider 的有限编译结果提升为 `T_SOL_floor` authority。

全量 coverage v3/rebuild 的并发策略同时更新为 workload 级任务调度：父进程只读取
benchmark 文件并串行写 artifact，worker 只做 semantic export/estimate/fusion；worker
启动顺序错开，`--worker-tasks-per-child` 控制计划回收（本次全量运行取 32），并由 3 GiB
RSS/超时 supervisor fail-closed。正常回收用单向 Pipe 发送 sentinel 后 join；只有超时、
RSS 越界或异常退出才强制终止。自动并发取
`min(CPU - 1, MemAvailable / 8 GiB)`；8 GiB 是为 Torch/ROCm native allocator、父进程
和 page cache 预留的总预算，而不是把不完整的 worker `VmRSS` 当作全部内存。在 32 核、
约 26 GiB 可用内存的本机，这会选择 3 workers，而非容易触发 OOM 的 CPU 核数级并发。

2026-07-14 的该策略实测完成 235 problems / 3,957 workloads 的全量 coverage；payload
checksum 为 `78892482dda7befcd5eddcdcb9b398c5729b949a4ffae72b637d1f80e0651028`，文件
checksum 为 `f78c1931feead5395502f9c2a48313d9b115082da2c8cc711b6f976f9e8b0612`，其中 564
项可进入 authority shape-aware sampling。由该 coverage 与 requirements（payload checksum
`9267b4b005075987a79e1d158a4b3167b4d1dba170f17b010f04bedcdfe7e262`）派生的 12-shard
采样计划 payload checksum 为
`fe8da9aa0ac2109de03df6c8d84b291e85a2c56d0ff5600f75229e3cbe033b52`。

所有 Torch export 的完整 stdout/stderr 与 traceback 仍由父进程 `QueueListener` 异步写入
`authority-analysis.log`，CLI stdout 只保留最终 JSON 汇总；worker 不再使用
`multiprocessing.Queue`，而是将完整日志 record 与结果通过其既有单向 Pipe 发给父进程。
这消除了跨进程日志 queue 的额外 semaphore，同时不截断诊断。八个
`094_time_decay_exponential_stabilization` workload 在 60 秒内未完成 export，明确记为
`authority_analysis_failed`，没有被删除或回退为 authority。对超时/RSS worker，先以
SIGTERM 转为 worker 内的 `SystemExit`，使 Torch 的 semaphore finalizer 有最多五秒的
协作式清理时间；只有仍卡在 native code 的进程才升级为 SIGKILL。该次全量运行的临时
resource-tracker 审计确认全部 semaphore `REGISTER/UNREGISTER` 成对，退出时无
`resource_tracker` 警告；因此资源卫生问题已关闭，而非被静默抑制。

### 七类缺口闭环矩阵（2026-07-14）

| 类别 | 当前状态 | 完成/阻断证据 |
| --- | --- | --- |
| 1. memory bandwidth 单位 | 已关闭 | 单一量纲换算函数、手算测试；旧产物不被追认。 |
| 2. AST shape/transpose 回退 | 已关闭 | 不能证明的 shape 只能 `inexact`；authority path 不接受 AST/FX fallback。 |
| 3. GEMM M/N/K 推导 | 已关闭 | lhs/rhs/output 一致性校验；`A @ B.T` 与不可证明 shape 均有回归测试。 |
| 4. fusion 语义边界 | 已关闭防误发布缺口 | 未验证长链使用更松的 `semantic_component.v1` 外部流量 envelope，不再把 compiler singleton 分区串行相加。 |
| 5. authority 与性能诊断混用 | 已关闭 | v5 仅 `T_SOL_floor` 可评分；provider prediction/fastest-known 仅在 diagnostics。 |
| 6. held-out 反例 | 已关闭防误发布缺口 | 更快的已校验 provider 实测会把 bound 变为 `unscored`，独立脚本可阻断 publication。 |
| 7. shape/layout/launch/occupancy roofline | smoke gate 已通过；完整 evidence 未重建 | 两次 ROCm 7.2 gfx1200 patched smoke 都有正的 `SQ_WAVE_CYCLES`/`SQ_BUSY_CYCLES`、有效 occupancy、raw checksum 和 clock reset provenance；仍缺新的完整 coverage、plan 与两轮完整 collection。 |

因此当前代码不会再将上述任一未知项误发为 official score；第 7 项已不再等待上游
counter 修复作为前置条件，但仍必须重建 ROCm 7.2 full-suite 输入、执行两轮完整采样，
并完成 V5 bound、baseline、candidate 与 publication gates，才可发布 authority bound。

## 审计范围与口径

本结论基于上述 main revision 的源码，以及工作区
`out/gfx1200-full-suite-closure/` 中同 revision 生成的本地 gfx1200 closure 工件。
`out/` 不进入 Git，因此下列工件统计是本次可复核快照，不是仓库永久常量；重建
bound、hardware model 或 authority slice 后必须重新统计。

用于衡量 bound 松紧度的比值是：

```text
tightness_ratio = T_SOL / T_b
```

其中 `T_b` 是同机 release scoring baseline。`T_b` 只是已知实现的实测上界，不是
理论最优值，因此 `tightness_ratio < 1` 不能单独证明 bound 正确；但极小比值能证明
bound 相对已有实现非常松，而 `tightness_ratio >= 1` 会直接与已知可达 latency
矛盾。

## P0：Memory Bandwidth 单位换算错误（已修复）

硬件校准 probe 明确将 memory profile 标为 `GB/s`：

- `hardware_calibration/hip_probe.py` 对 compute 返回 `TFLOP/s`，对 memory 返回
  `GB/s`；
- stream-copy HIP probe 使用 `bytes / elapsed_ms / 1e6`，结果同样是 `GB/s`；
- 生成 hardware model 时 profile value 原样写入，没有转换为 `TB/s`。

修复前，单算子与融合组 memory bound 都按以下公式计算：

```text
T_memory_ms = Bytes / (profile_value * 10^12) * 1000
```

对 `GB/s` profile，正确换算应使用 `10^9`。当前实现相当于把例如 275 GB/s
解释为 275 TB/s，使 memory bound 系统性缩小 1000 倍：

- 单算子路径：`core/scoring/amd_sol/math.py::bound_for_estimate()`；
- 融合组路径：`core/scoring/amd_sol/builder.py::_group_memory_bound()`。

这不是 roofline 模型天然保守导致的误差，而是确定的单位正确性缺陷。它会使
memory-bound workload 的 `T_SOL` 过低，并可能把 limiting resource 错判为 compute。

当前工作树已将换算收敛为带单位语义的 `memory_transfer_bound_ms()`：

```text
T_memory_ms = Bytes / (bandwidth_GB/s * 10^9) * 1000
```

单算子路径和融合组路径共用该函数；单元测试以 200,000,000 bytes / 100 GB/s
验证两条路径都得到 2 ms。

对当前 365-workload authority slice 做“仅将 memory bound 乘 1000”的反事实复算后，
`T_SOL/T_b` 中位数从约 `0.088340` 上升到约 `0.618676`。同时有 22 项变为
`T_SOL >= T_b`；这不表示简单改常数即可完成修复，而是说明单位问题修正后，shape、
流量和 fusion partition 的其他错误会立即暴露。

## P0：静态 AST Shape 传播可生成错误 GEMM FLOP（已修复）

历史完整 suite bound 生成曾显式调用 `build_static_bound_graph()`，以避免对大量动态 reference
执行 FX trace。该静态 AST extractor 在节点没有可证明的 `output_shape` 时，会回退到
第一个可用输入 tensor 的 shape；该回退本身不会自动把节点 confidence 降为
`inexact`。

已确认的实际模式是：

```python
torch.matmul(A, B.T)
```

其中：

```text
A shape       = [11948, 7168]
B shape       = [256, 7168]
真实输出 shape = [11948, 256]
```

静态图保留了对 `B` 的依赖，却没有将 `B.T` 的转置 shape 传递给 matmul；matmul
输出 shape 推导失败后回退为 `A` 的 `[11948, 7168]`。GEMM estimator 随后从错误
输出末维得到 `N=7168`，而真实 `N=256`：

```text
错误 T_SOL = 22.1866 ms
实测 T_b   =  1.6062 ms
T_SOL/T_b  = 13.8131
```

当前工件的粗粒度一致性扫描还发现，大量 GEMM estimate 的推导 `N` 与声明输出末维
不一致。对多算子、多输出 workload，这个扫描只能作为风险信号；但上述纯 GEMM
实例已经足以证明静态 transpose/output-shape 链路存在正确性问题。

当前工作树把 `.T`/`.mT` 作为显式 logical-view 节点传播精确 shape，并要求 GEMM
的 lhs、rhs 与 output 满足完整矩阵乘法和 batch-broadcast 一致性。无可证明 output
shape 的静态 GEMM 会降为 `inexact`，不再用第一个输入的回退 shape 授予
`supported`。测试覆盖 `matmul(A, B.T)` 的精确 M/N/K，以及 shape 不可证明时的
降级路径。

全量 closure 流程现已改为生成 `authority-coverage.json`，并由
`build_authority_bound_graph()` 直接构建 `torch.export` authority graph；export fallback
会以 `semantic_graph_provider_required` 留在 blocker ledger，不能再经
`static-coverage.json` 作为 v5 bound 的输入。这一流程修复不追认既有 static 工件，须以
新 coverage 重建全部 bound 与 publication evidence。

authority capture 对 concrete workload 使用 meta/FakeTensor 输入，只保留 concrete
shape/dtype 合同而不在 host 上分配真实 benchmark tensor。因此极大 shape 的 capture
若失败会留下 export blocker，而不会以 OOM 为由回退到 AST 或缩小 workload。

## 修复前后重建快照

修复前本地完整 closure 的静态 authority-eligible workload 有 543 份，生成：

| 状态 | 数量 |
| --- | ---: |
| `scored` | 442 |
| `degraded` | 101 |

多节点 fusion group 共 266 个：

| Fusion confidence | 数量 |
| --- | ---: |
| `supported` | 30 |
| `inexact` | 236 |

修复前 365-workload authority slice 的原始 `T_SOL/T_b` 分布为：

| 统计量 | 数值 |
| --- | ---: |
| 最小值 | `0.000001628` |
| 中位数 | `0.088340` |
| 均值 | `0.205433` |
| 最大值 | `0.987962` |

这意味着通过当前 gate 的一半 workload，其 bound 仍不足已知 baseline latency 的
8.9%。极小值主要出现在短时延 activation、低并行度或 memory/launch 主导路径。

365 项还是从 441 个完成 primary 与独立 rerun 的 workload 中移除 76 个
`T_b <= T_SOL` sanity blocker 后得到的幸存集合。因此“365 项没有 SOL 矛盾”只能
证明显式矛盾已被过滤，不能反推估算模型已经准确。

修复后使用相同 benchmark root、gfx1200 v3 hardware model 与 v2 fusion evidence
在临时重建目录重新生成了完整静态 coverage 和 bound（不覆盖旧 closure）：

| 指标 | 修复前 | 修复后 |
| --- | ---: | ---: |
| 静态 authority-eligible workload | 543 | 512 |
| 退役 bound 集合中 `scored` | 442 | 426 |
| 退役 bound 集合中 blocked | 101 | 86 |

少出的 31 个 eligible workload 不是静默丢失：严格 shape 证明把 4 个此前被错误
支持的问题降为 `inexact`，覆盖 64 个 workload。与旧 authority-441 的实测 baseline
按 UUID 相交后，425 个新 bound 中仍有 49 个 `T_SOL >= T_b`，最差比值为 2.313862。
这说明两项 P0 的实现修复已生效（上述 GEMM 样例从 22.1866 ms 降至 0.7924 ms），但
尚不足以让 bound 恢复发布资格。

## 融合建模缺口

当前 fusion registry 最多识别一个受支持 producer 后接一个单消费者
elementwise/activation epilogue。更长的 pointwise 链、分支、多个输出、完整
normalization、attention、MoE 与 SSM/Mamba 数据流通常会被拆成 singleton 或保留
`inexact`。

代表性 HIP fusion probe 现从 `torch.export` authority graph 提取精确相邻两算子
subgraph 的 shape/dtype signature；其 schema pattern 明确标为
`diagnostic_reduction_epilogue.v1`。这是编译候选和 trace 诊断，**不会**匹配或提升包含
更长数据流的 `semantic_component.v1` authority group。旧 AST 两节点 signature 不再能
被误解释为完整 semantic component 的物理 fusion 证明。

该 registry 现在只描述可由 provider 验证的**物理** fusion pattern；它不再决定
authority floor 的 component 边界。对无法归入该 registry 的连通子图，
`semantic_component.v1` 以理想化内部流量消除构建较松的全图 lower-bound envelope。
因此 registry miss 本身不能再把 compiler 的 singleton partition 当成强制
materialization，也不会使 authority reducer 对 group latency 求和。

这带来三类数值风险：

1. 对 singleton 分别计算 `max(compute, memory)` 后求和，可能重复计算实际可被长链
   融合消除的中间流量；
2. 两节点 group 只证明边界 tensor 流量，尚不能表达 register/LDS tile liveness、
   occupancy、spill 或跨阶段流水；
3. fusion validation 中的 fused/unfused 实测 performance 被记录在 sidecar，但
   group 升级目前只要求 matching case 的 `capacity_status == "passed"`，实测绝对
   latency 和 ratio 没有参与数值 bound 校准。

当前 266 个多节点 group 中只有 30 个得到 matching validation，剩余 236 个直接构成
101 份 degraded bound 的主要 blocker。扩大 fusion authority 前，应先修复单位与图
正确性，否则新增 validation 只会给错误输入授予更高 confidence。

修复后的 49 个剩余 `T_SOL >= T_b` 进一步定位出一个必须在再发布前关闭的 gate：
若多算子图只被分成多个 `singleton.v1` group，当前实现会把每个节点的 stream-copy
流量逐段相加，却仍把 aggregate 标为 `scored`。例如 9 节点
`025_video_latent_gelu_activation` 的新 bound 为 0.3200 ms，而实测 baseline 为
0.1383 ms。这是未被验证的长链 fusion 被当作独立 kernel 串行执行的模型假设，不应
被提升为精确 authority；在获得 matching fusion evidence 或保守的全图流量模型前，
这类图应降为 `inexact`/blocked。

## Roofline 结构性缺口

即使两个 P0 问题修复，当前标量 profile roofline 仍不是精确 latency model。它尚未
表达：

- kernel launch、event 和同步的最小成本；
- grid 大小、wave 数、尾块、occupancy 与低并行度利用率；
- GEMM 的 transpose、layout、alignment、tile、padding 与 accumulation dtype；
- memory stride、transaction、L2/cache 命中与广播复用；
- reduction 的跨 wave/workgroup 阶段和同步；
- LDS/VGPR/SGPR、spill、instruction mix 以及 compute/memory overlap。

因此一个全局 TFLOP/s 或 GB/s profile 被用于所有 shape 时，大矩阵可能接近实测 roof，
小矩阵和短算子则会非常松。这里应引入按 operation、dtype、path、shape/layout 与规模
分桶的可达吞吐 envelope，而不是用单个校准常数继续扩大 `supported` 覆盖。

## 修复与验收顺序

建议按以下顺序处理：

1. 已完成：修正 GB/s 换算；单算子、融合组两条路径共用同一量纲函数并有手算测试。
2. 已完成：修复 transpose/view/output shape 传播；无法证明时降为 `inexact`。
3. 已完成：GEMM dimension inference 校验 lhs/rhs/output 的矩阵乘法一致性。
4. 已完成：对非物理 fusion pattern 的连通子图使用 `semantic_component.v1` 全图
   外部流量 envelope；没有 non-overlap proof 时 reducer 取最大组件约束，且物理 fusion
   pattern 才要求 matching capacity validation。
5. 在第 4 项后重建全部 v5 bounds、baseline sanity、authority slice 和 publication
   evidence；旧 checksum 可以证明旧文件未变化，不能替代数值修正后的重建。
6. 建立按算子族和 shape 分桶的 held-out 数值验证，至少报告
   `T_SOL > T_fastest_known` violation 与 `T_SOL/T_fastest_known` 分布。
7. 再扩展长链/分支 fusion、tile-liveness 和 shape-aware hardware roof。

本次两个 P0 的实现验收已满足：

- GB/s 单位测试能以手算样例验证 ms 结果；
- `matmul(A, B.T)` 的 M/N/K 与声明输出一致；
- 不可证明的 shape 不再得到 `supported`；
- 对完整 closure 已重新统计 bound；
- 但 baseline sanity、authority slice 与 publication checksum 必须在 fusion gate 修复后
  从修复后的源码和工件重新生成。
