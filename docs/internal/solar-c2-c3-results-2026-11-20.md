# Solar C2 容量标定与 C3 终态覆盖报告（2026-11-20）

> [!IMPORTANT]
> 本报告为 M2 阶段 **C2（RX 9060 XT 容量标定与合格 `LLM_CORE` 目标视图冻结）** 与 **C3（双路径逐工作负载形式化分析覆盖扫描）** 在执行基线 `b45e6f9af61bf1f669fd36f217a72f37c5757d61` 上的**有界终态收尾（Bounded Terminal Closeout）**：
> - **C2 判定：`COMPLETE`（通过）**——实测显存档位 `mem-12gib`，冻结 `36` 条准入决策、`20` 个 `core` 算子定义、`180` 个工作负载与 `360` 个覆盖行。
> - **C3 判定：合法 `INCOMPLETE`（触达 12 小时活动预算上限）**——按 [M0 LLM_CORE 覆盖率规范](solar-m0-llm-core-coverage-spec-2026-11-07.md) §6–§7，`r2` 活动在累计 **`43,200.38 s`（`12.0 h`）** 时由调度器自动停止于 `activity budget exhausted`，完成 **`168 / 360` 行**（`40 passed`、`111 failed`、`16 unproved`、`1 budget_exhausted`、`192 not_reached`、`0 infrastructure_error`），经独立无设备终态容器全量复核并完成干净 GPU 交还。官方评分契约、默认 IR 后端与原生算子注册表均未改动，`D0` 待展开。

相关文档：
- [C2 与 C3 冒烟检查点（2026-11-19 `r1`）](solar-c2-c3-smoke-2026-11-19.md)
- [M0 LLM_CORE 覆盖率规范](solar-m0-llm-core-coverage-spec-2026-11-07.md)
- [演化路线图](solar-evolution-roadmap-2026-11-05.md)
- [下一轮迭代计划](solar-next-iteration-plan-2026-11-07.md)
- [HANDSOFF](../../HANDSOFF.md)
- [文档索引](../README.md)

---

## 1. 执行溯源、恢复记录与校验身份

### 1.1 运行序列与中断恢复

1. **执行源码冻结**：基线为提交 `b45e6f9af61bf1f669fd36f217a72f37c5757d61`（`Align RDNA configuration resolution with WGP and VRAM carveouts`），仅针对 [hardware.py](../../src/sol_execbench/core/platform/hardware.py) 的 `CONFIGURATION_TEMPLATE` 对齐 RDNA4 `WGP * 2 == declared CU` 与受界固件显存保留偏差（相对先前基线 `622c6ba3` 仅此 1 个源文件差异）；[standard-16gb.yaml](../../problems/LLM_CORE/targets/configurations/rx9060xt/standard-16gb.yaml) 与 `LLM_CORE_V2` 清单均保持逐字节未改。
2. **`r1` 检查点与服务重启中断（`data/outputs/solar-c2-c3-20261119/`）**：`r1` 完成了 C2 与 40 行 C3 smoke 检查点（见 [冒烟检查点记录](solar-c2-c3-smoke-2026-11-19.md)）；随后首次 `--resume` 在 `row-046` 遭遇宿主机服务重启中断，留下 `interrupted_in_progress`。按生产契约（禁止直接续跑或删改含 `interrupted_in_progress` 的目录），经受权恢复工具 `python -m sol_execbench.core.bench.clock_recovery 19a91a17f8a44ea3bc3b1e0b35fe9670` 核验并释放 GPU 租约与准入门禁（退出 `0`，0 GPU 进程、时钟 `AUTO`、`READY`），原 `r1` 目录完整保留。
3. **`r2` 全流程干净重建与闭环（`data/outputs/solar-c2-c3-20261119-r2/`）**：在新的干净活动目录重建并顺序通过 `prepare` → `readiness` → `b2-representatives` → `c2` → `c3-smoke` → `audit` → `c3-resume` → `c3_final_audit`。

### 1.2 `r2` 阶段执行与产物指纹摘要

| 阶段 / 产物（均在 `data/outputs/solar-c2-c3-20261119-r2/` 下） | SHA-256 / 核心摘要 |
|---|---|
| `freeze.json` | `ec6c3cea7e1135e5b54013069db2f0312b407ebd4894d5e5d18bed3b9c991e16`（基线 `b45e6f9af61bf1f669fd36f217a72f37c5757d61`，`changed_vs_e2_r2=["sol_execbench/core/platform/hardware.py"]`） |
| `preflight/container-readiness.json` & `host-run/readiness.json` | `verified=true`，`returncode=0`，宿主机 wall `2.54 s` |
| `preflight/representatives-verdict.json` & `host-run/b2-representatives-closeout.json` | `passed=true`（`8/8`），容器 `47.05 s` / 宿主机 `56.36 s`，`clean=true` |
| `llm-core/capacity-evidence.json` & `llm-core/c2-summary.json` & `host-run/c2-closeout.json` | `c05a4fd7775c203dc965cbc82f65b0e68cd59b94264f8570e5e3c91eeb9c7e93` / `0ad8eadd9537d15c4e536ffb3e24c65697ab223abf3f328d23287c84a588fd44`；`status="passed"`，容器 `5.65 s` / 宿主机 `14.78 s`，`clean=true` |
| `llm-core/smoke-summary.json` & `exposure-ledger.json` & `cpu-audit.json` | `746d043448363d7dfbc44e90c2c13f9a777aee524e2efba2056eedd9356af50f` / `e1cbd9a3e2c61d5a5675326b512d36b16c22826ae48ca10ee819ad9599a25bc7` / `b412b5414d3b323d25b5bc53f4855fb2485f333e4a858122d99b90c41808d524`（`40/360`：`16 passed`、`24 failed`，`454.07 s` / 宿主机 `469.24 s`，`clean=true`；终态复核确认 C3 后此 5 份文件逐字节未变） |
| `llm-core/coverage/target-coverage.json` & `host-run/c3-resume-closeout.json` | `86dd98e5c1b2380afec56b618095df53604e82e5775725e9820a66a6730c917a`；容器 `returncode=0`，宿主机 wall `42,773.85 s`，`clean=true`（`AMDSMI_DEV_PERF_LEVEL_AUTO`，`gpu_process_count=0`） |
| `llm-core/c3-summary.json` | `ba77a820ae99e1e370d16d3e7e21ef92e59fa64b31515fb0b8ca02b810485b67` |
| `llm-core/c3-exposure-ledger.json` | `e6969c96cfaf74bb7008fae3cfcd9b5db5ed034f0558c8fd2686459eeb548588` |
| `llm-core/c3-final-audit.json` & `host-run/c3-final-audit.json` | `91fc0ebd9974f796581257fcee2e4bafea0244ccf15797c694247bae2a5b26ed`；`audit_passed=true`，无设备（`gpu_initialized=false`、`rocr_visible_devices=""`），宿主机 wall `10.45 s` |

---

## 2. C2 结论：`COMPLETE`（显存配额与目标视图）

- **实测容量**：`runtime_total_bytes = 17,095,983,104` B（空闲 `17,026,777,088` B），连续稳定可分配 `15,300,820,992` B，安全余量后 `usable_budget_bytes = 13,005,697,843` B，最终落入 **`mem-12gib`**（`12,884,901,888` B；`capacity_probe_digest = b20fdcb63bde77ce0f30b25c3b9af69cdbcbcd15d751fb789ebb2d1e5e5562b4`）。
- **冻结目标视图**：`36` 条准入决策（`generated = 20`、`profile_not_selected = 16`），对应 `20` 个 `core` 算子定义（`linear=8`、`norm_activation=4`、`position=2`、`attention=6`）、`180` 个唯一工作负载与 `360` 个计划覆盖行；`workload_view_digest = 65b391133652d85ebef583867ec7aeac10d9a2b187499492b5bdc52859ff81a4`（与 `r1` 完全一致），`r2` target-view 清单 SHA-256 为 `c2323243fa7deb8c9551eb530aec2495b000a2124aec19208669a356b8bd5dce`。

---

## 3. C3 终态结果（`168 / 360` 已执行，`192 / 360` 因 12 小时预算停止）

### 3.1 总体指标与停机边界

- **覆盖完成度**：计划 `360` 行（`20` 定义 × `9` 槽位 × `2` `IRPath`）；**已执行 `168` 行**（`40` 行 `smoke` + `128` 行非 `smoke`），**未到达 `192` 行**（均标为 `"activity budget exhausted"`）；`matrix_complete=false`、`costs_observed=false`、`complete=false`、`performance_authority=false`。
- **完整执行 vs 仅 smoke 定义**：`8 / 20` 个算子定义完成全部 `18 / 18` 行覆盖（全部 `6` 个 `attention` 定义以及 `linear.batched_bias`、`linear.batched_matmul`）；`12 / 20` 个定义仅完成 `smoke`（`2 / 18` 行：`linear.{grouped_matmul,grouped_silu,matmul_bias,matmul_gelu,matmul_relu,matmul_silu}`、`norm_activation.{add_rmsnorm,bias_gelu,rmsnorm,swiglu}`、`position.{rope_interleaved,rope_yarn}`），停机首行位于 `row-168`（`llm.linear.grouped_matmul.v2` / `latency-low` / `make_fx_aten`）。
- **时间与证据体积**：总运行 `elapsed_seconds = 43,200.38 s`（`smoke = 454.07 s`、非 `smoke = 42,746.31 s`，对比上限 `43,200 s` / `12.0 h`）；单行 wall 分布最小 `3.86 s`、中位 `13.29 s`、均值 `257.15 s`、最大 `9,216.46 s`；总证据目录体积 `255,773,489` B（**`243.92 MiB` / `0.238 GiB`**，远低于 `50.0 GiB` 上限）。
- **生产原始状态 vs M0-C0 §5 报告归一分类**（原始状态总计 `40 passed / 111 failed / 16 unproved / 1 budget_exhausted / 192 not_reached / 0 infrastructure_error`；按 M0-C0 §5 展开语义子因后，原 `111` 项 `failed` 拆分为 `89` 项 `unsupported_or_unproved`、`16` 项验证 `failed` 及 `6` 项 `budget_exhausted`）：

| M0-C0 §5 报告归一分类 | 行数 | 来源与细分子原因计数 |
|---|---:|---|
| `passed` | 40 | `passed: 40`（原始 `passed: 40`） |
| `unsupported_or_unproved` | 105 | 原始 `failed` 中的 `incomplete_tile_aware_bound: 63`、`external_reads_unproved: 11`、`ir_unsupported_op: 8`、`missing_resource_rule: 7`（小计 `89`），加上原始 `unproved` 的 `conversion_not_proven_device_oom: 16` |
| `failed` | 16 | `exact_replay_failed: 8`、`gradient_verification_failed: 5`、`numerical_equivalence_failed: 3`（均来自原始 `failed`） |
| `budget_exhausted` | 7 | 原始 `budget_exhausted` 的 `worker_timeout: 1`，加上原始 `failed` 中的 `graph_extraction_device_oom: 4` 与 `mapper_timeout_surfaced_as_toolchain_unavailable: 2` |
| `not_reached` | 192 | `activity budget exhausted: 192` |
| `infrastructure_error` | 0 | — |

### 3.2 按 `IRPath` × `operation_family`、`role` 与 `slot` 分布

| 维度 | 分类 | `passed` | `failed` | `unproved` | `budget_exhausted` | `not_reached` | 小计 |
|---|---|---:|---:|---:|---:|---:|---:|
| **`IRPath`** | `make_fx_aten` | **36** | 37 | 11 | 0 | 96 | 180 |
| **`IRPath`** | `torchview_extended_einsum` | **4** | 74 | 5 | 1 | 96 | 180 |
| **`family`** | `attention` | 24 | 68 | 16 | 0 | 0 | 108 |
| **`family`** | `linear` | 8 | 39 | 0 | 1 | 96 | 144 |
| **`family`** | `norm_activation` | 8 | 0 | 0 | 0 | 64 | 72 |
| **`family`** | `position` | 0 | 4 | 0 | 0 | 32 | 36 |
| **`role`** | `smoke` | 16 | 24 | 0 | 0 | 0 | 40 |
| **`role`** | `development`（`*-low`） | 12 | 44 | 8 | 0 | 96 | 160 |
| **`role`** | `holdout`（`*-high`） | 12 | 43 | 8 | 1 | 96 | 160 |

按 9 个槽位细分的执行计数（已执行 `8` 个完整定义在各非 smoke 槽位占 `16` 行，剩余 `12` 个定义占 `24` 行 `not_reached`）：

| `slot` | `make_fx_aten`（`passed/failed/unproved/not_reached`） | `torchview_extended_einsum`（`passed/failed/unproved/budget_exhausted/not_reached`） | 单行耗时均值 / 最大 (s) |
|---|---|---|---|
| `smoke` | `12 / 8 / 0 / 0` | `4 / 16 / 0 / 0 / 0` | `11.35 / 16.16` |
| `latency-low` | `4 / 4 / 0 / 12` | `0 / 8 / 0 / 0 / 12` | `20.08 / 47.84` |
| `throughput-low` | `3 / 5 / 0 / 12` | `0 / 8 / 0 / 0 / 12` | `437.42 / 2,217.09` |
| `irregular-low` | `5 / 3 / 0 / 12` | `0 / 8 / 0 / 0 / 12` | `12.62 / 16.60` |
| `capacity-low` | `0 / 3 / 5 / 12` | `0 / 5 / 3 / 0 / 12` | `641.00 / 9,216.46` |
| `latency-high` | `4 / 4 / 0 / 12` | `0 / 8 / 0 / 0 / 12` | `14.31 / 23.93` |
| `throughput-high` | `3 / 5 / 0 / 12` | `0 / 8 / 0 / 0 / 12` | `1,296.68 / 7,230.59` |
| `irregular-high` | `5 / 3 / 0 / 12` | `0 / 8 / 0 / 0 / 12` | `12.98 / 17.79` |
| `capacity-high` | `0 / 2 / 6 / 12` | `0 / 5 / 2 / 1 / 12` | `236.55 / 1,177.36` |

### 3.3 8 个完整执行定义的 9 槽行为矩阵（揭示跨尺度不均一性）

在完整覆盖全 9 槽的 `8` 个定义中，没有任何一个 `make_fx_aten` 定义表现出 9 槽均一（`slot_status_uniform_make_fx = false`），证明 **`smoke` 无法外推至全槽位**：

| `semantic_id` | `make_fx_aten`（`smoke` → 8 非 `smoke` 槽位） | `torchview_extended_einsum`（全 9 槽位） | 关键尺度/实现发现 |
|---|---|---|---|
| `llm.attention.gqa_causal.v2` | `passed` × 7（含 `latency/throughput/irregular`），`capacity-{low,high}` **`unproved`** | `capacity-low` `unproved`、`capacity-high` `failed`（提取 OOM），其余 7 槽 `failed`（tile-aware bound） | `make_fx` 凭 fake-static 通过取图，在验证阶段报 OOM；`torchview` 在 `capacity-high` 即触发 15.92 GiB Eager 跟踪 OOM |
| `llm.attention.masked_softmax.v2` | `passed` × 7，`capacity-{low,high}` `unproved` | `capacity-low` `unproved`、`capacity-high` `failed`（提取 OOM），其余 7 槽 `failed` | 同上 |
| `llm.attention.mqa_decode.v2` | `passed` × 7，`capacity-{low,high}` `unproved` | **全 9 槽 `failed`**：8 槽 `ir_conversion` 不支持 `expand_as`，`capacity-high` 提取 OOM | `torchview` 全槽前端阻断 |
| `llm.attention.local_global.v2` | **`smoke` 与 `latency-{low,high}` `passed`（3/9）**；**`throughput-{low,high}` 与 `irregular-{low,high}` `failed`（4/9）**；`capacity-{low,high}` `unproved` | `capacity-low` `unproved`，其余 8 槽 `failed`（含 `throughput-high` 触达 7,200 s mapper 超时） | **`make_fx` 跨尺度回归**：同一定义在 `latency` 通过而在 `throughput`/`irregular` 失去完整 tile-aware bound |
| `llm.attention.attention_sink.v2` | `capacity-high` `unproved`，其余 8 槽 `failed`（`effect_boundary` 7 槽 + `capacity-low` 2×BMM tile bound 失败耗时 `9,216.46 s`） | `capacity-high` `unproved`；`smoke/latency/throughput` 5 槽 `gradient_verification_failed`；`irregular-{low,high}` 与 `capacity-low` 3 槽 `numerical_equivalence_failed` | `torchview` 在不同形状上分别暴露梯度与前向数值等价差异（`max_abs=0.15 ~ 6.5`） |
| `llm.attention.sliding_window.v2` | `capacity-{low,high}` `unproved`，其余 7 槽 `failed`（缺 `arange` 的 `amd_resource_v3` 规则） | `capacity-high` `unproved`，其余 8 槽 `exact_replay_failed`（缺失 `Model.hidden-tensor_8.Output`） | 两条路径在各自的原生规则/外张量映射层稳定失败 |
| `llm.linear.batched_bias.v2` | **`smoke` 与 `irregular-{low,high}` `passed`（3/9）**；**`latency/throughput/capacity` 6 槽 `failed`**（tile-aware bound） | 全 9 槽 `failed`（tile-aware bound） | **`make_fx` 跨尺度回归**：不规则形状可闭合而规整 `latency/throughput/capacity` 失去 tile bound |
| `llm.linear.batched_matmul.v2` | **`smoke` 与 `irregular-{low,high}` `passed`（3/9）**；`latency/throughput/capacity` 6 槽 `failed` | 8 槽 `failed`；`row-167` `capacity-high` 触达活动边界 **`budget_exhausted`**（`worker_timeout`） | 同上，且 `row-167` 揭示 worker 超时与子 PGID 清理竞态 |

### 3.4 就绪、验证子维度、形式指标与两路径对照

- **就绪阶段计数（`readiness_stage_counts`）**：
  - `make_fx_aten`：`graph_extraction` `84 passed`、`ir_conversion` `84 passed`、`conversion_verification` `73 passed / 11 failed`（全为 `capacity` 显存不足 `conversion_not_proven`）；`make_fx_real_retry_observability.observable_in_current_artifacts = false`。
  - `torchview_extended_einsum`：`graph_extraction` `80 passed / 4 failed`、`ir_conversion` `72 passed / 8 failed / 4 not_run`、`conversion_verification` `51 passed / 21 failed / 12 not_run`。
- **验证子维度（`verification_dimension_counts`，严格区分不外推）**：
  - `numerical`：`124 passed`、`11 failed`、`21 not_observable`、`204 not_reached`。
  - `gradient`：`124 passed`、`5 failed`、`16 not_observable`、`215 not_reached`。
  - `effects`：`145 not_observable`（attestation 未单独导出 alias/mutation）、`215 not_reached`。
- **形式分析指标（`formal_metrics_counts`）**：`formal_attempted = 124`、`formal_passed = 40`；`40` 个通过样本均满足 `external_reads_quality = exact`、`bound_kind = capacity_constrained_tile_aware_v1`，限制资源分布 **`mfma = 32`、`valu = 8`**，且三视图（`cache_aware`、`ideal_fused`、`unfused`）均达到 `available:ready = 40`。
- **两路径成对比较（`180` 组唯一 `(semantic_id, slot, workload_uuid)`）**：
  - 全矩阵成对状态：`not_reached__vs__not_reached = 96`、`failed__vs__failed = 36`、`passed__vs__failed = 32`、`unproved__vs__failed = 6`、`unproved__vs__unproved = 5`、`passed__vs__passed = 4`、`failed__vs__budget_exhausted = 1`。
  - 共同通过样本仅 `4` 组（均在 `smoke` 的 `norm_activation` 家族，与 `r1` 冒烟完全一致：`flops=0`、`macs=0`、`external_read_bytes` 与 `lower_bound_seconds` 双路径全等；`add_rmsnorm` 与 `rmsnorm` 因 `sfu` vs `valu` 资源归因差异使 `resource_work_equal=false`，其余 2 组全等）。

### 3.5 各阶段资源消耗（`stage_cost_summary`）

| 阶段 | 调用次 | 中位 wall (s) | 最大 wall (s) | 累计 wall (s) | 设备峰值显存 | Host RSS 峰值 |
|---|---:|---:|---:|---:|---:|---:|
| `readiness:graph_extraction` | 168 | 0.493 | 2.579 | 183.72 | 15,967,125,504 B（~14.87 GiB） | 1,645,727,744 B（~1.53 GiB） |
| `readiness:ir_conversion` | 164 | 0.084 | 0.183 | 11.97 | 850,034,688 B（~810.7 MiB） | 1,646,514,176 B（~1.53 GiB） |
| `readiness:conversion_verification` | 156 | 3.431 | 25.055 | 796.28 | 16,538,042,368 B（~15.40 GiB） | 1,994,858,496 B（~1.86 GiB） |
| `formal:graph_extraction` | 123 | 0.472 | 2.252 | 107.42 | 4,163,801,088 B（~3.88 GiB） | 1,646,063,616 B（~1.53 GiB） |
| `formal:ir_conversion` | 123 | 0.083 | 0.143 | 8.56 | 850,034,688 B（~810.7 MiB） | 1,646,850,048 B（~1.53 GiB） |
| `formal:conversion_verification` | 123 | 3.562 | 24.481 | 736.56 | 14,525,162,496 B（~13.53 GiB） | 1,995,317,248 B（~1.86 GiB） |
| `formal:formal_analysis` | 123 | 0.333 | 9,179.630 | 40,447.62 | — | 1,997,795,328 B（~1.86 GiB） |
| `formal:mapper` | 159 | 0.132 | 7,200.115 | **40,272.21** | — | 1,997,795,328 B（~1.86 GiB） |

> **关键发现**：全活动 `43,200 s` 中 **`93.2%`（`40,272.21 s`）消耗在 CPU 侧 `formal:mapper`（Timeloop `linear-pruned` 搜索）**，GPU 阶段仅占约 `4.2%`。例如 `row-046` 两个 4D BMM 的映射耗时达 `9,179.6 s`（`total wall 9,216.46 s`），`throughput-high` 24 个多行累积 `20,746.8 s`。

---

## 4. 按受影响定义去重排序的缺口清单与基础设施发现

按 `llm-core/c3-summary.json` 中的 `gap_inventory`（已排除 `192` 行未到达行，并按受影响唯一算子定义数降序排序），**无任何一项符合 `L3-d` 新前端准入条件**（全部 `12` 类缺口的 `l3d_frontend_candidate = false`）：

| 排序 | `gap_id`（阻塞层） | M0-C0 分类 / 子因 | 影响定义数（`make_fx` / `torchview`） | 影响行数 | 关键证据与后续建议范围 |
|---:|---|---|---:|---:|---|
| 1 | `formal_analysis:incomplete_tile_aware_bound` | `unsupported_or_unproved` / `incomplete_tile_aware_bound` | **12**（`8` / `11`） | 63 | 覆盖 4 项 `attention`、`batched_bias`/`batched_matmul` 规整槽位及 `matmul_*`（`sample: row-003`） |
| 2 | `conversion_verification:conversion_not_proven_device_oom` | `unsupported_or_unproved` / `conversion_not_proven_device_oom` | **6**（`6` / `5`） | 16 | 全部 6 个 `attention` 的 `capacity-{low,high}` 在 15.92 GiB 设备验证回放 OOM（`sample: row-054`） |
| 3 | `graph_extraction:device_oom_eager_trace` | `budget_exhausted` / `graph_extraction_device_oom` | **4**（`0` / `4`） | 4 | `torchview` 在 4 项 `attention` 的 `capacity-high` 取图阶段即超出 15.92 GiB（`sample: row-071`） |
| 4 | `formal_analysis:external_reads_unproved:read_semantics_unproved` | `unsupported_or_unproved` / `external_reads_unproved` | **2**（`2` / `2`） | 4 | `rope_interleaved` 与 `rope_yarn` 缺索引读取语义证明（`sample: row-036`） |
| 5 | `formal_analysis:external_reads_unproved:effect_boundary` | `unsupported_or_unproved` / `external_reads_unproved` | **1**（`1` / `0`） | 7 | `attention_sink` 7 槽位存在跨作用域写后读边界（`sample: row-000`） |
| 6 | `formal_analysis:missing_amd_resource_v3_rule:arange` | `unsupported_or_unproved` / `missing_resource_rule` | **1**（`1` / `0`） | 7 | `sliding_window` 7 槽位缺少 `arange` 的 `amd_resource_v3` 资源规则（`sample: row-010`） |
| 7 | `formal_analysis:mapper_timeout_7200s` | `budget_exhausted` / `mapper_timeout_surfaced_as_toolchain_unavailable` | **1**（`1` / `1`） | 2 | `row-082`/`083`（`local_global` / `throughput-high`）运行 `7,230 s` 触发 mapper `7,200 s` 硬超时，但上游 `OrojenesisRunner` 仅返回 `toolchain_unavailable` |
| 8 | `conversion_verification:exact_replay_failed` | `failed` / `exact_replay_failed` | **1**（`0` / `1`） | 8 | `torchview` 转换 `sliding_window` 缺失 `Model.hidden-tensor_8.Output`（`sample: row-011`） |
| 9 | `ir_conversion:unsupported_op:expand_as` | `unsupported_or_unproved` / `ir_unsupported_op` | **1**（`0` / `1`） | 8 | `torchview` 不支持 `mqa_decode` 的 `expand_as` 严格转换（`sample: row-009`） |
| 10 | `conversion_verification:gradient_numerical_mismatch` | `failed` / `gradient_verification_failed` | **1**（`0` / `1`） | 5 | `attention_sink` 在 5 槽上反向数值不匹配（`max_abs` up to `6.5`，`sample: row-001`） |
| 11 | `conversion_verification:numerical_equivalence_failed` | `failed` / `numerical_equivalence_failed` | **1**（`0` / `1`） | 3 | `attention_sink` 在 `irregular-{low,high}` 与 `capacity-low` 前向数值不匹配（`matched_ratio=0.9991..0.9998`，`max_abs=0.146..2.95`，`sample: row-045`） |
| 12 | `outer_bridge:activity_budget_worker_timeout` | `budget_exhausted` / `worker_timeout` | **1**（`0` / `1`） | 1 | `row-167` 在 12h 边界触发 `192.224 s` 超时并引发下方嵌套进程组残留写入漂移 |

### 独立无设备审计发现的嵌套进程组（Nested-PGID）超时清理竞态

在 `llm-core/c3-final-audit.json` 对全量 `168` 条已执行行进行无设备隔离审计时，确认以下精确机制：
- 前 **`167` 行（`row-000 .. row-166`）** 与 `row-167` 全部 `readiness/` 及语义收据达到 **0 SHA-256 偏差（`non_timeout_transient_artifact_mismatches = 0`）**，且全部 `168` 行均通过 `authenticate_resumed_attempts` 完整语义收据复核与 `resume_identity_matches_full_audit = true`（`f5d3a641...`）。
- 仅 `row-167`（`llm.linear.batched_matmul.v2` / `capacity-high` / `torchview_extended_einsum`，`budget_exhausted`）暂存目录内 **`row-167/attempt-00/.formal.mvob697p/orojenesis/Model.bmm/timeloop-mapper.oaves.csv`** 的磁盘文件（`230,476,701` B，SHA-256 `5e434dc2...`，mtime `1795138567.106`）晚于 `target-coverage.json` 保存时间（记录 SHA-256 `afd5138c...`，mtime `1795138563.917`）**`3.189 s`**。
- 根因为 `run_solar_worker`（[runner.py](../../src/sol_execbench/core/solar_bridge/runner.py)）以进程组 `PGID_1` 启动 worker，而内部 `OrojenesisRunner` 通过独立的进程组 `PGID_2` 启动 `timeloop-mapper`；当 `run_solar_worker` 触达 `192.224 s` 预算超时强杀 `PGID_1` 并返回、调度器随即将 `.formal.mvob697p/` 下文件纳入 SHA-256 哈希后，孤立的 `PGID_2` 子进程在容器 `--init` 回收前继续刷盘约 `3.2 s`，从而使生产入口 `target_coverage._resume_or_create` 因 `coverage resume artifact drift` 阻止基于原目录的 `--resume`（`standard_resume_blocked_by_timeout_transient_drift = true`）。该问题已作为明确待修项留待 **`D0`** 决策。

---

## 5. 暴露隔离账本（`c3-exposure-ledger.json`）

- 依据 `c3-exposure-ledger.json`（SHA-256 `e6969c96cfaf74bb7008fae3cfcd9b5db5ed034f0558c8fd2686459eeb548588`），`r2` 已暴露工作负载共 **`168` 行**（`smoke = 40` 行 / `20` 个 UUID，`development = 64` 行 / `32` 个 UUID，`holdout = 64` 行 / `32` 个 UUID），剩余 **`192` 行**未到达（`development = 96`、`holdout = 96`）。
- 凡已暴露于 C3 失败分析的 `development` / `holdout` 工作负载均不得作为后续针对性修复的“未暴露独立留出”验证集使用。
