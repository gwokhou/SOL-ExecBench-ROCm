# Solar C2 容量标定与 C3 冒烟检查点结果（2026-11-19）

> [!IMPORTANT]
> 本次记录对应 M2 阶段在干净执行基线 `b45e6f9af61bf1f669fd36f217a72f37c5757d61` 上完成的 **C2（GPU 显存容量标定与合格 `LLM_CORE` 目标视图冻结）** 与 **C3 smoke（40/360 行串行冒烟检查点与独立无设备审计）**。
> **C3 当前处于 `INCOMPLETE` 检查点状态**：已完成前 `40` 行（`common` 尺度切片），剩余 `320` 行处于 `not_reached`，待确认后通过 `--resume` 继续执行。官方评分契约、默认 IR 后端与原生算子注册表均未改动，`D0` 尚未开始。

相关文档与规范：
- [M0 LLM_CORE 覆盖率规范](solar-m0-llm-core-coverage-spec-2026-11-07.md)
- [演化路线图](solar-evolution-roadmap-2026-11-05.md)
- [下一轮迭代计划](solar-next-iteration-plan-2026-11-07.md)
- [HANDSOFF](../../HANDSOFF.md)
- [文档索引](../README.md)

---

## 1. 执行基线与预检门禁

### 1.1 运行前最小修复

在启动 C2 前，核实 [standard-16gb.yaml](../../problems/LLM_CORE/targets/configurations/rx9060xt/standard-16gb.yaml) 声明 `visible_compute_units: 32` 与名义 `visible_memory_bytes: 17179869184`（16 GiB），而 RX 9060 XT（`gfx1200`）的 ROCm 运行时设备探测返回 RDNA4 工作组处理器数 `multi_processor_count: 16`（WGP）及保留 80 MiB 固件显存后的可见总显存 `17095983104` B。

在不改动 [standard-16gb.yaml](../../problems/LLM_CORE/targets/configurations/rx9060xt/standard-16gb.yaml)（保持逐字节一致）或 release manifest 的前提下，通过提交 `b45e6f9af61bf1f669fd36f217a72f37c5757d61`（`Align RDNA configuration resolution with WGP and VRAM carveouts`）对 [hardware.py](../../src/sol_execbench/core/platform/hardware.py) 实施最小修复，并在 [test_hardware.py](../../tests/sol_execbench/core/platform/test_hardware.py) 补充单测：

- `_resolve_compute_units`：在 `CONFIGURATION_TEMPLATE` 上支持 RDNA（`gfx10`/`gfx11`/`gfx12`）运行时 `WGP * 2 == declared CU` 对齐。
- `_matches_declared_memory`：在 `CONFIGURATION_TEMPLATE` 上允许 `<= min(512 MiB, nominal // 32)` 的受界固件显存保留偏差。
- 回归测试：`test_rx9060xt_standard_16gb_resolves_rdna4_wgp_and_firmware_carveout` 与 5 用例参数化测试 `test_configuration_template_rejects_mismatched_wgp_or_memory`。

### 1.2 干净检出、冻结清单与容器/代表槽预检

本次战役由 `scripts/internal/solar/c2_c3_campaign.py` 与 `scripts/internal/solar/c2_c3_host.py` 编排，输出目录位于 `data/outputs/solar-c2-c3-20261119/`：

| 门禁项目 | 产物路径（`data/outputs/solar-c2-c3-20261119/` 下） | 关键结果与校验摘要 |
|---|---|---|
| 运行冻结 | `freeze.json` | SHA-256 `4dce5050341210d8e074db80ccee5acda75236a35f7e81c16e49ff3efbc43d03`；`execution_commit=b45e6f9af61bf1f669fd36f217a72f37c5757d61`（基于新干净分离检出）；`src/` 清单 SHA-256 `808ce8b6717ababdfe4a8c5b40bd844397473e22939a035fde596ef5f2daff35` |
| 相对基线源码差异 | `preflight/source-diff-vs-prior.json` | 相对先前基线 `622c6ba3`：`changed_vs_e2_r2 == ["sol_execbench/core/platform/hardware.py"]`（仅单文件差异） |
| 容器无 GPU 就绪 | `preflight/container-readiness.json` | `verified=true`，`gpu_initialized=false`，`definitions=36`，容器 wall `1.78 s` |
| B2 代表槽校验 | `preflight/representatives-verdict.json` & `host-run/b2-representatives-closeout.json` | `passed=true`，`8/8` 槽位通过（`6` 个 `ready` + `2` 个精确拒绝），容器 wall `46.97 s` / 宿主机 wall `56.51 s`，`clean=true` |

---

## 2. C2：实测显存标定与合格目标视图冻结

C2 在容器中基于 `rx9060xt/standard-16gb.yaml` 执行一次真实 GPU 显存探测（宿主机总耗时 `15.08 s`，阶段容器 wall `5.77 s`），输出 `llm-core/capacity-evidence.json`（SHA-256 `cb47591cdaf8a101bc37897d79b861c1590a6b483a85c60f8ffe19ee78e5d6bb`）与 `llm-core/c2-summary.json`（SHA-256 `dbf06a42a505a1592c4b19972e9b758723331c2ec850ba64241b8e230d645293`），并在结束后完成干净 GPU 交还（`c2-closeout.json` `clean=true`）。

### 2.1 `GPUMemoryQuotaEvidence` 标定结果

| 字段 | 实测值 | 说明 |
|---|---|---|
| `status` | `passed` | 显存配额探测与目标视图冻结均通过 |
| `runtime_total_bytes` | `17,095,983,104` B | ROCm 运行时报告总显存（名义 16 GiB 扣除 80 MiB 固件保留） |
| `runtime_free_bytes` | `17,026,777,088` B | 探测前空闲显存 |
| `stable_allocatable_bytes` | `15,300,820,992` B | 连续探测可稳定分配上限 |
| `usable_budget_bytes` | `13,005,697,843` B | 纳入工作预算余量后的可用上限 |
| `capacity_class_id` | `mem-12gib`（`12,884,901,888` B） | 选定的标准显存容量档位（12 GiB） |
| `capacity_probe_digest` | `b20fdcb63bde77ce0f30b25c3b9af69cdbcbcd15d751fb789ebb2d1e5e5562b4` | 容量探测摘要 |

### 2.2 冻结的目标覆盖矩阵规模与摘要

| 字段 | 值 |
|---|---|
| `hardware_configuration_id` | `9af8995dcb03589c02ee0d3b975d1411e5145604ac89d170f666a64c3d1a5aa2` |
| `generation_cohort_id` | `9de3bed7fe8eacdc0e87cb78e26b6fde663a1e3db0ca0145d385ef3a6426ceb8` |
| `distribution_id` | `e75f7e9cebdc7bf16ea1a5a898492f5c6f1fb153500211bb742980bf0e55c70c` |
| `workload_view_digest` | `65b391133652d85ebef583867ec7aeac10d9a2b187499492b5bdc52859ff81a4` |
| `target_view_manifest_sha256` | `3ab6162c989cd756301885374543f8a28ad49b4baa0e7627764223ef9aed7534` |
| `decisions_count` | `36`（`generated=20`，`profile_not_selected=16`） |
| `problems_count` | `20`（`linear=8`、`norm_activation=4`、`position=2`、`attention=6`） |
| `workloads_count` | `180` |
| `planned_coverage_rows` | `360`（`2` 条 `IRPath` × `180` 个工作负载） |

已选入 `core` 队列的 20 个算子定义对应的 `common` 尺度样本规模包括：`linear`（`matmul_*` `21,055` 元素，`M=328, K=64, N=64`；`batched_*`/`grouped_*` `2,335` 元素，`M=36, K=64, N=64, B=2`）、`norm_activation`（`32,735` 元素，`M=511, N=64`）、`position`（`rope_interleaved` `61,631` 元素 `S=962`，`rope_yarn` `63,503` 元素 `S=992`）、`attention`（`3,311..9,855` 元素）。

---

## 3. C3 Smoke（`40/360` 行）：结果分类与逐行阻塞明细

C3 smoke 在无额外并行的严格串行模式下执行前 `40` 个覆盖行（`20` 个 `common` 工作负载 × `2` 条 `IRPath`），输出 `llm-core/smoke-summary.json`（SHA-256 `c43d5b9c0753cda33044ebca6f5bc5aea0fa2835a8bfc131523d23f7fd1dc7d9`）与 `llm-core/exposure-ledger.json`（SHA-256 `ea350f98b9640fd40f1d2566038ab1e36e508dc3a71771a5f1df6d8fa601ad35`），并在结束后完成干净 GPU 交还（`c3-smoke-closeout.json` `clean=true`）。

### 3.1 总体统计、就绪阶段与形式指标覆盖

- **覆盖进度**：`spec_id="M0-C0-20261107-r1"`，总矩阵 `360` 行；已执行 `40 / 360` 行（`smoke_rows_count=40`）；`16 passed`、`24 failed`、`320 not_reached`（`deferred_rows_count=320`）；`matrix_complete=false`、`complete=false`、`performance_authority=false`，`identity_sha256="9a4effca21cbc71611c27b0b73acd51c31ede23219fcd1ef114a595319971cbe"`。
- **异常/分类边界**：`0` 例 `infrastructure_error`、`0` 例 `budget_exhausted`、`0` 例按基础设施未分类的 `unsupported`/`unproved`（注：`formal_analysis` 失败在账本中记录为 `reason_code="formal_analysis_failed"`，由 `classify_failure` 统一归类为 `failed` 并原样保留 `formal.message`）。
- **时间与存储成本**：`smoke_elapsed_seconds=449.49042280099275`（smoke 上限 `7,200 s`；宿主机 wall `465.32 s`），单行平均 `11.24 s`、最大 `16.16 s`；`smoke_evidence_bytes=6,899,965` B（约 `6.58 MiB`）。
- **就绪与形式子维度（`readiness_stage_counts` / `formal_metrics_counts`）**：
  - `make_fx_aten`：`graph_extraction` `20 passed`、`ir_conversion` `20 passed`、`conversion_verification` `20 passed`（注：`make_fx_real_retry_observability.observable_in_current_artifacts=false`，因当前 `make_fx_backend` 回退不标注 `operator_graph.yaml`）。
  - `torchview_extended_einsum`：`graph_extraction` `20 passed`、`ir_conversion` `19 passed / 1 failed`、`conversion_verification` `17 passed / 2 failed / 1 not_run`。
  - 形式分析指标（`formal_metrics_counts`）：`formal_attempted=37`、`formal_passed=16`；`16` 个通过样本的 `external_reads_quality` 均为 `exact`，且三种视图（`cache_aware`、`ideal_fused`、`unfused`）均达到 `available:ready=16`。

### 3.2 按 `IRPath` × `operation_family` 分解

| `IRPath` | `operation_family` | 总行数（已测） | Readiness 通过 | `PASSED` | `FAILED` | 失败阶段分布 |
|---|---|---:|---:|---:|---:|---|
| `make_fx_aten` | `linear` | 8 | 8 | 4 | 4 | `formal_analysis` (4) |
| `make_fx_aten` | `norm_activation` | 4 | 4 | 4 | 0 | — |
| `make_fx_aten` | `position` | 2 | 2 | 0 | 2 | `formal_analysis` (2) |
| `make_fx_aten` | `attention` | 6 | 6 | 4 | 2 | `formal_analysis` (2) |
| **`make_fx_aten` 小计** | — | **20** | **20** | **12** | **8** | `formal_analysis` (8) |
| `torchview_extended_einsum` | `linear` | 8 | 8 | 0 | 8 | `formal_analysis` (8) |
| `torchview_extended_einsum` | `norm_activation` | 4 | 4 | 4 | 0 | — |
| `torchview_extended_einsum` | `position` | 2 | 2 | 0 | 2 | `formal_analysis` (2) |
| `torchview_extended_einsum` | `attention` | 6 | 3 | 0 | 6 | `ir_conversion` (1), `conversion_verification` (2), `formal_analysis` (3) |
| **`torchview_extended_einsum` 小计** | — | **20** | **17** | **4** | **16** | `ir_conversion` (1), `conversion_verification` (2), `formal_analysis` (13) |

### 3.3 24 行失败原因明细

#### `make_fx_aten`（8 例失败，均已完成 graph/IR/verification 就绪）

1. **`linear`（4 例 `llm.linear.matmul_{bias,gelu,relu,silu}.v2`）**：`formal_analysis` 失败（`reason_code="formal_analysis_failed"`），`strict analysis did not produce a complete tile-aware bound`。
2. **`position`（2 例 `llm.position.rope_{interleaved,yarn}.v2`）**：`formal_analysis` 失败，`external reads are unproved: read_semantics_unproved`。
3. **`attention`（2 例）**：
   - `llm.attention.attention_sink.v2`：`formal_analysis` 失败，`external reads are unproved: effect_boundary`。
   - `llm.attention.sliding_window.v2`：`formal_analysis` 失败，`operation 'arange' has no amd_resource_v3 rule`。

#### `torchview_extended_einsum`（16 例失败：3 例前端/验证失败 + 13 例形式分析失败）

1. **`ir_conversion`（1 例 `llm.attention.mqa_decode.v2`）**：`strict_conversion_failed` — `cannot exactly convert Model.expand_as (expand_as): Unsupported operation: expand_as`。
2. **`conversion_verification`（2 例 `attention`）**：
   - `llm.attention.sliding_window.v2`：`exact_replay_failed` — `missing external tensor Model.hidden-tensor_8.Output`。
   - `llm.attention.attention_sink.v2`：`gradient_verification_failed` — `numerical mismatch: matched_ratio=0, required=1, max_abs=6.5`。
3. **`position`（2 例 `llm.position.rope_{interleaved,yarn}.v2`）**：`formal_analysis` 失败，`external reads are unproved: read_semantics_unproved`。
4. **`linear`（8 例全部：`llm.linear.{batched_bias,batched_matmul,grouped_matmul,grouped_silu,matmul_bias,matmul_gelu,matmul_relu,matmul_silu}.v2`）与 `attention`（3 例：`llm.attention.{gqa_causal,local_global,masked_softmax}.v2`）**：共 `11` 例在 `formal_analysis` 失败（`reason_code="formal_analysis_failed"`），`strict analysis did not produce a complete tile-aware bound`。

### 3.4 双路径对照、阶段资源成本与剩余 320 行预算外推

- **工作负载成对状态分布（`status_pair_counts`，共 `20` 组）**：`passed__vs__passed=4`、`passed__vs__failed=8`、`failed__vs__failed=8`。
- **双路径共同通过样本（4 个 `norm_activation` 算子）**：两路径在 `flops=0`、`external_read_bytes`、`lower_bound_seconds` 与三视图（`cache_aware`/`ideal_fused`/`unfused` 均为 `available`）上完全一致：
  - `llm.norm_activation.add_rmsnorm.v2`：`external_read_bytes=65,664`、`lower_bound_seconds=4.096e-07`、`resource_work_equal=false`（两条路径将 `32,704` 个元素分别归因于 `sfu.fp32` 与 `valu.fp32`）。
  - `llm.norm_activation.bias_gelu.v2`：`external_read_bytes=65,664`、`lower_bound_seconds=4.096e-07`、`resource_work_equal=true`。
  - `llm.norm_activation.rmsnorm.v2`：`external_read_bytes=65,536`、`lower_bound_seconds=4.092e-07`、`resource_work_equal=false`（两条路径同样将 `32,704` 个元素分别归因于 `sfu.fp32` 与 `valu.fp32`）。
  - `llm.norm_activation.swiglu.v2`：`external_read_bytes=65,664`、`lower_bound_seconds=4.096e-07`、`resource_work_equal=true`。
- **各阶段实测开销（`stage_cost_summary`）**：

| 阶段 | 调用次数 | 中位耗时 (s) | 最大耗时 (s) | 累计耗时 (s) | 设备峰值显存 | Host RSS 峰值 |
|---|---:|---:|---:|---:|---:|---:|
| `readiness:graph_extraction` | 40 | 0.478 | 2.182 | 33.12 | 85,087,744 B（~81.15 MiB） | 1,643,200,512 B（~1.53 GiB） |
| `readiness:ir_conversion` | 40 | 0.051 | 0.118 | 2.36 | 83,945,984 B（~80.06 MiB） | 1,643,986,944 B（~1.53 GiB） |
| `readiness:conversion_verification` | 39 | 2.832 | 3.760 | 91.34 | 170,590,720 B（~162.69 MiB） | 1,995,411,456 B（~1.86 GiB） |
| `formal:graph_extraction` | 37 | 0.465 | 2.190 | 26.75 | 85,087,744 B（~81.15 MiB） | 1,643,728,896 B（~1.53 GiB） |
| `formal:ir_conversion` | 37 | 0.049 | 0.120 | 2.12 | 83,945,984 B（~80.06 MiB） | 1,644,253,184 B（~1.53 GiB） |
| `formal:conversion_verification` | 37 | 2.838 | 3.767 | 90.99 | 170,590,720 B（~162.69 MiB） | 1,994,215,424 B（~1.86 GiB） |
| `formal:formal_analysis` | 37 | 0.204 | 2.739 | 20.13 | — | 1,996,095,488 B（~1.86 GiB） |
| `formal:mapper` | 32 | 0.223 | 1.632 | 13.94 | — | 1,995,788,288 B（~1.86 GiB） |

- **剩余 `320` 行预算外推（仅下界参考）**：按 `40` 行耗时（`449.49 s`，单行均值 `11.24 s`、峰值 `16.16 s`）与证据体积（`6.58 MiB`）线性外推，剩余 `320` 行预计耗时 `3,595.92 s`（vs 剩余活动预算 `42,750.51 s`）、总证据约 `0.0578 GiB`（vs `50.0 GiB` 上限），`fits_remaining_activity_budget_linearly=true`。由于剩余队列包含更大尺度的 `capacity-high` 工作负载，验证重放与映射阶段单行耗时可能上升，该外推仅作下界参考。

---

## 4. 独立无设备审计与 GPU 状态交还

| 检查项 | 产物路径 | 验证结果 |
|---|---|---|
| 独立无设备审计 | `llm-core/cpu-audit.json`（SHA-256 `851acbfec7ee1219ceb59a0b5788639712add40b1d20e557c8ef6845ec45cbd3`） | `audit_passed=true`、`gpu_initialized=false`、`rocr_visible_devices=""`、`cli_preflight_output="validated metadata: 360 rows; no worker executed"`、`resume_identity_matches_full_audit=true`（`9a4effca21cbc71611c27b0b73acd51c31ede23219fcd1ef114a595319971cbe`），宿主机 wall `5.61 s` |
| GPU 交还（代表槽 / C2 / C3 smoke） | `host-run/{b2-representatives,c2,c3-smoke}-closeout.json` | 3 阶段均回报 `clean=true`（风扇/功耗回落 `AUTO`、GPU 进程数为 `0`、L1/L2 锁释放、3 项阶段凭据均完整回传） |

## 5. 下一步决策点

C2 已完整冻结 `LLM_CORE` 目标视图，C3 前 `40` 行冒烟证明执行管线、预算记录、`--resume` 身份一致性与 GPU 隔离交还均工作正常；后续经授权的 `r2` 12 小时全活动有界推进及终态无设备审计结论见 [终态 C2 与 C3 结果报告（2026-11-20）](solar-c2-c3-results-2026-11-20.md)。
