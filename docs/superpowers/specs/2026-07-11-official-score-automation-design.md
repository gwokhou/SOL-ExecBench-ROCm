# Official Score 自动化与统一口径设计

## 目标

实现可由 dataset runner 与 CLI 自动生成的 `official_score_evidence.v1`，并将
official score 的 suite 聚合统一为固定 workload 分母：所有失败、缺失证据或被
official gate 阻断的 workload 均按 0 分贡献到 suite score。该变更解决
`docs/sol_score_gap_and_amd_reuse_report.md` 第 7 节第 4、5 项：固定失败/未评分
的分母策略，以及将 official score evidence 接入 runner、CLI 和 sidecar。

## 范围与非目标

范围包括：

- 复用现有 `official_score.py` 的 workload-level authority gate；
- 为 suite evidence 定义、序列化并测试固定分母零分聚合；
- 将官方 score 生成接入现有 dataset 执行和派生报告路径；
- 以 CLI 显式输入的 scoring baseline、聚合策略与可选 measured-baseline coverage
  report 构建可追溯 sidecar；
- 更新评估契约、分母策略与用户文档。

本次不校准 gfx1200 硬件模型、不扩展 SOL bound 算子覆盖，也不把 provisional 或
degraded bound 升格为 authority。它们仍由现有 blocker 阻止 official authority。

## 决策

### 统一的聚合策略

唯一支持的官方策略名为 `fixed_suite_denominator_zero_for_blocked`。

- suite 分母为本次运行预期写入 official evidence 的全部 workload 数；
- 有 workload authority 的项贡献其计算出的 SOL score；
- 无 authority 的项保留 `score: null`、`status: "blocked"` 和全部 blocker，
  但在 suite 聚合时贡献 `0.0`；
- suite score 为所有贡献值的算术均值；空 suite 的 score 为 `null`，而不是除以零；
- `score_authority` 仅当 suite 非空且没有 blocked workload 时为 true。任何 blocked
  workload 都不能因零分记账而被伪装成正式已评分。

suite 输出会同时包含总 workload 数、authoritative/scored 数、blocked 数、
zero-scored 数、blocker 汇总和聚合策略。`amd_native_score` 保持它现有的诊断/派生
语义，不改变为 official 口径。

### 数据流与工件

```text
trace + workload + SOL bound + scoring baseline
                 ↓
        amd_native_score.v1（诊断输入）
                 ↓  + policy + optional coverage report
       official_score evidence gate（逐 workload）
                 ↓
 official_score_evidence.v1 sidecar（固定分母零分聚合）
```

dataset runner 在已生成每个 problem 的 AMD-native score 后，收集整套 suite score，
调用 `build_official_score_suite_evidence()`，并将其 JSON 写入用户指定的 sidecar
路径。每项保留 AMD score、trace/timing、SOL bound、baseline、hardware model 与
派生证据引用；suite 还记录 policy 与 coverage precondition 的结果。

官方 sidecar 只能由显式的 official 运行请求写出。它需要：

- 有效的 `sol_execbench.scoring_baseline.v1`；
- 策略 `fixed_suite_denominator_zero_for_blocked`；
- 若 baseline source 为 `measured_baseline_registry`，则提供确认通过的 coverage
  report；
- 每个 workload 继续通过现有的 bound、硬件、模型、warning 和 latency gate。

任何不满足项应产生 blocked workload/zero contribution（当 suite 已可构建）或在
CLI 输入层清晰拒绝缺少 required official inputs 的请求；不会回退到
`reference_latency` 并宣称 official score。

### CLI 与运行接口

在现有 dataset-run 命令的 scoring 参数组中增加 official-output 参数，而不是新增
一个脱离运行上下文的 post-processing 命令。接口应要求 official sidecar 输出路径
和 official aggregation policy；复用现有 `--scoring-baseline` 输入，并为 measured
baseline registry/coverage report 提供明确、成对的输入选项。

普通 AMD-native 报告仍可在没有 official 参数时生成。只要请求 official sidecar，
CLI 会在运行开始前校验参数组合，并在结束时写入单一版本化 JSON 工件。这样可避免
人工离线拼接 score、baseline、trace 和 coverage 来源。

### 错误处理

- policy 缺失或非支持值：拒绝 official-output 请求，并显示接受的唯一策略名；
- scoring baseline 缺失、格式无效或不包含 workload：相应 workload 保持 blocked，
  suite 以零分反映缺口；若根本未提供 baseline artifact，则 CLI 对 official-output
  请求失败；
- coverage report 不确认：保留 `baseline_coverage_failed` 及具体原因，相关项 0 分；
- missing/degraded/unsupported bound、未验证硬件/模型和非有限 latency：复用稳定的
  现有 blocker，不生成 authority；
- sidecar 写入失败：使运行明确失败，避免声称已经生成 official 工件。

## 测试与验收

单元测试验证 policy 规范化/拒绝、混合 scored-blocked suite 的固定分母均值、全阻断
suite 得 0、空 suite 得 null、计数和 blocker summary 的稳定序列化，以及任何 blocked
项使 suite authority 为 false。

dataset/CLI 集成测试用小型 fixture 验证：带 scoring baseline 的运行写出 official
sidecar；sidecar 引用来源 AMD score 与输入 evidence；阻断项确实进入分母；缺少
official 输入或参数组合被拒绝；仅生成 AMD-native 报告的现有调用保持兼容。

验收时，`official_score_evidence.v1` 是唯一 official 输出，且可从 JSON 同时确定
policy、固定分母、零分项、所有 blocker 和输入 provenance。文档必须明确该自动化
不会把当前 gfx1200 provisional/degraded evidence 升级为 leaderboard authority。

## 文档更新

更新 `docs/EVALUATOR-CONTRACT.md`、`docs/internal/RDNA4-DENOMINATOR-POLICY.md`、
`README.md` 或对应 CLI 文档，说明 official 与 AMD-native derived score 的区分、唯一
aggregation policy、命令用法、sidecar 内容和 authority 边界。
