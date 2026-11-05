# L3-a 静态 fake 取图试点 + stage 测量报告：预固定验收规格

制定日期：2026-11-05（实施前预固定）。本文固定 [演进路线](solar-evolution-roadmap-2026-11-05.md)
L3-a 的验收口径，并把 L4 第 1 项（各 stage 时间/内存测量报告）合并为同一次测量活动。
规格先于实施结果固定；实施后只允许追加结果，不允许回改本文的分母、oracle 或阈值。

## 0. 交付物与闭合判定

L3-a 闭合当且仅当以下四项全部成立：

1. **M（测量报告）**：R16 双路径 16 槽位的 per-stage 墙钟时间与峰值内存报告，
   写入 `data/outputs/solar-l3a-20261105/measurement/`，§2 有效性 gate 通过。
2. **P（隔离试点）**：仅 `make_fx_aten` 路径的静态 fake（FakeTensorMode，具体 shape）
   取图试点在 §3 分母上执行，逐项 + 聚焦样例产出 §4 oracle 判定。
3. **D（决策）**：按 §5 预固定规则 D1–D4 评估并记录"是否改造生产取图路径"的结论。
4. **登记**：HANDSOFF 追加 L3-a 条目；roadmap L3-a 标注状态；生产默认路径零改动
   （结构保证：试点代码只新增模块，不注册进 backend registry，不修改任何默认值）。

## 1. 测量方法与词汇（预固定）

- 驱动方式：worker 子进程按生产适配器（`analyze_workload` → `invoke_solar` → `solar.api.analyze`）
  的同一请求构造驱动**真实** `solar.api.analyze()`；stage 归因通过在 worker 进程内对
  生产函数（`solar.pipeline.analysis` 命名空间的四个 stage 函数、`solar.api` 的发布函数、
  `solar.graph.make_fx_backend.trace_make_fx_reference`、`ArchitectureProfile.load`、
  `graph_analyzer.OrojenesisRunner`）包一层计时探针实现，不修改任何生产源码。
- 设备：主运行 `cuda:0`（gfx1200，正式设备 gate 照常）；`--device cpu` 为回退模式。
  测量是工程测量：不申请 STABLE_PEAK 时钟锁，槽位串行执行。
- mapper：pinned 可信 home（`sha256 18591892…d77a`），`require_orojenesis=True`。
- 每 stage 行固定字段：`stage`、`status`（passed/failed/not_run）、`wall_seconds`
  （`perf_counter`）、`peak_rss_kb`（`/proc/self/status` VmHWM，stage 前
  `clear_refs:=5` 重置）、`gpu_peak_allocated_bytes` / `gpu_peak_reserved_bytes`
  （`torch.cuda.reset_peak_memory_stats`，仅 cuda 模式）、失败时 `reason_code`。
- 归因字段（每 stage 桶）：`reference_calls`（次数与累计秒）、`input_factory_calls`、
  `mapper_runs`（次数、累计秒、采样到的 mapper 子进程 VmHWM 峰值，50ms 采样）。
- 计入的 stage 词汇（固定枚举，额外行只允许前缀 `component/`）：
  `workload_context_load`、`architecture_load`、`graph_extraction`、`ir_conversion`、
  `conversion_verification`、`formal_analysis`、`publication_extract_bound`、
  `publication_finish_artifacts`、`publication_manifest`、`publication_verify_tree`、
  `input_generation_standalone`（在主管线完成后单独测量，避免污染后续 stage 峰值）。
  make_fx 路径另有 `component/make_fx_trace`（trace_make_fx_reference 子区间）。
- 进程级：worker 记录 `import_torch_seconds` 与退出前 VmHWM（=整进程峰值）。
- 每槽位一个子进程（进程组 + 超时 3600s + 有界输出捕获）；campaign 汇总按 stage
  给出 min/median/mean/p95/max。

## 2. M 的分母、负例与有效性 gate（预固定）

- 分母：`run_r16_curated_qualification.R16_ITEMS`（冻结 8 workload）×
  {`make_fx_aten`, `torchview_extended_einsum`} = 16 槽位。
- 负例（先行声明）：R08/torchview 槽在 `graph_extraction` 以
  `graph_extraction_failed` 失败（与 L1 curated gate 的声明一致）；该槽位记录
  到失败为止的 stage 行，其余 stage 记 `not_run`，**不**计为 campaign 失败。
- 有效性 gate（全部满足才判 M 通过）：
  1. 16/16 槽位各有 record（worker 崩溃/超时 → 该槽位 failed，campaign 失败，不许静默跳过）；
  2. 除已声明预期失败槽位外，其余槽位 10 个主管线 stage 全 `passed` 且
     `wall_seconds > 0`、`peak_rss_kb > 0`；
  3. R08/torchview 恰在 `graph_extraction` 失败且 `reason_code = graph_extraction_failed`；
  4. cuda 模式下每完成槽位 `gpu_peak_allocated_bytes` 非空；
  5. 汇总报告含身份绑定：git commit、corpus manifest sha256、mapper sha256、
     device 与 gfx target、torch/ROCm 版本、`utc_timestamp()`。

**勘误（2026-11-05，campaign 执行中追加，依据先于本 campaign 的证据）**：
第 2/3 条最初写成"除 R08/torchview 外 15 槽位全 passed"，漏声明了第二个预期
失败——R08/make_fx 槽：readiness 三阶段通过（与 L1 curated gate 一致），但
formal bound 提取以 `formal_analysis_failed`（"external reads are unproved:
copy_boundary, read_semantics_unproved, unproven_alias"）拒绝，失败行落在
`publication_extract_bound`。这不是本 campaign 的新回归：L1 证据（HANDSOFF
2026-10-29 快照"14 份中 12 个可计价"；`r16-terminal-ledger.json` 只给 R06 记
正式 bound）从未声称 R08 可计价。修正后的 gate 规则：R08/make_fx 的前缀
stage（至 `formal_analysis`）全 passed，恰在 `publication_extract_bound`
失败且 outcome `reason_code = formal_analysis_failed`。分母、其余阈值不动。

## 3. P 的分母（预固定）

- 8 个 make_fx 槽位（R01–R08 的 `make_fx_aten` 侧，cuda:0）。
- 3 个聚焦样例（`scripts/internal/solar/l3a_samples.py`，测试与 campaign 共用）：
  - `mutation_inplace`：含 `aten.add_` 原地写与别名返回；
  - `scalar_constants`：非张量标量参数进入算子参数；
  - `data_dependent_branch`：`if x.sum() > 0` 数据依赖分支。

## 4. P 的 oracle（预固定）

对每个 make_fx 槽位，真实取图（生产 backend）与 fake 取图（试点模块，同一
`ReferenceGraphSerializer` 序列化）各产出一份 `operator_graph.yaml`：

- **O1 语义投影等价（tier-1，必须）**：逐层比较投影
  `{type, semantic_op(kind/target/exact_target/overload/arguments/kwargs/effects),
  tensor_shapes, tensor_dtypes, source_input_index}`、`outputs` 列表、
  `graph_signature.user_inputs/user_outputs`。允许节点命名漂移（tensor_names、
  connections 不在投影内）；其余任何差异判不等价。
- **O2 文档全等（tier-2，只报告）**：整份 YAML dict 全等，作为观测指标不计 gate。
- **O3 下游真实回放（必须）**：fake 图经 `convert_operator_graph`（aten）+
  `verify_callable_conversion`（生产 corpus 策略、真实回放）通过。
- **O4 聚焦样例（必须）**：mutation——`effects.mutates` 投影等价且 O3 通过；
  constants——`arguments` 中 `{"value": …}` 投影等价；
  data-dependent——fake 取图必须**抛错拒绝**（不得产出图）；真实取图对该样例的
  行为只记录不判 gate。
- **O5 收益测量（只报告）**：同一槽位 real vs fake 的 `graph_extraction` 墙钟、
  trace 子区间墙钟、GPU 峰值差。

## 5. 决策规则 D1–D4（预固定）

"建议改造生产 make_fx 取图为 fake" 当且仅当全部满足：

- **D1**：8/8 make_fx 槽位 O1+O3 通过。
- **D2**：3/3 聚焦样例 O4 通过。
- **D3**：O5 中 fake 的 `component/make_fx_trace + observe` 墙钟（以
  `graph_extraction` 行计）≤ 0.7× real 于 ≥6/8 槽位，且 GPU 峰值不升高（≤ real）
  于 ≥6/8 槽位。
- **D4**：结构检查通过：试点未改任何默认路径（git diff 仅新增模块与脚本）。

任一不满足 → 记录 no-go/defer 及对应数字；L3-a 仍按 §0 闭合（试点已执行、
决策已记录），改造本身不属于 L3-a 交付物。

## 6. 证据目录与 schema

```
data/outputs/solar-l3a-20261105/
  measurement/<ir_path>/<R-id>/record.json     # sol_execbench.solar_l3a_measurement.v1
  measurement/measurement-report.json          # 同 schema，聚合 + 有效性 gate 判定
  fake-pilot/<R-id>/pilot.json                 # sol_execbench.solar_l3a_pilot.v1
  fake-pilot/focused/<sample>.json             # 同 schema
  fake-pilot/pilot-summary.json                # 聚合 + D1–D4 判定
  worker-logs/<slot>.{out,err}.log             # 有界
```

record 内不复制既有生产 artifact 本体，仅登记其 sha256。

## 7. 边界与不做的事

- 不修改生产默认路径、backend registry、任何既有 schema 或验收语义。
- 测量探针只存在于 campaign 脚本与 worker 进程内（monkeypatch 包裹），不进 `src/`。
- 试点模块进 `src/solar/graph/`（新增文件，孤立可引用，与 L1-c `topk_gather.py`
  同模式），不注册 backend、不进公共 CLI。
- 墙钟为工程测量：报告明示无时钟锁、宿主可能有后台负载；结论用于相对比较与
  L4 优化排序，不作为正式 benchmark 证据。
