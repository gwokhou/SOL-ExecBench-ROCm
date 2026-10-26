# SOLAR 阶段 2 CPU 逐槽位对照（2026-10-26）

本报告对应 `projection-api/attempt-s2-02`，源码集合摘要为 `fe67511b9d4d6322a61a39cf293d3034ab1601b96c0018b9da62ba45c79272b4`。采集前后摘要相同，原始 BASE-01 的 627 个工件绑定重新核验通过。完整数值和三视图字段保存在 `projection-api/comparison-02.json`。

32 个 F 槽位均尝试；30 个语义／计量完成，2 个 torchview `tril` 转换拒绝。20 个来源合格的槽位完成公开分析和复用，完整分析内容逐字段相等；其余 12 个复用为 `not_run/source_analysis_not_eligible`。已计量槽位与冻结数学 oracle 的差异为零。R 组 16 个槽位仍为 `not_run`，不从分母删除。

以下 T 为 `torchview_extended_einsum`，A 为 `make_fx_aten`；“容量缺证”表示 `capacity_coverage_incomplete`，“读缺证”表示 `external_reads_unproved`。旧 BASE-01 没有三视图字段，记为 `not_recorded`，不倒填。

| 样本 | 路径 | BASE-01／当前计量 | 当前 unfused／ideal／cache | 完整分析＝复用 | 相对基线数值变化 |
| --- | --- | --- | --- | --- | --- |
| F01-D | T | completed / completed | 可用/可用/容量缺证 | 是 | 无 |
| F01-D | A | completed / completed | 可用/可用/容量缺证 | 是 | 无 |
| F01-Q | T | completed / completed | 可用/可用/容量缺证 | 是 | 无 |
| F01-Q | A | completed / completed | 可用/可用/容量缺证 | 是 | 无 |
| F02-D | T | completed / completed | 可用/可用/可用 | 是 | 无 |
| F02-D | A | completed / completed | 可用/可用/可用 | 是 | 无 |
| F02-Q | T | completed / completed | 可用/可用/可用 | 是 | 无 |
| F02-Q | A | completed / completed | 可用/可用/可用 | 是 | 无 |
| F03-D | T | completed / completed | 可用/可用/容量缺证 | 是 | 无 |
| F03-D | A | completed / completed | 可用/可用/容量缺证 | 是 | 无 |
| F03-Q | T | completed / completed | 可用/可用/容量缺证 | 是 | 无 |
| F03-Q | A | completed / completed | 可用/可用/容量缺证 | 是 | 无 |
| F04-D | T | completed / completed | 可用/可用/可用 | 是 | fused_bytes: 264.0 → 392.0 |
| F04-D | A | completed / completed | 可用/可用/可用 | 是 | fused_bytes: 264.0 → 392.0 |
| F04-Q | T | completed / completed | 可用/可用/可用 | 是 | fused_bytes: 392.0 → 648.0 |
| F04-Q | A | completed / completed | 可用/可用/可用 | 是 | fused_bytes: 392.0 → 648.0 |
| F05-D | T | completed / completed | 可用/可用/可用 | 是 | 无 |
| F05-D | A | completed / completed | 可用/可用/可用 | 是 | 无 |
| F05-Q | T | completed / completed | 可用/可用/可用 | 是 | 无 |
| F05-Q | A | completed / completed | 可用/可用/可用 | 是 | 无 |
| F06-D | T | not_run / not_run | 未到达/未到达/未到达 | 来源不合格／未复用 | 无 |
| F06-D | A | completed / completed | 读缺证/读缺证/读缺证 | 来源不合格／未复用 | 无 |
| F06-Q | T | not_run / not_run | 未到达/未到达/未到达 | 来源不合格／未复用 | 无 |
| F06-Q | A | completed / completed | 读缺证/读缺证/读缺证 | 来源不合格／未复用 | 无 |
| F07-D | T | completed / completed | 读缺证/读缺证/读缺证 | 来源不合格／未复用 | 无 |
| F07-D | A | completed / completed | 读缺证/读缺证/读缺证 | 来源不合格／未复用 | 无 |
| F07-Q | T | completed / completed | 读缺证/读缺证/读缺证 | 来源不合格／未复用 | 无 |
| F07-Q | A | completed / completed | 读缺证/读缺证/读缺证 | 来源不合格／未复用 | 无 |
| F08-D | T | completed / completed | 读缺证/读缺证/读缺证 | 来源不合格／未复用 | valu.fp32: 0 → 8 |
| F08-D | A | completed / completed | 读缺证/读缺证/读缺证 | 来源不合格／未复用 | valu.fp32: 0 → 8 |
| F08-Q | T | completed / completed | 读缺证/读缺证/读缺证 | 来源不合格／未复用 | valu.fp32: 0 → 12 |
| F08-Q | A | completed / completed | 读缺证/读缺证/读缺证 | 来源不合格／未复用 | valu.fp32: 0 → 12 |

三视图中 12 个无需收缩容量 mapper 的槽位具有可用 cache-aware 估算；另 8 个收缩槽位缺容量证明。采集器历史摘要字段 `complete_cache_aware=0` 只反映其既有正式容量资格计数，不能解释成所有新视图都不可用。真实 mapper、GPU 实验和正式发布均为 0。

`check-09` 的 197 项受影响回归及 Ruff、format、ty、coupling、readability、文档检查全部通过。先前宽范围回归、失败尝试和两轮完整采集均保留；本结果不替代后续源码检查点。嵌套 mapper 的外层超时清理仍需独立 CPU 进程测试，不能以启动期超时测试代替。

阶段状态和下一步见 [HANDSOFF](../../HANDSOFF.md#solar-evolution)；能力范围见 [S2 检查点](solar-stage2-cpu-checkpoint.md)。
