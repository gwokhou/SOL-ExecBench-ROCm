# E0 成本调用计数勘误（B1）

日期：2026-11-08。修订标识：`M0-E0-20261107-r2-counts`。
本次用户明确同意算术勘误：成本候选预热与计时调用上限从23,040更正为25,920；
保留144个session、6小时总GPU预算以及全部检查。
此修订只更正派生计数，不修改输入族、实例派生、候选、oracle、容差或正式评分契约。

原 [E0 r1](solar-m0-judge-preregistration-2026-11-07.md) 字节保留，SHA-256：
`4e8bea92afed0c345bad1a335e0833969f394ed152c24048f27500aadde9bc19`。
后续 E2 的活动身份须同时绑定该原件和本勘误；逻辑 case 继续沿用 r1 标识，
不得借计数修订重新抽样或将开发实例认定为新留出。

## 1. 依据与修正

原文的“10 warmup、50 iterations、3 trials”与现行实现一致，但23,040的派生数错误。
[measure_solution_latency](../../src/sol_execbench/core/bench/eval_timing.py)
逐 trial 调用 `_measure_solution_trial`，每次都传入完整 `warmup_runs`；
[event timing](../../src/sol_execbench/core/bench/timing.py) 和 CPU 对应路径均逐 trial 预热。
因此每个session候选预热+计时调用为 `3 × (10+50) = 180`，
全成本活动为 `24块 × 3重复 × 2模式 × 180 = 25,920`。
没有把预热从每trial改成每session，也没有增加session或重复数。

## 2. 固定生产协议的附加调用

以下是正常完整执行的上限；前置拒绝或早停只会减少实际调用，不能补选或重试填满。
固定 `min_measurement_time_seconds=null`；不启用 counter replay 或自适应采样。
12个候选均为普通 PyTorch callable，不触发 compiled callable 的可选编译器预热。

| 活动 | session数 | 每session候选调用 | 每session reference 调用 | 全活动上限 |
| --- | --- | --- | --- | --- |
| 正确性 | 432逻辑单元×3=1296 | 10个正确性round + 1个计时调用 = 11 | 10个正确性round + 1个timing准备 + 1个计时验证reference = 12；`benchmark_reference=false` | 候选14,256；reference15,552 |
| 成本 | 24块×3×2=144 | 10个正确性round + 180个预热/计时 = 190 | 10个正确性round + 1个timing准备 + 180个独立reference基准调用 + 180个逐调用验证reference = 371；`benchmark_reference=true` | 候选27,360，其中预热/计时25,920；reference53,424 |
| 两活动合计 | 1440 | — | — | 候选41,616；reference68,976 |

调用依据为 [run_correctness_rounds](../../src/sol_execbench/core/bench/eval_correctness.py)
的10轮、[timing准备](../../src/sol_execbench/core/bench/eval_workload_execution.py)
的单次请求，以及 [ReferenceService](../../src/sol_execbench/core/bench/reference_service.py)
的逐调用求值和独立reference基准。逐计时输出重验不再次执行候选；
附加oracle只消费可信输入/输出，不允许为了测量而悄悄重跑候选。
这些数字是Python可调用对象的调用数，不是设备dispatch数；缓存清理、输入生成、同步、
初始化与进程/IPC成本另外留证，不能从上表推出GPU工作量或耗时。

## 3. 修订时点与不变项

修订发生在 B1 只读代码核查时。E1 的72个开发输入/288个候选单元已经暴露；
其负结果继续保留，未据此改候选、种子或阈值。E2成本实验没有启动，h0密钥、commitment、
实例均未生成；真实LLM_CORE目标视图尚未生成。
该修订不是看过成本结果后放宽预算：6小时GPU、12小时CPU oracle、300s成本session、
120s正确性单元/单次oracle、20GiB证据、jobs=1及内存上限全部维持原规定。
预算到限保留原分母并标INCOMPLETE。

本文件只补全 B1 所需的计数与解释。E2仍须通过B2及后续执行授权；
若实际活动需要修改冻结源码或改变调用协议，必须重新冻结B1，不能沿用本次资格身份。
