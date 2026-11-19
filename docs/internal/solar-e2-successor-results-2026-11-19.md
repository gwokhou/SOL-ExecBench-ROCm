# E2 裁判辨错与成本实验结论

2026-11-19 收尾。**修正版 E2 完成全部预登记分母，终态核验通过。裁判未达到 E0 的整体通过标准，原因是 LayerNorm L1 的 15 个逻辑单元未满足三次重复一致。** 必检违例全部检出，已独立认证的合法对照在数值正确性阶段零误拒。计时校验的拒绝和独立标签不确定项另列。 [E0 验收条件](solar-m0-judge-preregistration-2026-11-07.md#6-验收产物与变更规则) · [终态核验](../../data/outputs/solar-e2-successor-closeout-20261118/terminal-verification.json)

## 执行与证据

| 分母 | 完成 | 失败、中断或未到达 |
| --- | ---: | ---: |
| 正确性逻辑单元 | 432 / 432 | 0 |
| 正确性会话 | 1,296 / 1,296 | 0 |
| 在线成本会话 | 144 / 144 | 0 |
| 离线回放 | 72 / 72 | 0 |

三个任务各完成 432 个正确性会话。开发实例 `d0`、`d1` 和本次独立留出 `h0` 各有 432 个会话。12 个候选的 SHA-256、留出承诺与冻结身份均通过核对。独立审计复现了 36,477 份实际输入的身份，核验了相应的原始观测和 12,603 份离线诊断与原始工件的绑定。四个宿主阶段均以退出码 0 结束。 [独立审计](../../data/outputs/solar-e2-successor-reaping-frozen-20261118/successor-confirmation/independent-evidence-audit.json) · [宿主终态](../../data/outputs/solar-e2-successor-reaping-frozen-20261118/host-run/completed.json)

GPU 宿主阶段墙钟为 2 小时 59 分 59 秒，累计 oracle CPU 时间为 45 分 10 秒，最终证据分配量为 12.37 GiB，分别低于 6 小时、12 小时和 20 GiB 上限。证据空间数字采用终态实际分配量。GPU 交还回执记录时钟恢复 `AUTO`、评测进程数归零、两级锁释放和所有者清空。 [GPU 交还收据](../../data/outputs/solar-e2-successor-reaping-frozen-20261118/successor-confirmation/gpu-closeout.json) · [资源台账](solar-e2-successor-budget-2026-11-18.md)

## 辨错与裁判判定

| 任务 | P1 必检原契约违例已检出 / 已认证 | 数值阶段确认误拒 / 已认证合法 P0、P3 | 不确定对照 | P2 预登记见证 |
| --- | ---: | ---: | ---: | --- |
| GEMM | 27 / 27 | 0 / 216 | 0 | G2 小量保真 27 / 27 |
| 求和归约 | 27 / 27 | 0 / 216 | 0 | R2 小量保真 27 / 27 |
| LayerNorm | 27 / 27 | 0 / 126 | 90 | L0 方差不稳定 27 / 27 |
| 合计 | 81 / 81 | 0 / 558 | 90 | 81 / 81 |

54 个 G2/R2 的 P2 会话通过了原裁判，独立诊断确认其违反更强的小量保真条件。27 个 L0 的 P2 会话由原裁判数值拒绝，独立诊断确认了方差不稳定见证。其余 486 个会话保留为描述性观测。 [正确性全表](../../data/outputs/solar-e2-successor-reaping-frozen-20261118/successor-confirmation/report/correctness.json) · [见证索引](../../data/outputs/solar-e2-successor-closeout-20261118/counterexample-and-witness-index.json)

648 个 P0/P3 对照中的 90 个被列为不确定，全部来自 LayerNorm L0/L1，均观测到实际参考输出与独立 oracle 的分歧。原裁判在这 90 个会话中给出 48 次通过、42 次数值拒绝。它们保留在实验全分母中，单列于已认证合法对照的误拒率。

已认证的 558 个合法对照中，555 个通过整条原评测路径。其余 3 个均为 LayerNorm L1/P3，正确性阶段通过独立认证，计时校验因输出与实际参考不一致而给出 `REWARD_HACK`。这 3 个计时输入的独立诊断均为不确定，且观测到参考与 oracle 分歧。记录保留原标签、失败阶段和实际调用，关于是否存在误拒或候选行为变化的归因仍未证实。 [例外与计时证据索引](../../data/outputs/solar-e2-successor-closeout-20261118/counterexample-and-witness-index.json)

15 个重复不一致的逻辑单元均来自 LayerNorm L1，分布为 P0 2 个、P2 5 个、P3 8 个。其中 13 个的原裁判 verdict 有变化，10 个的独立分类有变化，两组有重叠。每个单元保留三次原始结果；E0 的重复一致性条件因此未通过。整个正确性分母中的原裁判 verdict 为 987 个 `PASSED`、306 个 `INCORRECT_NUMERICAL` 和 3 个 `REWARD_HACK`。 [分阶段对账和重复索引](../../data/outputs/solar-e2-successor-closeout-20261118/counterexample-and-witness-index.json)

## 附加诊断成本

成本口径为诊断侧与原检查侧的总会话墙钟，在每个完整配对内先求比值和差值，再按任务取中位数。配对只覆盖冻结的 `w0`、`d0`、P0/P3 与四种输入族，共 24 个块、72 对。两侧分别从空的会话级磁盘缓存开始，各自预热。同一块匹配输入构造与分布，各侧独立派生实际输入。进入配对统计的两侧均通过且完成 190 次候选调用、371 次参考调用。 [成本全表](../../data/outputs/solar-e2-successor-reaping-frozen-20261118/successor-confirmation/report/cost.json)

| 任务 | 可比较配对 / 总配对 | 墙钟比值中位数 | 增加墙钟中位数 | 比值范围 |
| --- | ---: | ---: | ---: | ---: |
| GEMM | 24 / 24 | 1.54× | 4.44 秒 | 1.44–1.63× |
| 求和归约 | 24 / 24 | 1.16× | 0.98 秒 | 1.13–1.19× |
| LayerNorm | 18 / 24 | 2.96× | 12.50 秒 | 2.88–18.84× |

66 / 72 对进入墙钟统计。另 6 对均为 LayerNorm P3，其中 L0 三对和 L1 一对的两侧均提前数值拒绝，另两对的原检查侧在计时校验中给出 `REWARD_HACK`，诊断侧提前数值拒绝。这 6 对的 12 个会话有完整终态，保留各自的实际调用数和原判定；配对差值、比值保持空值。LayerNorm 完整路径中耗时比最高的是 L3/P0 的第三次重复。 [不可比较配对和原始 trace 索引](../../data/outputs/solar-e2-successor-closeout-20261118/counterexample-and-witness-index.json)

成本全表逐会话保存准备、编译、当前检查、参考进程与 IPC、附加 oracle、证据采集及进程 RSS 峰值。IPC 时间包含 oracle 等待，分项之间存在重叠。候选与参考进程 RSS 分别最高约 1.27 和 1.30 GiB，采用各进程单独峰值的口径。72 个原检查侧离线回放在 GPU 交还后完成，宿主墙钟为 23 分 39 秒，其中 oracle CPU 时间约 11 分 40 秒。表中在线比值是诊断的会话级运行开销。 [成本分项和离线记录](../../data/outputs/solar-e2-successor-reaping-frozen-20261118/successor-confirmation/report/cost.json)

## 留存与路线图

- 原 E2 在证据预算处停止，保留 325 个完成、1 个中断、970 个未到达的正确性会话，144 个成本会话未到达。 [原 E2 台账](solar-e2-execution-2026-11-08.md)
- 第一次正式后继运行在计时线程启动时停止，保留 500 个完成、1 个故障、795 个未到达的正确性会话，144 个成本会话和 72 个离线回放未到达。终态核验重新校验了这次停止的 13,757 份文件。 [停机证据清单](../../data/outputs/solar-e2-successor-worker-reap-20261118/stopped-activity-manifest.json)
- 本次修正版使用独立留出和冻结包。结果用于 E2 研究验收，正式评分契约沿用原版本。M2 的 E2 证据交付已闭合，C2/C3 仍未启动，D0 等待覆盖矩阵终态。 [报告索引](../../data/outputs/solar-e2-successor-reaping-frozen-20261118/successor-confirmation/report/report-index.json) · [路线图](solar-evolution-roadmap-2026-11-05.md)
