# SOLAR 演进线：现状综合与下一阶段设计

制定日期：2026-11-05。本文是调研综合加设计提案，不是已完成工作的验收记录。输入：
[SOLAR 边界](../SOLAR-BOUNDARY.md)、[阶段任务规格](solar-evolution-tasks.md)、
[升级机会分析](solar-upgrade-opportunities-2026-10-11.md)、[阶段 1–5 交付](solar-resume-2026-10-26.md)、
[mapper 后续与 GPU 提案](solar-mapper-followup-2026-10-26.md)、HANDSOFF 的 2026-10-26/29 条目，
以及 2026-10-30 落 main 的阶段实现提交与 2026-11-05 落 main 的 HIP scope 提交
（`97355a7a` 等五个）。仓库级进度仍只在 [HANDSOFF](../../HANDSOFF.md#solar-evolution) 维护；
本文若与其冲突，以 HANDSOFF 为准。

## 0. 现状判定（哪些门已关、哪些开着）

已闭合（有证据绑定）：

- 阶段 1–5（访问区域并集、复用与三视图、反向需求与扫描、受限计算、收缩顺序搜索）
  已落 main；`final-f32-01` 采集 32/32 槽位零数学差异，关闭 BASE-01 的 F04/F08/F06。
- 真实受信 Orojenesis mapper 四点验证通过（F03-D）；GPU canary 两路径已通过
  （CHECKPOINT-0553，2026-10-26 05:53 UTC）。早期文档"GPU canary 未执行"的叙述已被取代。
  注意：该 checkpoint 证据存于外部研究工作树
  `~/Documents/Codex/2026-10-26/task/solar-stages1-5/data/outputs/`，本仓库 `data/outputs/`
  无副本；HANDSOFF 中的仓库相对路径不指向现存文件。已决策（2026-11-05）：外部证据
  不复制入仓库，L0 只登记路径与内容摘要。
- 2026-10-29 科学快照：R 组 14 份 readiness 回执、12 个可计价、12 个完整 cache-aware 模型结果；
  语义运行时身份 `005d6c03…` 未变，允许严格复用 `5d37` GPU 回执。
- P0 资源审计已刷新到 ROCm 10.1 / Torch 2.14（`rx9060xt-amd-resource-v6`）。
- 生产 eager 范围资格：`evaluate --profile rocprofv3-scope` 已验收（34 workload × 2 seed 扩大
  验收全合格）。BASE-01 时期的一项提取回归测试当前通过（2026-11-05 复验）。

开着的门（按主题归组，均为各文档明确声明，不是本文新发现）：

| 组 | 缺口 | 出处 |
| --- | --- | --- |
| A 正式资格 | R06 两个动态读取未计价、R08 两个边界拒绝（L0 已判终态：合法域外预期拒绝，保留于分母）；R04 已恢复（5d37 回执严格复用，L0 复核）；完整 corpus 发布资格未取得 | ACCEPTANCE-LEDGER（L0 登记）、10-29 快照 |
| B 调度因果 | 完整跨 lane 调度目标未关闭：内存复制域依赖、默认 stream、Graph、后台线程、跨进程；一切 scope 资格固定 `schedule_complete=false` | hip-torch-scope / evaluate-hip-scope 验收 |
| C 表达能力 | 动态形状（ShapesSpec → 自有形状域契约）、静态 fake 取图、2.14 算子覆盖差分 | 升级机会 P1 |
| D 吞吐工程 | mapper CPU 配额（当前每 mapper 全核）、有界批处理；均要求先测各 stage 时间 | 升级机会 P2 |
| E 架构扩展 | gfx942/CDNA3 非正式适配需真机核验（counter 拼写、MFMA 探针、MI300X profile、内存类工作集） | HANDSOFF P1 |
| F 研究对照 | Origami 分析式 tile 选择、grouped GEMM/CK 对照 | 升级机会 P3 |

推断而非证明的事项：阶段 1–5 与 176 路径整合演练"已通过 10-30 的
`f9d0d994`/`cb41d916` 落 main"是从提交内容与 HANDSOFF 时间线推断的；
严格对应关系需按整合 handoff 的 main preimage 复核后再声称。

## 1. 设计原则（继承自既有边界，不新增放宽）

1. 能力与资格分离：CPU 能力推进不自动产生正式资格；资格结论按资格链单独验收。
2. fail-closed 不变：`schedule_complete`/`performance_authority` 缺省仍为 `false`，
   只在受限域内凭独立验收翻转，且永不回退到时间戳排序当因果。
3. 边界不反转：调度因果证据留在 `sol_execbench` 诊断 sidecar / 性能模型，经 bridge 消费
   SOLAR 结果；SOLAR 不反向依赖 profiler、候选时间或 benchmark 定义。
4. 每条线独立可审查：预固定分母与 oracle、负例必测、身份与预算进缓存键、
   schema 变更原子更新生产者/读者/fixtures/CLI/文档。
5. 证据目录留在未提交 `data/outputs/`，文档只绑定摘要与路径。

## 2. 演进线设计

六条线（L0–L5），线间依赖见 §3。编号是文档引用号，不是公共 API 或 schema 标识。

### L0 状态账实对齐（立即，纯 CPU/文档）

HANDSOFF 最新条目已领先于多份 10-26 文档的"未执行 GPU"叙述。交付：HANDSOFF 与
resume/mapper-followup 文档的交叉引用修正（追加状态不改写历史值）；
`97355a7a` 等 main 新提交在 HANDSOFF 的对应登记；按整合 handoff
（`~/Documents/Codex/2026-10-26/task/solar-integration-prep-20261029/`）的
main preimage 复核"176 路径整合已落 main"的推断并登记结论；登记外部工作树中
GPU checkpoint / 整合 handoff 证据的路径与内容摘要（已决策：不复制文件本体，
2026-11-05）；对 R04/R06/R08 逐槽位
建立"当前状态 + 下一步动作 + 所需前置"的台账。不修改任何生产实现。

优先级声明：[升级机会分析](solar-upgrade-opportunities-2026-10-11.md)（10-11）把形状域
列为 P1 主要投资；本路线把 L1 资格闭合排在 L3 之前。差异原因是排序时点：该分析的
优先级产生于 R16 已 12 ready 与 GPU canary 通过之前——资格闭合的边际成本在当时
被高估。若 L1 因硬件窗口不可得而停摆，L3-b 自动回到首要位置。

### L1 正式资格闭合线（最高优先；硬件窗口是唯一硬前置）

目标：把 R 组 16 槽位推向终态并取得（或明确拒绝）完整 corpus 正式发布资格。

- L1-a（L0 已改判）容器 EROFS 环境前置：R04 两槽位经 5d37 回执严格复用已是
  ready/priceable/cache-aware 终态，EROFS（容器默认 `/home/sol-execbench/.triton`
  只读，errno 30）不再是槽位阻塞；但它仍是 L1-d 在容器内新跑资格链的必要
  前置——落实 CHECKPOINT-0553 建议的有界临时 tmpfs 并留证据。
- L1-b（L0 已闭合）R08 终态：两路径均为 expected_rejection（torchview 卡
  graph_extraction、make_fx 卡 conversion_verification 的梯度检查），按合法
  域外预期拒绝保留在 16 槽分母内，无需代码变更；不因追求 16/16 通过而放宽
  提取或梯度验证。
- L1-c R06 动态读取计价：R06 是 `torch2flydsl/moe_topk_softmax`（E=256,K=8,T=64），
  阻塞点是路由与**数据依赖索引**的外部读取——属 external-reads/受限需求证明域
  （阶段 1/4 家族）的扩展，与 L3-b 形状域（符号维度范围）无关。已决策（2026-11-05）：
  建设受限动态索引读取的证明词汇，把 R06 推向可计价终态；词汇冻结在 top-k
  路由类按 K 限界读取，不通用化。若冻结词汇无法覆盖 R06 实际语义，回退为
  显式拒绝并记录理由，不从分母移出、不放宽验证。
- L1-d 资格链执行：既有 `qualify-static → canary → full` 链在独占窗口
  （`SOLAR_GPU_APPROVED`、STABLE_PEAK、原生 L1/L2 锁）下跑完，
  复用 `evaluate --profile rocprofv3-scope` 的生产范围资格作为逐 workload 佐证。
  **2026-11-05 执行结果（终态：经路径 (a) 闭合）**：宿主前置全部就绪（锁/UUID/
  mapper/sudoers），baseline 工作区物化、qualify-static gate 通过。全分母
  qualify-canary 结构性受阻（112 项审计 79 ready / 33 failed：3 项 R08 预期拒绝、
  20 项原生算子覆盖缺口属 L3-c 域、10 项特定轴点数值失配；R04-w0 本身 ready）。
  按本路线图 sanctioned 的路径 (a)，以现有 `audit_corpus_stage_readiness(
  selected_item_ids=…)` 接缝对冻结 R16 分母执行受控资格
  （`scripts/internal/solar/run_r16_curated_qualification.py`，R08 声明式预期
  拒绝）：**双路径 curated gate 全部通过**——15 ready + 1 声明预期拒绝 = 16/16
  终态；R06 双路径额外取得正式 `capacity_constrained_tile_aware_v1` bound
  （L1-c 词汇生效）；R08 make_fx 槽正当升级为 ready（集成树的验证修复，
  三阶段真实通过）。全分母链保持 INCOMPLETE 并推迟至 L3-c。证据：
  `data/outputs/solar-l1-release-20261105/r16-terminal-ledger.json` 与
  `campaign-summary.json`。

验收骨架：16/16 槽位各有 qualified / rejected（含稳定理由码）终态；
发布资格或逐项拒绝清单；全部 GPU 证据绑定设备 UUID、时钟协议与源码摘要。

### L2 调度因果与性能解释线（中期；关键依赖在 SDK 回执）

目标：在受控执行器域内把"完整跨 lane 调度"从永久开放变为受限域可资格；
性能模型获得已验证调度边（`schedule_evidence.py` 的 unverified 路径出现 verified 分支）。

里程碑（顺序）：

1. M-a 内存复制域依赖资格：dispatch 归属已验证，复制域依赖关系未验证——
   按 torch-scope 验收声明的缺口设计账本/回执核对。
2. M-b 默认 stream 与后台线程语义：默认 stream 的保守语义定义（不推断隐式同步
   之外的关系）、后台线程提交的显式拒绝或受限支持。
3. M-c 跨 lane 调度模型验证：仅消费已资格的事件边，预固定正/负例分母
   （含"时间序相近但无事件边不得成边"负例）。
4. M-d 性能模型消费：sol_execbench 侧性能模型把 verified 调度边用于
   实际执行 vs SOL 下界的差距解释；`performance_authority` 仍为 `false`，
   差距解释是诊断结论不是性能授权。

外部依赖与替代：SDK 可核验 drop/flush/finalization 回执（trace-scope 验收声明的
future work）若上游不可得，替代路径是扩展受控执行器账本域（Graph/IPC 除外）。
Graph capture 与 IPC 在本线内保持显式拒绝。

### L3 表达能力线（中期；CPU 为主，可与其他线并行）

按升级机会 P1 的三步，顺序固定：

- L3-a 静态 fake 取图试点：仅 `make_fx_aten` 路径，默认路径不动。先测
  输入生成/trace/转换/回放/Orojenesis 各 stage 耗时与峰值内存，再决定改造。
  验收：输入/输出元信息、mutation、常量、数据依赖拒绝路径、与真实回放对照。
- L3-b 形状域契约：自有约束 AST（symbol、range、派生、整除、assumptions、
  source binding）；张量计算与形状/断言节点分离计价；第一阶段只在域验证后的
  具体 shape 上实例化现有 IR，artifact 记录实例与来源域。
  验收词汇：共享/派生维度、合法边界、越界、整除、shape branch（拒绝）、
  nonzero（区间）、alias/mutation。R06 是首个真实消费者。
- L3-c 2.14 算子覆盖差分：从实际 corpus 出发（matmul/addmm/bmm、attention、
  normalization、reduction、view/alias、mutation、低精度转换、显式 backward），
  双固定路径比对精度/alias/mutation/梯度/资源计数；产出覆盖矩阵而非直接扩 allowlist。
- L3-d 新前端（阶段 6，默认延后）：Affine Loop IR／LLM 修复／JAX 前端
  （stage0 C11 归 P3、"由实际覆盖缺口驱动"）。触发条件是 L3-c 覆盖矩阵指出
  两条现有路径的系统性缺口；[论文演进记录](solar-paper-evolution-2026-10-19.md)
  指出 v2 论文的真实跃迁正是可执行 Affine Loop IR 流程，故本项有论文对齐意义。
  在触发前不投入。

边界：两条固定路径与"不自动回退"政策不变；不把上游 `repr(SymBool)` 当可执行协议。

### L4 吞吐工程线（按测量推进，不预设方案）

现状修正：mapper 并发上限框架**已存在**——`solar_bridge/resource_policy.py` 的
`formal_mapper_job_limit`（可见逻辑核÷单 mapper 线程数）已被
`release_solar_runner` 与 `solar_bridge/scan.py` 消费。剩余真实缺口收窄为：

1. 各 stage 时间/内存测量报告（取图、验证、转换、分析、Orojenesis、发布），
   作为一切后续优化的前置；
2. 单 mapper 线程数仍等于全部可见逻辑核（`orojenesis_mapper_thread_count`）——
   评估该策略在批量场景下是否合理，或改为可配置；
3. 核查其余并发路径（批次资格、compare-paths）是否同样经过 `resource_policy`；
4. 有界队列（Python 3.14 `Executor.map(buffersize=…)`）；解释器并行仅在验证
   torchview 全局 patch 与扩展兼容后考虑，取图保持进程隔离。

收益对象是 L1 的 corpus 批量资格与 L3 的扫描吞吐；不改变任何结果语义。

### L5 架构扩展线（gfx942/CDNA3，非正式；独立硬件）

按 HANDSOFF P1 已列清单执行：counter 拼写核验（`rocprofv3-avail`）、
CDNA3 MFMA 探针（含 deferred fp8 FNUZ / int8 布局）、MI300X profile 实测替换、
`cdna3_total_memory_class.v1` 工作集推导（wave64 变体）、
`pytest -m requires_cdna3 -n 0` 全链。全部产物标 `inexact`，正式资格保持
gfx1200-only；不与 L1 争抢 gfx1200 窗口。

研究对照（升级机会 P3 的 Origami/grouped GEMM/CK）不单列线：作为 L1 完成后
可选的外层对照任务，任何时候不得替代 pinned Orojenesis。

## 3. 依赖、排序与并行

```text
L0 ──► L1 ──────────► (正式发布资格)
 │        └─ L1-c 自含（阶段 1/4 证明域扩展），不依赖 L3
 │   L4测量 ──► L4实施 ─┐（服务 L1 批量、L3 扫描）
 ├──► L3-a ──► L3-b ──► L3-c ──► (L3-d 新前端，仅当覆盖矩阵触发)
 └──► L2 规格/SDK 调研 ──► L2 M-a ──► M-b ──► M-c ──► M-d
L5 独立硬件，随时可并行
```

- 关键路径：L0 → L1（唯一硬前置是 gfx1200 独占窗口的可得性）。
- L3-a/L4-测量立即可做（纯 CPU），不与 L1 争资源。
- L2 前期是规格与 SDK 能力调研；M-a 不依赖 L1/L3。
- L1-b 的判定结果决定 R08 是否转化为 L3-c 的覆盖项。
- 建议节奏：L0 立即；L1 与 L3-a/L4 并行启动；L2 规格与 L3-b 设计评审合并进行。

## 4. 风险与开放问题

| 风险 | 影响 | 缓解 |
| --- | --- | --- |
| gfx1200 独占窗口不可得 | L1 全线停摆 | L3/L4/L2 规格不受阻；窗口按维护窗口协调（README 锁协议） |
| 容器 EROFS 在 L1-d 复现 | 资格链新跑失败被误判为代码回归 | 资格链运行前落实有界 tmpfs；失败时先对照 CHECKPOINT-0553 已知症状 |
| SDK 回执上游不提供 | L2 M-a 起无法按首选路径验证 | 替代：受控执行器账本域扩展；Graph/IPC 维持拒绝 |
| 形状域范围蔓延 | L3-b 吞噬多个周期 | 首批词汇冻结在升级机会文档所列；超出即另立任务 |
| R06 证明域扩展蔓延 | 为单槽位过度建设动态索引词汇 | 已决策建设但词汇冻结在 top-k 路由类按 K 限界读取；无法覆盖则回退显式拒绝 |
| 整合对应关系未严格证明 | 现状叙述失真 | L0 中按整合 handoff 的 preimage 复核后登记 |
| gfx942 证据停留 spec-derived | 非正式结论被误引用 | 全部 `inexact` 标注；正式资格链不消费 |

## 5. 与既有文档的关系

本文不替代任何验收文档；各线进入实施时按惯例另立预固定验收规格
（分母、oracle、负例先行），实施进度只登记进 HANDSOFF。
阶段 1–5 的既有接口（reuse/scan/order/restricted）保持现合同；
本文所有新能力均以"新增受限域"方式接入，不修改已验收语义。
