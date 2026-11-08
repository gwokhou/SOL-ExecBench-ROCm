# SOLAR 演进线：现状综合与下一阶段设计

制定日期：2026-11-05；规划更新：2026-11-08。本文是调研综合加设计提案，不是已完成工作的验收记录。输入：
[SOLAR 边界](../SOLAR-BOUNDARY.md)、[阶段任务规格](solar-evolution-tasks.md)、
[升级机会分析](solar-upgrade-opportunities-2026-10-11.md)、[阶段 1–5 交付](solar-resume-2026-10-26.md)、
[mapper 后续与 GPU 提案](solar-mapper-followup-2026-10-26.md)、HANDSOFF 的 2026-10-26/29 条目，
以及 2026-10-30 落 main 的阶段实现提交与 2026-11-05 落 main 的 HIP scope 提交
（`97355a7a` 等五个）。仓库级进度仍只在 [HANDSOFF](../../HANDSOFF.md#solar-evolution) 维护；
本文若与其冲突，以 HANDSOFF 为准。

当前执行顺序由 [下一轮迭代计划](solar-next-iteration-plan-2026-11-07.md) 细化：
版本基线 → 裁判辨错 / LLM_CORE 覆盖 → 证据驱动的后续决策。
2026-11-08 已按后续授权完成 E1/C1 的实现和 CPU 检查，详见
[CPU 验收记录](solar-e1-c1-cpu-acceptance-2026-11-08.md)；没有运行 GPU、真实 mapper、容量探针或生成留出实例。
§2 保留 L0–L5 的历史设计与日期记录。

**M0 已于 2026-11-08 完成只读清点与文档冻结**：
[B0](solar-m0-baseline-evidence-2026-11-07.md)、[E0](solar-m0-judge-preregistration-2026-11-07.md)、
[C0](solar-m0-llm-core-coverage-spec-2026-11-07.md)。旧 full 的489个引用工件字节匹配，但语义源码身份
已漂移；刷新仍在 B1/B2。E0 冻结432个候选—case单元；C0 固定core20项、合格视图180槽/360行。
E1/C1 的最小诊断执行器、对照实现和 target-view 审计接缝已完成 CPU 验证；
[B1](solar-b1-execution-freeze-2026-11-08.md) 已固定执行提交 `160b9988`、镜像、mapper和资源预算；
宿主/容器各288项CPU检查通过。当前下一依赖是B2，CPU结果不授予硬件资格。
E0成本次数的算术冲突已按用户批准另建[r2-counts勘误](solar-e0-cost-count-erratum-2026-11-08.md)，
原冻结文件和实验机制保持不变。
E1 开发集保留11项 LayerNorm P3 与 CPU reference 的原容差分歧；它们尚非已确认合法对照误拒，
不得通过删项、换种子或改正式容差清除。真实 LLM_CORE 矩阵仍待 C2/C3。

## 0. 现状判定（哪些门已关、哪些开着）

规划锚点为 main `8641b9dc78d00250ba1cf75d5e9434f9839e5ef4`。
提交说明记录 CPU 整合检查，没有当前合并版本 GPU 验收；下表的历史闭合不自动转移到该 SHA。

| 主题 | 截至 2026-11-07 的已登记状态 | 下一步 |
| --- | --- | --- |
| 阶段 1–5 / L0 | 已集成；早期 mapper 与 GPU checkpoint 有历史记录 | B0 核对身份与索引；不重启阶段开发 |
| L1 / AKA 资格 | R06 证明域已闭合；make_fx 在 `39329249` 上完成 static 163 → canary 112 → full 163 ready | B1/B2 冻结当前执行版本并刷新必要资格；不将 ready 等同完整形式回放 |
| 外部读取证明 | 原 128 行阻塞清零；127 行 G4 完整，1 行大卷积回放内存 miss | 保留缺口和原分母，不裸跑重试 |
| L2 调度与 M-d | 受限 eager 域及 opt-in 生产供数均闭合；Graph/IPC/跨进程仍拒绝 | 作为基线能力；不重复建设 M-d |
| L3-a / L3-b | fake-static 已生产接入；形状域阶段 1 闭合 | 请求级形状域/符号化 mapper 仍未启动 |
| L3-c / L4 | AKA 矩阵闭合；吞吐配额、有界提交审计闭合 | C 线扩至 LLM_CORE core；按实测成本再提优化 |
| 裁判辨错 | 已有数值覆盖审计提案，没有本计划的有效性实验结果 | E 线固定三类任务、对照和独立 oracle |
| L3-d / L5 / 研究对照 | 新前端默认延后；gfx942 非正式且仍需真机 | 只按明确覆盖缺口或真实消费者另行立项 |

外部阶段 1–5 checkpoint 仍在
`<external-solar-workspace>/data/outputs/`；
继承 2026-11-05 决策：只登记路径与内容摘要，不复制文件本体。
若需声称旧 176 路径整合演练与 main 的严格对应，B0 仍须核对 handoff preimage；
时间线或提交标题不足以代替该证明。

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

六条线（L0–L5）保留既有设计编号，不是公共 API 或 schema 标识。
下列旧目标和带日期结果是历史脉络；当前待办以 §0、§3 和下一轮计划为准。

### L0 状态账实对齐（立即，纯 CPU/文档）

HANDSOFF 最新条目已领先于多份 10-26 文档的"未执行 GPU"叙述。交付：HANDSOFF 与
resume/mapper-followup 文档的交叉引用修正（追加状态不改写历史值）；
`97355a7a` 等 main 新提交在 HANDSOFF 的对应登记；按整合 handoff
（`<external-integration-workspace>/`）的
main preimage 复核"176 路径整合已落 main"的推断并登记结论；登记外部工作树中
GPU checkpoint / 整合 handoff 证据的路径与内容摘要（已决策：不复制文件本体，
2026-11-05）；对 R04/R06/R08 逐槽位
建立"当前状态 + 下一步动作 + 所需前置"的台账。不修改任何生产实现。

历史优先级声明（2026-11-05）：[升级机会分析](solar-upgrade-opportunities-2026-10-11.md)（10-11）把形状域
列为 P1 主要投资；本路线把 L1 资格闭合排在 L3 之前。差异原因是排序时点：该分析的
优先级产生于 R16 已 12 ready 与 GPU canary 通过之前——资格闭合的边际成本在当时
被高估。若 L1 因硬件窗口不可得而停摆，L3-b 自动回到首要位置。

### L1 正式资格闭合线（历史 make_fx 链已闭合；新版本刷新归 B 线）

原目标：把 R 组 16 槽位推向终态并取得（或明确拒绝）完整 corpus 正式发布资格。

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
  **2026-11-07 追记：全分母链闭合（make_fx 分母）**——L1-d 延迟的全分母
qualify-canary/full 已在重建于当时 HEAD `39329249` 的 workspace 上经生产 CLI 三级链
（static 163 → canary 112/112 ready → full **163/163 ready**）完成，
证据 `data/outputs/solar-qualification-mfx-20261107/`；torchview 分母链
仍开放（42 就绪缺口，见 L3-c 矩阵），不构成本闭合前置。详见 HANDSOFF
2026-11-07 资格链条目。
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

**2026-11-07 执行结果（受限域闭合；M-d 随后也已闭合，见后文）**：预固定验收规格
[solar-l2-schedule-causality-acceptance-2026-11-06.md](solar-l2-schedule-causality-acceptance-2026-11-06.md)。
SDK 回执上游不可得判定成立，按 sanctioned 替代路径扩展受控执行器账本域：
`hip_sdk_ledger` v2 新增引擎复制完成行与 launch/memcpy 原始 stream 句柄；
`hip_torch_scope`/`hip_evaluate_scope` v2 携带 `copy_domain` 与
`schedule` 资格，`schedule_complete` 在受控 eager 域（复制 lane 共观察绑定、
事件边 verified、无默认流混布）首次可资格；GPU 分母六用例 × 3 = 18/18
（15 个 `schedule_complete=true`，default_mix 三次正确拒绝），证据
`data/outputs/solar-l2-schedule-20261106/`。`performance_schedule_evidence`
v3 增加 `event_wait` 边与 `cross_lane_event_dependencies` 状态；
`predict_hw` 消费 verified 边（无边时行为不变）。M-d 的生产供数管道
（scope↔counter dispatch 标识桥、显式 opt-in）当时按规格 §7.4 登记遗留。
Graph/IPC/跨进程维持拒绝。详见 HANDSOFF 2026-11-07 L2 条目。

**2026-11-07 后续闭合**：[M-d 验收](solar-l2-md-feed-acceptance-2026-11-07.md)
已完成生产 scope archive bridge、显式 `--schedule-scope` 和真实双采集核验；
默认无 opt-in 行为保持不变，`performance_authority=false`。M-d 不再是遗留待办。

原里程碑（已执行的顺序）：

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

原升级机会 P1 的分步设计与历史结果：

- L3-a 静态 fake 取图试点：仅 `make_fx_aten` 路径，默认路径不动。先测
  输入生成/trace/转换/回放/Orojenesis 各 stage 耗时与峰值内存，再决定改造。
  验收：输入/输出元信息、mutation、常量、数据依赖拒绝路径、与真实回放对照。
  **2026-11-05 执行结果（终态：闭合）**：预固定验收规格
  [solar-l3a-fake-static-pilot-acceptance-2026-11-05.md](solar-l3a-fake-static-pilot-acceptance-2026-11-05.md)
  （含一项日期勘误：R08/make_fx 声明为第二个预期失败）。测量 gate 16/16
  通过（与 L4 第 1 项同一活动交付）；试点 D1–D4 全过、adopt=true（8/8 槽
  O1+O2+O3 全绿，提取墙钟 11–196×，GPU 峰值不升）。当时将生产改造列为后续线，
  另立验收。详见 HANDSOFF 2026-11-05 L3-a 条目。
  **2026-11-07 后续闭合**：[生产接入](solar-lineb-fake-production-acceptance-2026-11-07.md)
  已完成 163/163 图字节对账和 3 个代表槽 GPU 验证；4 个 rope 槽使用声明式真实取图回退。
  11–196× 是先前试点的提取时间收益，不是整条分析链或当前合并版本的新测量。
- L3-b 形状域契约：自有约束 AST（symbol、range、派生、整除、assumptions、
  source binding）；张量计算与形状/断言节点分离计价；第一阶段只在域验证后的
  具体 shape 上实例化现有 IR，artifact 记录实例与来源域。
  验收词汇：共享/派生维度、合法边界、越界、整除、shape branch（拒绝）、
  nonzero（区间）、alias/mutation。R06 是首个真实消费者。
  **2026-11-05 执行结果（终态：阶段 1 闭合）**：预固定验收规格
  [solar-l3b-shape-domain-acceptance-2026-11-05.md](solar-l3b-shape-domain-acceptance-2026-11-05.md)。
  `src/solar/shapes/` 自有约束 AST 落地；R06 冻结域全分母验证 + 负例精确
  拒绝；结构认证 structure_stable（topk.k 由 K 符号解释）且 shape branch
  正确拒绝；实例来源域经既有 `analysis_metadata` 通道写入 solar-analysis/
  manifest，计价不变性成立。阶段 2（请求级形状域字段、符号化 Orojenesis）
  未启动，另行立项。详见 HANDSOFF 2026-11-05 L3-b 条目。
- L3-c 2.14 算子覆盖差分：从实际 corpus 出发（matmul/addmm/bmm、attention、
  normalization、reduction、view/alias、mutation、低精度转换、显式 backward），
  双固定路径比对精度/alias/mutation/梯度/资源计数；产出覆盖矩阵而非直接扩 allowlist。
  **2026-11-06 执行结果（终态：闭合）**：预固定验收规格
  [solar-l3c-coverage-matrix-acceptance-2026-11-06.md](solar-l3c-coverage-matrix-acceptance-2026-11-06.md)。
  326 行全量矩阵 + canary 对账 0 漂移 + 家族 7/8（mutation 诚实为零）。
  头条：make_fx 就绪覆盖 163/163 严格优于 torchview 121/163；形式边界层
  当时 128 行 external-reads 未证明（normalization 家族主导）；conv 家族在
  torchview 侧撞 mapper 7200s 生产预算（5/5）。可执行缺口清单已产出；
  当时全分母资格链仍为 L1 延迟项，被矩阵解锁。详见 HANDSOFF 2026-11-06 条目。
  **截至 2026-11-07 的后续状态**：128 行证明阻塞已清，127 个完整 G4 与 1 个内存 miss 分列；
  make_fx 全分母资格链闭合。下一轮 C 线测 LLM_CORE，不复用 AKA 的 163/163 作为新语料结论。
- L3-d 新前端（阶段 6，默认延后）：Affine Loop IR／LLM 修复／JAX 前端
  （stage0 C11 归 P3、"由实际覆盖缺口驱动"）。触发条件是 L3-c 覆盖矩阵指出
  两条现有路径的系统性缺口；后续判断以 C 线的真实高价值模式及层级归因为依据，
  不把容量不足或 adapter 未接通当作前端缺陷。[论文演进记录](solar-paper-evolution-2026-10-19.md)
  指出 v2 论文的真实跃迁正是可执行 Affine Loop IR 流程，故本项有论文对齐意义。
  在触发前不投入。

边界：两条固定路径与"不自动回退"政策不变；不把上游 `repr(SymBool)` 当可执行协议。

### L4 吞吐工程线（按测量推进，不预设方案）
**2026-11-07 执行结果（终态：闭合）**：预固定验收规格
[solar-l4-throughput-acceptance-2026-11-07.md](solar-l4-throughput-acceptance-2026-11-07.md)。
第 2 项：全核默认经测量判必要（conv mapper 全核仍撞 7200s 预算），
新增 `SOLAR_OROJENESIS_MAPPER_THREADS` 运维覆盖；第 3 项：四路径审计
落表，qualification `--jobs` 接入共享 `validate_formal_job_count`
（原仅 >0 校验）；第 4 项：生产路径无 `Executor.map` 调用点（盘点
闭合），全部提交点按构造有界，解释器并行以源码事实测试钉桩拒绝。
详见 HANDSOFF 2026-11-07 L4 条目。

现状修正：mapper 并发上限框架**已存在**——`solar_bridge/resource_policy.py` 的
`formal_mapper_job_limit`（可见逻辑核÷单 mapper 线程数）已被
`release_solar_runner` 与 `solar_bridge/scan.py` 消费。以下是闭合前的任务分解，
完成状态见上段；不再作为当前待办：

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

2026-11-07 起采用 [下一轮计划](solar-next-iteration-plan-2026-11-07.md) 的 B/E/C/D 任务：

```text
B0 ─┬─ E0 → E1 ─┐
    └─ C0 → C1 ─┴→ B1 → B2 ─┬─ E2 ───────┐
                             └─ C2 → C3 ─┴→ D0
```

- P0：版本基线与身份闭合；必要接入代码先完成，再统一冻结，避免采集中改树使资格失效。
- P1：三任务辨错实验与单配置 LLM_CORE core 全阶段矩阵；CPU 准备可并行，GPU 段串行。
- 里程碑：规格冻结 → 执行基线 → 两项证据报告 → 后续决策；各 gate 见下一轮计划。
- 条件项：L3-b 阶段 2、L3-d、新 DP、L5 与研究对照保持未启动，由 D0 提出独立范围。
- E1/C1 与B1已完成；B2及后续硬件活动仍须按既有依赖和授权执行，不因路线图更新自动启动。

## 4. 风险与开放问题

| 风险 | 影响 | 缓解 |
| --- | --- | --- |
| 旧 GPU 证据被套到合并 HEAD | 基线结论失真 | B0 身份对照、B1 冻结、B2 刷新必要资格 |
| oracle/比较精度不足 | 把裁判误差归给候选 | E1 独立自检；不确定单列；原契约与更强保证分开 |
| 留出结果参与修规则 | 确认结果失去独立性 | 记录暴露；修复后换新预登记确认集 |
| LLM_CORE 与 AKA 契约混用 | 虚假分母或未经验证的通过 | C0/C1 显式接入，保留生成及身份门禁 |
| 容量/mapper/回放成本超预算 | OOM 或只剩容易样本 | 固定资源上限；原分母保留，INCOMPLETE 不改写为通过 |
| gfx1200 独占窗口不可得 | B2/E2/C2/C3 GPU 段停留待执行 | 完成 CPU 准备和证据索引，不替代硬件验收 |
| 单配置结果被外推 | 新语料/跨硬件主张过宽 | 分 profile/路径/阶段/配置报告；L5 另行立项 |

## 5. 与既有文档的关系

本文不替代任何验收文档；下一轮任务分解和验收标准见
[迭代计划](solar-next-iteration-plan-2026-11-07.md)，其 E0/C0/B1 负责在运行前补全冻结件。
各线进入实施时按惯例固定验收规格
（分母、oracle、负例先行），实施进度只登记进 HANDSOFF。
阶段 1–5 的既有接口（reuse/scan/order/restricted）保持现合同；
本文所有新能力均以"新增受限域"方式接入，不修改已验收语义。
