# L2 调度因果与性能解释线：预固定验收规格

制定日期：2026-11-06（实施前预固定）。本文固定
[演进路线](solar-evolution-roadmap-2026-11-05.md) L2（B 组调度因果）的验收口径。
规格先于实施结果固定；实施后只允许追加结果与带先验证据的日期勘误，
不允许回改分母、oracle 或阈值。

## 0. SDK 事实基础（预固定时的先验证据）

- 宿主 ROCm 10.0 头文件（`/opt/rocm/core-10.0/include/rocprofiler-sdk/`）提供
  `ROCPROFILER_BUFFER_TRACING_MEMORY_COPY` 与 `ROCPROFILER_CALLBACK_TRACING_MEMORY_COPY`；
  buffer 记录（`rocprofiler_buffer_tracing_memory_copy_record_t`）携带
  correlation、thread、GPU 起止时间、方向、字节数、src/dst agent，**不含 queue**。
- HIP API callback 参数（`hip/api_args.h`）在 async memcpy 家族携带 `hipStream_t stream`。
- rocprof-sdk-tool JSON 导出已含 `memory_copy` buffer 域（既有 2026-10-11 复合用例
  导出观察为 0 行：该 workload 的连续张量复制走了 `copy_helper` kernel 路径）。
- SDK 无独立可核验的 drop/flush/finalization 回执通道（trace-scope 验收已声明）。
  **判定：本线采用路线图 sanctioned 的替代路径——扩展受控执行器账本域**；
  Graph/IPC/跨进程维持显式拒绝。
- 既有真实账本中出现的 memcpy API：`hipMemcpyAsync`、`hipMemcpyWithStream`。

## 1. 范围与闭合判定

L2 闭合当且仅当以下六项全部成立：

1. **M-a 复制域依赖资格**：受限执行器域内，复制两条子路径全部可资格：
   - 引擎复制（`memory_copy` buffer 域）：采集器新增结构化 memcpy 完成
     行（corr、方向、字节、GPU 起止、agents）；宿主 memcpy API 行绑定
     stream 参数；三方核对（API 行 ↔ 账本 memcpy 行 ↔ 导出 memory_copy 行）
     按 corr 严格相等，且时间戳、字节数、方向逐项一致。
   - kernel 化复制（既有隐式 helper dispatch）：归属维持现状，lane 与
     事件端点资格与 kernel 一致，不因本线改动放宽或收紧。
   - lane 绑定：引擎复制经 stream→queue 共观察绑定（同一 stream 上已
     验证的 kernel dispatch 提供 queue；一份共观察、全 capture 一致）。
     绑定失败或冲突 → 稳定 reason code 拒绝复制域资格，不静默降级。
2. **M-b 默认 stream 与后台线程语义**：legacy 默认 stream（句柄 0）原值
   照存，绝不推断隐式同步；默认 stream 工作与其他 lane 并存时不授予
   完整调度资格（reason: `default_stream_implicit_sync_not_inferred`）。
   后台线程提交维持显式拒绝（单提交线程），以测试钉住。
3. **M-c 跨 lane 调度模型 verified 分支**：`performance_schedule_evidence`
   原子升级新版本：边 reason 增加 `event_wait`（跨 lane 事件边，basis 为
   已验证的 `sdk_enqueue_order` 链接）；仅当 scope 验证、单 workload
   身份、全 lane 已知（含复制）、无默认 stream 混布、复制域资格成立、
   事件边已验证时发出跨 lane 边并可判 AVAILABLE；任何缺失 → fail-closed
   reason code。**时间序相近但无事件边不得成边**（必测负例）。
4. **受限域调度资格 artifact**：新 schema（单当前版本，原子升级生产者/
   读者/fixtures/CLI/文档，删除旧路径）：`schedule_complete` 仅在 M-a/M-b/M-c
   全部前置成立时为 true；`performance_authority` 恒 `false`。普通
   （非 scope）profile 不自动升级。
5. **M-d 性能模型消费**：调度估计消费 verified 跨 lane 边（事件强制串行
   计入 lane 间关键路径），差异只出现在诊断解释字段，不改变测量值；
   无 verified 边时行为与现状一致，不回退时间戳排序。
6. **登记**：HANDSOFF 条目 + roadmap 状态 + 测试/lint 全绿 + 提交。

## 2. 冻结词汇（预固定）

- **引擎复制 API 核心集**（授予复制域资格的唯一来源）：
  `hipMemcpyAsync`、`hipMemcpyWithStream`、`hipMemcpy2DAsync`、
  `hipMemcpy3DAsync`。其余 `hipMemcpy*` async 变体出现时账本行照存、
  复制域资格以 `copy_api_unsupported` 拒绝；`hipMemPrefetchAsync` 属
  unified memory 域，排除在复制域外，出现即 `copy_api_unsupported`。
- **复制方向**：H2D、D2H、D2D 授予资格；H2H 与未知方向拒绝
  （`copy_direction_unsupported`）。
- **reason codes**：`copy_domain_unqualified`、`copy_api_unsupported`、
  `copy_direction_unsupported`、`copy_queue_unresolved`、
  `copy_stream_conflict`、`copy_channel_mismatch`、
  `default_stream_implicit_sync_not_inferred`、
  `cross_lane_event_dependencies_unverified`（既有，保留）。
- lane 定义不变：`(queue_id, stream_id)` 二元组；复制的 queue 来自共观察
  绑定，stream 来自宿主 API 参数。

## 3. CPU 分母（预固定）

对新增/改动的每个验证器与模型，正例 + 负例矩阵：

1. 账本 memcpy 行模型与三方核对：删账本行、删导出行、corr 错绑、改
   bytes/方向/GPU 时间戳、footer 破坏、drop count 非零——全部稳定
   reason code 拒绝。
2. lane 绑定：共观察正例；无共观察（`copy_queue_unresolved`）；同一
   stream 绑出两个 queue（`copy_stream_conflict`）；stream 句柄 0
   （走 M-b 语义，不参与 lane 绑定）。
3. 事件链接端点扩展到引擎复制端点：producer/consumer 为复制时的边
   生成正例；端点篡改（换 queue、换 corr）负例。
4. M-b：默认 stream 与其他 lane 并存 → `schedule_complete` 不可为 true；
   第二提交线程 → 既有单 tid 拒绝路径（如已有测试钉住则引用，不重复）。
5. M-c 模型：verified 边正例；时间相近无事件边负例；scope 未验证、
   混合 workload 身份、lane 缺失、复制域未资格负例。
6. M-d：有/无 verified 边的预测对比只改诊断解释字段；测量字段逐位不变。
7. schema 原子升级：旧版本读者、fixtures、兼容再导出全部删除
   （registry 单当前版本规则）；新版本字段往返序列化测试。

## 4. GPU 分母（预固定；gfx1200 独占窗口，协议与既有 scope 验收一致）

三个用例 × ≥3 次独立进程，全部绑定设备 UUID、STABLE_PEAK、源码 SHA，
数值过 CPU reference，drop count 为零、封口成功：

- **用例 A（复制 lane）**：eager GEMM + pinned host 张量的 H2D/D2H 传输
  （pinned 迫使引擎路径产生 `memory_copy` 行），复制与 kernel 同 stream。
  验收：三方核对相等；复制 lane 绑定成立；same-lane 顺序边覆盖复制。
- **用例 B（复制事件端点）**：大矩阵跨 stream event 链，producer 侧
  RECORD 前的最后提交是引擎复制（复制流先传输入再 RECORD），consumer
  流 WAIT 后接 kernel。验收：事件边端点解析到复制；跨 lane 边 reason
  为 `event_wait`；`schedule_complete=true` 成立（若用例内无默认 stream
  混布且其余前置全成立）。
- **用例 C（默认 stream 混布）**：legacy 默认 stream 工作与显式 stream
  并存。验收：资格正确**拒绝**完整性（`default_stream_implicit_sync_
  not_inferred`）——本负例即验收正例。
- **回归**：既有 torch-scope 三用例（GEMM/复合/event 链）重放不因新增
  行类型与验证器破坏既有资格判定。

若 pinned 传输在该栈上仍走 kernel 路径而非引擎路径，用例如实记录
`memory_copy=0` 与 kernel 化复制归属，此时用例 A/B 的引擎复制部分以
注入真实结构的合成副本（CPU 分母同构）替代，GPU 分母如实登记
“引擎路径未观察到”，不伪造。

## 5. 有效性 gate（预固定）

1. CPU：新增测试全绿、既有 suite 不回归；Ruff、format、ty、耦合、
   可读性、schema 检查全绿。
2. GPU：三用例 × 3 进程全过（用例 C 以正确拒绝计过）；证据目录
   `data/outputs/solar-l2-schedule-20261106/`（执行日为准），只登记
   摘要与路径，不入库本体。
3. schema：`hip_sdk_ledger`（新增行类型）与调度证据/资格 artifact 的
   版本升级原子完成，旧版本全部移除。
4. `performance_authority` 与 `adversarial_attestation` 恒 false；
   普通 profile summary 不自动升级。

## 6. 边界（预固定，闭合后亦不得回改）

- Graph capture、IPC、跨进程维持显式拒绝；不因本线闭合放宽。
- 不从时间戳推断因果；不把 stream 0 解释为默认流同步；不从 queue 归属
  推断跨 stream 顺序。
- M-d 输出是诊断结论，不是性能授权；差距解释不改变任何测量。
- 复制域词汇不外溢到非复制域（prefetch 等明确排除）。

## 7. 执行结果（2026-11-07 追加，分母与 oracle 未回改）

### 7.1 CPU 分母

- 账本 v2 模型（`SDKMemcpy` 行 + `SDKAPI.stream`）、导出读取器
  `memory_copy` 域、`hip_sdk_memcpy` 三方核对与共观察绑定、`hip_sdk_links`
  复制端点、`hip_sdk_schedule` 混布语义、`performance_schedule_evidence`
  v3 verified 分支、`predict_hw` 事件边串行——全部按 §3 落地；
  新增 33 个测试，受影响面（bench/cli/driver）1537 passed，
  Ruff/format/ty/耦合/复用/生产可达性全绿（schema 检查对
  `sol_execbench` 家族 0 违例）。
- 可读性基线与 SOLAR 内部 schema 串的既有红项属并行 L3-e 线，未动。

### 7.2 GPU 分母（gfx1200 / GPU-a30fa29c1cca1511，ROCm 10.1 + Torch 2.14，
STABLE_PEAK host guard，证据 `data/outputs/solar-l2-schedule-20261106/`）

六用例 × 3 独立进程 = 18/18 采集并资格通过（rc=0）：

| 用例 | schedule_complete | default_stream_present | 引擎复制 | 事件边 |
| --- | --- | --- | --- | --- |
| gemm（回归） | true ×3 | false | 3 | 0 |
| composite（回归） | true ×3 | false | 3 | 0 |
| event_chain（回归） | true ×3 | false | 6 | 1 |
| copy_lane | true ×3 | false | 5（H2D+D2H 绑定 queue） | 0 |
| copy_event | true ×3 | false | 7 | 1（端点=引擎复制） |
| default_mix | **false ×3** | true | 3 | 0 |

- 受限域调度完整性 15/15 成立；default_mix 三次均被
  `default_stream_implicit_sync_not_inferred` 正确拒绝（负例即验收正例）。
- `performance_authority` 与 `adversarial_attestation` 恒 false。
- 回归三用例资格判定与 2026-10-11 基线一致（implicit 覆盖、事件边数）。

### 7.3 实施中探明的机制事实（先验证据，登记不改判分母）

1. **大拷贝分块**：64 MiB 引擎复制以同一 correlation 产生多个完成记录；
   三方核对按 multiset 相等 + 分块字节和等于宿主 `sizeBytes` 验证。
2. **流身份双空间**：SDK stream ordinal（lane 共观察用）与宿主
   `hipStream_t` 句柄（默认流判据用）互不映射；torch 显式 stream 可获得
   ordinal 0。默认流语义只能用句柄 0（采集器记录 launch/memcpy 的原始
   stream 参数）。
3. **WAIT barrier 的竞争窗口**：4096³ fp32 matmul 在 RX 9060 XT 上
   ~0.5 ms，与 record→wait 的 host 提交间隔同量级；producer 工作不足时
   runtime 合法地不产生 GPU barrier（2026-10-11 边界 4 的机制解释）。
   copy_event 用 8 次 matmul 铺队列后 barrier 稳定可观察。
4. **teardown 面**：pinned-copy 路径的封口后 teardown 新增
   `hipEventCreateWithFlags`/`hipStreamIsCapturing`/
   `hipEventRecordWithFlags`（零 kernel、零复制），已加入审计白名单。
5. **导出通道**：`--memory-copy-trace` 使 rocprofv3 导出 `memory_copy`
   域，三方核对在真机成立；采集器经 `include_memory_copy` 显式启用。

### 7.4 M-d 生产供数管道（明确未接线）

M-d 的消费面已实现并有 CPU 分母（`predict_hw` 的
`cross_lane_verified`/`event_edges`、builder 的
`_WorkloadDiagnosticInputs` 透传、无边时行为逐位不变）。生产
`build_performance_diagnostic` 尚未自动读取 scope archive：两 artifact 的
dispatch 标识空间不同（scope 的 int correlation/dispatch id ↔
counter evidence 的 str dispatch_id），标识桥需真机 counter+scope 双采集
验证，超出本次 GPU 分母。按"能力与资格分离"与"普通 profile 不自动升级"
政策，该管道登记为后续接线任务（跨 artifact 桥 + 显式 opt-in 参数），
不并入本线闭合声明；`performance_schedule_evidence` v3 的
`cross_lane_event_dependencies` 字段已就绪等待供数。

### 7.5 闭合判定

§1 第 1–4、6 项全部成立；第 5 项（M-d）按 §7.4 部分成立——消费能力与
fail-closed 缺省闭合，生产供数管道显式登记为后续任务。**L2 受限域调度
因果资格闭合**（schedule_complete 在受控 eager 域首次可资格并已获证据），
M-d 管道接线遗留。
