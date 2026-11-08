# M0 / C0：LLM_CORE 覆盖规格冻结

制定窗口：2026-11-07—08；冻结日期：2026-11-08（Asia/Shanghai）。
规格标识：`M0-C0-20261107-r1`，文件日期沿用本轮规划日期，不是生产 schema。
状态：**C0 只读规格冻结；C1/C2/C3 未开始**。
依据 [B0](solar-m0-baseline-evidence-2026-11-07.md) 的源码/语料摘要和
[迭代计划](solar-next-iteration-plan-2026-11-07.md)。没有生成 target view、运行提取或 GPU 探针。

## 1. 固定目标与两个分母

- 源语料：`problems/LLM_CORE/releases/LLM_CORE_V2/manifest.yaml`，原始文件 SHA
  `8de8b6367380fe3200907bf6f62d93dcc9b77b3040df8201e29ef03a568169e7`；
  36个 Definition 和36条 generation rule 的文件摘要已逐项匹配 manifest。
- 请求唯一 profile：`core`。核心清单恰好20项，`profile_minimum_generated_definitions.core=20`。
- 目标声明：[standard-16gb.yaml](../../problems/LLM_CORE/targets/configurations/rx9060xt/standard-16gb.yaml)，
  原始文件 SHA=`df0d88a8a4b96cab7e1b08bbd9c8ca8d0644cdc70b95cc468e3c15b3c6805b0c`。
  指定 AMD Radeon RX 9060 XT、gfx1200、standard SKU、32 CU、16 GiB；当前未观察设备。
  不用泛 gfx1200 模板冒充此配置，B2 不满足时停止本活动而不是静默改目标。
- IR 路径固定为 `make_fx_aten` 与 `torchview_extended_einsum`；前者为本研究主线，
  后者为独立比较。无自动跨路径回退，无选择最佳路径后的合并通过率。
- 形式分析使用正式 `RX_9060_XT` profile、受信 mapper、`capacity_constrained_tile_aware_v1`。
  不以 `analyze_workload_diagnostic` 的 roofline 结果填补 formal/cache-aware 缺失。

第一分母是36条 Definition 决策清单，必须全部出现；其中16条 profile外，core内20条等待测量后生成。
第二分母来自合格 target view：必须 **G=20、N=9G=180、矩阵2N=360行**。
不足20时 C2 判 target-view INCOMPLETE；即使产生诊断性部分视图，也不能改称合格 core 分母。
因此上一版计划中的2N公式仍有效，但本冻结规格下不再允许用任意更小的 G 达成验收。

每 Definition 的9槽固定如下；总计20 smoke、80 development、80 holdout：

| 槽 | 角色 |
| --- | --- |
| `smoke` | smoke |
| `latency-low`, `throughput-low`, `irregular-low`, `capacity-low` | development |
| `latency-high`, `throughput-high`, `irregular-high`, `capacity-high` | holdout |

role/slot IDs 来自已冻结规则，不从形状大小反推。正式矩阵中两路径对应角色行数为40/160/160。

## 2. Definition 决策清单（36/36）

下表是 profile 选择结果，不是硬件/容量/提取通过结论。所有 ID 属于同一冻结 manifest，
对应 definition/rule 路径及完整 SHA 由该 manifest 提供；不复制第二套可漂移的注册表。

| semantic_id | family | C0 范围 |
| --- | --- | --- |
| `llm.linear.matmul_bias.v2` | linear | core，待 C2 |
| `llm.linear.matmul_relu.v2` | linear | core，待 C2 |
| `llm.linear.matmul_gelu.v2` | linear | core，待 C2 |
| `llm.linear.matmul_silu.v2` | linear | core，待 C2 |
| `llm.linear.batched_matmul.v2` | linear | core，待 C2 |
| `llm.linear.batched_bias.v2` | linear | core，待 C2 |
| `llm.linear.grouped_matmul.v2` | linear | core，待 C2 |
| `llm.linear.grouped_silu.v2` | linear | core，待 C2 |
| `llm.norm_activation.rmsnorm.v2` | norm_activation | core，待 C2 |
| `llm.norm_activation.swiglu.v2` | norm_activation | core，待 C2 |
| `llm.norm_activation.add_rmsnorm.v2` | norm_activation | core，待 C2 |
| `llm.norm_activation.bias_gelu.v2` | norm_activation | core，待 C2 |
| `llm.position.rope_interleaved.v2` | position | core，待 C2 |
| `llm.position.rope_yarn.v2` | position | core，待 C2 |
| `llm.attention.gqa_causal.v2` | attention | core，待 C2 |
| `llm.attention.mqa_decode.v2` | attention | core，待 C2 |
| `llm.attention.sliding_window.v2` | attention | core，待 C2 |
| `llm.attention.local_global.v2` | attention | core，待 C2 |
| `llm.attention.masked_softmax.v2` | attention | core，待 C2 |
| `llm.attention.attention_sink.v2` | attention | core，待 C2 |
| `llm.advanced_attention.mla_compressed.v2` | advanced_attention | profile_not_selected |
| `llm.advanced_attention.indexed_sparse.v2` | advanced_attention | profile_not_selected |
| `llm.kv_cache.paged_write.v2` | kv_cache | profile_not_selected |
| `llm.kv_cache.cache_gather.v2` | kv_cache | profile_not_selected |
| `llm.kv_cache.cache_scatter.v2` | kv_cache | profile_not_selected |
| `llm.kv_cache.cache_compress.v2` | kv_cache | profile_not_selected |
| `llm.moe.route_top2.v2` | moe | profile_not_selected |
| `llm.moe.shared_expert.v2` | moe | profile_not_selected |
| `llm.moe.grouped_expert.v2` | moe | profile_not_selected |
| `llm.moe.normalized_topk.v2` | moe | profile_not_selected |
| `llm.moe.fused_gate_up.v2` | moe | profile_not_selected |
| `llm.moe.routed_residual.v2` | moe | profile_not_selected |
| `llm.quantization.fp8_per_token.v2` | quantization | profile_not_selected |
| `llm.quantization.mxfp8_block.v2` | quantization | profile_not_selected |
| `llm.quantization.mxfp4_block.v2` | quantization | profile_not_selected |
| `llm.quantization.int8_weight_only.v2` | quantization | profile_not_selected |

core 家族分母为 linear8、norm_activation4、position2、attention6。
其他4家族本轮 absence 原因是 profile 选择，不是能力缺失或成功；不宣称覆盖完整 LLM_CORE_V2。

## 3. C1 接入边界和数据流

现状：`audit_corpus_stage_readiness` 与 `run_solar_release_qualification` 使用
`AKACorpusManifest.load`，不能直接传 LLM_CORE manifest。
这是一项 outer bridge 接入任务，不是 SOLAR IR 不支持的证据。

冻结接法：

1. 由 [corpus.py](../../src/sol_execbench/core/dataset/corpus.py) 的现行生成流程写出 target view；
   读取时使用 [CorpusTargetViewManifest](../../src/sol_execbench/core/dataset/corpus_models.py)
   的版本与身份校验，交叉核对 manifest/规则/Definition/生成 Workload 原始摘要。
   验证 coverage、全部 decisions、完整9槽、UUID唯一性与 hardware/容量观察绑定。
2. 以 target view 中的 GeneratedProblem/GeneratedWorkloadRecord 找到已物化 `problem_dir`
   与 `workload_uuid`；不重建另一套 workload，不将 LLM_CORE 改写成 AKA 清单。
3. 构造既有 [SolarStageAuditRequest](../../src/sol_execbench/core/solar_bridge/models.py)，
   交给 `run_solar_stage_worker`。收集三个正式阶段及原始工件；仅 ready 行进入下一步。
4. 对 ready 行构造既有 `SolarWorkerRequest`，经 `run_solar_worker` / `analyze_workload`
   获取正式结果。硬件/时钟/进程隔离、真实数值与梯度回放、receipt 检查不得绕过。
5. 报告层将生成身份和原生产工件连接起来。复用原 artifact/schema owner；独立序列化的矩阵
   若需要新 envelope，在 C1 提交中归属 bridge 所在 domain registry，不替代生产工件、不得新增旁路“通过”。

C1 的必检输入边界：错误manifest/hash、重复UUID、漏一个槽、scope路径越界、硬件/容量身份冲突、
跨路径错配、旧schema、篡改工件、partial视图、旧执行源码恢复，均应拒绝。
验证过的同身份完整输入可运行；单行 unsupported 与上游未通过均保留对应状态。
这些是未来定向检查清单，本次没有实现或执行测试。

resume 键至少绑定 spec、source/semantic runtime、manifest/rule/Definition/workload、
target-view/cohort/distribution/hardware observation、IR、policy、mapper与预算参数；
已暴露但身份相同的行可继续核对，不因此重新成为未暴露留出。
变化只能新活动或现行验证器认可的严格复用，不能仅凭 UUID 匹配就跳过验证。

## 4. C2 容量门禁和暴露边界

生产 `generate_corpus` 的 `require_complete_profile` 默认 false；本活动必须显式 true。
按现行 `_eligibility` 顺序保留 profile/target/dtype/quantization/capability/insufficient_capacity 理由。
每 Definition 采用共同 scale、9槽一同成功或一同拒绝；不逐槽缩小，也不因 OOM 修改已封存视图。
target descriptor 的 `max_tensor_bytes=reference_ipc_limit_bytes=1072693248` 保持不变。

容量测量发生在 B2 合格且另获执行授权后的 C2。raw capacity evidence、capacity class、
hardware_configuration_id、distribution_id、generation_cohort_id、workload_view_digest 都必须留存。
类内硬件审计字段不能悄悄改变 workload；这些身份通过生产模型判断，不复制其哈希算法作为替代验证器。
部分视图可作为失败诊断材料，但不得喂给 C3 的正式360行活动。

完整视图只由评测侧持有；公开开发投影沿用
[build_agent_view](../../src/sol_execbench/core/generalization/views.py)，**smoke 和 holdout 均不对候选开发公开**。
本轮不生成 Agent 候选；仍记录谁在何时查看了具体轴、UUID、资源量和失败反例。
C1 完成与 B1 source freeze 在 C2/C3 之前，避免利用留出形状修接入。
C3 报告后暴露的项可以用于下一轮开发，却不能再充当该修复的独立确认样本。
公开规则的可推导性意味着这里只声称有审计的暴露隔离，不声称不可推知。

## 5. 行契约、阶段和失败分类

行键：`spec_id + target_view_digest + semantic_id + workload_uuid + ir_path`。
同一键的多次尝试存 attempt，不增加行数、不覆盖第一次失败。
必须记录 source/语义环境、生成/输入身份、slot/role/family、实际 policy、原始结果和所有工件 SHA。

| 维度 | 原始来源及冻结解释 |
| --- | --- |
| 准备/硬件 | prepare/architecture/outer_bridge；失败不可归给算子前端 |
| 提取 | graph_extraction + operator_graph；make_fx 声明的 real retry 单独标记 |
| 转换 | ir_conversion + 路径图；保留原 op/attribute/reason |
| 数值/梯度/effects | conversion_verification + attestation；分别记录实际验证范围、数值/梯度结果和 alias/mutation |
| 外部读取 | access/analysis 证据；缺证明词汇标 unproved，不从 ready 推算通过 |
| 容量与计数 | formal_analysis/formal_acceptance、bound kind、resource_work/MACs/FLOPs与 witness |
| 三视图 | unfused、ideal_fused、cache_aware 各自状态与 assumptions；UNAVAILABLE 保留原因 |
| 成本 | 各 stage 墙钟、host RSS/device峰值、mapper耗时/线程、timeout；不能测到的字段为null+原因，不能填0 |

独立维度无法从生产工件拆分时记 `not_observable` 并说明来源，不能把整个 verification pass
复制成未实际检查过的子项 pass。forward-only 不自动豁免转换器的梯度合同。

报告分类（不改变生产 enums）：

| 报告分类 | 使用条件 |
| --- | --- |
| passed | 该阶段有对应通过证据；formal 还须满足既有形式条件 |
| unsupported / unproved | 原 reason 明确支持域/证明域拒绝；保留细分原因 |
| failed | 数值/梯度/不变量失败，或已有支持域内回归；不自动称前端缺口 |
| not_reached | 上游失败、预算停机或先决条件未满足；指向阻断行/阶段 |
| not_applicable | 按现行合同确实无需此维度，附条款；不是缺证据的替代 |
| not_observable | 工件未暴露可拆分的观测；报告缺少哪项证据 |
| budget_exhausted | worker/mapper超时、预设内存或活动总预算到限 |
| infrastructure_error | worker崩溃、工具缺失、IPC/镜像/文件环境错误；必须归因，不能当作算子不支持 |

意外 OOM 原因未明时先属 infrastructure_error；只有有配额/资源证据时才归 budget_exhausted。
未知 reason 保留 raw 值并标需归因，禁止默认映射为 unsupported 或 pass。

## 6. 执行顺序、预算和差分

固定排序：按 semantic_id，再按 smoke、development四槽、holdout四槽；每 workload 先 make_fx 再 torchview。
先完成20个 smoke 的40路径行，确认身份/隔离/输出结构有效后才展开其余320行。
合法的 unsupported 不要求 smoke 全部 ready；身份/基础设施问题阻断扩展。
formal 仅运行 ready 行，不能在共同支持集内选成功者替代全部360行。

工作量上限：36项生成决策、180 workloads、360次三阶段审计、最多360次完整formal worker。
一次活动只有一个目标视图，jobs=1；GPU 段串行。每项默认只尝试一次，环境修复后的重试另记，
无自动超时重试。mapper单次上限7200s、formal worker上限14400s沿用现行入口；
三阶段审计上限3600s沿用历史资格活动。所有超时进入身份，不能借调整参数复用原回执。

研究预算上限：C2 容量活动占用最多30分钟；C3 smoke占用最多2小时，C3全活动累计最多12小时
（含smoke，不能重置），证据总量最多50 GiB；单worker host内存为
`min(16 GiB, B1确认物理内存的1/2)`。mapper线程由现行资源政策解析并在B1冻结，jobs=1。
显存约束采用实际 target view/生产分配策略，不额外伪造容量类。
这些是计划的停止上限，不是完成时间预测；现有单项上限乘360可能远超总窗口，因此
预算用尽允许得出 INCOMPLETE，不允许删行或抬高预算来宣称闭合。
B1 可进一步收紧，放宽必须在结果揭示前新规格登记并另获执行范围授权。

共同成功项对比 raw MACs/FLOPs、资源类别计数、外部读取、bound和三视图可用性；
同单位的整数计数要求逐项相等或逐项解释差异，时间不预设应逐位相等。
差异保留两侧 artifact/hash、assumptions 和输入；不取较低下界或较高通过率拼新路径。
stage计时/峰值缺少测量接缝由C1解决；不能以重复耗时相减推断未直接测到的阶段成本。

## 7. 验收与下一步触发

**C0 本次完成条件**：manifest与72个来源文件摘要核对，36条范围清单、20项core及角色/行数固定；
接入数据流、缺证据分类、资源预算和后续必检样例均已冻结。无需现时存在 target view。

**C2 合格条件**：观察配置符合模板，生产容量与身份门禁通过，strict完整视图20项/180槽，
36条 decisions 保留，封存前后摘要一致。失败不进入合格 C3。

**C3 完成条件**：360行精确对账、每阶段都有证据或合法阻断/不适用说明、共同成功差分完成、
family/role/path/stage成本与失败汇总可复核、缺口逐条有复现和owner。
预设预算失败和未解决基础设施错误使活动 INCOMPLETE；unsupported/unproved 可构成完整的有效发现。
not_observable 若影响必需的通过判定或成本交付，同样不得宣称该维度完成。
完成不要求360行全部 ready/formal，也不代表LLM_CORE获得正式发布资格。

产物为 target view、36项清单、360行矩阵、成本表、两路径差分、缺口表与暴露记录。
缺口表按受影响 Definition 数（不以重复槽放大）、语义重要性、阻塞层和预计改动范围排序。
只有共同系统性提取/IR表达缺口才进入 L3-d 讨论；容量不足、适配缺口、数值回放或证明词汇
不足不能直接作为 LLM/Affine Loop/JAX 前端的立项依据。本轮不扩 registry、不改默认backend。
