# L3-c 2.14 算子覆盖差分（覆盖矩阵）：预固定验收规格

制定日期：2026-11-06（实施前预固定）。本文固定 [演进路线](solar-evolution-roadmap-2026-11-05.md)
L3-c 的验收口径。规格先于实施结果固定；实施后只允许追加结果与带先验证据的
日期勘误，不允许回改分母、oracle 或阈值。

## 0. 范围与闭合判定

L3-c 闭合当且仅当以下五项全部成立：

1. **M（覆盖矩阵）**：全部 163 个 scored workload × 两条固定路径 = 326 行，
   每行携带六维覆盖判定（§2）+ 算子清单 + 冻结家族标签；产出矩阵与家族
   汇总，**不修改任何 allowlist/registry**。
2. **D（双路径差分）**：对双路径均 ready 的 workload，逐项比对两路径的
   资源计数（macs/flops）：一致或差异逐项记录（差异是发现，不是失败）。
3. **W（canary 墙归因）**：L1 canary 的 33 个失败（10
   `native_attribute_unsupported` + 10 `native_operation_unsupported` +
   10 `gradient_verification_failed` + 3 `graph_extraction_failed`）逐项
   归因到（家族，缺口类型），形成可执行的覆盖缺口清单。
4. **R（对账）**：新鲜 torchview 全量审计在 L1 canary 的 112 项分母上
   与 L1 结果逐项一致（status + failure_stage + reason_code）。
   已知背景：canary 证据产于 `12859e3e`，其后 main 只新增孤立模块
   （fake 试点/形状域），未改 readiness 行为；任何漂移必须先以日期勘误
   解释才可继续。
5. **登记**：HANDSOFF 条目 + roadmap 状态 + 测试/lint 全绿 + 提交。

## 1. 分母与执行方式（预固定）

- 分母：`problems/AMD_AKA/manifest.yaml` 全部 scored workload（43 题
  163 项）× {`make_fx_aten`, `torchview_extended_einsum`}。
- 就绪审计：生产接缝 `audit_corpus_stage_readiness`（cuda:0，逐槽隔离
  worker，超时 3600s，支持 resume），产物为 L1 同构的 `matrix.jsonl`。
- 形式分析：对每路径 ready 槽经生产桥 `analyze_workload`（formal，
  pinned mapper，cuda:0）跑完整分析；`operator_graph.yaml` 提供算子
  清单与 effects，`solar-analysis.yaml` 提供 totals。
- 失败槽不跑分析（fail-closed 原样保留），其覆盖缺口以审计的
   stage/reason/message 为准。

## 2. 六维覆盖判定（预固定）

| 维度 | 判定来源 | 取值 |
| --- | --- | --- |
| extraction | 审计 stages | passed / failed:`graph_extraction_failed` |
| conversion（语义） | 审计 stages | passed / failed:`native_operation_unsupported` / failed:`native_attribute_unsupported` / failed:`strict_conversion_failed` |
| verification_precision | 审计 stages | passed / failed:`conversion_not_proven` |
| verification_gradient | 审计 reason | passed / failed:`gradient_verification_failed` / not_reached |
| effects（alias/mutation） | 分析期 operator graph | has_alias / has_mutation 布尔；无分析则为 gap |
| formal_analysis + resource_counting | `analyze` 结果 | analyzed（totals 非空） / failed:`reason` / not_reached |

## 3. 冻结家族规则（预固定，多标签，首中即标）

按 problem 目录名 token 与算子 target 子串分类（大小写不敏感，次序固定；
名字 token 保证失败行在无算子清单时仍可归因——本 corpus 问题名描述性强）：
`explicit_backward`（名 bwd/backward）、`attention`（名/算子 attention、
sdpa、scaled_dot_product）、`low_precision`（名/算子 quant、fp8、i8、
mxfp8、quantize、dequantize）、`convolution`（名/算子 conv）、`matmul`
（名 matmul/gemm/matrix；算子 matmul/mm/bmm/einsum/linear）、
`normalization`（名/算子 norm、softmax）、`reduction`（名 sum/reduction/
pool；算子 sum/mean/max/min/arg/topk/reduction/pool）、`view_alias`
（名 transpose；算子 view/reshape/permute/transpose/slice/select/expand/
repeat/concat/cat/split/gather/index/scatter/embedding）。其余算子落
`elementwise`（点风格）或显式 `other`（不得静默丢弃）；effects 含 mutation
的行额外加 `mutation` 标签。规则表实现为冻结字面量并与测试共用。

## 4. 有效性 gate（预固定）

1. 326/326 行存在，无静默跳过（worker 崩溃 → 行标记 worker_error，campaign 失败）；
2. R 对账（§0.4）逐项一致；
3. 每个分析成功行的算子清单非空、totals 非空（flops 或 macs ≥ 0 且
   resource_work 存在）；
4. 每个算子都命中家族规则（含 other 兜底），矩阵无未分类算子；
5. 家族汇总覆盖路线图 8 个命名家族的 presence 判定（≥1 workload 即
   present；0 也如实记录为 absent）；
6. W 归因表 33 项全部分类完成；
7. 身份绑定：git commit、manifest sha256、mapper sha256、device/gfx、
   torch 版本、`utc_timestamp()`。

## 5. 证据目录

```
data/outputs/solar-l3c-20261106/
  audit/<ir_path>/{matrix.jsonl,summary.json,workloads/...}   # 生产接缝原样
  analysis/<ir_path>/<problem>/<workload>/{...生产分析工件..., record.json}
  coverage-matrix.jsonl        # 326 行 × §2 六维 + 算子 + 家族
  family-summary.json          # 家族 × 路径 × 维度 计数
  canary-wall-attribution.json # 33 项归因
  differential.json            # 双 ready 槽的资源计数比对
  l3c-summary.json             # gate 判定 + 身份
```

## 6. 边界与不做的事

- 只产出矩阵与归因；不扩 allowlist、不改 registry/转换器/验证器。
- 覆盖判定沿用既有 fail-closed reason code 词汇，不新增通过语义。
- GPU 为共享前提：cuda:0、正式设备 gate、pinned mapper；无 STABLE_PEAK
  时钟声明（工程测量口径与 L1 审计一致）。
- 双路径 totals 差异只记录不裁决——是否等值属于后续覆盖修复线的
  oracle 工作。
