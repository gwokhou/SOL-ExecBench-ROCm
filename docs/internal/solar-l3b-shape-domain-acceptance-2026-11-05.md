# L3-b 形状域契约（阶段 1：具体实例化）：预固定验收规格

制定日期：2026-11-05（实施前预固定）。本文固定 [演进路线](solar-evolution-roadmap-2026-11-05.md)
L3-b **第一阶段**的验收口径。规格先于实施结果固定；实施后只允许追加结果与带先验
证据的日期勘误，不允许回改分母、oracle 或阈值。

## 0. 范围与闭合判定

L3-b 阶段 1 闭合当且仅当以下五项全部成立：

1. **A（自有约束 AST）**：`src/solar/shapes/` 生产模块提供
   `ShapeSymbol`（name/range/divisibility/provenance source binding）、
   `DimExpr`（整系数仿射：符号线性组合 + 常数）、`TensorBinding`
   （张量名/输入输出角色/逐维 DimExpr 或固定整数）、`Assumption`
   （符号间关系 ≤/==）、`IntervalDim`（仅区间界定的维，nonzero 类，
   含 provenance）、`ShapeDomain`（域标识 + 以上全部），
   带规范序列化（to_dict/from_dict 往返）与稳定身份
   （canonical JSON checksum）。不把上游 `repr(SymBool)`/ShapesSpec
   当可执行协议；AST 完全 SOLAR 自有。
2. **V（验证器）**：`ShapeDomain.validate_assignment` 对具体符号赋值给出
   validated / rejected + 稳定 reason code（枚举先行固定，见 §3）；
   `bind_tensors` 从具体张量形状解出符号赋值。区间维按实例供给具体值，
   超界拒绝。
3. **S（结构稳定性认证）**：`structural_projection`（算子 target/
   exact_target/overload/effects/常量参数序列）+
   `certify_structure`：同一 reference 在多个域内实例的投影一致且差异
   常量可由符号赋值解释 → `structure_stable`；存在不可解释的结构差异 →
   `shape_branch_rejected`（拒绝全域复用声明，只保留逐实例）。
   常量解释阶段 1 限单符号；仿射解释留待后续（记录为限制，不是缺口豁免）。
4. **P（实例工件记录来源域）**：R06 真实分析（make_fx，CPU，pinned
   mapper）经现有 `AnalysisRequest.analysis_metadata` 通道携带
   `shape_domain` 块（domain_id、domain_sha256、实例符号赋值）；
   `solar-analysis.yaml` 含该块，`manifest.yaml` 以内容哈希绑定之。
   该通道是调用方自有元数据，不改任何生产 schema；ConversionRequest
   增加形状域字段属于后续阶段。
5. **分离计价不变性**：同一实例带/不带 shape_domain 元数据各跑一次完整
   分析，张量计价（analysis totals/resource_work）逐项相等——形状域层
   不进入张量计账；验证/语义/工件合同不受扰动。

## 1. 分母（预固定）

- **R06 域消费者**：`torch2flydsl/moe_topk_softmax` 全部 4 个 workload
  （w0 E=256,K=8,T=64；w1 E=256,K=8,T=1024；w2 E=128,K=4,T=256；
  w3 E=64,K=2,T=64）+ 冻结合成域内点：
  - 边界点 B-max = (T=1024, E=256, K=8)（全部上界同时命中）
  - 内部非语料点 I = (T=128, E=192, K=6)
- **R06 冻结域声明**（实现为 campaign 内冻结字面量，身份入证据）：
  - 符号：`T` ∈ [64,1024] 整除 64；`E` ∈ [64,256] 整除 64；`K` ∈ [2,256] 整除 2
    （宽于语料实值 {2,4,8}，使 `K<=E` 假设非退化并能承载 §2 的假设违反
    负例探针，例如 K=100,E=64 是域内各 range 合法但 topk 语义非法的点）；
    provenance 均为 `definition.axes:<name>`
  - 假设：`K <= E`
  - 张量绑定：gating→[T,E]，bias→[E]，topk/route_scale→标量（零维），
    weights→[T,K]，ids→[T,K]（T、E、K 均为共享维度）
- **聚焦样例**（`scripts/internal/solar/l3b_samples.py`，测试与 campaign
  共用）：`shared_and_derived`（双输入共享 B，输出 [2*B] 派生维）、
  `nonzero_interval`（输出维仅区间界定）、`shape_branch`（按维度值分支）、
  `alias_mutation`（复用 L3-a mutation 语义样例）。

## 2. 聚焦词汇 oracle（预固定）

| 词汇 | oracle |
| --- | --- |
| 共享维度 | 同一符号绑定多个张量维度；不一致的具体形状 → `shape_mismatch` |
| 派生维度 | 输出维 `2*B` 由赋值求值；不符 → `shape_mismatch` |
| 合法边界 | 恰在 lower/upper 的赋值 validated（含 B-max 三上界同时命中） |
| 越界 | 超出 range → `out_of_range` |
| 整除 | 域内值但不满足整除 → `divisibility_violated` |
| 假设违反 | 各符号均在 range 但 `K<=E` 不成立 → `assumption_violated` |
| shape branch | 两域内实例结构投影差异不可由符号解释 → `shape_branch_rejected`；R06 的 `topk.k` 随 K 变化必须被解释为符号参数而非差异 |
| nonzero 区间 | 实例供给区间内具体值 validated；超上界 → `interval_exceeded` |
| alias/mutation | 域验证不扰动既有语义：样例经生产 `audit_conversion`（CPU）三阶段通过 |

## 3. 稳定 reason codes（预固定枚举）

`validated`；`out_of_range`、`divisibility_violated`、`assumption_violated`、
`shape_mismatch`、`interval_exceeded`、`unknown_symbol`、
`shape_branch_rejected`（认证层）。不在此列的状态不得出现于工件。

## 4. 有效性 gate（预固定）

L3-b 活动判通过当且仅当：

1. §1 分母内 4 个 workload + 2 个合成域内点全部 `validated`；
2. §2 全部词汇断言按行通过（负例必须以指定 reason code 拒绝，
   不得静默通过）；
3. R06 结构认证：w0/w1/I 三实例 `structure_stable`（topk.k 由 K 解释）；
   `shape_branch` 样例认证为 `shape_branch_rejected`；
4. 实例分析（P）：`solar-analysis.yaml` metadata 含 `shape_domain` 块且
   `domain_sha256` 与冻结域声明 checksum 相等；`manifest.yaml` 绑定的
   solar-analysis 哈希覆盖该块；
5. 分离计价不变性：带/不带元数据两次分析的 totals 与 resource_work
   逐键相等；
6. 序列化往返保持 `stable_identity` 不变；
7. 全部新测试 + 既有 solar/scripts 套件通过；ruff/ty 全绿。

## 5. 证据目录

```
data/outputs/solar-l3b-20261105/
  r06-domain-verdicts.json      # 分母 + 负例探针逐项判定
  structure-cert.json           # 三实例投影 + 差异解释记录
  instance-analysis/with-domain/    # 真实分析（含 shape_domain 元数据）
  instance-analysis/control/        # 同实例无元数据对照
  l3b-summary.json              # gate 判定 + 身份绑定（git/manifest/mapper sha）
```

## 6. 边界与不做的事

- 不修改 ConversionRequest/图/IR/verification/manifest 的既有 schema；
  阶段 2（请求级形状域字段、符号化 Orojenesis）另立规格。
- 域验证是**逐实例成员判定**，不构成全域形式化证明；全域复用只以
  `structure_stable` 的有限实例证据声明（样本点先行固定）。
- torchview 默认路径不受任何影响；全部活动走 make_fx_aten 与 CPU。
- 不引入上游 ShapesSpec/SymBool 运行时依赖；AST 与验证纯 SOLAR 自有。
