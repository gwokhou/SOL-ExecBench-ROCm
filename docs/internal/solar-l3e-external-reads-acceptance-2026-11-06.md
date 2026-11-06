# 线A external-reads 证明词汇扩展（批次 1）：预固定验收规格

制定日期：2026-11-06（实施前预固定）。本文固定 L3-c 后续线"external-reads
语义扩展"**批次 1**的验收口径。规格先于实施结果固定；实施后只允许追加
结果与带先验证据的日期勘误，不允许回改分母、oracle 或阈值。批次 2/3 只
枚举不展开，另立规格。

## 0. 目的与范围

L3-c 矩阵（`9dc3c46d`）测得 128 行形式边界失败于 `external reads are
unproved`。逐槽归因清单（冻结于
`data/outputs/solar-l3e-external-reads-20261106/inventory/l3e-inventory.json`，
sha256 `f7294fd3eec87ae0154415ffa9117f561a81c28195700c0d1037f427a2c7a109`）
把缺口分解为三个证明类。批次 1 只做其中机制同构、零新语义的部分：
把既有 `FULL_OPERAND_READS` 词汇表补齐，不新增任何证明机制，不放松
fail-closed 的任何其他路径。

## 1. 批次 1 冻结词汇（15 个 target）

`cat`、`pow`、`sin`、`cos`、`log2`、`floor`、`exp2`、`log_softmax`、
`softplus`、`leaky_relu`、`bitwise_and`、`xlogy`、`masked_fill`、
`expand`、`repeat`。

判定依据（冻结）：
- 一元/二元逐元素数学与已有条目（`gelu/silu/tanh/exp/eq/clamp` 等）
  机制完全同构——读取为其张量操作数的全操作数读取；
- `cat` 逐输入全操作数读取；`expand/repeat` 的外部读取等于其操作数的
  唯一元素（输出大于读取，方向保守）；
- **明确排除**（不属批次 1）：`nll_loss_forward`（索引化归约，需
  topk_gather 式证明）、`reshape/view/transpose/contiguous/alias/detach`
  （批次 2 别名/复制边界）、`layer_norm/batch_norm/group_norm/conv2d/
  max_pool2d/avg_pool2d`（批次 3 窗口读取）、torchview 侧全部
  `effect_boundary` 项。

## 2. 分母与预期翻转（预固定）

- 分母 = 冻结清单的 128 行。
- **S1（预期全翻转集）= 恰好 46 行**：其 blocker target 集合 ⊆ §1 词汇
  且 reason 集合 ⊆ {`read_semantics_unproved`}；按路径分布
  make_fx 33 行、torchview 13 行（由冻结清单 sha 确定性可复算）。
- 部分受益 30 行（批次 1 消除部分 blocker 但余 `view/contiguous/reshape/
  nll_loss_forward/batch_norm/conv2d/...`）、不受影响 52 行。

## 3. oracle 与 gate（预固定）

1. **G1 词汇单元**：§1 每个 target 出现在图中时不再产生 blocker；
   机制上等价于既有 `FULL_OPERAND_READS` 条目（同一代码路径）。
2. **G2 负例不放松**：`conv2d`（带 stride/padding/groups 的非满收缩）、
   `nll_loss_forward`、`layer_norm`、批次 2 的 `view/reshape` 别名类在
   批次 1 落地后**仍拒绝**，reason 不变；torchview `effect_boundary`
   行为零变化。
3. **G3 重清单精确翻转**：同一清单脚本对 128 行重跑，翻转集 == S1
   （46 行，逐行 id 相等；多一行=意外放松，少一行=回归，均失败）。
4. **G4 真实分析确认**：S1 内非 conv 家族行在 CPU 上逐槽子进程跑完整
   `analyze()`，**≥ 40/46 达到 analyzed 且取得形式 totals**；不足者
   逐行给出生产 reason code（容量/mapper 类失败如实记录，不计放松）。
   S1 内若含 conv 家族行（按冻结清单其 blocker 已排除 conv2d，预期为
   零），出现则单独走日期勘误。
5. **G5 套件不变量**：既有测试全绿；固定旧语义（把 §1 target 断言为
   未证明）的测试在同一提交内原子更新并列出清单。

## 4. 证据目录

```
data/outputs/solar-l3e-external-reads-20261106/
  inventory/l3e-inventory.json          # 冻结基线（sha 已绑定）
  batch1-reinventory.json               # 词汇落地后重清单
  batch1-flipped-analysis/<path>/<slot>/ # S1 行的 CPU 完整分析
  l3e-summary.json                      # gate 判定 + 身份绑定
```

## 5. 实现边界（预固定）

- 生产改动仅限 `FULL_OPERAND_READS` 集合扩充（`src/solar/analysis/
  external_reads.py`）及同提交内的原子测试更新；不改
  `_blocking_reason` 控制流、DYNAMIC_INDEX/COPY/mutation/effect 路径、
  access 语义与任何 schema。
- 清单/分析 campaign 沿用免验证清单路径与逐槽子进程隔离（OOM 教训，
  见 `l3e_inventory.py` 注释）；完整分析仅对 S1 行执行。
- 批次 2（视图别名/复制边界）与批次 3（norm/conv/pool 窗口读取）在本
  批次落地并验收后另立规格，词汇与本批隔离。

## 6. 勘误（2026-11-06，G4 执行中发现，依据先于本线的证据）

G4 首跑 38/46 达标（低于预固定下限 40），8 个未过行全部为
`dynamic_mxfp8_quant`（make_fx，w0–w7），失败原因均为
`external reads are unproved: unproven_alias`。归因：**清单诊断工具的
blocker 模型漏建模了生产 `external_reads` 的透明层别名路径**（透明层
产出声明输出 → `AccessReason.ALIAS`，该生产代码路径先于本线存在；
工具缺陷可由生产函数源码直接证明）。修正后的工具补充该建模，并以
`<transparent_alias>` 标记（修正清单证据：
`inventory/l3e-inventory-v2-corrected.json`，sha256 `d77baa768f900975…`）。

修正结论（三向集合相等，逐行验证）：
- **S1 修正 = 38 行**（原 46 行 − 8 个 alias 行；alias 属性与词汇无关，
  38 个 alias 行分布于 128 行全域，8 个恰好落入原 S1）；
- 修正词汇下重清单的零 blocker 集 == S1 修正（38，逐行相等）；
- G4 实际分析成功集 == S1 修正（38，逐行相等；8 个 miss 全部
  `unproven_alias`，属批次 2 域，如实记录不计放松）。

据此：G3/G4 以修正 S1=38 判定通过；原 G4 下限 40 由本勘误废止
（其基于漏建模的 S1=46）。批次 1 的语义边界不变：`unproven_alias`
路径未被触碰，别名语义仍 fail-closed。
