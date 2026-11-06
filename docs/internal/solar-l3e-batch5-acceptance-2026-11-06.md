# 线A 批次5（make_fx view 未声明别名家族）：预固定验收规格

制定日期：2026-11-06（实施前预固定）。前置：批次4（`1fc71291`）。

## 0. 语义（预固定）

make_fx 图中 `view`（含 `_unsafe_view` 规范化）的 ATen schema 常缺
alias_info，access 层别名声明检查误拒。torch 语义：`Tensor.view`
**无条件**返回视图（不可视图化会抛错而非拷贝）——别名性由语义保证，
与声明无关。本批把 `view` 并入批次2 的 stride 判定分支：

- 输入 tracked 且行主序连续 → 经 `_reshaped`（`reshape_access`）证明
  纯视图；
- 输入 tracked 但不连续 → 维持 `UNSUPPORTED_VIEW`（保守欠近似，
  view 兼容性一般规则留待后续）；
- 输入未跟踪（内部）→ `MATERIALIZED`（内部数据移动）。

分析侧 `view` 已在 `STATIC_VIEW_TARGETS` 流程内，无词汇变更。
排除：reshape 拷贝分支、transparent_alias、conv、索引化 loss。

## 1. 分母与预期

分母 = 批次4 清单仍阻塞行。**S5 = 4 行**（l3n44_mingpt_block make_fx
w0–w3，blockers ⊆ {view, transpose} 且 reason ⊆ {unproven_alias}）。
上界 6。view 分量在混合行中的清除不构成翻转，如实记录。

## 2. Gate

G1：view(连续输入) 解析为 tracked 视图；view(非连续) 仍
UNSUPPORTED_VIEW；view(内部输入) MATERIALIZED。G2：reshape/contiguous/
clone/loss 家族零变化。G3：翻转 == S5（4，精确）。G4：S5 行 CPU 完整
分析全部 analyzed 且 totals 非空。G5：套件全绿 + 原子更新清单。

## 3. 证据

`data/outputs/solar-l3e-external-reads-20261106/{batch5-reinventory.json,
batch5-flipped-analysis/}`。
