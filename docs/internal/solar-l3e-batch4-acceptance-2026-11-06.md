# 线A 批次4（layer_norm torchview de-opaque）：预固定验收规格

制定日期：2026-11-06（实施前预固定）。前置：批次3 已闭合（`1d8ce971`）。

## 0. 范围（预固定）

torchview 原生注册表的 `layer_norm` 仍标 `_LIBRARY`（opaque）。layer_norm
是已复核的纯归一化函数，读取全部张量操作数（批次3 已入分析侧词汇）——
与批次3 gelu de-opaque 完全同构，移入 `_NO_EFFECTS`。**仅 layer_norm**：
batch_norm/group_norm 虽同类，但当前残余清单中无其 torchview-opaque 行
（留待出现时同法处理）；kl_div/cross_entropy/nll_loss 维持 opaque；
`_unsafe_view`/view 继承（mingpt 4 行 + view×20）与 conv 窗口覆盖明确
排除（批次5 域）。

## 1. 分母与预期（预固定）

- 分母 = 批次3 清单（`batch3-reinventory.json`）仍阻塞行。
- **S4 = 3 行**（l1n40_layernorm torchview w0–w2，blocker ==
  layer_norm:effect_boundary）。上界 5。

## 2. Gate（预固定）

1. G1：layer_norm torchview 图 effects 无 opaque；分析侧词汇已有。
2. G2 负例：kl_div/cross_entropy/nll_loss/conv2d/`_unsafe_view` 零变化。
3. G3：重清单翻转 == S4（3，精确）。
4. G4：S4 行 CPU 完整分析全部 analyzed 且 totals 非空。
5. G5：套件全绿；固定旧语义测试原子更新并列清单。

## 3. 证据

`data/outputs/solar-l3e-external-reads-20261106/{batch4-reinventory.json,
batch4-flipped-analysis/}`。
