# 线A 批次3（norm 词汇补全 + gelu 复核）：预固定验收规格

制定日期：2026-11-06（实施前预固定）。前置：批次1（`2d6bc9b9`）、批次2
（`63e968ef`）已闭合。

## 0. 范围与语义（预固定）

1. **norm 家族词汇补全**：`layer_norm`、`batch_norm`、`group_norm` 加入
   `FULL_OPERAND_READS`——归一化读取其全部张量操作数（x/weight/bias，
   batch_norm eval 另含 running 统计量），逐元素-归约读取与既有条目机制
   同构，sound。按修正清单，无纯 norm-blocker 行，预期翻转 0（家族补全，
   混合行的 norm 分量由此消除）。
2. **gelu 复核去 opaque**：torchview 原生注册表把 gelu 与 norm/kl_div
   一起标 `opaque_library_call=True`。gelu 是纯逐元素函数，与已复核的
   relu/elu 同类，移入 `_NO_EFFECTS`。**仅 gelu**；kl_div、cross_entropy
   （索引化 loss 语义）、nll_loss 保持 `_LIBRARY`。
3. **明确排除**：conv2d 窗口覆盖（需区域机器，批次4）、nll 索引词汇、
   `<transparent_alias>`（输出别名域）。

## 1. 分母与预期（预固定）

- 分母 = 批次2 后清单（`batch2-reinventory.json`）的仍阻塞行。
- **S3 = 15 行**（全部 torchview，blockers ⊆ {gelu} 且 reason =
  effect_boundary；即 l1n26_gelu 与 l2n86/l2n99 家族的 gelu 槽）。
- 上界 20；部分行保持阻塞。

**勘误（2026-11-06，G3 执行中）**：上句自身的严格定义（blockers ⊆
{gelu}）确定性给出 **S3 = 12 行**；冻结时的计数 15 误纳入了 3 行
l1n40_layernorm 的 torchview `layer_norm: effect_boundary` 行——其
blocker target 是 layer_norm 而非 gelu，被同一句定义排除（其
de-opaque 属批次4 候选，不在本批）。S3 修正为 12；上界同步修正为 15。

## 2. Gate（预固定）

1. G1：gelu 图（torchview）effects 不再含 opaque；norm 词汇单元通过。
2. G2 负例：kl_div/cross_entropy/nll_loss 行为零变化；make_fx gelu 行
   （本就通过）不受影响。
3. G3：重清单翻转 == S3（15，精确；多/少均失败）。
4. G4：S3 行 CPU 完整分析全部 analyzed 且 totals 非空（15/15）。
5. G5：套件全绿；固定旧 gelu-opaque 语义的测试原子更新并列清单。

## 3. 证据

`data/outputs/solar-l3e-external-reads-20261106/{batch3-reinventory.json,
batch3-flipped-analysis/, l3e-batch3-summary.json}`。
