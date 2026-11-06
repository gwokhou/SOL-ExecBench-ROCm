# 线A 批次2（视图/复制/别名语义）勘察与设计草案

日期：2026-11-06。本文是批次2 实施前的语义勘察记录，不是验收规格。
批次2 验收规格须在本文开放问题解决后另行预固定。

## 1. 残余缺口（修正清单 v2 实证）

128 行中的 82 行仍被阻塞，视图/复制/别名类 blocker 分布：
`view` 36（unproven_alias 20 + copy_boundary 16 继承）、`reshape` 24
（全部 unproven_alias，torchview 侧）、`contiguous` 12（copy_boundary）、
`alias`/`detach` 16（copy_boundary，多为继承）、另有
`<transparent_alias>` 38 行（声明输出经视图链别名外部存储）。

## 2. 语义链路（已打通）

- **reason 的产生在转换侧 access 层**（`ir/access_graph.py::
  resolve_graph_accesses`），分析侧（`analysis/external_reads.py::
  _blocking_reason`）只消费：
  - STATIC_VIEW 分支：要求视图层每个输出槽在 schema `effects.aliases`
    中声明无条件 alias(input 0)；未声明 → `ALIAS`（unproven_alias）。
  - `COPY_TARGETS = {clone, copy, contiguous}` 一律 `COPY`
    （copy_boundary），分析侧仅豁免 `clone`。
  - **reason 继承**：视图层输入的 producer 不可解析时，输出继承其
    reason（`view.Output` 的 copy_boundary 即继承自上游 contiguous）。
- **make_fx vs torchview 差异**：make_fx 的别名效应来自 ATen
  FunctionSchema（`reference_serializer._schema_effects`）；torchview 的
  来自其语义模块。torchview 的 `reshape` 语义**未声明**输出别名 → 全部
  ALIAS；`reshape_access`（`ir/access_views.py`）本身已有"行主序连续性
  保持则可证"的证明器（否则 UNSUPPORTED_VIEW）。
- **透明层声明输出别名**（`<transparent_alias>` 38 行）：输出经视图链
  别名外部存储，是界/容量语义问题，不是读取计数问题。

## 3. 候选证明设计（待规格化）

1. **contiguous 的 stride 可判定证明**：源 `TensorAccess` 携带 strides；
   输入已连续 → 无操作别名（视图不变、读归 base）；不连续 → 拷贝
   （读 = 源操作数全元素，输出独立存储 materialized）。两分支均可证。
   `copy`（copy_，写突变语义）与 `clone`（已在分析侧豁免但 access 侧
   仍 raise COPY——注意其输出不进 `maps.tensors`，消费者读会被跳过，
   需核实是否存在读取漏计）保持不变。
2. **torchview reshape 别名声明**：在其语义模块补 alias(output_slot ←
   input0) 声明（与 make_fx schema 对齐），随后复用既有
   `reshape_access` 证明器；不可证分支维持 UNSUPPORTED_VIEW。
3. **视图层非物化输出**：分析侧 STATIC_VIEW 分支现要求 reason ⊆
   {MATERIALIZED}。候选放宽为 ⊆ {MATERIALIZED, STATIC}——读取经
   `view.base_id` 归账语义不变；**前置核实**：下游（union/capacity/
   fusion）对非物化输出的不变量。
4. **transparent_alias（38 行）不在批次2 范围**：声明输出别名外部存储
   涉及 bound 语义（输出即外部存储），单独立项。

## 4. 开放问题（规格化前必须解决）

1. clone 输出被 access 层 raise 后不进 tensors，消费者是否漏计读取
   （若漏计是现 bug，须先修）？
2. reason 继承链是否可能把已证明层误标（继承 vs 重算边界）？
3. `expand`/`repeat`（批次1 已入 FULL_OPERAND_READS）在 access 层的
   视图语义（broadcast 视图 vs 物化）是否与分析侧一致？
4. torchview 语义模块中 alias 声明的准确位置与影响面
   （`extended_einsum/torchview/semantics.py`）。

## 5. 建议实施顺序

先解决 §4.1（潜在读取漏计——正确性优先），再 3.1（contiguous），
再 3.2（torchview reshape 声明），3.3（MATERIALIZED/STATIC 放宽）
最后且须独立负例护栏。每步独立预固定规格 + 重清单三向相等验证
（沿用批次1 的方法学与勘误纪律）。
