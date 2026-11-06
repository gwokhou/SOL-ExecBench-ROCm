# 线A 批次2（contiguous/reshape 条件别名判定）：预固定验收规格

制定日期：2026-11-06（实施前预固定）。前置：批次1 已闭合（`2d6bc9b9`），
批次2 勘察（`ca0c23f1`）的 §4.1 疑似漏计已经实证解除（clone→matmul
探针 exact 448B = x+w 全计入），§4.3 同构解除，§4.4 根因落定。

## 0. 语义设计（预固定）

torch 语义：`reshape`/`contiguous` 的输出别名是**条件的**（可能视图可能
拷贝），torchview 语义如实声明 `conditional: true`。条件性可用输入视图
的**行主序连续性**静态判定（决策变量在 access 层可得）：

1. **视图分支**（输入 tracked 且 strides == 行主序连续）：
   - 条件别名按无条件接受（别名在连续输入下必然成立）；
   - `reshape` 复用既有 `reshape_access`（行主序保持）证明；
   - `contiguous` 为无操作：输出视图不变。
2. **拷贝分支**（输入不连续）：
   - `reshape`/`contiguous`/`clone` 的输出是**新物化存储**：access 层
     将其 reason 记为 `MATERIALIZED`（而非 COPY），使下游视图可正常
     解析（修复继承链把 copy_boundary 传染给消费者的行为）；
   - 拷贝层的读取在分析侧按全操作数计入（读取按 base 求并集，重计
     无害）。
3. `copy`（copy_，写突变语义）不变；`<transparent_alias>`（声明输出
   别名外部存储，38 行）不在本批范围；`view` 非条件别名路径不变。

## 1. 生产改动面（预固定）

- `ir/access_graph.py::resolve_graph_accesses`：STATIC_VIEW 分支接受
  {reshape, contiguous} 的条件别名当且仅当输入视图行主序连续；拷贝
  分支按 §0.2 记 MATERIALIZED；contiguous 从 COPY 全拒中移出。
- `analysis/external_reads.py`：`contiguous`、`reshape` 加入
  `FULL_OPERAND_READS`；COPY 全拒集合排除 `contiguous`。
- 不改 torchview 语义声明、schema、序列化与其余拒绝路径。

## 2. 分母与预期（预固定）

- 分母 = 修正清单 v2 的 82 个仍阻塞行。
- **S2（必须翻转集）= 12 行**（全部 torchview，blockers ⊆
  {reshape, contiguous, view-inherited} 且 reason ⊆ {unproven_alias,
  copy_boundary}）。
- **上界 = 20 行**：因继承级联不可完全预计算，翻转集允许 ⊇ S2 且
  ≤ 20；超出 20 行视为意外放松，campaign 失败。
- 4 个部分行（l1n44_mingpt_block，余 layer_norm/transpose）**必须保持
  阻塞**。

## 3. oracle 与 gate（预固定）

1. **G1 单元**：连续输入的 reshape/contiguous 视图分支产出可解析视图；
   非连续输入的拷贝分支输出 reason == MATERIALIZED 且下游视图不再继承
   copy_boundary；clone 下游视图同理由 copy_boundary 变为可解析。
2. **G2 负例**：`copy_` 突变、`<transparent_alias>` 行、layer_norm/
   transpose/nll_loss/conv2d 阻塞行为零变化；非连续输入 reshape 的
   输出不得进入 tracked tensors（拷贝语义）。
3. **G3 重清单**：翻转集 ⊇ S2(12) 且 ≤ 20；部分行保持阻塞。
4. **G4 分析确认**：翻转集中 ≥ S2 全部行 CPU 完整分析 analyzed 且
   totals 非空（继承批次1 方法；conv 家族行如出现按同规则处理）。
5. **G5 套件**：全量测试通过；固定旧语义的测试原子更新并列清单
   （已知：clone copy_boundary 继承相关断言若存在）。

## 4. 证据目录

```
data/outputs/solar-l3e-external-reads-20261106/
  batch2-reinventory.json
  batch2-flipped-analysis/
  l3e-batch2-summary.json
```

## 5. 边界

- 每个新通过路径必须有"决策变量可得"的静态判据（strides）；不得引入
  数据依赖判定。
- 批次3（norm/conv/pool 窗口读取）与 transparent_alias（输出别名外部
  存储）另行立项。
