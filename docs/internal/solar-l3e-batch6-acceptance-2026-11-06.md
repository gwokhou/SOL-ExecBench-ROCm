# 线A 批次6（transparent_alias 内部物化输出别名）：预固定验收规格

制定日期：2026-11-06（实施前预固定）。前置：批次5（`36c7e27c`）。

## 0. 语义（预固定；勘察实证嵌入）

批次2 勘察把 `<transparent_alias>` 定性为"声明输出经视图链别名**外部**
存储"。批次6 勘察（实施前，38 行逐行实证）推翻该定性：冻结清单内全部
42 个 `<transparent_alias>` 条目无一例外是 `identity`/`detach`/`view`
在**未跟踪（内部物化）输入**上的 proven view——其声明输出的存储是图
自身 compute 层新分配的内部存储，不是外部存储。真正别名外部存储的
视图链（输入在 `access.tensors` 中被跟踪）在冻结清单中未出现，但语义
上仍属未证明域。

冻结判定（分层，可静态判定）：

- transparent 层的输出为声明图输出，且其**每个**张量输入都未跟踪
  （不在 `access.tensors`）且 reason 恰为 `MATERIALIZED`
  （`materialized_value`，即图自有新存储）→ 该输出别名内部物化存储，
  无外部读取问题，**不产生** `ALIAS` reason；
- 任一张量输入被跟踪为外部基的视图（在 `access.tensors` 中），或未
  跟踪但 reason 不是 `MATERIALIZED`（如 `UNSUPPORTED_VIEW`/`COPY`/
  `EFFECT`/`DYNAMIC_INDEX`——存储归属未证明，可能是外部存储的别名）
  → 维持 `ALIAS`（unproven_alias），fail-closed 不变。

理由：`MATERIALIZED` 的语义就是"图自有的新分配存储"（clone/拷贝分支/
内部数据移动），别名它不引入任何外部存储可见性；读取仍按消费者归账
到外部基，输出写入已在内部生产者处计价。批次2 已为读取侧接受
"resolved STATIC 视图等同物化值"，本批把同一逻辑补到声明输出门。

## 1. 分母与预期翻转（预固定）

分母 = 批次5 清单（`batch5-reinventory.json`，sha256
`b7f9ef3f72582b713ab9c5ea21d0d708e7f464ac094aa248e5e54acc297567d1`）
中仍有阻塞的行。**S6 = 22 行**（blocker 集 ⊆ {`<transparent_alias>`}，
全部 make_fx，由该 sha 确定性可复算）：

- `instruction2triton/rmsnorm_bwd` w0–w3（4）；
- `torch2flydsl/dynamic_mxfp8_quant` w0–w7（8）；
- `torch2hip/gpumode_positionwise_ffn` w0–w3（4）；
- `torch2hip/l1n40_layernorm` w0–w2（3）；
- `torch2hip/l2n37_groupnorm_fused` w0–w2（3）。

部分受益 16 行（清除 transparent 分量但仍余 pool/loss/conv blocker，
不构成翻转，如实记录）：`l1n42_maxpool2d` w0–w2（余
`max_pool2d_with_indices`）、`l1n95_cross_entropy` w0–w4（余
`nll_loss_forward`）、`l2n52_conv_activation_batchnorm` w0–w4（余
`conv2d`）、`l2n98_matmul_avgpool_gelu_scale_max` w0–w2（余
`avg_pool2d`）。上界 24。

## 2. Gate（预固定）

1. **G1 判定单元**：identity/detach/view(内部物化输入, declared) 三类
   通过且不产生任何 reason；同构造但输入被跟踪为外部基 → `ALIAS`；
   未跟踪但 reason 为 `UNSUPPORTED_VIEW` 的输入 → `ALIAS`。
2. **G2 负例不放松**：外部基视图链上的声明输出别名仍拒绝；批次1–5 的
   全部负例（conv2d 非满收缩、nll_loss_forward、非连续 view、索引化
   loss、torchview effect_boundary 家族）reason 零变化。
3. **G3 重清单精确翻转**：修正镜像后的清单脚本重跑分母，翻转集 == S6
   （22 行，逐行 id 相等；多一行=意外放松，少一行=回归，均失败）；
   16 个混合行的 blocker 列表中 `<transparent_alias>` 分量全部消失且
   其余分量不变。
4. **G4 真实分析确认**：S6 全部 22 行在 CPU 上逐槽子进程跑完整生产
   `analyze()`（pinned mapper），全部 analyzed 且 totals 非空；
   不足者逐行给出生产 reason code（容量/mapper 类失败如实记录，
   不计放松）。
5. **G5 套件不变量**：既有测试全绿；固定旧语义（把内部物化声明输出
   别名断言为 unproven_alias）的测试在同一提交内原子更新并列出清单。

## 3. 证据

`data/outputs/solar-l3e-external-reads-20261106/{batch6-reinventory.json,
batch6-flips.json,batch6-flipped-analysis/}`。
