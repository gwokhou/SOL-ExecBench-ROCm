# 线A 批次8（conv2d 窗口覆盖泛化）：预固定验收规格

制定日期：2026-11-06（实施前预固定）。前置：批次7（`d3731348`）。

## 0. 语义（预固定；勘察实证嵌入）

conv2d 的外部读取同样是唯一元素**并集**（权重/偏置在窗口覆盖下每个
元素都被使用；输入被感受野并集覆盖）。批次7 的逐维覆盖判定泛化到
conv：对每个空间维（D、kernel k、stride s、padding p、dilation d），
e = d·(k−1)+1，n = ⌊(D+2p−e)/s⌋+1，全覆盖 ⟺ n ≥ 1 ∧ s ≤ e ∧
(n−1)·s − p + e ≥ D。与 pool 的差异只在参数来源（冻结）：

- kernel 取自 weight 形状的空间维（weight 必须为 4 维、空间维 ≥1）；
- stride/padding/dilation 从 conv 位置参数解码（stride=3、padding=4、
  dilation=5），标量广播；
- `transposed=True` 或非零 `output_padding`（转置卷积是 scatter，读
  模式不同）→ 维持拒绝；
- `groups` 任意（分组只划分输入通道，不改变覆盖并集；weight 仍全读）；
- 输入必须 4 维 NCHW、空间维为正整数；不满足 → 维持
  `read_semantics_unproved`。

既有 `_full_contraction_reads`（1×1 stride1 pad0 满收缩特判）保留不动；
新增覆盖判定与其 OR。覆盖数学提取为 pool/conv 共享谓词（单一来源）。

## 1. 分母与预期翻转（预固定）

分母 = 批次7 清单（`batch7-reinventory.json`，sha256
`0a6641b1c035195781bc39b525f8b4faeb6978d25abee50971d484028e2f9c90`）
中仍阻塞的 14 行，全部 make_fx、blocker 恰为 {conv2d}。

**S8 = 14 行**（勘察实证四族代表行 stride 1×1、padding 0、dilation 1、
kernel 3×3，覆盖成立；由该 sha 确定性可复算）：
`l1n63_conv2d` w0–w2、`l1n82_conv_depthwise` w0–w2（groups=8）、
`l2n17_instancenorm_fused` w0–w2、`l2n52_conv_activation_batchnorm`
w0–w4。上界 14。

G4 预声明诚实口径：L3-c 曾实测 torchview 侧 l2n52 全 5 槽撞 pinned
mapper 7200s 生产预算（~43 CPU-hours/槽）；本批 14 行全为 make_fx 路径
（L3-c 中 make_fx conv 行仅败于 external-reads），预期在逐槽子进程
预算内完成，但若仍有容量/mapper 类失败，逐行如实记录生产 reason code
不计放松（沿用批次1 的 G4 诚实记录条款）。

## 2. Gate（预固定）

1. **G1 词汇单元**：kernel 3×3 stride 1 覆盖成立 → 零 blocker；
   stride>extent、尾部不覆盖、transposed、非 4 维输入 → 保持
   `read_semantics_unproved`；1×1 特判行为不变。
2. **G2 负例不放松**：批次1–7 全部负例 reason 零变化；pool 覆盖负例
   不受共享谓词提取影响。
3. **G3 重清单精确翻转**：翻转集 == S8（14 行，逐行 id 相等）；分母
   清空（128 行 inventory 全零阻塞）。
4. **G4 真实分析确认**：S8 全部 14 行 CPU 完整生产分析全部 analyzed
   且 totals 记录在案；容量/mapper 类失败逐行如实记录（见 §1 诚实
   口径）。
5. **G5 套件不变量**：既有测试全绿；固定旧语义（把 strided conv 断言
   为 `read_semantics_unproved` 的批次1 测试保持——stride 2 仍不覆盖）
   无需更新的，列空清单。

## 3. 证据

`data/outputs/solar-l3e-external-reads-20261106/{batch8-reinventory.json,
batch8-flips.json,batch8-flipped-analysis/}`。
