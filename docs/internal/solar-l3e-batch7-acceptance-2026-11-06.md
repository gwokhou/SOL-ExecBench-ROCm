# 线A 批次7（pool 窗口覆盖 + 融合/索引化 loss + torchview de-opaque）：预固定验收规格

制定日期：2026-11-06（实施前预固定）。前置：批次6（`3756a2e7`）。

## 0. 语义（预固定；勘察实证嵌入）

三个证明类，词汇全部冻结在封闭家族，不新增访问表示：

1. **融合 loss 全操作数读取**（torchview 原生 `cross_entropy`、`kl_div`）：
   `cross_entropy = log_softmax(全读 input) ⊕ 按目标 gather(子集，并集不增)
   ⊕ 全读 target`——两操作数外部读取均为全操作数，**精确**；`kl_div`
   逐元素作用后归约，两操作数全读，**精确**。出现第三个张量操作数
   （如 `weight`）时读取为按类 gather（数据依赖子集），全操作数声明会
   过计 → 维持拒绝。
2. **分解索引化 loss**（make_fx `nll_loss_forward`）：target/索引操作数
   全读（**精确**——每个索引值都被读取以决定 gather 位置）；data 操作数
   是按位置 K=1 gather（计数恰为 numel(target)，位置数据依赖）。冻结
   词汇：data 操作数为**内部**（corpus 实证：log_softmax 的输出）时该层
   只贡献 target 全读；data 操作数为**外部**时保持 `DYNAMIC_INDEX`
   fail-closed——计数化 gather 的访问表示不在本批建设。torchview 原生
   `nll_loss` 保持 opaque（同因，无 corpus 行依赖它）。
3. **pool 窗口覆盖**（`max_pool2d`/`max_pool2d_with_indices`/
   `avg_pool2d`，两路径）：外部读取是唯一元素**并集**，与读多重性无关。
   窗口覆盖即可判：对每个空间维（尺寸 D、kernel k、stride s、padding p、
   dilation d），感受野跨度 e = d·(k−1)+1，窗口数
   n = ⌊(D+2p−e)/s⌋+1；全覆盖 ⟺ n ≥ 1 ∧ s ≤ e ∧
   (n−1)·s − p + e ≥ D（首窗起点 −p ≤ 0 恒成立）。`ceil_mode=True`
   保持拒绝（torch 的尾窗丢弃规则不在本批建模，fail-closed）。属性从
   语义参数解码（make_fx 位置参数 [tensor, kernel, stride, …]，
   torchview operands[1] + attributes；缺省 stride=kernel、p=0、d=1、
   c=False；标量 kernel 广播到各空间维）。覆盖不成立 → 维持
   `read_semantics_unproved`。
4. **torchview registry de-opaque 复审**（批次3/4 gelu、layer_norm 同
   类）：`kl_div`、`cross_entropy`、`max_pool2d`（含
   `max_pool2d_with_indices` alias）从 `_LIBRARY` 移入已复审无副作用
   类；`nll_loss` 保持 opaque。

## 1. 分母与预期翻转（预固定）

分母 = 批次6 清单（`batch6-reinventory.json`，sha256
`5782de2eb88dc1c7eda08a53806680a8bc21a559262b1a6d384788ab55864dfa`）
中仍阻塞的 37 行。**S7 = 23 行**（blocker 集 ⊆ {`cross_entropy`,
`kl_div`, `max_pool2d`, `nll_loss_forward`, `max_pool2d_with_indices`,
`avg_pool2d`} 且覆盖判定成立，由该 sha 确定性可复算）：

- torchview 12：`l1n95_cross_entropy` w0–w4（cross_entropy）、
  `14007_kd_loss` w0–w3（kl_div）、`l1n42_maxpool2d` w0–w2（max_pool2d）；
- make_fx 11：`l1n42_maxpool2d` w0–w2（max_pool2d_with_indices，kernel
  2×2 stride 2×2 全覆盖）、`l1n95_cross_entropy` w0–w4
  （nll_loss_forward，data 内部）、`l2n98_matmul_avgpool_gelu_scale_max`
  w0–w2（avg_pool2d，kernel 1×16 stride 1×16 全覆盖）。

不受益 14 行：conv2d 家族（批次8 域）。上界 23。

## 2. Gate（预固定）

1. **G1 词汇单元**：§0.1 两 target、§0.2 内部-data nll、§0.3 覆盖成立
   的三 pool target 在代表图中零 blocker；覆盖不成立的 pool
   （stride>kernel、尾部不覆盖）仍 `read_semantics_unproved`；外部
   data 的 nll 仍 `dynamic_index`；带第三张量操作数的 cross_entropy
   仍拒绝。
2. **G2 负例不放松**：`nll_loss`（torchview）effect_boundary 零变化；
   conv2d 家族 reason 零变化；registry 其余 `_LIBRARY` 条目零变化；
   批次1–6 全部负例保持。
3. **G3 重清单精确翻转**：翻转集 == S7（23 行，逐行 id 相等）；14 个
   conv 行 blocker 逐字不变。
4. **G4 真实分析确认**：S7 全部 23 行 CPU 完整生产分析全部 analyzed
   且 totals 记录在案；不足者逐行给生产 reason code（容量/mapper 类
   失败如实记录，不计放松）。
5. **G5 套件不变量**：既有测试全绿；固定旧语义的测试（含批次1 的
   `test_indexed_loss_still_blocks`，其图恰为内部-data nll 形状）在同
   一提交内原子更新并列出清单。

## 3. 证据

`data/outputs/solar-l3e-external-reads-20261106/{batch7-reinventory.json,
batch7-flips.json,batch7-flipped-analysis/}`。
