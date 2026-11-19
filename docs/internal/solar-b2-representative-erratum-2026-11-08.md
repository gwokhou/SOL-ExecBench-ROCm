# B2 代表槽预期独立勘误

日期：2026-11-08。**用户已批准，生效**。原 r2/r3/r4 冻结及失败记录不覆盖。
本勘误修正本轮选择代表槽时对历史证据的错误推广，不修改 M0、生产实现或正式评分契约。

## 问题与证据

[B1 修订冻结](solar-b1-refreeze-2026-11-08.md) 将小卷积
`torch2hip/l1n63_conv2d/aka-l1n63_conv2d-w0` 的两条 IR 路径均要求 ready。
该预期错误地把批次 8 的 **make_fx** 完整分析证据推广到了 torchview。
[批次 8 原规格](solar-l3e-batch8-acceptance-2026-11-06.md) §1 明确 14 行全部为 make_fx。

早于本次活动的 `data/outputs/solar-l3c-20261106/audit/` 两条矩阵对同一
definition SHA `5872de3bb47153c4895497ee7040fc99fe4530a8ce1da0802df59b4ddf5eb1c2`
分别记录：make_fx ready；torchview 在 IR conversion 以
`native_attribute_unsupported` 拒绝，消息为
`conv2d does not accept dilation, groups, padding, stride`。
两份矩阵摘要为：

- torchview：`208730d8094b839b17212643048e7258b361a2075d06c6b8207675f6a5b33ea4`。
- make_fx：`6cf1b34f9eb98a0baa0d6c039d674c1fe0ac393f237374604f1ec5359614f172`。

本轮先遇到 MIOpen 默认用户数据库目录位于只读 HOME 的环境错误；
独立 r5 诊断仅改变 MIOpen 用户数据库/内核缓存目录后，取图通过，随后复现上述同一拒绝。
诊断不授予 B2 或资格链通过。原生 owner 均已恢复 AUTO；GPU 子进程已排水。
随后无 GPU 容器在旧 `160b9988` 和当前 `043a6820` 的精确 checkout 上重放保存图，
均复现该拒绝且 GPU 未初始化；结果与源码身份分别留在 B2 的
`preflight/cpu-conv-old-b1-git-mounted.log` 和 `preflight/cpu-conv-current-b2.log`。

## 已批准的独立修订

1. 保留 4 个 workload × 两条 IR 的 **8 槽分母**、全部输入/种子/检查与预算。
2. 仅将上述 **torchview 小卷积一槽**改为既存边界的预期拒绝：
   必须在 `ir_conversion` 返回 `native_attribute_unsupported`，并与上述消息精确一致。
   make_fx 小卷积仍须 ready。其余 7 槽预期不变。
3. 新预期合计为 **6 ready + 2 精确边界拒绝**。原 r4 不改判通过；
   记录为环境失败并暴露代表预期错误，r5 仅为归因记录。
4. 新建执行冻结后从全部 8 槽重新执行；完成后才允许启动
   static 163 → canary 112 → full 163 的原 make_fx 资格链。
5. 冻结 MIOpen 可写目录 `/tmp/miopen-db` 和 `/tmp/miopen-cache`；
   保持固定镜像/源码、只读根文件系统、原 2 GiB tmpfs 与累计 6 小时截止时刻。
   不以新修订重置预算，不跳过资格项，不授予性能或 formal-analysis 权威。

用户明确批准“批准独立勘误并继续 B2”。原 ready 预期不在旧记录中改判；
后续验收按新冻结独立登记。扩展 torchview 卷积属性支持仍须另立前端任务；E2/C2/C3 继续不启动。
