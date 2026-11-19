# B2 历史检查点：准入通过，代表预期勘误待确认

日期：2026-11-08。**B2 为 INCOMPLETE；AKA 资格链尚未启动。**
本节状态为用户批准前的历史检查点。用户随后已批准独立勘误；r6 的新执行结果另行登记，
不在本记录中改写原失败或把诊断计为通过。
后续 B2 已通过，见[独立验收记录](solar-b2-acceptance-2026-11-08.md)。
CPU 审计分类修复完成。当前停在代表回归，等待
[代表槽独立勘误](solar-b2-representative-erratum-2026-11-08.md)的用户决定。
E2/C2/C3 未启动，没有生成留出实例或 LLM_CORE target view。

## 已完成

- 审计分类只接受 `core/bench/judge/executor.py` 中精确空值
  `CUDA_VISIBLE_DEVICES`；非空值、其他路径和设备工具调用仍拒绝。
  宿主和固定容器分别 **18 passed**。Ruff、format、ty、coupling、readability、
  schema、reachability 和空白检查通过。修复为本地 DCO 提交
  `043a6820b263ae6b214ab3816f970dd7f64dee03`，未推送。
- [B1 修订冻结](solar-b1-refreeze-2026-11-08.md)保留原 B1、M0 与 E0 勘误，
  使用新的独立干净 checkout。739 个生产文件、镜像、语义实现、mapper 和输入未变；
  原宿主/容器各 288 项基线按原身份保留，本轮未冒称重跑。
- 原生 `run_docker.sh` 容器验证、UUID/BDF、gfx1200、32 CU／16 WGP、
  可见显存与运行时 smoke 通过；使用既有共享 L1/L2 和 host clock owner。
  主准入证据在 `attempt-03-miopen/gpu-admission.json`，`performance_authority=false`。

## 代表回归与归因

固定分母仍为 4 workload × 2 IR 路径。r4 尝试了 torchview 的 4 槽：

| 槽 | r4 原预期 | 实际观察 |
| --- | --- | --- |
| R06 routing w0 | ready | ready，真实转换/语义回放完成 |
| R08 rmsnorm_bwd w0 | `graph_extraction_failed` | 精确理由匹配，预期边界拒绝 |
| R04 attention w0 | ready | 取图阶段 `miopenStatusUnknownError`，环境失败 |
| l1n63_conv2d w0 | ready | 取图阶段同一 MIOpen 环境失败 |

2/4 已执行槽符合原预期；make_fx 的 4 槽未启动，不能计作已通过。
失败分母与原记录保存在 `attempt-03-miopen/`，未删槽或改判。

r5 是单独冻结的诊断活动，只重放同一个小卷积 torchview 槽：

1. 原缓存环境重现错误，MIOpen 日志明确为只读文件系统，路径
   `/home/sol-execbench/.config/miopen`。
2. 仅将 MIOpen 用户数据库和内核缓存改到 `/tmp/miopen-db`、`/tmp/miopen-cache`，
   取图通过，IR conversion 返回 `native_attribute_unsupported`，消息为
   `conv2d does not accept dilation, groups, padding, stride`。
3. 2026-11-06 的原 L3-c 矩阵对相同 definition 摘要已记录上述拒绝，make_fx 才是 ready。
   批次 8 规格也明确其 14 行均为 make_fx。r2 把该证据推广到两路径，是本轮冻结错误。
4. 将该已保存图在**无 GPU 设备**的固定容器中用旧 `160b9988` 和当前 `043a6820`
   分别转换，两者均复现同一拒绝，GPU 未初始化。没有改前端或正式评分规则。

r5 诊断不算代表回归通过，也不授予资格。提议的勘误保持 8 槽，改为
6 ready + 2 精确边界拒绝，尚未批准或执行。批准后必须重新冻结并重跑全部代表槽，
再执行原 static 163 → canary 112 → full 163；不复用旧 SHA 的门禁。

## 运行控制与收尾

原 6 小时活动窗口从首次原生 B2 启动起累计，重试没有重置；证据上限 20 GiB。
内存上限 16,081,639,424 bytes 在内核中按页向下对齐为 16,081,637,376 bytes，
没有提高额度。启动器的 `target_not_tested` 处理、内存对齐和 CU/WGP 口径修订
分别留有冻结及原失败日志。只读根文件系统与 2 GiB tmpfs 保持不变。

所有已结束原生 owner receipts 均为 restored、无 release blocker。
`pending-erratum-closeout.json` 另核对了实际 AUTO、空 GPU 进程、无运行容器、
准入 ready／owner=null，以及原 L1/L2 inode 20452135／20514166 可用。

本次没有运行 mapper、formal-analysis、容量探针、E2/C2/C3 或官方评分。
历史完整 formal 覆盖仍单列：127 行 G4 完整，1 行大卷积 replay miss；本轮没有重试或补齐。
torchview 全语料资格也没有因此闭合。

## 证据索引

原根目录：`data/outputs/solar-b2-20261108/`（不提交原始产物）。
用户批准后的续跑前，以下 116 个绑定文件及清单已逐摘要核对并完整保存在
`data/outputs/solar-b2-20261108/checkpoint-before-approved-erratum/`；
复核本历史检查点时以该不可覆盖副本为根，避免与续跑的顶层日志混用。

- `freeze-r2.json`、`launch-preflight-correction.json`、`freeze-r3.json`、
  `freeze-r4.json`：执行前冻结及顺序修订，旧件保留。
- `freeze-r5.json`：MIOpen 单槽诊断冻结，SHA-256
  `2d23d941259b1d268aded7ed8ff2cab05e3f1e99ea7891388d72bac85d55200b`。
- `attempt-00-preflight/`、`attempt-01-memory-alignment/`、`attempt-02-cu-wgp/`、
  `attempt-03-miopen/`：原控制文件、失败输出与 owner receipts。
- `miopen-diagnosis/`：两种缓存配置的原生 worker 请求、响应、日志和保存图。
- `preflight/cpu-audit.xml`、`preflight/container-cpu-audit.xml`：两组各 18 项检查。
- `preflight/cpu-conv-current-b2.log`、`preflight/cpu-conv-old-b1-git-mounted.log`：
  两个精确 checkout 的 CPU 重放，均返回 reproduced=true。
- `preflight/conv-historical-evidence.json`：两份旧矩阵及批次 8 规格的摘要。
- `pending-erratum-closeout.json`：独立收尾观察，不作为性能或发布收据。
- `checkpoint-manifest.json`：116 个本轮文件摘要；复核原 B1 的 38 个绑定工件、
  M0 三份规格、E0 勘误、三个原有研究文档及 739 个生产文件均保持原字节。
  检查点清单 SHA-256 为
  `c0ac8c616cc354f6e188769ed921d750b2c2282c36678c5252ad450a5bc9b503`。

首次 CPU 重放脚本遗漏源码验证必需参数，旧 worktree 容器随后还需只读挂载 Git 元数据；
这两次诊断工具错误及日志均保留，最终成功重放的日志明确如上。
更新后的台账、路线图和文档索引通过当前文档检查及 `git diff --check`；
文档仍在工作区，未另建提交或推送。
