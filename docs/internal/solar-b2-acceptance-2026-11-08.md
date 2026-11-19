# B2 验收：当前执行基线与 AKA 资格链刷新

日期：2026-11-08。**B2 通过，M1（E1/C1/B1/B2）完成。**
本次按用户批准的[独立代表槽勘误](solar-b2-representative-erratum-2026-11-08.md)
重新冻结并重跑全部 8 槽，随后完成原 AKA make_fx 资格链。
E2/C2/C3 尚未启动；没有留出实例、容量探针或真实 LLM_CORE target view。

## 执行身份

- 执行提交 `043a6820b263ae6b214ab3816f970dd7f64dee03`，独立干净 checkout：
  `data/outputs/solar-b2-20261108/execution-source/`。
- 当前冻结 `freeze-r6.json` SHA-256：
  `b43c845bf141da84a9384bcc7bef61a09166de345a64df633c149cc8460aa495`。
- 固定镜像 `sha256:1b92e7b1a59987aaef4449e7e6eec0fdb138de209b16e09e214f0dc5e17ddaaa`；
  Python 3.14.4、Torch 2.14.0+rocm10.1.0，语义实现摘要
  `d26c9feb84c803d0ea2d0a5ddf67a30cd8db064c4b83eed2df331af83ffe9da7`。
- 739 个生产文件摘要仍为
  `047b6fc082fd7450ca7d3a8f2555b0b90f63c2ccbd73b915401cc007afd7330d`。
  本次仅增加此前已提交的 CPU 审计精确分类修复，没有改变生产实现或正式评分契约。
- 目标 RX 9060 XT / gfx1200，UUID `<redacted-gpu-uuid>`，
  BDF `<redacted-pci-bdf>`；物理 32 CU、运行时 16 WGP，实际可见显存 17,095,983,104 bytes。
  这些是设备属性核验，不是容量分配探针结果。

原 B1 的 38 个绑定工件、M0 三份规格、已批准 E0 算术勘误与三个原有研究文档均保持原字节。
CPU 审计修复在宿主／固定容器各 **18 passed**；原各 288 项 CPU 基线按原身份保留，
没有把新增检查称作全套重跑。修复为本地 DCO 提交，未推送。

## 代表槽回归

| 已有 workload w0 | torchview | make_fx |
| --- | --- | --- |
| R04 dot_product_attention | ready | ready |
| R06 moe_topk_softmax | ready | ready |
| l1n63_conv2d 小卷积 | 精确预期拒绝：`ir_conversion` / `native_attribute_unsupported` | ready |
| R08 rmsnorm_bwd | 预期拒绝：`graph_extraction_failed` | ready |

**8/8 符合已批准规格：6 ready + 2 边界拒绝。**
小卷积拒绝消息还须精确匹配 `conv2d does not accept dilation, groups, padding, stride`。
ready 槽包括真实语义／梯度验证，不仅是 FakeTensor 取图；全部 stage 工件摘要另经收尾核对。
原失败回归与 r5 诊断保留在[历史检查点](solar-b2-checkpoint-2026-11-08.md)，不改判为通过。

## AKA 三级资格链

| Gate | 分母与结果 | gate.json SHA-256 |
| --- | --- | --- |
| static | 163 项输入检查通过 | `adc0c16911cab910ad87ae3e50ad1441c0c514fdcedcf7239158208f1524a725` |
| canary | 112/112 ready | `d205a35ed4cf5f65351eb00e42835593580c1efa853db601ba3591a4f27c6b93` |
| full | 163/163 ready，覆盖 43 个 problem | `329eac6a88553223e37e3d0c60feed5aa144509fb1259b9de58d173c3b201695` |

原生产 `run_solar_release_qualification` 和 `require_solar_release_qualification`
完成链式验收；三个 gate 均绑定上述执行提交，父 gate 摘要逐级相等，
`performance_authority=false`。没有跨 SHA 拼接旧资格。

GPU 活动结束后，在固定镜像的**无 GPU、只读容器**中再次调用原生产验证器：
static/canary/full、分母、全部 readiness 收据及源码／语义运行时绑定均通过，
`gpu_initialized=false`。full 的取图、转换、语义验证三个阶段均为 163，失败清单为空。
这证明当前集成基线的 readiness 资格；没有执行 formal-analysis 或发布官方分数。

## 资源预算与收尾

jobs=1、mapper 线程配置 32、CPU affinity 0–31 保持冻结。mapper 仅核验身份，没有执行。
内存上限 16,081,639,424 bytes；内核实际按页向下对齐为 16,081,637,376 bytes，禁用 swap。
MIOpen 数据库／内核缓存显式放入原 2 GiB tmpfs 的可写目录；只读根文件系统保持。

r6 控制器墙钟 **1452.7 秒**，原生主要 owner 从 08:40:37 到 09:04:55 UTC。
收尾完整性快照的累计活动墙钟为 3107.8 秒，包含此前诊断、等待和重新冻结，
低于原 21600 秒上限；没有重置预算。证据与执行 checkout 合计约 49.6 MiB，
低于 20 GiB 上限。这些时长仅为活动预算记录，不是性能结果。

本轮三个原生 owner receipts 全部 restored、无 release blocker。
独立最终检查确认 AUTO、GPU 无残留进程、无运行容器、共享准入 ready／owner=null，
原 L1/L2 inode 20452135／20514166 可用；GPU 已释放。

## 证据与后续边界

证据根 `data/outputs/solar-b2-20261108/`：

- `freeze-r6.json`、`approved-representative-erratum.md`：冻结与批准原文。
- `gpu-admission.json`、`representatives-verdict.json`：设备准入与 8 槽判定。
- `qualification/{static,canary,full}/gate.json`：三级 gate；canary/full 各自的
  `readiness/` 保存全矩阵、阶段工件和语义收据。
- `result.json`：r6 生产执行通过；`preflight/final-chain-cpu-replay.json`：独立 CPU 复核。
- `host-owner-closeout.json`、`final-closeout.json`、`final-integrity.json`：原生恢复、
  实际资源释放与完整性检查。
- `checkpoint-before-approved-erratum/`：原 116 个绑定文件及清单逐摘要保留。
- `final-manifest.json`：绑定本轮及保留历史的 1,717 个文件，SHA-256
  `0c670dcdc8f3891845704f398c7ad1bd093e13a4f107d785de831c9dbe071389`。

未证实范围继续保留：历史完整 formal 证据为 127 行 G4 完整、1 行大卷积 replay miss，
本次未重试；torchview 全语料资格仍开放；AKA ready 不外推为 LLM_CORE 覆盖。
E1 的 11 项 LayerNorm P3 reference/oracle 分歧也没有因 B2 通过而改判。

下一阶段按原依赖为 E2 与 C2→C3；本次授权止于 B2，未进入这些任务。
验收文档、HANDSOFF、迭代计划和路线图仅作状态更新，不修改 M0 冻结原文。
当前文档检查与 `git diff --check` 通过。文档更新仍在工作区，未另建提交或推送；
历史清理改动和其他研究文档保持原有工作区状态。
