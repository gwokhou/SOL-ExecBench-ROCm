# B1 修订冻结：CPU 审计修复与 B2 执行身份

日期：2026-11-08。用户授权先修复 CPU 审计分类，再执行 B2；E2/C2/C3 不在本次范围。
原 [B1](solar-b1-execution-freeze-2026-11-08.md) 与其独立 checkout、收据保持不变。

## 修复与版本

`test_rocm_migration_residue_audit.py` 只将裁判 CPU worker 的精确空值
`CUDA_VISIBLE_DEVICES` 设置归类为设备隔离。新增四个场景：精确空值通过；
非空设备选择、其他源码路径、设备工具调用仍拒绝。没有修改裁判、正式 checks 或评分合同。

执行提交为 `043a6820b263ae6b214ab3816f970dd7f64dee03`，本地 DCO 提交，未推送。
相对原 `160b9988`，增加原 B1 台账提交及上述单文件测试修复；739 个生产源文件摘要仍为
`047b6fc082fd7450ca7d3a8f2555b0b90f63c2ccbd73b915401cc007afd7330d`。
独立、干净的执行副本位于 `data/outputs/solar-b2-20261108/execution-source/`。
当前工作区的历史文件清理和三个原有未提交研究文档不混入执行副本。

宿主及原冻结容器分别通过 18 项审计检查；Ruff、format、ty、coupling、readability、schema、
production reachability 和空白检查通过。原 288 项 CPU 基线按未变的生产源、锁文件和镜像绑定保留，
本次不将新增 18 项称为重新执行了全套基线。最终无设备容器核对了执行源码、语义 runtime、
mapper provenance、线程与输入摘要，GPU 未初始化。

## 保持的环境与预算

- 镜像仍为 `sha256:1b92e7b1a59987aaef4449e7e6eec0fdb138de209b16e09e214f0dc5e17ddaaa`；
  容器 Python 3.14.4、Torch 2.14.0+rocm10.1.0、语义实现 `d26c9feb…` 不变。
- mapper 二进制/来源与 B1 相同，仅校验未执行；32 线程、CPU affinity 0–31、jobs=1。
- readiness 单 worker 3600 秒；B2 总窗口 21600 秒、证据 20 GiB；不提高原上限。
  串行容器整体限 16,081,639,424 bytes 且禁用 swap，以保证每个 worker 不超过原内存上限。
  总墙钟由宿主进程组控制，证据预算由活动监视器执行；原生 owner 负责容器排水和时钟恢复。
- 目标仍为 UUID `<redacted-gpu-uuid>`、BDF `<redacted-pci-bdf>`、
  RX 9060 XT / gfx1200 / 32 CU。名义 SKU 为 16 GiB；宿主只读查询的实际可见显存为
  16304 MiB（17,095,983,104 bytes），与历史观察一致。此值不是容量分配探针结果。
- 使用文档约定的共享 L1/L2 inode、`gpu-admission/rx9060xt.json` 和 SOL 私有 owner 目录；
  `run_docker.sh` 执行容器准入与 host guard，不借用独立的新锁绕过其他项目。

## B2 的固定选择

代表回归为 4 个已有 workload × 两条固定 IR，共 8 槽：R04 attention w0、R06 routing w0、
`l1n63_conv2d-w0`（B=4,C=8,H=W=32,O=16）与 R08 rmsnorm_bwd w0。
小卷积已有 batch8 的完整分析记录；本次不重试已知大卷积 formal replay miss。
前 3 个 workload 两路径均须 ready；R08 make_fx 须 ready，torchview 须按历史预登记的
`graph_extraction_failed` 拒绝。不得在见到结果后更改预期、删槽或切换路径。

代表回归通过后使用原生产资格接口，刷新 make_fx 的 static 163 → canary 112 → full 163，
并再次验证父 gate 摘要与全部 readiness receipts。这条链不执行全语料 formal analysis，
也不产生官方分数；ready 与完整容量分析继续分开登记。

冻结清单 `data/outputs/solar-b2-20261108/freeze-r2.json` SHA-256：
`78feba45ad029a713c26fdce342f126046de08ab165afbd323191bd25fb0434d`。
它绑定两个活动控制文件、输入摘要、执行提交、镜像、mapper、代表槽和预算；
运行日志、CPU JUnit 与身份核对在同一证据根。M0 原文与 E0 独立算术勘误保持原字节。

首轮在只读 `--preflight-only` 返回 `target_not_tested` 时停止，未启动 GPU。
原生 wrapper 的 `--record-container-validation` 明确支持在 `probes_allowed=true`
时进行验证。启动器仅对该精确状态继续进入原生验证流程，其他拒绝仍终止；
未修改生产门禁、资格规则、源码/镜像或预算。旧启动器和失败日志保存在
`attempt-00-preflight/`。执行前补充冻结为 `launch-preflight-correction.json`，
SHA-256 `72248bb890c637095f693b2b62c9e2472be86ff13bf0014b61f5ac82ce8f6688`，
绑定原冻结清单与修正后的启动器摘要，不覆盖原记录。

## 运行控制修订：r3

随后原生容器准入完成，但控制器在任何代表槽前拒绝了 cgroup 的内存读回。
无设备容器核对确认：内核按 4096-byte 页将 16,081,639,424 向下对齐为
**16,081,637,376 bytes**，差值 2048 bytes。该次原生 owner 已排水、恢复 AUTO 并释放准入；
日志、控制器和 owner receipts 留在 `attempt-01-memory-alignment/`。

`freeze-r3.json` 绑定上述精确生效值，原内存上限不提高；
同时固定从首次原生 B2 启动起的累计截止时刻，后续重试不重置 21600 秒总预算。
控制器对预算观测失败同样终止。执行提交、生产源码、镜像、mapper、输入、代表槽预期及资格链不变。
r3 SHA-256 为 `613b3e8b5cc88d432247d4a3cd4fd398f2558ca8cbf59a40f4cb86db6227482c`；
它保留 r2 及预检补充记录的摘要链接。

## 当前生效的运行控制：r4

r3 在 GPU 属性核对时停止，尚未运行代表槽；UUID、BDF、gfx1200 已通过，
控制器错误地要求运行时 `multi_processor_count` 等于物理 CU 数 32。
冻结源码的 `matrix_bf16_bf16_wmma.hip:40` 已说明 gfx12 在该字段报告 WGP，
每个 WGP 含两个 CU。r4 分别固定宿主物理 **32 CU** 和运行时 **16 WGP**，
保留原始运行时属性观察；不放宽设备身份、代表槽或资格预期。r3 失败日志及已恢复
AUTO 的 owner receipts 留在 `attempt-02-cu-wgp/`。

`freeze-r4.json` SHA-256 为
`ee62abbee4982fb1651ad2cf2dde9bbfb5762a32cda4532852af50b9f4fab61b`，
链接 r3 摘要并绑定修正后的控制文件及上述源码依据；执行源码和镜像不变。
本轮以 r4 为准，沿用 r3 的同一累计截止时刻与页对齐内存值。

## r5 诊断与当前停止点

r4 准入通过后，torchview 代表回归在 MIOpen 只读缓存错误处停止，资格链未启动。
r5 仅对同一小卷积进行原缓存／可写缓存两进程对照，冻结摘要
`2d23d941259b1d268aded7ed8ff2cab05e3f1e99ea7891388d72bac85d55200b`。
可写缓存解决取图错误后暴露既存的 torchview 卷积属性拒绝；原 L3-c 矩阵与旧／新源码
CPU 重放均确认，这不是本轮新增回归，而是上述“两路径小卷积均 ready”预期的错误推广。

当前 B2 为 **INCOMPLETE**。完整状态见[B2 检查点](solar-b2-checkpoint-2026-11-08.md)。
[代表槽勘误](solar-b2-representative-erratum-2026-11-08.md)待用户确认；
原 r4 不改判，不继续 AKA 链，E2/C2/C3 不启动。r5 是诊断记录，不能替代 B2 回归验收。

## 用户批准后的 r6 重新冻结

用户已明确批准独立勘误并继续 B2。以上 r4/r5 停止点是批准前历史记录，原结果不改写。
批准前检查点的 116 个绑定文件和清单完整复制到
`data/outputs/solar-b2-20261108/checkpoint-before-approved-erratum/`，逐一核对原摘要。

当前执行冻结为 `freeze-r6.json`，SHA-256
`b43c845bf141da84a9384bcc7bef61a09166de345a64df633c149cc8460aa495`。
它链接 r5 诊断和 r4 执行冻结，并绑定已批准勘误原文及控制文件摘要。
同一 8 槽独立重跑，预期为 6 ready + 2 边界拒绝；小卷积 torchview 必须精确匹配
`ir_conversion`、`native_attribute_unsupported` 和原历史消息，其余预期不变。
MIOpen 用户数据库和内核缓存显式置于原 2 GiB tmpfs 内的可写目录。

执行提交仍为 `043a6820`；739 个生产源文件、镜像、输入和全部正式检查未变。
沿用原累计截止时刻，不重置 6 小时窗口。代表槽全部匹配后才能开始原 AKA 三级资格链；
E2/C2/C3 仍不在本次执行范围。

r6 后续已完成 8/8 代表槽及 static 163 → canary 112 → full 163，
独立 CPU 收据重放和原生 owner／资源收尾通过。完整结果见
[B2 验收](solar-b2-acceptance-2026-11-08.md)；上述 r4/r5 历史停止点保留，不改判。
