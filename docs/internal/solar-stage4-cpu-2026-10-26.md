# SOLAR S4：2026-10-26 CPU 逐槽位证据

阶段完成状态见 [HANDSOFF](../../HANDSOFF.md#solar-evolution)；能力、数值域和拒绝边界见 [S4 契约](solar-stage4-cpu-checkpoint.md)。本次采集使用与 BASE-01 相同的锁定 CPU 镜像，未运行 GPU、真实 mapper 或 R 工作负载。

## 来源和失败分母

源码内容摘要：`258616884e5e06de4dfb7c17a6126ca0074d575748a5cf7b48f9aae4df76d90d`。主采集和补采集各自的前后源码完全一致，两次也相同。镜像 `sha256:0ec2e6cba29e673ebe5ad83b7049e5044647fc340174d1c855379efaf36409dd`；Python 3.14.4、Torch 2.14.0+rocm10.1.0。完整环境、命令和逐点绑定见未提交目录 `data/outputs/solar-stages1-5-20261026/demand/`。

32 个 F 槽位中，24 个公共原图分析和受限计数完成；8 个 F07/F08 槽位因源读取或效果未证明而不进入公共复用。24 个 IEEE 默认请求均明确拒绝零消去，原图仍保留。受限计数结果中，20 个有完整 ideal-fused 估计；4 个 F04 保留 reduction 边界，完整受限时间不可用。

主采集有 4 个 F02 收集失败：脚本错误地直接索引 MAC oracle，但 F02 的独立标准在资源 oracle 中。失败时原图和受限工件已生成；原记录保留。修正收集脚本后只在新目录补跑这 4 个槽位，并核对 VALU/reduction oracle。共 36 次工作进程尝试，逻辑槽位仍为 32，不隐藏失败，也不把补采集写回原目录。生产源码没有因此修改。

F06 另完成 8 个带宽扫描点、8 个恢复点和 4 个必要资源查询；正向公式和逆向分组缩放一致，恢复引用完全一致。每个扫描点和受限结果的资格均为 false。所有公开复用的 dense 分析与本轮完整分析逐字段一致。F01–F05 的原图总量与 S3 原封存值一致。

## 逐槽位

T1 = make_fx → ATen；T2 = torchview → Extended。时间未列为设备测量；这里比较模型计数与证据状态。

| 样本 | 路径 | 公共原图 | dense MAC | 受限 MAC | 受限 ideal-fused | 采集说明 |
| --- | --- | --- | ---: | ---: | --- | --- |
| F01-D | T2 | 完成 | 60 | 60 | available | no_structural_selection |
| F01-D | T1 | 完成 | 60 | 60 | available | no_structural_selection |
| F01-Q | T2 | 完成 | 385 | 385 | available | no_structural_selection |
| F01-Q | T1 | 完成 | 385 | 385 | available | no_structural_selection |
| F02-D | T2 | 完成 | 0 | 0 | available | F02 收集补跑成功 |
| F02-D | T1 | 完成 | 0 | 0 | available | F02 收集补跑成功 |
| F02-Q | T2 | 完成 | 0 | 0 | available | F02 收集补跑成功 |
| F02-Q | T1 | 完成 | 0 | 0 | available | F02 收集补跑成功 |
| F03-D | T2 | 完成 | 84 | 84 | available | no_structural_selection |
| F03-D | T1 | 完成 | 84 | 84 | available | no_structural_selection |
| F03-Q | T2 | 完成 | 165 | 165 | available | no_structural_selection |
| F03-Q | T1 | 完成 | 165 | 165 | available | no_structural_selection |
| F04-D | T2 | 完成 | 0 | 0 | unavailable | restricted_model |
| F04-D | T1 | 完成 | 0 | 0 | unavailable | restricted_model |
| F04-Q | T2 | 完成 | 0 | 0 | unavailable | restricted_model |
| F04-Q | T1 | 完成 | 0 | 0 | unavailable | restricted_model |
| F05-D | T2 | 完成 | 0 | 0 | available | no_structural_selection |
| F05-D | T1 | 完成 | 0 | 0 | available | no_structural_selection |
| F05-Q | T2 | 完成 | 0 | 0 | available | no_structural_selection |
| F05-Q | T1 | 完成 | 0 | 0 | available | no_structural_selection |
| F06-D | T2 | 完成 | 48 | 30 | available | restricted_model |
| F06-D | T1 | 完成 | 48 | 30 | available | restricted_model |
| F06-Q | T2 | 完成 | 70 | 30 | available | restricted_model |
| F06-Q | T1 | 完成 | 70 | 30 | available | restricted_model |
| F07-D | T2 | formal_analysis_failed | — | — | not_run | source_analysis_not_eligible |
| F07-D | T1 | formal_analysis_failed | — | — | not_run | source_analysis_not_eligible |
| F07-Q | T2 | formal_analysis_failed | — | — | not_run | source_analysis_not_eligible |
| F07-Q | T1 | formal_analysis_failed | — | — | not_run | source_analysis_not_eligible |
| F08-D | T2 | formal_analysis_failed | — | — | not_run | source_analysis_not_eligible |
| F08-D | T1 | formal_analysis_failed | — | — | not_run | source_analysis_not_eligible |
| F08-Q | T2 | formal_analysis_failed | — | — | not_run | source_analysis_not_eligible |
| F08-Q | T1 | formal_analysis_failed | — | — | not_run | source_analysis_not_eligible |

## F06 的具体增量

| 样本（两路径相同） | dense MAC | 受限 MAC | dense 外部读 | 受限外部读 | 输出写入 | dense / 受限 compulsory I/O |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| F06-D | 48 | 30 | 88 B | 88 B | 48 B | 136 / 136 B |
| F06-Q | 70 | 30 | 116 B | 100 B | 40 B | 156 / 140 B |

F06-D/Q 的 MFMA 工作均由原图 96/140 降为受限 60 work units；mask 的整数选择工作 16/35 和完整写入 64/140 B 保留，未把初始化当成免费消失。矩形 Q 的后两个收缩行不被三角结构使用，权重读取从 14 减为 10 个 fp32 元素。以上均来自独立坐标枚举，未从数值样本估计密度。

相对原始 BASE-01，新增 torchview 三角能力使两个原先拒绝的 F06 槽位可以通过原生回放与验证；dense MAC 未变。三角外部读现在有结构证明，F06 的公共分析因此从不可发布的诊断阶段进入可用的原图模型分析，但没有取得可信容量、设备性能或正式发布资格。受限变换的 cache-aware 视图一律不可用，不能继承 dense 容量证据。

原始 BASE-01 的 627 个绑定文件已再次校验；417 个范围外既有源码文件与 S3 检查点相同。记录见 `preservation.json`。R 集合 16 槽位全部 `not_run`。阶段 5 搜索另行实现，不能用本阶段的计数对比声称已完成收缩顺序搜索。
