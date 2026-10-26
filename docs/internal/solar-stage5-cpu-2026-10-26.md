# SOLAR S5：2026-10-26 CPU 搜索证据

当前完成状态见 [HANDSOFF](../../HANDSOFF.md#solar-evolution)，支持域和接口见 [S5 契约](solar-stage5-cpu-checkpoint.md)。原始 BASE-01 保留不变，所有结果来自 CPU 验证和模型计算。

## 环境、来源和失败分母

源码摘要 `a225aaa60853091b985fdc5f4db8f7ca17de2b7d603cc483874ac7f963ffb0cc`。完整回归、主采集和补采使用相同最终源码；两次采集的前后清单均完全一致。锁定镜像 `sha256:0ec2e6cba29e673ebe5ad83b7049e5044647fc340174d1c855379efaf36409dd`，Python 3.14.4、Torch 2.14.0+rocm10.1.0；没有 GPU 节点或网络。日志、命令、工件和完整环境在未提交的 `data/outputs/solar-stages1-5-20261026/order/`。

最终 `check-06` 通过 1445 项相关 CPU 测试，以及 schema、Ruff、format、ty、coupling、readability 和文档检查。早期类型／格式／序列化问题和测试前提错误全部保留。支持域拒绝测试使用已通过语义验证的 IR；部分批量矩阵／重复输入在公共分析阶段已因读取或资源证明不足而拒绝，并没有为完成搜索测试扩展这些上游能力。

全部 32 个 F 槽位进入主采集：24 个公共源分析完成，原图总量与 S4 完全相同；8 个 F07/F08 因源读取／效果未证明而不进入搜索。24 个有效源中，F01/F03 的 8 个槽位属于搜索域，另外 16 个显式拒绝，不会从分母中消失。

4 个 F03 收集脚本曾错误地要求改价后的 total 对象与原 profile 完全相等。实际差异只有 resource_seconds、lower_bound_seconds 和 lower_bound_components；计数、流量和资源工作量没有变化。修正收集器后，在独立目录补采这 4 个槽位，并继续核对独立正向公式。生产代码未修改，原失败和已生成工件保留。F 槽位共 36 次工作进程尝试。

最终接受的 12 个 F 搜索情景加 2 个严格数值情景，共含 24 个候选记录和 24 个恢复候选，其中 2 个明确数值拒绝。若把失败收集器已经执行的候选也算入实际工作，共保留 40 个新候选工件；没有将失败尝试删掉或混入最终逻辑分母。

## F 槽位状态

T1 = make_fx → ATen；T2 = torchview → Extended。F01 只有原始二矩阵结合；F03 有原始左结合和右结合。

| 样本 | 路径 | 源／搜索状态 | 原图 MAC | 最终新候选数 | 说明 |
| --- | --- | --- | ---: | ---: | --- |
| F01-D | T2 | completed | 60 | 1 |  |
| F01-D | T1 | completed | 60 | 1 |  |
| F01-Q | T2 | completed | 385 | 1 |  |
| F01-Q | T1 | completed | 385 | 1 |  |
| F02-D | T2 | unsupported_domain | 0 | 0 | contraction_order_domain_unsupported |
| F02-D | T1 | unsupported_domain | 0 | 0 | contraction_order_domain_unsupported |
| F02-Q | T2 | unsupported_domain | 0 | 0 | contraction_order_domain_unsupported |
| F02-Q | T1 | unsupported_domain | 0 | 0 | contraction_order_domain_unsupported |
| F03-D | T2 | completed | 84 | 4 | 收集器补采成功 |
| F03-D | T1 | completed | 84 | 4 | 收集器补采成功 |
| F03-Q | T2 | completed | 165 | 4 | 收集器补采成功 |
| F03-Q | T1 | completed | 165 | 4 | 收集器补采成功 |
| F04-D | T2 | unsupported_domain | 0 | 0 | contraction_order_domain_unsupported |
| F04-D | T1 | unsupported_domain | 0 | 0 | contraction_order_domain_unsupported |
| F04-Q | T2 | unsupported_domain | 0 | 0 | contraction_order_domain_unsupported |
| F04-Q | T1 | unsupported_domain | 0 | 0 | contraction_order_domain_unsupported |
| F05-D | T2 | unsupported_domain | 0 | 0 | contraction_order_domain_unsupported |
| F05-D | T1 | unsupported_domain | 0 | 0 | contraction_order_domain_unsupported |
| F05-Q | T2 | unsupported_domain | 0 | 0 | contraction_order_domain_unsupported |
| F05-Q | T1 | unsupported_domain | 0 | 0 | contraction_order_domain_unsupported |
| F06-D | T2 | unsupported_domain | 48 | 0 | contraction_order_domain_unsupported |
| F06-D | T1 | unsupported_domain | 48 | 0 | contraction_order_domain_unsupported |
| F06-Q | T2 | unsupported_domain | 70 | 0 | contraction_order_domain_unsupported |
| F06-Q | T1 | unsupported_domain | 70 | 0 | contraction_order_domain_unsupported |
| F07-D | T2 | source_ineligible | — | 0 | formal_analysis_failed |
| F07-D | T1 | source_ineligible | — | 0 | formal_analysis_failed |
| F07-Q | T2 | source_ineligible | — | 0 | formal_analysis_failed |
| F07-Q | T1 | source_ineligible | — | 0 | formal_analysis_failed |
| F08-D | T2 | source_ineligible | — | 0 | formal_analysis_failed |
| F08-D | T1 | source_ineligible | — | 0 | formal_analysis_failed |
| F08-Q | T2 | source_ineligible | — | 0 | formal_analysis_failed |
| F08-Q | T1 | source_ineligible | — | 0 | formal_analysis_failed |

## F03 的选择结果

| 样本（两路径相同） | 原始／右结合 MAC | 未改 profile | 计算受限假想 profile |
| --- | --- | --- | --- |
| F03-D | 84 / 70 | ideal-fused 时间平价，保留原始结合 | 1.68 / 1.40 模型秒，选择右结合 |
| F03-Q | 165 / 224 | ideal-fused 时间平价，保留原始结合 | 3.30 / 4.48 模型秒，保留原始结合 |

假想情景把 fp32 MFMA 模式设为 100 work units/s，专门验证计算成本选择；它不是硬件吞吐率或设备时延。独立 oracle 同时核对每个候选的 ideal-fused 全局最大值和 unfused 逐层串行和。cache-aware 缺少适用容量证明，保持 unavailable。每个原始 baseline 单独保存；候选没有覆盖原始 F03 计量。

## 数值拒绝、恢复与边界

额外的两路径样本使用 1×1 fp32 矩阵 1e20、1e-20、1e-20，先按 atol=rtol=max_error_cap=gradient_atol=gradient_rtol=0 的策略验证原始左结合，再原封不动地验证候选。原始通过，右结合数值失败；失败没有视图或可比较时间。两条路径均恢复原来的成功与失败记录，没有放宽策略重试取得通过。

独立小链枚举和真实四矩阵五候选的测试证据保留在 check-06；五个 MAC 为 76、80、84、102、110。测试还覆盖受限 IEEE 情景不可用、工作预算在 input factory 前拒绝、廉价巨大空形状、总预算／单候选超时、规则漂移、损坏工件及非预期不变量上抛。

原始 BASE-01 的 627 项绑定及 417 个范围外既有源码文件再次复核未变，见 preservation.json。R 的 16 个槽位、真实 mapper 和 GPU 实验均为 not_run；搜索和假想投影没有发布或性能权威。
