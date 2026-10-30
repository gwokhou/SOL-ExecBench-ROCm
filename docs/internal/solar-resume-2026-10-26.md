# SOLAR 阶段 1–5 本轮实施与 CPU 验收

日期：2026-10-26。机器：`guohao-B850-AI-TOP`。
分支：`codex/solar-stages1-5-resume`；基线：
`edc9ad211857f208e73efdd631ec2e472c11a581`。
工作树：`/home/guohao/Documents/Codex/2026-10-26/task/solar-stages1-5`。

用户在 BASE-01 安全拆分后明确授权继续阶段 1–5。先前封存是范围决策；本轮按
依赖检查和适配 `c9c0e5b6` 的候选模块，不合并其 foundation 中的 86 项无关改动。
主工作区保护清单保存各文件内容摘要、权限及 git 状态。没有合入 main、推送、
安装软件或恢复 `p1-successor-start700`。

## 阶段结果

| 阶段 | 已实现及实际覆盖 | 本轮通过数 | 证据尝试 |
| --- | --- | ---: | --- |
| 1 | base/storage/access/effects 合同，预算化区域并集，双路径读取计量，G01–G10；F04/F08 修正 | 919 | `stage1-full-02` |
| 2 | 完整验证包复用、失效身份、三种视图及 CLI/外层桥；损坏、资格传播和进程清理 | 1140 | `stage2-full-02` |
| 3 | 固定模型必要条件反推、有限串行扫描、失败/未运行点保留、恢复身份和预算 | 1233 | `stage3-full-02` |
| 4 | 结构零和反向需求；dense/受限情景分开，F06 双路径和数值/别名/索引边界 | 1364 | `stage4-full-01` |
| 5 | 固定形状纯矩阵链搜索，逐候选原始策略数值及梯度验证，稳定选择、预算及恢复 | 1445 | `stage5-full-01` |

各行是当时阶段源码的累积测试数量，不能相加为独立用例总数。最终阶段测试耗时
97.42 秒；Ruff、format、ty、coupling、readability、schema 检查全部通过。
最终测试绑定源码摘要：
`ec176be36d93db331f71a18a27426a08c82691f28f39ad56a8f2114603263277`。
前后摘要相同。测试范围为 `tests/solar`、SOLAR 外层桥及受影响的 CLI、性能模型
桥和 release scoring 读者；未声称整个仓库所有测试通过。

最终 `final-f32-01` 采集为 **32/32 槽位语义验证和计量完成、数学差异零**，
没有失败槽位。完整 cache-aware 为 0，正式资格为 0，GPU 实验为 0；R 组
16 槽位均未运行。前后源码摘要与最终阶段测试相同。
阶段 1 曾保留两个 torchview 三角算子拒绝；阶段 4 接入其语义后，本次最终
采集已确认两路径全部完成。F/G/M 数学规格自检与实现回归分别留证；仅规格
自检不算未来功能验收。

## 对外合同与边界

详细公共合同见 [SOLAR boundary](../SOLAR-BOUNDARY.md)。当前版本为 access v1、
analysis v6、request manifest v7、IR verification v8、path comparison v2；reuse、
exploration、restricted computation、order search 均为 v1。旧/未知工件版本在
外部读取入口先拒绝；没有旧版本兼容 reader。

- `solar reprice`：读取完整、摘要固定的 verified bundle，不执行 reference、重取图
  或反序列化 pickle。新速率/资源/容量情景重算相应部分，不继承正式资格。
- `solar requirements`、`solar scan`：仅给固定模型的必要约束，不保证硬件达标。
  默认扫描总预算 300 秒、单点 30 秒，最多 64 点、8 轴，串行运行。
- 受限计算默认 IEEE 域拒绝消零。调用方须明确断言有限操作数与中间值、无溢出、
  忽略有符号零；保留初始化和输出存储，未知边界不凭样本稀疏率推断。
  默认上限为 200000 坐标工作、1000000 收缩项、5 秒；dense 工件保持不变。
- `solar search-orders`：最多 12 个有序二维矩阵、64 候选；默认总计 300 秒、
  单候选 30 秒、1000000 元素和 10000000 MAC。候选沿用原始 reference、
  seeds/patterns、容差及梯度策略。失败不删除，不改容差；原图基线始终保留。

三种视图不是正式 SOL 字段的替代。缺完整容量证据时 cache-aware 不可用，不能
用 ideal fused 冒充。受限计算和搜索结果都没有正式发布或实测性能权威性。

## 环境、失败及可复核证据

证据根：`data/outputs/solar-resume-20261026/`。每次尝试独立保存 environment、
source-before/source-after、checks、stdout/stderr 和 JUnit。各阶段另有
`stageN-source-snapshot.tar.gz` 和 `stageN-checkpoint.json`，包含该阶段源码与
测试；摘要清单与固定基线用于复核。生成结果不提交到 Git。

固定的已安装镜像：
`sha256:0ec2e6cba29e673ebe5ad83b7049e5044647fc340174d1c855379efaf36409dd`。
Python 3.14.4、PyTorch 2.14.0+rocm10.1.0；运行时 HIP 标识 7.16.26385。
源码只读挂载、无网络、无 GPU 节点、4 CPU quota、8 GiB、`pytest -n 0`，
OMP/MKL 各 1 线程。未修改宿主或 Kernel-Cause 环境。

早期失败保留在各阶段 `*-01` 等目录：阶段 1 拆开未来视图引用、更新版本和
外部读取证明 fixture；阶段 2 拆开阶段 3 测试，修复外层超时后独立 mapper
进程组可能残留的问题，验证正常结束、超时及中断的清理；阶段 3 修正格式。
最终通过不删除此前失败记录。清理测试使用 CPU 工具替身，不能充当真实 mapper
数学/工具链验收。

首次全仓文档检查仅因隔离树缺少历史图片/报告引用而失败。已从主工作区
复制 30 个被引用文件（约 6.8 MB）到隔离树的忽略目录，未执行历史脚本，
并用 `doc-dependency-receipt.json` 记录来源与摘要；它们只满足历史文档引用，
不参与本轮实现/测试资格。

## 尚未验收及最小下一步

以下时间是建议上限/预算，不是已测运行时间，也不表示已有执行许可。

| 门槛 | 最小步骤与资源预算 | 前置条件及授权 |
| --- | --- | --- |
| 真实 mapper | 先只读定位并核验已有受信任二进制/构建来源，预算 5 分钟；再取 F03-D、两条 IR、64/4096 B 容量点，1–2 CPU affinity、单作业，整体上限 5 分钟 | 当前固定镜像 `/opt` 仅有 ROCm，宿主项目及相关缓存未定位到正式安装；不能据此断言全机没有。若需安装/构建新工具，需另行授权。真实 mapper 验收安排单独确认 |
| R 组 canary | R01 的两个路径先行，单 GPU、串行，2–4 CPU；整体上限 10 分钟，失败立即记录并停止扩大 | 核对 gfx1200/软件身份、现有资格门槛与设备空闲；与 Kernel-Cause 协调独占窗口，确认本机 GPU 执行授权 |
| R 组其余槽位 | 保持冻结的 8 个 UUID、16 个路径槽位；每槽位上限 2 分钟、总预算 30 分钟，失败保留 | canary 通过后再确认资源窗口；R08 可按明确原因拒绝，不把 16/16 全通过作为假设 |
| GPU 性能/正式资格 | 先制定单独配对测量与资格方案，再固定重复数、容差和预算 | 本轮零 GPU、零正式资格；不自动恢复旧 start700，不从模型 MAC/时间变化推断实测加速 |

阶段 1–5 的 CPU 实现完成不关闭上述门槛。R 组输入身份仍按冻结清单核验，
16 槽位全部为未运行；真实 mapper 与 GPU 实验均为零。

后续增量：同一源码已完成 F03-D 双路径、64/4096 B 的四点真实 mapper 验证，
结果与尚待批准的 GPU 方案见[后续记录](solar-mapper-followup-2026-10-26.md)。
上文零 mapper 是该轮 CPU/F32 收据的历史边界，不能用来覆盖后续独立工具证据。
