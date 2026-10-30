# SOLAR 真实 mapper 小样本验证与 GPU 待批准方案

2026-10-26，本机 `guohao-B850-AI-TOP`，隔离分支
`codex/solar-stages1-5-resume`。这是 [阶段 1–5 CPU 交付](solar-resume-2026-10-26.md)
之后的独立增量验证；旧收据、F32 的原分母与原始日志保持不变。

## 实际执行的真实 mapper 验证

本轮源摘要仍为
`ec176be36d93db331f71a18a27426a08c82691f28f39ad56a8f2114603263277`，
运行前后相同。没有安装、构建新工具或改变全局配置。

可读的既有 mapper 位于：
`/home/guohao/PyCharmMiscProject/CrossSOL-Eval/out/paper-a-local-20261005-v3/infrastructure-amd-v3/prerequisites/external/amd-orojenesis`。
它是 62767848 字节的静态链接 ELF；二进制 SHA-256
`18591892b1ecec3264ec729b0e457ec9f22422993f656ece40dba809c032d77a`
与本仓库受信列表一致。当前 `OrojenesisRunner` 完整校验了来源、构建和 artifact
清单，包括源提交 `97d52178bf9a9c209bf79be96b87c164bcd35625`。
provenance SHA-256：
`99efa80834e837cfe14eef9d039812397ac1a0145f93134d49ae8017fe7ff460`。

使用原固定镜像，源码和 mapper 只读挂载，无网络、无 GPU；2 CPU quota，
affinity 为 30、31，4 GiB 内存，单作业。内部总预算 260 秒、每个 mapper 进程
30 秒，容器外层 290 秒加 10 秒终止宽限。验证主体实测 2.86 秒（不含镜像启动
和 Python 导入）。没有替换或 monkeypatch mapper runner。

| 路径 | 缓存容量 | dense MAC | 真实 cache-aware 流量 | 实际 mapper CSV |
| --- | ---: | ---: | ---: | ---: |
| torchview_extended_einsum | 64 B | 84 | 420 B | 4 |
| torchview_extended_einsum | 4096 B | 84 | 196 B | 4 |
| make_fx_aten | 64 B | 84 | 420 B | 4 |
| make_fx_aten | 4096 B | 84 | 196 B | 4 |

四个 F03-D 投影点均可用；同一路径两个容量点复用相同固有计数，容量单调性
成立。每个输出经过完整工件加载/摘要复验；共保留 16 个真实 mapper 原始 CSV
及其配置、日志、曲线和来源绑定。此前单元测试中的 392 B 是有意构造的合成
曲线，不能代替本次 420 B 的实际工具结果。

[真实执行结果](../../data/outputs/solar-resume-20261026/mapper-followup/real-mapper-01/result.json)
与 [工具链校验记录](../../data/outputs/solar-resume-20261026/mapper-followup/real-mapper-01/toolchain.json)
在独立证据目录。此结果关闭 F03-D 两路径、两个容量点的最小真实工具验证；
不代表全部图/容量域均通过，也没有 GPU 性能或正式发布资格。该情景为
hypothetical，`publication_eligible=false`。

## GPU canary 与 R16：仅准备，未执行

当前只读 sysfs 身份：AMD 目标 PCI `0000:05:00.0`、`card1`、`renderD128`，
设备 ID `1002:7590`，unique ID `a30fa29c1cca1511`，显存 17095983104 字节。
目标卡 `gpu_busy_percent` 返回 `EBUSY (16)`，不能把它当作空闲证据。没有
运行 GPU 探测 kernel、canary、R 输入或性能测试。

[完整计划](../../data/outputs/solar-resume-20261026/mapper-followup/gpu-proposal/plan.json)
已经固定 8 个原 UUID × 两路径、输入文件摘要、最终源码摘要和 worker IPC v2。
[控制器](../../data/outputs/solar-resume-20261026/mapper-followup/gpu-proposal/controller.py)
及 [启动脚本](../../data/outputs/solar-resume-20261026/mapper-followup/gpu-proposal/proposed_gpu.sh)
已生成；无 GPU 容器内验证了 16 个请求 schema、输入/源码摘要及 Python 语法。
未验证 GPU 运行，不能称它们已完成设备验收。

拟议最小目标是现有 `run_solar_stage_worker` 的提取、严格转换和语义/梯度回放，
沿用 workload 的容差、固定 3 seeds × 3 patterns，不做候选计时。它不是正式
生产资格 gate，不替代仓库大批次的 `qualify-static → qualify-canary →
qualify-full` 链，也不执行 82 槽位旧 cross-path focus。R 的完整计量/容量及
正式发布仍需后续单列，不能把 readiness 通过写成全部 R 验收完成。

| 步骤 | 固定规模 | 预算与停止条件 |
| --- | --- | --- |
| canary | R01 `aka-l1n1_square_matmul-w0`，M=K=N=64、fp32，两路径共 2 槽位 | 单槽 120 秒；主体总预算 540 秒、容器硬上限 600 秒；任何失败停止扩大 |
| R16 readiness | 原冻结 R01–R08，每个两路径，共 16 槽位；R08 保留历史边界预期 | canary 两路径通过且源码一致后才可启动；单槽 120 秒、主体总预算 1740 秒、容器硬上限 1800 秒；保留失败和未运行点 |

两步独立运行，R16 会重新执行 R01，因此若全部执行，共 18 次 readiness 尝试。
R07 最大矩阵为 1823×781×511、fp16；其余形状与 UUID 均见冻结验收清单。
每次仅一个 GPU worker、2 CPU affinity、8 GiB 主存、2 GiB共享内存。仅映射
`/dev/kfd` 与 `/dev/dri/renderD128`，`ROCR_VISIBLE_DEVICES=0`，控制器必须看到
恰好一个 `gfx1200`，否则拒绝。固定已验证镜像、源码只读、无网络，沿用
`expandable_segments:False`，不改时钟、不跑 profiler、不安装依赖。

需要先确认 GPU 执行授权和独占窗口，并核对设备节点、资源占用、固定栈及现有
audit 前置条件。Kernel-Cause 当前零 GPU 不等于目标卡已空闲；EBUSY 未解决前
不能自行认定可用。批准后实际命令如下，当前没有执行：

```bash
SOLAR_GPU_APPROVED=yes bash data/outputs/solar-resume-20261026/mapper-followup/gpu-proposal/proposed_gpu.sh canary
# 仅在 canary 两路径通过、资源窗口与 R16 范围获确认后：
SOLAR_GPU_APPROVED=yes bash data/outputs/solar-resume-20261026/mapper-followup/gpu-proposal/proposed_gpu.sh r16
```

所有新结果只写隔离树的 `gpu-proposal/outputs/`；输出目录必须不存在。旧
start700 保持暂停，没有合并 main 或 push。

## 读取限制与结果确定性

mapper 文件名枚举在宿主部分系统/容器目录及固定镜像的两个 `problems/local`
目录遇到 OS `Permission denied`。这是读取拒绝，不是安装或执行审批拒绝。
动作、已观察到的确切目标和原因列于
[限制记录](../../data/outputs/solar-resume-20261026/mapper-followup/access-limits.json)。
未对这些受拒目录提权重读或改道访问。宽范围枚举输出曾截断，故不能声称完成
全机工具盘点；这不影响另一个可读、独立工件已通过正式校验并实际完成四点
验证的确定结果。GPU 是否可独占使用仍不能确认。
