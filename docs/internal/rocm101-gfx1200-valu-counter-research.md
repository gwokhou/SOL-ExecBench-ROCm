# ROCm 10.1 / gfx1200 VALU 零计数调查

研究日期：2026-10-11。范围：官方固定版本文档、官方仓库 issue/PR 的第一手反馈，以及本仓库已有采集产物。本次后台研究没有运行 GPU、修改电源策略或重新采集计数器。

**结论**

首要假设是 PMC 的电源状态前置条件未满足。ROCm **10.1.0** 文档明确说明，RDNA3/RDNA4 在 `AUTO` 下会关闭部分硬件块的 perfmon 时钟；`STABLE_STD` / `profile_standard` 才启用这些块的 PMC。文档描述的是计数功能前置条件，不能仅理解为降低测量方差的建议。[固定版本文档](https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/docs-10.1.0/how-to/using-rocprofv3.html#setting-gpu-performance-level-for-pmc-profiling)

本地已有记录符合该症状，但还没有当前 ROCm 10.1 / gfx1200 的电源状态干预对照，因此**不能宣布 VALU 根因已证实或已修复**。也没有证据支持把它定性为 kernel replay 独有缺陷、硬件不支持 VALU、或者 GEMM 性能退化的原因。

**本地证据与边界**

| 观察 | 证据 | 可以说明什么 |
| --- | --- | --- |
| application 与 kernel replay 都有六个 dispatch；`SQ_WAVES` 均为 32，`SQ_INSTS_VALU` 全为 0 | [既有回执](../../data/outputs/stack-upgrade-followup-20261011/counter-replay-final/report.json) | 两种采集模式共同出现异常；replay 模式不是目前最有区分力的变量 |
| 向量加法及原地更新结果正确 | 同一回执、[探针](../../scripts/internal/upgrade/counter_probe.py) | 输出正确不能证明计数器有效；六个 dispatch 还包含 fill 和 copy，不能全部等同于算术核心 |
| 使用 gfx1200 镜像、`ROCR_VISIBLE_DEVICES=GPU-a30fa29c1cca1511`、每个 PMC 带 `:device=0` | [实际命令](../../data/outputs/stack-upgrade-followup-20261011/counter-replay-final/pilot.json) | 已有明确设备限定，不应先删掉限定重试所有 GPU |
| counter CSV 的 `Agent 1` 对应 agent 表中的 gfx1200；agent 表也枚举 gfx1036 | [agent 表](../../data/outputs/stack-upgrade-followup-20261011/counter-replay-final/kernel/probe_agent_info.csv)、[CSV](../../data/outputs/stack-upgrade-followup-20261011/counter-replay-final/kernel/probe_counter_collection.csv) | 枚举到 iGPU 不等于选中它采集；不能把 warning 单独当成错卡证据 |
| 升级性能采集前后均记录 `AMDSMI_DEV_PERF_LEVEL_AUTO` | [前置快照](../../data/outputs/stack-upgrade-followup-20261011/timing-clocks-before.json)、[后置快照](../../data/outputs/stack-upgrade-followup-20261011/timing-clocks-after.json) | 支持 AUTO 假设，但这不是每次 replay dispatch 同时取得的状态证明；下次 PMC 需独立前后核验 |

**官方说明与社区反馈**

1. **已合并的官方文档变更。** PR #7455 于 2026-06-23 合并，说明 Navi3x/Navi4x 的 AUTO 与 perfmon clock gating 关系。应以最终 10.1 文档的“部分硬件块”为准，不能沿用 PR 初始描述中“全部 profiling 被禁用”的泛化。[官方 PR](https://github.com/ROCm/rocm-systems/pull/7455)

2. **相同计数器症状，设备不同。** #5953 报告 RX 9070 XT / gfx1201 / TheRock 7.13 nightly 上 `SQ_WAVES`、`SQ_BUSY_CYCLES` 正常，而 `SQ_INSTS_VALU`、GL2C、TA 为零，且可用列表仍列出这些计数器。它与本地症状接近，但不是当前 gfx1200 / 10.1 的复现。[原始报告](https://github.com/ROCm/rocm-systems/issues/5953)

3. **重要的撤回归因。** #156 作者最初怀疑底层 firmware/kernel，后来关闭问题并说明此前只试过 `high`，没有试 stable；在 gfx1201 / Radeon AI PRO R9700 上用 `STABLE_STD` 后，读请求、写请求、GRBM 计数恢复。该回复把 VALU 也列为受影响项，但提供的恢复数值针对 GL2C/GRBM；不能冒充它给出了 VALU 的逐项复测值。[作者更正](https://github.com/ROCm/rocprofiler-sdk/issues/156#issuecomment-5066100645)

4. **同型号 GPU 的相邻计数器验证。** #8523 是 RX 9060 XT / gfx1200 / ROCm 7.2.0 的 `SQ_WAVE_CYCLES` 零值报告。AMD 维护者建议 stable；报告者随后给出 AUTO 全零、`STABLE_STD`、`STABLE_PEAK`、`STABLE_MIN_SCLK` 均为正的对照，并由 AMD 确认。报告者与该项目相关，不能把它包装为独立第三方复现；更不能把 wave-cycle 的证据直接当成 10.1 VALU 修复。报告者把现象解释为 dVFS transitions，这个解释应与官方的 perfmon gating 说明区分。[AMD 建议](https://github.com/ROCm/rocm-systems/issues/8523#issuecomment-5060191074)、[复测表](https://github.com/ROCm/rocm-systems/issues/8523#issuecomment-5093611718)、[AMD 确认](https://github.com/ROCm/rocm-systems/issues/8523#issuecomment-5106056576)

5. **尚未发布的提醒改进。** 查询当日 PR #9605 为 open、未合并，拟对选中 gfx11/gfx12 agent 的 AUTO 状态增加 warning，不自动改 GPU 状态。它是可观测性提案，不是本机已安装的修复。作者还报告 gfx1100 切回 AUTO 后可能暂时保留启用的 perfmon 状态；这只是该主机观察，提醒我们不要要求 AUTO 的反向复测一定再次归零。[PR 状态与实测说明](https://github.com/ROCm/rocm-systems/pull/9605)

**计数器命名、设备限定和 replay**

`rocprofv3-avail info --pmc -d 0` 可输出目标的计数器名称、description、dimensions、block/expression；`pmc-check` 用于判断同组可采集性。可用列表和组检查都不能替代对非零、缩放和语义的实测验收。继续保留 UUID/BDF 核验，并区分 AMD-SMI 索引、HIP 可见索引和输出 agent ID。[10.1 可用性工具](https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/docs-10.1.0/how-to/using-rocprofv3-avail.html)

本次没有取得可靠的 **10.1 安装版 gfx1200** 证据，证明 `SQ_INSTS_VALU_VOPD` 是可直接替换 `SQ_INSTS_VALU` 的别名。不要猜名字、套 CDNA 公式或修改 raw event selector。若本机确实列出 VOPD/VALU 子类，先保存定义，再用反汇编确认会触发该类指令；子类计数不能直接充当总 VALU。`SQ_WAVES` 是活跃性对照，不是 VALU 替代指标。上述是下一步方法建议，不是新的硬件事实。

10.1 的 kernel replay 仍为 beta，通过显式选项启用；application replay 则按 PMC 组重跑程序。因此应先用普通单组 application 路径排查功耗状态，再验证 replay，而不是用两个相同的零判定 replay 一致性。[10.1 replay 说明](https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/docs-10.1.0/how-to/using-rocprofv3.html#kernel-replay-beta)

**最小后续验证建议（未执行）**

1. 在独立 PMC 窗口中锁定同一 gfx1200，保存 GPU UUID/BDF、镜像 ID、驱动与 profiler 版本、当前性能级别和计数器定义。保留原来的 AUTO 性能实验，避免改变它的解释范围。
2. 先使用原有探针与单组 `SQ_INSTS_VALU:device=0` 的 application 采集，保存 AUTO 现状。另采 `SQ_WAVES` 对照；若想同组，先跑该版本 `pmc-check`。
3. 在获得该主机电源策略修改授权的独立窗口内，仅把核验后的目标 GPU 暂时设为 `STABLE_STD`，立即读回，再跑相同镜像、输入和 kernel；退出时恢复并核验原状态。不要使用不带 GPU 选择的设置命令，不要放宽 sysfs 权限，不要与主线程 GEMM 计时并行。
4. 如果 VALU 恢复，进一步使用固定 ISA 的纯 VALU 循环，检查操作数、循环次数或 wave 数变化下的计数缩放，再比较 application/kernel replay。即使只做到“恢复非零”，也只能确认前置条件相关，不能宣布定量计数已校准。
5. 如果 stable 读回成功但 VALU 仍零，保留未过滤原始数据，核验实际 kernel ISA、gfx1200 event 定义、计数器分组及版本，再组织最小上游复现。此时才提高 selector/SDK/AQLprofile/driver 缺陷的优先级。暂不替换计数器、更改驱动、patch profiler 或接入生产默认路径。

**当前处置**：将既有 VALU 结果继续标记为不可用于诊断。GEMM 性能原因先依靠不带 PMC 的计时、生成代码和实际库调度证据；PMC 电源状态实验取得的数据应单独归档，不能与 AUTO 性能结果直接拼接成因果结论。
