# 通用性能退化调查 Agent：GitHub 补充调研

核查日期：2026-10-11。只读核查 README、关键代码和公开产出；未安装或执行候选项目。这里比较的是版本升级后通过 benchmark 发现退化、定位责任变化和验证机制的 Agent，不把模型质量评测中的 regression 混入。

## 结论

通用的“Agent + 性能测量 + 假说实验”已经有人实现；自动性能巡检也已有真实运行产出。不能把这些能力当作空白。以下项目尚不足以单独证明存在一个已成熟覆盖跨厂商算子、跨依赖栈升级、多因子干预和提交级机制验证的完整替代品。GPU 专项项目应与本笔记合并比较，不能据此断言全市场没有直接竞品。

| 项目 | 已核验的能力 | 与目标的差距、证据等级 |
|---|---|---|
| [agent-sh/perf](https://github.com/agent-sh/perf) | 基线、负载边界、资源约束、假说、代码路径、profiling、单变量实验、反复测量、恢复和证据记录；包含真正的 JavaScript 阶段执行器 | 是可执行插件，不只是愿景。面向编辑器内调查；未看到上游 release 触发、依赖栈重建或组件版本交叉矩阵。README 的二分是负载参数边界，不是提交二分。未在本次核查中找到可审计的已完成真实调查数据集 |
| [github/gh-aw：Daily CLI Performance](https://github.com/github/gh-aw/blob/e19bf2b96f816952561795eb81ecd9f5df227fd0/.github/workflows/daily-cli-performance.md) | 定时检查近期 Go/依赖变化、运行 benchmark、历史基线比较、保存结果和创建退化 issue | 有真实自动产出；核查案例证明检测与分派，不能证明闭环因果归因。主对象是 Go CLI 编译性能 |
| [Neil4561/isaaclab-bisection-agent](https://github.com/Neil4561/isaaclab-bisection-agent) | 按候选 commit 重建固定运行栈；先验证好坏端点、噪声和阈值，再二分；记录测量、栈变化和不可判定原因 | Alpha，有实现；LLM 只帮助有界环境探测/恢复，不能改 GOOD/BAD 判定。二分后的 trace 分析明确留作外部集成；不是完整自主实验归因系统 |
| [PerfBench / OpenHands-Perf-Agent](https://www.microsoft.com/en-us/research/publication/perfbench-can-agents-resolve-real-world-performance-bugs/) | 官方论文报告 81 个 .NET 真实性能缺陷；Agent 生成 benchmark、修改代码并测量验证；性能专用 Agent 约 20% 成功率 | 是直接相关研究，但主要是给定性能问题后的修复评测。论文关联 glGarg/PerfBench 路径在本次已获网络访问的 GitHub API 查询中返回 404，不能当作已经取得的可复用开源实现 |
| [android24/CausalPerf](https://github.com/android24/CausalPerf) | Android 场景的可证伪假说、受控干预、环境/统计门禁、因果证据与实验账本设计；已有契约和确定性参考代码 | 概念非常接近，但 README Status 明确没有可运行优化 Agent、真实 trace corpus、统计校准任务或生产 evaluator；不能把端到端示例数字当成已完成实验 |

## 最强的可复用调查流程：agent-sh/perf

核查版本 `7d1ffea9ec9c7ef5f0ed578b5daf777c022dcd0e`。

[README](https://github.com/agent-sh/perf/blob/7d1ffea9ec9c7ef5f0ed578b5daf777c022dcd0e/README.md) 的十阶段流程由 [scripts/perf-phase.js](https://github.com/agent-sh/perf/blob/7d1ffea9ec9c7ef5f0ed578b5daf777c022dcd0e/scripts/perf-phase.js) 落实：调用 baseline、constraint、profiling、optimization runner，并把结果写入 investigation state。假说仍由 Agent 提供 JSON，而非执行器自行生成；decision 接受传入的 verdict 和 rationale。

[perf-theory-tester](https://github.com/agent-sh/perf/blob/7d1ffea9ec9c7ef5f0ed578b5daf777c022dcd0e/skills/perf-theory-tester/SKILL.md) 明确要求一个变化、每臂至少两次、顺序运行、波动内判 inconclusive、实验后恢复；支持同一代码树用 `PERF_EXPERIMENT=0/1` 切换。它已经覆盖“带证据的实验调查”，不能简单归类为 profiler 包装。

仍需注意它的默认阶段包括负载二分和 CPU/内存约束，不一定适合算子升级排查；“single change”流程也不等于已经实现多因子交互实验。未执行，故不评价当前代码的运行可靠性。

## 已运行的巡检，不等于已归因：github/gh-aw

当前 workflow 版本 `e19bf2b96f816952561795eb81ecd9f5df227fd0`。文件包含 daily schedule、近期编译相关变化检查、真实 `make bench-performance`、历史中位数比较和 issue 输出。

[Issue #38870](https://github.com/github/gh-aw/issues/38870) 是 2026-06-12 自动创建的公开产出：报告 CompileSimpleWorkflow 从约 3.33 ms 增至 12.31 ms，并关联运行记录。报告的下一步仍是审查近期修改、pprof 和 benchstat。后续自动分派 Copilot，但评论报告启动失败，最终 issue 因两日期限自动关闭。**关闭状态不是修复证明。** 这个案例说明公开 Agent 运维流水线是真实的，也说明不能把“检测→开 issue→交给 coding agent”直接当成因果闭环成功。

## 带版本栈重建的二分：IsaacLab Bisection Agent

核查版本 `3fce0a3c496368a6019f9c97aafefe4c28662053`。

[engine.py](https://github.com/Neil4561/isaaclab-bisection-agent/blob/3fce0a3c496368a6019f9c97aafefe4c28662053/src/isaaclab_bisection/bisection/engine.py) 确有端点测量、信号验证、不可重现时 inconclusive、候选测试和二分逻辑；会输出 narrowed interval、skips、stack diff。环境不兼容与性能判定分开。

[LLM policies](https://github.com/Neil4561/isaaclab-bisection-agent/blob/3fce0a3c496368a6019f9c97aafefe4c28662053/docs/llm-policies.md) 明确 LLM 默认关闭，仅用于受限恢复和 container probe，数值判定及二分决策由确定性代码控制。[README](https://github.com/Neil4561/isaaclab-bisection-agent/blob/3fce0a3c496368a6019f9c97aafefe4c28662053/README.md) 说明真实 GPU sweep 是单独的硬件验证门禁，post-bisection trace analysis 是外部集成点。因此应称“Agent 可调用的版本重建与性能二分基础设施”，不应称已具备跨层根因证明。

## 设计重叠与证据成熟度：CausalPerf

核查版本 `090133a54badbaff0f72e3623a96eede8081128e`。[README Status](https://github.com/android24/CausalPerf/blob/090133a54badbaff0f72e3623a96eede8081128e/README.md#status) 明确区分模拟/参考实现与真实 Android 校准。其预注册预测、A1/B/A2、正确性和环境门禁、不可判定结果、账本可回放等设计，与“实验归因”产品有高度重叠。当前不存在可运行 Agent 的声明也同样明确；成熟度不能由篇幅或示例界面推断。

## 更广搜索中的排除项

- [ROCm/FlyDSL bisect-perf-regression](https://github.com/ROCm/FlyDSL/blob/main/.claude/skills/bisect-perf-regression/SKILL.md)：官方 GPU 回归二分 Agent skill，已转交 GPU 专项调研重点核查；不能遗漏。文中假 SHA 的 example session 不能算实际历史结果。
- [Ali-Sayed-Salehi/jit-dp-llm](https://github.com/Ali-Sayed-Salehi/jit-dp-llm)：LLM 缺陷预测和 Mozilla 性能回归数据上的离线 batch/bisect 模拟，与在线真实实验归因不同。
- [jalpatel11/Self-Healing-SRE-Agent](https://github.com/jalpatel11/Self-Healing-SRE-Agent)：README 将 performance regression analysis 列在 Future Extensions，不能以仓库描述中的 production-grade 认定已实现该能力。
- [uditgoenka/autoresearch](https://github.com/uditgoenka/autoresearch)：搜索命中含性能统计门禁和 regression skill；本次未深入核心实现，不计入已核验直接竞品，作为补充候选保留。

可守住的差异不是“Agent 会跑 benchmark / 会 bisect / 会解释 profiling”，而是把多仓库软件栈升级转成可重建、可干预、可复核的实验，明确区分退化检测、责任提交、执行机制和交互因果。是否已有项目覆盖这些，必须与 GPU 专项调研合并判断。
