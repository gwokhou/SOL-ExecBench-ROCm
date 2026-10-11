# 面向算子版本升级性能退化归因 Agent：GitHub 竞品调研

核查日期：2026-10-11。范围不限厂商、算子或硬件。核查公开仓库、关键源码、Agent/skill 定义、工作流、发布记录及部分真实 issue；没有安装或运行候选项目。源码存在不等于运行可靠，发布说明不等于相关模块全部开源，示例报告不等于真实成功案例。

## 结论

有人在做，而且已存在明显重叠的实现。不能再以“benchmark 报警后，让 Agent 二分并阅读 diff”作为主要创新点。最相关的技术组合是 TritonBench + TritonParse；最直接的 GPU 调查 Agent SOP 是 ROCm/FlyDSL 的回归二分 skill；通用受控实验流程则已有 agent-sh/perf。AMD TraceLens 和 NVIDIA TensorRT-LLM 也公开了性能分析 Agent。

本次尚未核验到一个公开项目，已证明稳定完成“上游版本触发 → 算子退化检测 → 多组件版本重建 → 交叉干预 → 提交及执行机制验证 → 可回放证据报告”的全链路。这是本次证据的边界，不能推导为不存在竞品或存在商业需求。

## 最相关项目

| 项目 | 已核验能力 | 还不能据此认定的能力 |
|---|---|---|
| [TritonParse](https://github.com/meta-pytorch/tritonparse) + [TritonBench](https://github.com/meta-pytorch/tritonbench) | 算子 benchmark 与性能二分连接；Triton 二分可继续追溯 LLVM 依赖更新，另有 PyTorch 二分；TritonBench 工作流支持 NVIDIA H100、AMD MI350 | 不能把 release 中所有 AI 功能视为开源可运行：部分 AI diff 分析依赖公开树中缺失的模块。也未证明任意算子库组合的多因子归因 |
| [ROCm/FlyDSL 回归二分 skill](https://github.com/ROCm/FlyDSL/blob/main/.claude/skills/bisect-perf-regression/SKILL.md) | 面向 GPU kernel，输入好坏提交和 benchmark，建立基线、重复测量、二分、读 diff 并提出原因 | 是 Agent 执行指南及脚本片段；不是已验证的 release 监听服务，也没有普遍的机制干预闭环证据 |
| [agent-sh/perf](https://github.com/agent-sh/perf) | 可执行阶段 runner，基线、假说、profiling、单变量实验、复测、恢复、证据日志 | 主要是编辑器内通用性能调查；其 breaking-point 二分是负载参数，不是提交；未核验跨组件版本构建矩阵 |
| [AMD-AGI/TraceLens Agent](https://github.com/AMD-AGI/TraceLens/blob/main/TraceLens/Agent/Analysis/README.md) | 比较前后 trace、语义对齐、按算子/系统瓶颈组织 Agent 分析和报告 | 输入主要是已采集 trace；未证明自主拉取历史依赖、逐版本重建或提交级因果验证 |
| [NVIDIA TensorRT-LLM 性能 Agent](https://github.com/NVIDIA/TensorRT-LLM/blob/main/.claude/agents/perf-profiling-specialist.md) | 官方 Agent 定义连接 workload profiling、nsys、ncu、host analysis 等技能；明确两版本 trace 对比和 CPU/GPU 瓶颈分类 | 本次只核验公开 Agent 定义；不能认定已自动监听升级、执行跨依赖二分或成功闭环归因 |
| [IsaacLab Bisection Agent](https://github.com/Neil4561/isaaclab-bisection-agent) | 逐提交固定栈重建、好坏端点验证、阈值/噪声处理、二分和测量归档 | Alpha；LLM 主要负责有界环境恢复，二分后的 trace 分析明确是外部集成点 |
| [GitHub Daily CLI Performance](https://github.com/github/gh-aw/blob/main/.github/workflows/daily-cli-performance.md) | 定时跑真实 benchmark，与基线比较并创建 issue；有真实公开产出 | 对象是 Go CLI；已核验案例只证明检测与分派，未证明完成根因归因 |
| [CausalPerf](https://github.com/android24/CausalPerf) | Android 场景的可证伪假说、受控干预、正确性/环境/统计门禁和实验账本设计 | README 明确尚无可运行优化 Agent、真实校准任务或生产 evaluator。属于方向重叠，不能称成熟替代 |

详细代码链接、固定 SHA 和证据边界见 [GPU 专项核查](gpu-regression-infrastructure-landscape.md) 与 [通用 Agent 核查](performance-agent-competitors-research.md)。

## 为什么不能只看名称和发布说明

- TritonParse 公开了真实二分实现，但公开 diff CLI 对 AI 模块的导入失败有显式兜底。需要逐项区分开源二分能力和未公开的 AI 实现。
- FlyDSL 的 example session 使用示例 SHA；它说明预期操作流程，不证明这些操作已在真实退化中成功完成。
- GitHub 的 [#38870](https://github.com/github/gh-aw/issues/38870) 是真实自动退化报告，但后续 Copilot 启动失败，最终按期限关闭。issue 已关闭不能作为归因或修复成功的证据。
- CausalPerf 的端到端示例属于设计展示，README Status 已明确真实 Agent 与校准尚未完成。

## 其他覆盖与排除

搜索覆盖了 PyTorch/Triton、ROCm/AMD-AGI、NVIDIA、JAX/XLA、IREE，以及 Intel/oneDNN、TVM 相关关键词和通用性能 Agent。检索组合包括 performance regression、benchmark、operator/kernel、bisect、root cause、agent，以及 repository description/README。没有搜索命中不能当成某生态不存在相关实现。

- [PyTorch operator benchmark](https://github.com/pytorch/pytorch/blob/main/benchmarks/operator_benchmark/README.md) 与 [TorchBench 二分](https://github.com/pytorch/benchmark/blob/main/bisection.py)：应复用的检测/版本定位基础设施，不因为没有 LLM 就忽略其替代价值。
- [IREE megabump](https://github.com/iree-org/megabump)：依赖更新和重建/二分操作的相关基础设施；不是已核验的自主性能归因 Agent。
- [bingxche/sglang-ci-bot](https://github.com/bingxche/sglang-ci-bot)：个人命名空间中的 Claude CI 调查工具，监测包括 ROCm、AITER 的 CI，并记录上下游 commit。已读工作流支持日志/代码调查和建议 bisect，未证明自动执行 GPU 重建和归因实验；不能标成 AMD 官方产品。
- [NVIDIA/LogSage](https://github.com/NVIDIA/LogSage)：GPU 集群训练日志故障分析相关，但不能直接视为算子版本升级性能归因产品。
- [FlagOS skills](https://github.com/flagos-ai/skills)：跨芯片验证、benchmark 和局部开关对照有复用价值；计划中的功能不计作已实现升级归因。
- [PerfBench](https://www.microsoft.com/en-us/research/publication/perfbench-can-agents-resolve-real-world-performance-bugs/)：性能缺陷修复 Agent 的直接相关研究。官方论文有真实缺陷评测，但本次关联 GitHub 仓库不可取得，不作为已核验开源实现。
- 模型能力 regression、纯遥测阈值检测、kernel 优化 Agent、微服务 AIOps 根因定位不自动纳入直接竞品；只有与本任务相关的已实现环节才计入。

## 对立项和实现边界的建议

以下是根据上述公开证据作出的产品判断，不是对整个市场的存在性证明。

已经拥挤的能力包括：定时 benchmark、回归阈值、git bisect、LLM 阅读 diff、trace 解释、单变量优化试验。不要从零重造一整套，也不宜用这些能力宣称首创。

值得验证的差异有四项：

1. **多个组件同时升级时自动拆因子。** 分别替换框架、算子库、编译器、运行时，并处理不兼容组合，而不是将环境失败判为性能退化。
2. **把责任提交与机制验证分开。** 找到提交后，用开关、回退、替换 kernel/算法或局部补丁验证预测；处理两个因素单独无影响、组合后才退化的情形。
3. **可靠且有预算的 GPU 实验。** 同一锁频策略、热身、缓存、编译状态和正确性检查；测量与 profiling 分离；记录机器与二进制身份；不稳定时允许不可判定。
4. **用真实历史退化评估 Agent。** 在隐藏最终答案的条件下比较归因正确率、错误归因率、GPU 时间、人工介入量和证据可回放性。不能让 Agent 直接读取含答案的 issue 再声称完成归因。

建议先以 TritonBench/TritonParse、FlyDSL skill、agent-sh/perf 加现成 profiling 工具构成基线，对同一组 AMD/NVIDIA 历史案例回测。若增量只是在这些工具间传参数，优先做集成或贡献上游；若能稳定解决基线无法处理的跨组件和交互问题，再独立产品化。此处只是调研建议，未安装工具、运行回测或修改生产代码。
