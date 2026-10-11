# GPU 算子版本回归基础设施与 Agent 近竞品

核验日期：2026-10-11。范围：公开 GitHub 仓库、官方文档和关键实现；没有运行 GPU 实验，也没有验证这些工具的实际成功率。下文“未见”仅限所查入口，不是证明不存在。

## 结论

已经有人实现“算子 benchmark → 版本二分 → 责任提交”，也有人把这一流程写成 Agent skill。不能再把“Agent 运行 benchmark 并分析版本 diff”作为独特性。

最接近的公开组合是 **TritonBench + TritonParse**；最接近直接 Agent 工作流的是 **ROCm/FlyDSL 的 bisect-perf-regression skill**；最接近跨版本性能解释的是 **TraceLens comparative agent**。这些工具尚不能据本次证据等同于通用的“自动监测新版本 → 多组件替换 → 竞争假设 → 干预确认机制 → 因果报告”产品。

## 能力核验

| 项目 | 已核验能力 | 关键边界 |
| --- | --- | --- |
| TritonBench + TritonParse | 指定 good/bad 和复现命令后自动构建、测量、判别、二分；H100 和 MI350 runner；TritonParse 可沿 LLVM bump 进入 Triton/LLVM commit-pair 测试和 LLVM 二分；另支持 PyTorch 二分 | 所查 bisect workflow 是手动 dispatch；性能判别仍由脚本给出。没有核验到自主选择通用软件栈因子的闭环 |
| ROCm/FlyDSL skill | Agent 接收 good/bad 与 benchmark，先验证端点、每点取三次中位数、二分、读取 diff、报告 root-cause hypothesis | 是 skill 和伪代码；假设解释后没有要求干预／revert 重测闭环；示例 SHA 和邮箱为示意，不能当真实成功案例 |
| AMD-AGI/TraceLens | 两个 trace 的性能差异；结构匹配以及 LLM 语义 kernel 名称统一；比较分析 Agent 和报告 | 输入是已有 trace；所查工作流没有负责新版本触发、安装版本、bisect 或组件交叉替换 |
| pytorch/benchmark | 回归检测、自动构建和 benchmark 二分、输出提交及 GitHub issue 信息；可选择 pytorch/vision/audio/benchmark 为目标 | 目标仓库预先选择；不是自主假设实验 Agent |
| AMD-AGI/Magpie | AMD/HIP 与 NVIDIA/CUDA，kernel 执行/比较/profiling、框架 benchmark、MCP、跨运行报告比较 | 是执行和观察层；所查 README 未提供升级因果调查闭环 |
| AMD-AGI/Hyperloom | Agent 根据 profiling 搜索框架及 kernel 优化，重新测量和保留/回滚候选；源码还收录 FlyDSL bisect skill | 主目标是优化现有 workload。不能因为有实验反馈和退化保护，就称为版本升级归因产品 |
| IREE package bisect | 下载每个提交的 compiler/runtime 包，独立 venv，执行用户 test script，自动定位 first-bad | 文档实例是编译正确性故障；可承载性能判别脚本，但未见其自身实现性能统计和 Agent 归因 |

### 1. TritonBench + TritonParse：已有跨厂商算子回归二分

TritonBench 固定快照 `dabe5714d2cfa76ad1ee8a3237ee730b4a5a8627`：

- [workflow](https://github.com/meta-pytorch/tritonbench/blob/dabe5714d2cfa76ad1ee8a3237ee730b4a5a8627/.github/workflows/bisect.yaml) 输入包括 good/bad commit、repro command、性能/功能模式、百分比阈值，以及 `h100` / `mi350`。
- [执行脚本](https://github.com/meta-pytorch/tritonbench/blob/dabe5714d2cfa76ad1ee8a3237ee730b4a5a8627/.ci/bisect/run.sh) 先验证 bad 相对 good 的退化，再调用 `tritonparseoss bisect`，不是另一个孤立的二分工具。
- [回归判别器](https://github.com/meta-pytorch/tritonbench/blob/dabe5714d2cfa76ad1ee8a3237ee730b4a5a8627/.ci/bisect/regression_detector.py) 是确定性脚本，不是 LLM 主导实验设计。
- [bisection 文档](https://github.com/meta-pytorch/tritonbench/blob/dabe5714d2cfa76ad1ee8a3237ee730b4a5a8627/docs/bisection.md) 有部分旧脚本名；使用时应以 workflow 与实际脚本为准。

TritonParse 固定快照 `219a2e124f8b80e942f20123f7d7103e765dea39`：

- [CLI](https://github.com/meta-pytorch/tritonparse/blob/219a2e124f8b80e942f20123f7d7103e765dea39/tritonparse/bisect/cli.py) 与 [README](https://github.com/meta-pytorch/tritonparse/blob/219a2e124f8b80e942f20123f7d7103e765dea39/tritonparse/bisect/README.md) 实现 Triton 二分、LLVM 描述符判断、commit-pair 测试和 LLVM 二分，支持恢复以及 found/ambiguous/aborted/error 结果，保留 candidate set、构建测试日志和 git bisect log。
- 这已经是在追踪上游依赖升级，不能把“跨仓库定位 LLVM bump”视为我们的未被满足需求。
- [公开兼容性修复 Agent](https://github.com/meta-pytorch/tritonparse/blob/219a2e124f8b80e942f20123f7d7103e765dea39/tritonparse/compat_builder/ai_fixer.py) 把构建错误与 LLVM API 变更交给 Claude/Codex 等客户端，允许读写与 Bash 工具，核验是否产生新提交。它处理构建兼容性，不等同于性能机制归因。
- **必须保留公开/内部边界**：[releases](https://github.com/meta-pytorch/tritonparse/releases) 宣布 AI diff root-cause，但当前 [diff CLI](https://github.com/meta-pytorch/tritonparse/blob/219a2e124f8b80e942f20123f7d7103e765dea39/tritonparse/diff/cli.py) 在两处导入 `tritonparse.diff.fb.ai.diff_analyzer`，该目录不在此 SHA 的公开递归文件树中；代码捕获 ImportError 并显示 AI analysis unavailable。不能仅按 release notes 把这个功能判为公开可运行。
- [README](https://github.com/meta-pytorch/tritonparse/blob/219a2e124f8b80e942f20123f7d7103e765dea39/README.md) 明确支持 NVIDIA/AMD 及 PTX/AMDGCN；这不自动证明每个二分构建脚本在两家平台都已验证。

### 2. FlyDSL：确有针对 GPU 性能退化的 Agent SOP

[固定版本 skill](https://github.com/ROCm/FlyDSL/blob/43ec5da9d934057b60f06d32ff05a8e044f9020c/.claude/skills/bisect-perf-regression/SKILL.md) 已具备端点确认、噪声提醒、重复采样、提交二分、异常处理、diff 检查和恢复工作目录。其 root-cause 部分明确要求给出简短 hypothesis，列举同步、tile、访存、layout 等常见模式。未见要求编译代码/PMC 证据或恢复旧实现确认机制。

因此它是直接同方向的工作，而不是仅名字相近的工具；但仍属于仓库内技能，不能据文件存在就声称是部署成熟的无人值守产品。Hyperloom 在 [vendored skill](https://github.com/AMD-AGI/Hyperloom/blob/e5f2be1d83d16b3e86fad0b8b873c9a4a8b283e3/src/kernelforge/knowledge/local_wiki/resources/languages/flydsl/skills/optimize/bisect-perf-regression.md) 也收录了该方向。

### 3. TraceLens：版本比较分析已经 Agent 化

[比较文档](https://github.com/AMD-AGI/TraceLens/blob/c5da90ee7d59e49bf75e8602f0211da1f6dcd8b3/docs/how-to/compare-traces.md) 明确覆盖代码、库或硬件变化，支持逐算子报告及 TraceDiff 结构匹配。

[semantic-comparison agent](https://github.com/AMD-AGI/TraceLens/blob/c5da90ee7d59e49bf75e8602f0211da1f6dcd8b3/TraceLens/Agent/Analysis/skills/analysis-orchestrator/agents/semantic-comparison-agent.md) 先对两个 trace 确定性提取/分类，再由 LLM 统一不同 kernel 名称所对应的语义操作，随后生成差异。它对“升级后名字与图结构变化，直接按名字匹配失败”这一难点已经有实质实现。

## 其余可复用基础设施

- PyTorch Benchmark：[二分器](https://github.com/pytorch/benchmark/blob/ab5e0c9b7efdb5b0e8e6e58f0c767ae8c46091fc/bisection.py)、[回归检测](https://github.com/pytorch/benchmark/blob/ab5e0c9b7efdb5b0e8e6e58f0c767ae8c46091fc/regression_detector.py)。它预定义目标仓库并协调其余包版本，已经自动构建、运行和查找 performance signal 的责任提交。
- Magpie：[README](https://github.com/AMD-AGI/Magpie/blob/63794dea07b8b12572e894511a1e8315d4982c26/README.md) 明确跨 AMD/NVIDIA，提供 MCP 和比较、benchmark、profiling，可作为实验执行层。
- Hyperloom：[README](https://github.com/AMD-AGI/Hyperloom/blob/e5f2be1d83d16b3e86fad0b8b873c9a4a8b283e3/README.md) 是有实际重测及正确性约束的多 Agent 优化流程；本次没有找到其主入口直接接收新旧版本并自动做通用依赖归因的证据。
- IREE：[package bisect](https://github.com/iree-org/iree/blob/3fd498c00d0dec30c179864178715753f536a599/build_tools/pkgci/bisect/README.md) 可以复用二进制包缓存及版本安装，避免每一步源码构建；记录了真实编译故障案例，不能把它改称性能归因成功案例。
- 对 JAX/XLA、Intel 和 NVIDIA 的 GitHub 关键词搜索未在本次有限范围找到更强的同类闭环。JAX 当前递归目录未见 bisect 命名文件，这只是检索结果，不意味着其内部或其他仓库没有相关服务。

## 可以保留的差异化假设

真正值得验证的是：在完整软件栈同时变化、版本间有 ABI/构建不兼容、测量有噪声时，Agent 能否自己选择有效的组件替换和机制干预，区分 CPU 调用路径、编译产物与库算法选择，并输出“已确认/支持/未排除”的证据等级。它应当复用现有 benchmark、bisect、trace 工具，而不是以重新实现这些能力为创新点。

此差异化仍是待评估假设；需要与上述工具用同一组历史及未知回归案例对照，以调查成功率、错误归因率、GPU 时间、人工补充次数和复现成本来判断。
