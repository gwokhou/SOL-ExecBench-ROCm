# SOLAR 之外的升级收益：ROCm 10.1.0 / Python 3.14 / PyTorch 2.14.0

研究完成日期：2026-10-11。基础提交 `e5a0d450`，分析过程中工作区已有依赖、Docker、版本矩阵与测试的未提交升级改动。本文以读取时源码为准，不修改这些实现，也不将正在进行的升级视为已经验证成功。仅新增本报告；没有安装、运行测试或执行 GPU 工作负载。

**判断**

相比 SOLAR 的形式分析，评测执行、原生编译、profiler 和环境观测更容易直接受益。最高优先级是减少采集/编译故障，并让性能诊断能使用明确的同步事件；其次是更新参考正确性与性能基线。Python 3.14 的工程便利不是核心 GPU 能力提升。

| 模块 | 最相关变化 | 收益类型 |
| --- | --- | --- |
| `rocm_profiler` | profiler 共存、退出处理、HIP event、受限 kernel replay | 底层修复可直接受益；新事件与回放方式需要接入 |
| `performance_model` / `decision` / `agent_feedback` | 显式等待、ATT 与计数器诊断 | 需要解析、建模和反馈契约改造 |
| 原生候选编译 | PyTorch extension 使用可自动释放的文件锁；TheRock SDK 发现 | 直接稳定性收益，加工具链一致性验证 |
| correctness / reference worker | ROCm 算子正确性修复与新编译路径 | 有条件直接受益，必须重建参考证据 |
| platform / clock / isolation | AMD SMI 容器进程、时钟 unavailable 表示修复 | 观测可靠性收益 |
| evidence / publication | Python 标准库 Zstandard | 可选简化，当前已经实现压缩，收益有限 |
| scoring / qualification | 无新的公式能力 | 需要新环境重新资格验证，不能继承旧证据 |

**Profiler 与性能诊断：最值得投入**

ROCm 10.1 修复 PyTorch 内置 profiler 与 rocprofv3 的冲突，改善异常退出和运行中工具配置，并提供 GPU 侧 HIP event 记录。这些变化与当前独立 worker、Torch/Triton 候选、超时清理的执行模式高度相关。[10.1 发布说明](https://rocm.docs.amd.com/en/docs-10.1.0/about/release-notes.html)

当前 [命令构造](../../src/sol_execbench/core/bench/rocm_profiler/commands.py) 已使用 `--hip-runtime-trace`，但 [profile summary](../../src/sol_execbench/core/bench/profile_summary/artifacts.py) 主要解析已有 CSV 和 JSON 字段，没有消费 HIP event 的完整依赖关系。产生新记录和真正使用新记录是两个验收步骤。

最具体的改造点是 [schedule_evidence.py](../../src/sol_execbench/core/bench/performance_model/schedule_evidence.py)：目前依靠 lane 与时间戳构造顺序，跨 lane 的先结束后开始并不能单独证明因果依赖。加入 event record/wait 的身份关联后，可把“观察到的时间顺序”和“显式同步依赖”分开。不能删除没有 event 的同 stream 或其他同步约束；应记录依赖覆盖范围。

下一步才是把已验证的等待归因送入 Decision 和 Agent feedback，例如“kernel B 被事件 E 阻塞，E 对应 A 的完成”，而不是直接建议修改 tile。当前 [hint 分类](../../src/sol_execbench/core/bench/profile_summary/models.py) 尚无专门的同步等待类别；新增类别需要与 producer、consumer、schema 和测试一起更新。事件只解释观测，不自动证明某个同步可以删除。

kernel replay 的价值是减少多组 PMC 的应用启动成本。但它对恢复内存类型、图执行、异步 copy 和并发有边界，应先限定 microbenchmark；不能把被 replay 串行化的调度拿去证明原始并发图，也不能把 replay 耗时当 canonical timing。[SDK replay 边界](https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/how-to/using-kernel-replay.html)

gfx12 ATT 修复适合验证本地 RDNA4 诊断；MI300/MI350 的 PC sampling 增强不能自动外推到 gfx1200。Compute Profiler 的 rocpd-only 输出变更也不代表本仓库直接调用的 rocprofv3 不再支持 CSV。[ROCm 10.1 说明](https://rocm.docs.amd.com/en/docs-10.1.0/about/release-notes.html)

**原生扩展编译：一个明确的直接收益**

PyTorch v2.14.0 的 `cpp_extension._jit_compile` 使用 `FileLock`，替代只靠锁文件存在与否判断的机制。进程被终止时操作系统会释放锁，降低崩溃/OOM 后后续构建被陈旧锁卡住的风险。该路径包含 HIP extension。[固定版本源码](https://raw.githubusercontent.com/pytorch/pytorch/v2.14.0/torch/utils/cpp_extension.py)

本仓库的 [build_ext.py](../../src/sol_execbench/driver/templates/build_ext.py) 直接调用 `ext.load`，所以这是无需重新设计框架即可利用的稳定性修复。实际收益取决于是否复用发生过异常退出的构建目录；每次全新 staging 时问题暴露概率较低。它不替代构建超时、进程组清理或完整产物验证。

[NativeCompileCache](../../src/sol_execbench/cli/evaluation/native_compile_cache.py) 已有 fcntl 锁，且 key 包含 compiler/python hash、Torch/HIP 和 image 身份。这项机制应保留；新栈使用不同缓存身份即可，不应删除所有旧缓存或将旧二进制冒充新构建。

新 SDK 的设备包选择和可迁移安装利于隔离 gfx1200 环境；TheRock 转型从 ROCm 7.14 开始，不能称为 10.1 首创。[官方迁移说明](https://rocm.docs.amd.com/en/docs-10.1.0/about/transition-guide-TheRock.html) v2.14 extension 源码能发现 `_rocm_sdk_core`；项目自己的 [runtime.py](../../src/sol_execbench/core/platform/runtime.py) 则优先显式路径、hipcc 推导和传统位置。因此应证明 Torch、编译器、headers、动态库与 profiler 实际指向同一套目标环境，而不只检查 PATH 或包版本。

**参考执行与正确性：价值不只是跑得更快**

PyTorch 2.14 修复了 ROCm 上大 rows 且非四倍 normalized size 的 LayerNorm、超大 channels-last nearest interpolation、quantized embedding_bag 访存等问题；也包含 gfx10/11/12 atomic-store wait 指令修复。收益仅适用于触发相应条件的 workload，本次没有确认 corpus 触发率。[2.14 完整发布说明](https://github.com/pytorch/pytorch/releases/tag/v2.14.0)

这直接关系到 [correctness rounds](../../src/sol_execbench/core/bench/eval_correctness.py) 使用的参考输出。建议对受影响 shape、dtype、布局以及历史失败案例优先做差分，区分“参考已修正”和“候选退化”；不能通过放宽容差吞掉差异。2.14 还改变部分 clamp/min/max 边界梯度语义，若 reference 明确计算梯度，需要针对相等边界单独检查。[2.14 兼容性说明](https://github.com/pytorch/pytorch/releases/tag/v2.14.0)

Origami、grouped GEMM 与 CK JIT 支持可改善满足条件的 torch.compile 候选或参考路径；源码没有证据表明所有参考都被统一 compile，所以不能预期 corpus 普遍提速。[PyTorch ROCm 发布说明](https://pytorch.org/blog/pytorch-2-14-release-blog/#rocm)

若参考耗时变化，reference-relative speedup 会变化；若正式 baseline 变化，SOL Score 也会变化。候选本身未改不代表分数应该保持原值。应在一致的新环境内重新测量候选与对应基线，记录编译/autotune 设置。

**计时和执行隔离：保持语义，升级观测**

[timing.py](../../src/sol_execbench/core/bench/timing.py) 的实际计时包含预热、独立数据地址、cache clear、device events、同步和结果验证。HIP event trace 适合诊断这些环节，并不是替换计时协议的理由。Python 发射开销和运行库路径发生变化时，也不能事先承诺当前 event 指标必然不变。

2.14 的 make_graphed_callables/hipBLASLt capture 修复不能直接转为当前候选收益：[reward_hack 规则](../../src/sol_execbench/core/bench/reward_hack/models.py) 明确限制隐藏 streams 和 graph capture。未来若开放图执行，需要另行定义测量与正确性规则，不能用性能特性绕过现有约束。[PyTorch 2.14 修复列表](https://github.com/pytorch/pytorch/releases/tag/v2.14.0)

[评测架构](../user/ARCHITECTURE.md) 使用独立 trusted reference / candidate 进程、JSON 控制消息和 safetensors 数据，没有靠 pickle 反序列化候选输入。不要为使用 InterpreterPoolExecutor 而将它们合并到一个进程。Python 3.14 的 multiprocessing 默认启动方式变化也不能直接套到这里的显式 subprocess/exec 链上。

**环境观测与调试：减少不完整证据**

AMD SMI 改善容器进程识别和完整容器 ID 报告，并修复 unavailable 时钟被显示为数值的情况。它们适合当前 [amd_smi.py](../../src/sol_execbench/core/platform/amd_smi.py)、[timing isolation](../../src/sol_execbench/core/bench/timing_isolation.py) 和 [clock lease](../../src/sol_execbench/core/bench/clock_lock.py)。[ROCm 10.1 AMD SMI 变更](https://rocm.docs.amd.com/en/docs-10.1.0/about/release-notes.html)

当前 process parser 只保留 pid/device/name，要利用容器归属需要增加字段与验证。读取失败也不应被解释为证明 GPU 空闲。CLI 修复不能消除 clock lock 的所有权和恢复责任。

HIP 还修复了部分 GPU 的 stream/signal 资源泄漏，新增 `hipInitDevice` 可初始化指定设备而不改变当前设备。前者可能改善重复评测的稳定性，后者是未来显式初始化的候选；当前没有证据证明 gfx1200 已触发该泄漏，也不需要仅为使用新 API 重写现有预热。[HIP 组件说明](https://rocm.docs.amd.com/en/docs-10.1.0/about/release-notes.html#hip-7-16-0)

ROCgdb 对 verbose trap 的诊断改进，可用于未来失败复现流程；当前 static kernel 模块读取编译产物与 ISA 信息，不会自动启动 debugger。若接入，应仅在隔离的失败重放中记录输出，避免把调试插桩混入正式计时。[ROCgdb 变更](https://rocm.docs.amd.com/en/docs-10.1.0/about/release-notes.html#rocgdb-builtin-verbose-trap-improvement)

**Python 基础设施和评分：收益有限，边界明确**

Python 3.14 的标准库 `compression.zstd` 可以减少第三方依赖，但本仓库 [publication packaging](../../src/sol_execbench/core/bench/performance_model/release/packaging.py) 已经用 zstandard 生成可复现归档，明确控制 compression level、checksum、content size 和 dictionary ID。迁移需要验证这些设置、流式读写、展开限制及产物 hash；格式可互读不代表压缩字节完全相同。[标准库文档](https://docs.python.org/3.14/library/compression.zstd.html)

`Executor.map(buffersize=...)` 可服务未来有界的 CPU 解析/哈希批处理；子解释器和 free-threading 需要独立兼容验证。Pydantic schema 层应优先验证 JSON Schema 和序列化稳定性，不应将延迟注解求值宣传为可直接获得的大幅性能提升。[Python 3.14 变化](https://docs.python.org/3.14/whatsnew/3.14.html)

读取时，新版依赖已出现在工作区 pyproject/Docker 中，但 [RDNA4 qualification 常量](../../src/sol_execbench/core/platform/rdna4_validation.py) 仍绑定 ROCm 7.2 / Torch 2.11 / Triton 3.6，[release trace 验证](../../src/sol_execbench/core/scoring/release_traces.py) 会精确检查它们。这是正在进行的升级还需要处理的边界，并非本文对升级最终结果的判定。不能仅改常量就授予新环境发布资格；需要新硬件测试、baseline 和绑定证据。

建议顺序：先验证编译/执行/观测基础链，重建 reference 与 timing 证据；再接入 HIP event 依赖及诊断反馈；随后限定试验 kernel replay 与新编译路径；最后按测得的瓶颈决定 Python 并行或压缩库迁移。CPU 回归通过不能代替 GPU 资格验证。
