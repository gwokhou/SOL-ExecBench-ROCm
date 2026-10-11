# SOLAR 升级收益分析：ROCm 10.1.0 / Python 3.14 / PyTorch 2.14.0

研究日期：2026-10-10。源码基准：`e5a0d450`；研究开始时 tracked worktree 无改动。本报告是源码与官方资料分析，没有安装目标环境、运行测试或获取新的 GPU 证据。文中的优先级是工程判断，不是实测加速承诺。

**结论**

SOLAR 最值得利用的是 PyTorch 2.14 的声明式动态形状；ROCm 10.1 的主要收益落在外围诊断、资源审计和验证稳定性；Python 3.14 的收益更偏批处理与维护。单纯升级依赖不会自动扩大可证明算子覆盖、改善 Orojenesis 的证明能力，或使理论下界更紧。

AMD 官方安装页已经列出 `rocm/pytorch:rocm10.1.0_ubuntu26.04_py3.14_pytorch_release_2.14.0`，因此组合本身有官方交付入口。镜像存在不能证明本仓库在 gfx1200 上通过验证；本次未下载镜像、解析 digest 或检查内部 wheel。[AMD 安装文档](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/install/3rd-party/pytorch-install.html)

**当前实现决定了哪些收益能落地**

| 本地事实 | 对升级分析的含义 | 源码 |
| --- | --- | --- |
| Python 固定 3.12；Linux x86_64 使用 Torch 2.11.0+rocm7.2、TorchVision 0.26.0+rocm7.2、Triton ROCm 3.6.0 | 需要整体更新解释器、wheel 来源、锁文件与容器契约 | [pyproject.toml](../../pyproject.toml)、[Dockerfile](../../docker/Dockerfile)、[目标清单](../../docker/rocm-targets.json) |
| 默认 `torchview_extended_einsum`，另有 `make_fx_aten`；失败时不自动切换 | 新 make_fx 能力不会自动改善默认路径 | [责任边界](../SOLAR-BOUNDARY.md)、[IR 契约](../../src/solar/ir/contracts.py) |
| make_fx 未传入动态形状声明，TensorSignature 是整数 shape；serializer 直接保存 `list(value.shape)` | 没有完整的符号表达式与假设持久化契约，不能只改一个调用参数 | [取图](../../src/solar/graph/make_fx_extraction.py)、[图契约](../../src/solar/graph/contracts.py)、[序列化](../../src/solar/graph/reference_serializer.py) |
| Extended IR 已为 nonzero 记录 lower/upper/trace_value；计算计数使用下界，容量分析使用上界 | 已有保守动态维度设计可扩展，但仍非通用符号代数 | [语义](../../src/solar/ir/extended_einsum/torchview/semantics.py)、[计数](../../src/solar/analysis/graph_context.py)、[融合](../../src/solar/analysis/fusion.py) |
| 下界由资源工作量、理论峰值及形式化流量约束计算；严格发布路径要求 Orojenesis | 实测 kernel 加速与理论下界是不同变化 | [形式分析](../../src/solar/analysis/formal_analysis.py)、[架构 profile](../../src/solar/rocm/profiles/RX_9060_XT.yaml) |

**优先级最高：ShapesSpec 改善形状契约和覆盖边界**

PyTorch 2.14 新增实验性 `ShapesSpec` / `ShapeVar` / `@dynamic_spec`，可表达共享维度、范围、派生维度和假设，供 export、compile 和 make_fx 使用。make_fx 接入限定 `tracing_mode="fake"`。声明维度使用 unbacked symbols，阻止按样例值静默专化；形状分支也可能因此报错，而非自动生成覆盖所有分支的图。[2.14 发布说明](https://pytorch.org/blog/pytorch-2-14-release-blog/#declarative-dynamic-shapes-with-dynamic_spec)

对 SOLAR 的直接价值，是让“这个图对应哪些输入”更明确。例如共同的 batch 维度 B、`2 <= B <= 128`、`B % 2 == 0` 可以成为提取与验证的一致前提。当前一次具体输入取图的结果，不能自然升级为整个范围的证明。

建议改造位置：`graph/make_fx_extraction.py` 接收显式声明，`reference_serializer.py` 与图/IR 契约保留规范化的符号关系，verification 在合法边界与越界输入上验证，analysis 明确每个 bound 是具体 shape 的结果还是整个域的保守结果。对既有 workload，先实例化具体 shape 计算下界，通常比直接引入参数化 Orojenesis 更容易审查。

不能丢掉 `B` 与 `2*B` 的关联后把两者当独立区间，也不能把 shape-dependent 分支失败回退成具体样例图后仍宣称覆盖整个域。新增提取语义还需要更新 provenance、精确 schema 与支持的 extractor/IR 配对。

v2.14.0 源码的 `to_jsonable()` 将 assumptions 保存为字符串表示；不能直接把它当作 SOLAR 稳定且可回放的约束格式。SOLAR 应拥有自己的可验证表示。[v2.14.0 dynamic_spec 源码](https://raw.githubusercontent.com/pytorch/pytorch/v2.14.0/torch/fx/experimental/dynamic_spec.py)

fake tracing 的潜在收益是减少取图阶段的真实张量运算，但当前 [make_fx backend](../../src/solar/graph/make_fx_backend.py) 在取图前仍真实调用 reference，转换验证也需要独立回放。因此换 fake mode 不等于整条链无 GPU 执行。`torch.export`、FakeTensor 和 `torch.compile` 本身不是 2.14 才有的功能。

**较高优先级：ROCm 10.1 的观测与验证能力**

10.1 的相关变化包括 HIP event GPU barrier 跟踪、gfx12 ATT occupancy 修复、PyTorch 内置 profiler 与 rocprofv3 共存修复，以及更灵活的 profiler 初始化。Compute Profiler 增加交互式 roofline 展示。[固定版本发布说明](https://rocm.docs.amd.com/en/docs-10.1.0/about/release-notes.html)

| 能力 | 本仓库的用途 | 接入边界 |
| --- | --- | --- |
| HIP event 等待记录 | 对比 SOLAR 理想资源下界与实际 stream 等待，解释差距来源 | 属于 `sol_execbench` 的 profile/performance sidecar；不能把等待时间直接加入 SOLAR 硬件理论峰值 |
| gfx12 ATT / profiler 共存修复 | 改善 gfx1200 上的指令、occupancy 诊断与 Torch/Triton 采集成功率 | 先验证具体 decoder、GPU 和输出；不能从发布说明推断本地已有问题全消失 |
| 交互式 roofline | 对照理论资源瓶颈与实测 kernel 行为 | profiler 的实测 roof 与 SOLAR 的形式化下界必须分别标识 |

当前 [profiler 命令](../../src/sol_execbench/core/bench/rocm_profiler/commands.py) 已使用 `--hip-runtime-trace`；新增事件仍需要解析、关联和报告，不能仅凭命令运行成功认定被模型消费。

本地 [SQ_WAVE_CYCLES workaround](../../scripts/patches/gfx1200_sq_wave_cycles/README.md) 处理的是 AUTO 时钟下计数为零的问题。gfx12 ATT occupancy 修复不是同一个故障，不能据此删除时钟锁或该 workaround。

**有价值，但必须单独验证：kernel replay**

10.1 引入 Beta kernel replay：在单次应用启动中重复 dispatch 并恢复受跟踪状态，可减少多组 PMC 采集的重复启动工作。rocprofv3 入口是 `--replay-mode kernel --kernel-replay-beta-enabled`，当前 CLI 模式仅用于 counters，不能与 ATT、PC sampling、SPM 合用。[10.1 发布说明](https://rocm.docs.amd.com/en/docs-10.1.0/about/release-notes.html#kernel-replay-feature-beta)

当前 SDK 文档进一步限定了恢复范围：不恢复 managed、host、hipMallocAsync 和 HIP graph-managed memory；HIP graph dispatch 不进行这种重放。存在 agent 级串行化、额外主机内存开销和异步 copy 限制。这是当前文档的设计边界，目标镜像中的具体实现仍要核实。[Kernel replay 文档](https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/how-to/using-kernel-replay.html)

本仓库 [counter_collection.py](../../src/sol_execbench/core/bench/rocm_profiler/counter_collection.py) 逐组启动受控应用，记录 pass 与输入身份。改用 SDK replay 时，必须重新证明输入/输出状态、dispatch 身份和计数组间对应关系，处理“不重放但未失败”的情况。最适合先试单 GPU、无并发外部状态、分配策略明确的校准 microbenchmark；不宜立即替代全部 workload 的采集协议。重放数据也不应作为无扰动 canonical timing。

**中等优先级：Origami 与新 GEMM 路径作为研究对照**

PyTorch 2.14 将 Origami 分析式 tile 选择用于 ROCm max-autotune，并扩展 ROCm grouped GEMM 的 Triton lowering（含 scaled FP8）与 CK JIT cpp_wrapper 路径。[2.14 ROCm 说明](https://pytorch.org/blog/pytorch-2-14-release-blog/#rocm)

这些能力适合在外围产生更强的参考实现，比较实际延迟、tile 选择与 SOLAR 容量约束。Origami 的预估延迟/候选选择不是形式化下界，不能代替 Orojenesis。是否覆盖 gfx1200、特定 dtype 和 shape，需要单独跑能力测试，不能把“ROCm 支持”泛化到所有 AMD 架构。

当前 SOLAR 没有调用 torch.compile 的生产链，不能期待这些优化自动改变 graph analysis。新增融合实现还可能改变 ATen 图和算子覆盖，因此需要比较数值等价、资源计数、IR 身份三者，不能只看执行更快。

**Python 3.14：工程收益高于数学能力收益**

`InterpreterPoolExecutor` 提供独立解释器的 CPU 并行；`Executor.map(buffersize=...)` 可限制未消费任务数。未来可试用于已序列化 IR 的离线验证和图分析，并控制排队内存。当前没有这项接入，且第三方扩展兼容性与序列化成本需要验证。[concurrent.futures 文档](https://docs.python.org/3.14/library/concurrent.futures.html)

free-threading 是单独的构建/运行选择，普通 3.14 不会自动关闭 GIL。[Python free-threading 文档](https://docs.python.org/3.14/howto/free-threading-python.html) 当前 [Torchview Recorder](../../src/solar/_vendor/torchview/recorder_tensor.py) 会替换 `torch.nn.Module.__call__` 和 torch 创建函数；多个取图任务共享解释器会互相干扰，这个风险在有 GIL 时也不能靠 GIL 解决。因此保持取图 worker 的进程隔离更合理；子解释器也不等价于进程级故障隔离。

Orojenesis 已是外部原生 mapper，且 [线程配置](../../src/solar/analysis/orojenesis/configuration.py) 使用进程可见 CPU；解释器升级不会自动加速其求解。并发多个 mapper 前要协调 CPU 配额，避免每个都使用全部可见 CPU。

3.14 还提供延迟注解求值、标准库 Zstandard、调试与解释器改进；这是维护或 I/O 层面的候选收益。Linux 默认 multiprocessing 启动方式变化需要检查，但不能据此断言显式 subprocess worker 受影响。CPython JIT 不等于 HIP/Triton kernel JIT。[Python 3.14 变化](https://docs.python.org/3.14/whatsnew/3.14.html)

**不应算作当前 SOLAR 的主要升级收益**

- NVGEMM/CuTeDSL、NCCL、CUDA 专属图特性：不能移植其宣传收益到当前 ROCm 目标。2.14 的 RDNA3 FlexAttention 调优也不是 gfx1200 RDNA4 的收益保证。[PyTorch 2.14 发布说明](https://pytorch.org/blog/pytorch-2-14-release-blog/)
- 新数学库、hipThreads、libhipcxx：可以服务未来 kernel 或 workload 开发，但当前 SOLAR 的 Python 图分析和形式证明不会自动调用它们。[ROCm 10.1 发布说明](https://rocm.docs.amd.com/en/docs-10.1.0/about/release-notes.html)
- 编译器生成更快的代码：可能改善验证耗时与实测表现，但不会增加物理 GPU 的理论算力；已有 ISA/runtime 审计要重新验证。

**建议实施顺序与验收证据（本次未实施）**

1. 先做普通 CPython 3.14 的基础迁移：更新 `.python-version`、Python 范围、Ruff target、完整 Torch/TorchVision/Triton wheel 组合、uv.lock、Docker 目标与配置测试；核实 `torch.version.hip` 和实际加载库，避免主机 SDK 与 wheel 内 runtime 混用。不要假设更换系统 `/opt/rocm` 就改变 wheel 的构建身份。
2. 保持两套 IR 路径行为，对固定小型 corpus 比较算子图、源参数绑定、dtype、alias/mutation、前向/梯度回放与资源计数；差异必须有解释。相关现有测试包括 `tests/solar/test_aten_ir.py`、`test_semantic_values.py`、`test_gradients.py`、`test_resource_audit.py`。
3. 将 ShapesSpec 作为独立功能接入与审查：包含共享维度、派生关系、边界值、越界、形状分支、数据依赖 nonzero、约束持久化和拒绝路径。数值抽样通过不等于全域证明。
4. 再升级外围 profiler：核实 CSV/rocpd 解析、事件关联和 kernel replay 的分配覆盖。Compute Profiler 改为 rocpd-only 不等于 rocprofv3 不能输出 CSV；本仓库直接调用后者，不能错误地宣布现有 CSV 路径必然失效。[ROCm 10.1 输出变更](https://rocm.docs.amd.com/en/docs-10.1.0/about/release-notes.html#profiling-output-update)
5. 重新获取 gfx1200 的编译指令、runtime probes、计数器与必要资源审计。当前 [资源证据](../../src/solar/rocm/audits/rx9060xt_resource_peaks_v3.json) 明确记录旧 ROCm/Torch 版本，只能作为旧环境证据。保持原有 nominal 理论峰值与实测审计的角色区别；不能因新编译器跑分更好就抬高 SOL bound 中的峰值。

优先投资顺序：**动态形状契约 > profiler 可靠性与解释能力 > 受限 kernel replay > Origami 对照 > Python 并行实验**。若只做依赖升级，合理预期是新环境兼容与部分底层修复；覆盖率、证明强度和吞吐收益都需要后续证据。
