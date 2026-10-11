# SOLAR 在 ROCm 10.1 / Python 3.14 / PyTorch 2.14 下的升级机会

分析日期：2026-10-11。初次分析的源码基准为 `1b431d7455a4ba045fbb9646a448b3b0f758aeb4` 加当时未提交的工作区。初次分析未修改生产实现、依赖或现有实验文件，也未运行新的 GPU 实验。

同日进展：P0 资源审计刷新已完成，正式 profile 已更新为 `rx9060xt-amd-resource-v6`，绑定 ROCm 10.1 / Torch 2.14 的运行证据，详见 [P0 完成报告](solar-p0-rocm101-completion-2026-10-11.md)。本文中的能力探针和研究观察保留初次分析时的范围；P1 及后续建议尚未实施。

## 结论与现状修正

基础栈迁移已经完成。SOLAR 后续最值得投入的是：**新环境资源证据的适用性、动态形状契约、取图与 IR 覆盖、受控的分析吞吐优化**。ROCm 的 profiler 新能力主要改善 SOLAR 周边的解释与诊断；不会自动增强形式证明。

10 月 10 日的[原研究稿](solar-rocm101-python314-torch214-research.md)仍可作背景，但其中“Python 3.12 / Torch 2.11 尚待迁移”“ShapesSpec 尚无本地实验”等状态不适用于当前工作区。本报告不覆盖或改写旧稿。

| 已核查的事实（P0 状态已补充更新） | 证据与范围 |
| --- | --- |
| 项目已锁定 Python 3.14、Torch `2.14.0+rocm10.1.0`、Triton 3.8 ROCm wheel | [pyproject.toml](../../pyproject.toml)、[Dockerfile](../../docker/Dockerfile) |
| 已有新栈验证镜像，本次 CPU 检查显示 Python 3.14.4、Torch `2.14.0+rocm10.1.0`、`torch.version.hip=7.16.26385` | 镜像 `sha256:0ec2e6cba29e673ebe5ad83b7049e5044647fc340174d1c855379efaf36409dd`；HIP 组件版本与 ROCm 总发行版不是同一编号 |
| 保留的全 CPU 回归记录为 2,499 passed / 51 skipped，其中 `tests.solar.*` 为 616 passed / 0 skipped | 本次核验 `validation-receipt.json` 所绑定 `cpu-accepted.xml` 的 SHA-256 并解析 JUnit；这是已有记录，不是本轮重跑，也不覆盖记录后发生的所有未提交变化 |
| ShapesSpec 已有六类独立开发实验 | 本次核验 `shapes-spec.json` 的 SHA-256；没有注册生产 IR 路径 |
| HIP event 的 JSON/rocpd reader、摘要和反馈已在工作区出现 | [hip_events.py](../../src/sol_execbench/core/bench/rocm_profiler/hip_events.py)、[hip_event_artifacts.py](../../src/sol_execbench/core/bench/rocm_profiler/hip_event_artifacts.py)；仍未取得跨 lane 事件因果关系资格 |
| 正式资源审计已从 ROCm 7.2 / Torch 2.11 更新至 ROCm 10.1 / Torch 2.14 | [RX profile](../../src/solar/rocm/profiles/RX_9060_XT.yaml)、[资源审计](../../src/solar/rocm/audits/rx9060xt_resource_peaks_v3.json)；仅佐证指令生成和实际运行，不证明达到无节流峰值 |

已有记录位置：[验证回执](../../data/outputs/stack-upgrade-followup-20261011/validation-receipt.json)、[CPU JUnit](../../data/outputs/stack-upgrade-followup-20261011/cpu-accepted.xml)、[ShapesSpec 实验](../../data/outputs/stack-upgrade-20261011/shapes-spec.json)。这些位于未提交 `data/`，不应随本文提交。

宿主项目 `.venv` 本次只检测到 Python 3.14.3、没有 Torch 包元数据，因此没有在宿主安装依赖；实际能力探针使用上述已有镜像。AMD [当前安装页](https://rocm.docs.amd.com/projects/ai-ecosystem/en/latest/frameworks/pytorch/install.html)确认这一发行组合，原 install-on-linux 链接已搬迁。

## 升级优先级

| 优先级 | 具体升级 | 主要收益 | 当前剩余工作 |
| --- | --- | --- | --- |
| P0：已完成 | 资源审计与软件适用性闭环 | 明确哪些旧物理事实仍可用，哪些编译/运行结论须刷新 | 已刷新 ISA emission、runtime probes 和 profile 证据绑定；适用范围见 P0 完成报告 |
| P1：主要功能投资 | ShapesSpec → 自有形状域契约 → 可验证 IR | 共享、派生、范围和整除约束不再隐含于单一样例 | 标量/断言表示、约束验证、序列化、具体化与 bound 域标识 |
| P1：较小独立增量 | 静态 fake 取图试点 | 可能减少取图中的真实计算和显存需求 | 输入/输出元信息、mutation、常量、数据依赖拒绝路径与真实回放验收 |
| P1/P2：按实际 corpus | 2.14 图与算子覆盖差分 | 避免新 ATen lowering 导致原支持工作负载失效 | 比较两条固定路径、精度、alias、mutation、前向/梯度和资源计数 |
| P2：外围诊断 | HIP event 依赖与 profiler qualification | 解释实际耗时与理想下界之间的差距 | 补齐事件 generation 与 dispatch 因果证据；保持 SOLAR 边界 |
| P2：性能工程 | 有界批处理与 mapper CPU 配额 | 减少排队内存、CPU 超卖 | 先量 stage 时间，再决定 executor 与配额实现 |
| P3：研究对照 | Origami / grouped GEMM / CK 路径 | 比较 tile 选择与容量约束 | 架构、dtype、shape 限定的独立对照，不替代 Orojenesis |

优先级是基于当前源码的工程判断，不是实测收益排序。

## P0：更新资源证据的适用性，而不是改大理论峰值

本节保留实施前的判断依据和建议；这些待办已由 [P0 完成报告](solar-p0-rocm101-completion-2026-10-11.md)记录的工作落实，不应视为仍待执行的采集计划。

初次分析时，[RX_9060_XT.yaml](../../src/solar/rocm/profiles/RX_9060_XT.yaml)引用旧软件栈的精度支持链接和 INT4 限制说明。其当时绑定的资源审计 `hip_version` 为 `7.2.26015`，`torch_version` 为 `2.11.0+rocm7.2`。

[architecture.py 的 `_verify_audit_contract`](../../src/solar/rocm/architecture.py)校验 schema、timing profile、gfx、设备名称和锁频状态，并不在这个边界比较当前 Torch/HIP 版本。因此“profile 被成功加载”本身不证明当前软件栈已重获资格。这不使旧审计失去历史效力，也不说明同一芯片的物理峰值变了。

建议将两种信息分别表达：

- 架构来源和名义上限：ISA、CU/存储层级、厂商吞吐规格。
- 实现适用性：精确 wheel、HIP/compiler 身份、编译指令、实际 runtime probes、目标 GPU、时钟和采集协议。

新环境正式使用前，核对已有新栈证据能否覆盖 resource-audit 的每个必需模式；缺项再有针对性地补采。现有六个 backend smoke tests 不能替代所有资源模式的审计。保持审计协议中的冻结 tuning、独立 held-out raw samples 和指令校验要求。

INT4 的旧限制文字应重新核实，而不是直接删除 exemption；FP8 也需保持 gfx1200 OCP 与 gfx942 FNUZ 的区别。新库声称支持低精度，并不足以把某精度加入正式能力矩阵。

## P1：动态形状接入的真实缺口

PyTorch 2.14 新增实验性 `ShapesSpec` / `ShapeVar` / `dynamic_spec`，支持共享维度、派生关系和假设；`make_fx` 要求 fake mode。它不会自动覆盖依赖形状的 Python 分支。[官方发布说明](https://pytorch.org/blog/pytorch-2-14-release-blog/#declarative-dynamic-shapes-with-dynamic_spec)

当前生产链仍有以下具体边界：

1. [ConversionRequest](../../src/solar/contracts.py)没有形状域字段；`TensorSignature.shape` 是 `tuple[int, ...]`。
2. [make_fx_extraction.py](../../src/solar/graph/make_fx_extraction.py)调用默认 real tracing，且将原 reference 包装成 `*tensor_values`。声明中的原参数名必须准确映射到 source input indices，不能直接套用包装前的签名。
3. [reference_serializer.py](../../src/solar/graph/reference_serializer.py)只为张量保存 metadata，并拒绝无张量输出与大多数非 ATen target。
4. [ATen allowlist](../../src/solar/ir/aten/conversion.py)没有承载完整符号形状/断言图；只放宽 allowlist 也不能补齐 executor 与资源计数语义。
5. Extended IR 对 `nonzero` 已有 `lower/upper/trace_value`，但不是通用符号代数。[当前实现](../../src/solar/ir/extended_einsum/torchview/semantics.py)
6. [graph_context.product](../../src/solar/analysis/graph_context.py)用动态下界做保守计数，[fusion._product](../../src/solar/analysis/fusion.py)用上界估容量。独立区间不能自动表达 `B` 与 `2*B` 的关联。

本次在无 GPU、无网络、源码只读挂载的现有镜像中，对 CPU `x + y` 做直接检查：

| 模式 | 当前 serializer 结果 |
| --- | --- |
| 默认 real，具体 shape `(8, 3)` | 通过，3 layers |
| fake，具体 shape `(8, 3)` | 通过，3 layers |
| fake + 共享 batch `2 <= B <= 16` | 拒绝：`FX node sym_size_int has no tensor outputs` |

动态图包含 `aten.sym_size.int`、Python 比较算子与 `aten._assert_scalar.default`。因此只增加 `dynamic_shapes=...` 无法打通生产链。该检查只覆盖这个小样例，未宣称整个 fake 路径合格。

已有 ShapesSpec 记录的有限回放结果也支持这一方向：共享维度、派生维度、范围、整除和 nonzero 的 fake 图均通过已测合法输入，并拒绝已测越界输入；shape branch 在 fake tracing 时以 `GuardOnDataDependentSymNode` 拒绝。整除用例 real 图只匹配 1/4 个合法回放 shape，fake 图匹配 4/4，说明追踪时具体化可能限制可回放域。这是该实验的观察，不是生产静态路径在其单一 shape 合约内不正确。

建议最小落地方案：

- 由 SOLAR 定义约束 AST / 类型化域，保存 symbol、range、派生关系、assumptions 和 source binding；不要把上游 `repr(SymBool)` 当成可执行协议。[上游版本源码](https://raw.githubusercontent.com/pytorch/pytorch/v2.14.0/torch/fx/experimental/dynamic_spec.py)
- 明确区分张量计算与形状/断言节点；后者不能按张量 FLOPs 计费，也不能为了通过转换而直接丢弃。
- 第一阶段先在经过域校验的具体 shape 上形成现有 IR，继续调用当前分析器；artifact 明确记录具体实例及来源域。通用参数化 Orojenesis 可以后续单独设计。
- 验收覆盖共享/派生关系、合法边界、越界、整除、shape branch、nonzero 以及 alias/mutation。有限数值回放与整个域的形式化证明保持不同声明。
- 变更时同步更新请求、图/IR、verification、manifest、schema registry、bridge 与 CLI 合约；保持现有两条固定路径、不自动回退的政策。

默认路径是 `torchview_extended_einsum`；上述 make_fx 增量首先作用于 `make_fx_aten`。如果要让默认路径受益，应明确设计其约束接收与绑定方式，而不是默默换默认 extractor。[SOLAR 边界](../SOLAR-BOUNDARY.md)

## 静态 fake 与覆盖差分可以分开推进

静态 fake 比完整符号 IR 更小，适合作为早期实验。它是已有 PyTorch 能力，本次升级提供了重新验证的环境，不应称为 2.14 首次提供。

[make_fx_backend.py](../../src/solar/graph/make_fx_backend.py)取图前仍执行 `observed = reference(*source_inputs)`，后续 verification 也执行真实 reference / IR。因此即使 trace 改为 fake，整条链仍有真实执行。先分别测输入生成、输出 metadata 获取、trace、转换、回放、Orojenesis 的耗时与峰值内存，才能判断改造收益。

算子覆盖应从实际 corpus 差分出发：matmul/addmm/bmm、attention、normalization、reduction、view/alias、mutation、低精度转换与显式 backward。每个新支持项都要贯穿提取、语义参数、effects、replay、资源分类与形式分析；不能因 Torch 支持 grouped GEMM 或 FP8 就直接宣称 SOLAR 已支持。

已有 616 个 SOLAR CPU 测试提供兼容性基础，不能代替旧/新栈的图语义差分或新 GPU 精度验证。源码级回归入口包括 `test_extraction_symmetry.py`、`test_reference_serializer.py`、`test_aten_ir.py`、`test_extended_native_v6.py`、`test_gradients.py` 和 `test_resource_audit.py`。

## ROCm 新能力主要放在外层

10.1 提供 HIP event barrier 活动、gfx12 ATT occupancy 修复、profiler 初始化/共存修复和 beta kernel replay；HIP event 的原生输出是 JSON/rocpd。[固定版本说明](https://rocm.docs.amd.com/en/docs-10.1.0/about/release-notes.html)

当前工作区已经实现事件解析和摘要，下一步应解决 **WAIT 对应哪次 RECORD、对应哪些 dispatch**。现有 [normalize_event_generations](../../src/sol_execbench/core/bench/rocm_profiler/hip_events.py)明确将 generation 保持为观察值；[schedule_evidence.py](../../src/sol_execbench/core/bench/performance_model/schedule_evidence.py)跨 lane 仍标为 unverified。这是有意保守的边界。没有因果证据时，不能仅按时间戳把等待连成调度边。

这类信息用于解释实际执行与 SOLAR 下界之差；应留在 `sol_execbench` 诊断 sidecar / performance model，经 bridge 使用 SOLAR 结果，不让 SOLAR 反向依赖 profiler 或候选时间。

kernel replay 已有 retained pilot，并非完全未试：两个模式数值检查通过，但 `SQ_INSTS_VALU` 全零，`counter_validity_qualified=false`。本次核验了报告 hash。当前不能推荐替代生产多 pass 采集，也没有速度收益结论。[保留报告](../../data/outputs/stack-upgrade-followup-20261011/counter-replay-final/report.json)

## Python 并发与编译模型：后续有条件投入

Python 3.14 的 `Executor.map(buffersize=...)` 可用于限制待消费批次；`InterpreterPoolExecutor` 提供独立解释器并行，但扩展兼容性、对象序列化和状态隔离需要验证。[标准库说明](https://docs.python.org/3.14/library/concurrent.futures.html)

优先保持取图进程隔离：[Torchview recorder](../../src/solar/_vendor/torchview/recorder_tensor.py)会临时替换全局 `torch.nn.Module.__call__` 和 torch 创建函数。同解释器内并发取图不因升级 Python 或关闭 GIL 就安全。普通 Python 3.14 也不自动成为 free-threaded build。[官方说明](https://docs.python.org/3.14/howto/free-threading-python.html)

现有 [pipeline](../../src/solar/pipeline/analysis.py)已能将设备阶段锁与后续形式分析分开，[parallelism tests](../../tests/solar/test_pipeline_parallelism.py)覆盖这种安排。应先给并发 Orojenesis mapper 分配总 CPU 预算；目前 [configuration.py](../../src/solar/analysis/orojenesis/configuration.py)让每个 mapper 使用全部进程可见 CPU。多个任务同时启动可能超卖 CPU，换 executor 不会自动解决。

Origami 的 2.14 变化是 ROCm max-autotune 默认启用分析式 tile 选择，适合做实际实现的比较对象；它的延迟预测不是形式化下界，也不能替代 pinned Orojenesis。[PyTorch ROCm 说明](https://pytorch.org/blog/pytorch-2-14-release-blog/#rocm)

当前 SOLAR 生产链没有 `torch.compile`。已有外层 24-workload 编译路径筛查还观察到部分新栈回退，所以不能按版本提升推导加速；这些外层数据也不等于 SOLAR 分析耗时。GPU 库、CPython 和 Orojenesis 应各自保持测量范围。

## 建议实施分组

1. **资源证据维护**：核查新栈已有资源模式证据，列出待补项，刷新确有证据的精度支持说明；新物理采集独立安排。
2. **小型取图增量**：静态 fake 的隔离试点，以及新旧图差分报告，保持默认路径不变。
3. **核心功能增量**：形状域、标量/断言表示、实例化与版本化证据闭环。先支持明确有限的约束词汇，不立即承诺全域参数化下界。
4. **按测得瓶颈再优化**：mapper 配额与有界队列；profiler 因果证据、replay、Origami 作为独立外层工作。

初次分析完成了源码审计、官方资料核查、三份 retained artifact 的 hash 核验、JUnit 汇总及一次 CPU-only serializer 探针，当时未重跑全套测试、未取得新 GPU 资源资格，也未实施上述升级。[官方来源核查附录](solar-upgrade-official-source-check-2026-10-11.md)。随后完成的 P0 采集、回归验证及证据边界另见 [P0 完成报告](solar-p0-rocm101-completion-2026-10-11.md)。
