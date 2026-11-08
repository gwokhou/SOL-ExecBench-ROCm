# 相对 SOL-ExecBench / SOLAR 论文与官方实现的增量和缺口

核查日期：2026-11-07。范围：排除 CUDA→HIP、工具链、ISA、产品参数和 profiler API 的对位移植，比较研究问题、算法、验证协议和可复现工作流。本次只检查论文、公开源码、当前本地源码及既有验收材料；没有运行测试、模型调用、mapper 或 GPU 实验，没有改动生产代码。

合并来源：本报告的首次源码对照，以及用户随后提供的另一 Coding Agent 独立分析。后者补充了 C01–C12 设计决策、运行时保护、GPU 所有权和未移植入口等线索；下文按 owning source 核对后吸收，不将两份分析的一致意见本身当成证据。

## 1. 比较版本与结论

| 对象 | 固定版本 |
| --- | --- |
| 本仓库 | `8641b9dc78d00250ba1cf75d5e9434f9839e5ef4` |
| SOL-ExecBench 论文 | [2603.19173v1](https://arxiv.org/html/2603.19173v1) |
| NVIDIA/SOL-ExecBench | [`a9fa0804c793d438e70850c33fe34426e66d53dd`](https://github.com/NVIDIA/SOL-ExecBench/tree/a9fa0804c793d438e70850c33fe34426e66d53dd)，查询时的 `main` |
| SOLAR 论文 | [2606.26383v2](https://arxiv.org/html/2606.26383v2) |
| NVlabs/SOLAR | [`62dab75349315c0844d8fd646631be598270672d`](https://github.com/NVlabs/SOLAR/tree/62dab75349315c0844d8fd646631be598270672d)，查询时的 `master`，2026-11-06 提交 |

**总体判断：本仓库与论文/官方代码是“批判性再实现＋治理强化＋研究协议扩展”的关系，不是功能超集。优势主要在更严格的评测执行协议、适用域明确的分析、可核验的复用与发布链，以及新增的跨硬件研究协议；独立 SOLAR 在通用源码前端、语言覆盖和某些 einsum 代数搜索上仍更广。没有证据支持本仓库在整体算法、覆盖率或预测准确度上全面超过论文或官方实现。**

这里区分三类增量：超出原论文研究范围的新协议；把论文思想落实得更完整的公开实现；对已有算法施加更明确的验证、预算和资格约束。后两者有实际价值，但不等于新理论。

重要的版本修正：此前本地研究比较的 SOLAR `master=d7e792df…` 已成为 `master-pre-v2-20261105`；旧 `release/llm4arch-sync-20261011=2e381641…` 也不代表当前默认分支。当前官方实现已有 Affine Loop 前端、严格回执、来源绑定和反向需求传播，不能继续沿用“上游没有这些机制”的历史判断。[当前 README](https://github.com/NVlabs/SOLAR/blob/62dab75349315c0844d8fd646631be598270672d/README.md)

[stage0 的 C01–C12](solar-evolution-stage0.md) 是 2026-10-20 冻结的设计记录，可以解释复用/改造/自建的选择，不能证明当前上游能力仍停留在当时。尤其 C01/C02 的上游前端/验证判断已被新版本取代；C11 的默认延后决策则在 [11 月路线图 L3-d](solar-evolution-roadmap-2026-11-05.md) 仍有明确记录。

## 2. 做得更进一步的部分

### 2.1 把正确性检查绑定到每一次计时调用

官方 ExecBench v1.1 先做多轮 correctness，再在计时阶段使用 shifting allocator 改变输入地址。本地每次 warmup/timed invocation 从可信参考服务获得新值，拒绝重复的输入 checksum，并把该次候选输出交回服务验证；参考执行和比较在计时区间之外。

这比“独立正确性检查 + 计时防地址缓存”更强地绑定了实际测量与正确计算。增量是逐次新值和逐次输出验证；原论文已有独立参考、进程隔离、多 seed、防作弊，不能把这些概念都算作本地原创，也不能由此声称完全防御恶意原生代码。

来源：[官方 eval_driver](https://github.com/NVIDIA/SOL-ExecBench/blob/a9fa0804c793d438e70850c33fe34426e66d53dd/src/sol_execbench/driver/templates/eval_driver.py#L413)、[官方 timing](https://github.com/NVIDIA/SOL-ExecBench/blob/a9fa0804c793d438e70850c33fe34426e66d53dd/src/sol_execbench/core/bench/timing.py#L281)、本地 [eval_timing.py](../../src/sol_execbench/core/bench/eval_timing.py)、[reference_service.py](../../src/sol_execbench/core/bench/reference_service.py)。

运行时防护应按机制比较：官方已有 elapsed_time 身份检查、线程数量检查、lazy tensor 拒绝和关键函数完整性检查，也已有 `ShiftingMemoryPoolAllocator`。本地继续增加 sealed strong-reference 身份保护、同步捕获短生命周期线程启动、设备切换检测等。确定性 AST 规则包含别名解析和部分 `getattr` 间接访问识别，但没有证据证明它普遍优于论文的 LLM judge；两者覆盖域不同。

本地独立可信 reference、私有继承管道、nonce、协议 token，以及候选不可见的参考源码/输出，是应保留的隔离实现增量。所查官方公开 driver 在同一进程载入 reference 和 candidate，但独立参考准备/IPC 的概念本来就在论文中，因此这是相对公开实现的落实与加固，不是超越论文的新隔离思想。Trace 条数不符或异常退出会拒绝该次结果，但也可能来自崩溃或不完整运行，不能直接认定已经证明伪造。这里的 Python guard 和协作式隔离都不是对任意恶意原生代码的安全证明。[官方 driver 加载顺序](https://github.com/NVIDIA/SOL-ExecBench/blob/a9fa0804c793d438e70850c33fe34426e66d53dd/src/sol_execbench/driver/templates/eval_driver.py#L168)

来源：[官方 reward_hack.py](https://github.com/NVIDIA/SOL-ExecBench/blob/a9fa0804c793d438e70850c33fe34426e66d53dd/src/sol_execbench/core/bench/reward_hack.py)、[官方 shifting allocator](https://github.com/NVIDIA/SOL-ExecBench/blob/a9fa0804c793d438e70850c33fe34426e66d53dd/src/sol_execbench/core/bench/io.py#L559)；本地 [runtime.py](../../src/sol_execbench/core/bench/reward_hack/runtime.py)、[static_review.py](../../src/sol_execbench/core/bench/reward_hack/static_review.py)、[evaluation_orchestrator.py](../../src/sol_execbench/driver/templates/evaluation_orchestrator.py)、[评测结果检查](../../src/sol_execbench/cli/evaluation/runtime.py)。

### 2.2 数值、语义验证与复用失效规则更系统

评测端的 `NumericCheck` / `CodeDistanceCheck` 强制 matched ratio 为 1，并有 exact、联合 tie-aware TopK ID/权重合同。官方通用容差默认 matched ratio 为 0.99，但允许 workload 覆盖，因此不能说官方所有问题都接受 1% 错误。原论文已讨论量化等专用比较；本地增量是具体的闭合合同和更严格的全元素要求。

SOLAR 端的正式转换强制验证，并检查输入执行后状态、输出/输入别名关系，以及策略启用时的梯度；复用绑定 Python/Torch 版本和本项目语义解释器源码摘要。图、access、参考源码、验证策略及完整文件清单也一并绑定。改变硬件假想参数不会继承原有发布资格。

官方最新 SOLAR 同样有实质性的 strict receipts v5，绑定 source、DSL、prepared reference、依赖文件、shape、dtype 和 checker。精确差别在于其 `strict=False` 旧入口仍可运行未数值验证的分析，mechanical 路径也不等于数值比较；本地正式路径将更广的语义检查和复用失效规则作为强制合同。两边都不是全输入形式等价证明，本地也没有为所有第三方二进制做完整环境证明。

来源：[官方 ToleranceSpec](https://github.com/NVIDIA/SOL-ExecBench/blob/a9fa0804c793d438e70850c33fe34426e66d53dd/src/sol_execbench/core/data/workload.py#L81)、[官方 receipts](https://github.com/NVlabs/SOLAR/blob/62dab75349315c0844d8fd646631be598270672d/solar/translate/receipts.py#L4)、[官方验证合同](https://github.com/NVlabs/SOLAR/blob/62dab75349315c0844d8fd646631be598270672d/docs/SOLAR_V2_CONTRACTS.md#L84)；本地 [workload.py](../../src/sol_execbench/core/data/workload.py)、[verify.py](../../src/solar/verification/verify.py)、[execution.py](../../src/solar/verification/execution.py)、[verified_program.py](../../src/solar/pipeline/verified_program.py)、[reuse_projection.py](../../src/solar/pipeline/reuse_projection.py)。

限制：全元素通过仍不能代替高辨错输入和高精度 oracle。本地仍有 norm weight=1/bias=0 启发式和 FP32 比较路径，不能据此宣称数值压力覆盖完整。[input_heuristics.py](../../src/sol_execbench/core/bench/input_heuristics.py)、[correctness.py](../../src/sol_execbench/core/bench/correctness.py)

### 2.3 三视图和有限容量分析比当前公开实现更完整

当前官方公开 `perf_model.py` 为三个标签都采用全图 compute/memory 的最大值；headline compute 仍只计 Tensor Core 部分，scalar/other 工作量主要作为信息输出。`graph_analyzer.py` 将 fused 和 fused-prefetched 的流量设为相同。其 Timeloop 命令是导出接口；本次检查的源码树没有对应的 Orojenesis 执行与资格管道。官方指南也明确列出这些实现限制。

本地则分别计算：

- unfused：逐算子 `max(compute_time, memory_time)` 后求和；
- ideal-fused：全图工作量按共享资源模式累加时间，再与必要 I/O 组合；
- cache-aware：消费已审计的容量证据，覆盖不完整时返回 unavailable。

本地还把固有工作量与资源归属、容量证据、速率计价分开，提供 `ReuseSession`、必要资源查询和有预算、可恢复的参数扫描。资源名称或吞吐常数替换属于移植；多资源组合、真实容量证据消费及视图适用性区分则超出了参数替换。

**这属于相对当前公开代码的实现推进。fusion、cache-aware、跨平台重投影、inverse roofline 本来就在论文中，不能宣称是本地提出的新方法。** 本地容量分析也只覆盖已支持的层、链和区域，不是任意 DAG 的最优缓存调度。

本地对 mapper 的可复现构建、SHA-256 allowlist 和 provenance 检查，以及外层发布强制 `capacity_constrained_tile_aware_v1`，进一步收紧了工具和发布资格。但是“roofline 估计升级为形式化下界”这个概括不准确：论文 Eq. 1 本来就是其模型假设下的下界，本地 Python API 仍接受 `roofline_eq1_v1` 并赋予 `sol_score_eligible`。严格发布资格、数值回放、静态计量推导和工具身份核验是不同的保证；组合起来不等于全程序形式证明。[评分合同](../SCORING-V3.md)、[当前 bound 合同](../../src/solar/contracts.py)、[mapper 身份验证](../../src/solar/analysis/orojenesis/identity.py)

来源：[官方 perf_model](https://github.com/NVlabs/SOLAR/blob/62dab75349315c0844d8fd646631be598270672d/solar/perf/perf_model.py#L1317)、[官方流量组合](https://github.com/NVlabs/SOLAR/blob/62dab75349315c0844d8fd646631be598270672d/solar/analysis/graph_analyzer.py#L3701)、[官方限制说明](https://github.com/NVlabs/SOLAR/blob/62dab75349315c0844d8fd646631be598270672d/docs/SOL_GUIDE.md#L570)、[Timeloop 导出](https://github.com/NVlabs/SOLAR/blob/62dab75349315c0844d8fd646631be598270672d/solar/cli/totimeloop.py#L16)；本地 [views.py](../../src/solar/analysis/views.py)、[orojenesis_evidence.py](../../src/solar/analysis/orojenesis_evidence.py)、[reuse.py](../../src/solar/reuse.py)、[inverse.py](../../src/solar/analysis/inverse.py)、[explore.py](../../src/solar/explore.py)。

### 2.4 将访问/受限计算从宽泛折扣收紧为带边界的集合分析

访问区域并集算法本身来自官方实现；本地 `access_regions.py` 明确保留归属。增量是结构化 storage/base/view 身份、预算，以及 exact / lower_bound / unavailable 的分离；超预算不会将 bounding box 冒充必要读取。

官方当前已有结构稀疏和 backward `needed_density` 传播，不能说本地首次引入反向需求。其部分分析采用标量密度估计和折扣；本地在受支持操作上枚举有预算的坐标/集合，合并多个消费者的实际需求，显式处理 effects、copy、动态索引边界。受限计算还要求调用方声明有限值、无溢出和忽略 signed-zero 的数值域；保留 dense 结果，将 restricted 结果单独输出，且禁止其直接取得分数/发布权威。

这可以称为更可审查、适用域更明确的受限分析，不能称为更广泛的稀疏支持，也不能声称调用方的数值域声明已经被证明。

来源：[官方稀疏/需求传播](https://github.com/NVlabs/SOLAR/blob/62dab75349315c0844d8fd646631be598270672d/solar/analysis/graph_analyzer.py#L357)、[官方密度折扣](https://github.com/NVlabs/SOLAR/blob/62dab75349315c0844d8fd646631be598270672d/solar/analysis/graph_analyzer.py#L2465)；本地 [access_regions.py](../../src/solar/analysis/access_regions.py)、[external_reads.py](../../src/solar/analysis/external_reads.py)、[demand.py](../../src/solar/analysis/demand.py)、[demand_contracts.py](../../src/solar/analysis/demand_contracts.py)。

### 2.5 从公式扩展到内容绑定的资格与发布链

原论文已有 SOL Score 和 reference/baseline 的区分。本地增加 corpus、baseline、candidate、SOLAR 和源码身份的联合核验，以及 static→canary→full 资格链；只有通过绑定检查后才评分。对于 baseline≤SOL 或 candidate<SOL，返回需要审计的错误，不静默给分。

这是公开可复现的发布/评测工程增量，而不是新评分公式。哈希只能证明内容一致，不能独自证明发布者身份。结论也只针对检查到的公开实现，不推断 NVIDIA 非公开服务没有相应措施。

来源：[官方公开评分函数](https://github.com/NVIDIA/SOL-ExecBench/blob/a9fa0804c793d438e70850c33fe34426e66d53dd/src/sol_execbench/sol_score.py)；本地 [formula.py](../../src/sol_execbench/core/scoring/formula.py)、[release_verifier.py](../../src/sol_execbench/core/scoring/release_verifier.py)、[release_solar_qualification.py](../../src/sol_execbench/core/scoring/release_solar_qualification.py)。

### 2.6 新增跨硬件研究协议和因果诊断层

LLM_CORE 将问题语义、生成规则和具体 workload 分开：容量归入固定 class，以 distribution/cohort 标识会改变生成语义的条件，同一 scale 生成完整九槽或整题跳过。跨硬件协议区分硬件、容量、配置变化，隐藏具体 holdout，固定 candidate portability digest，报告全目标/共同支持集、missingness、分布漂移和按 Definition 分组的置信区间。

这超出原版固定题集的评测范围；它没有证明跨硬件 workload 同分布，也尚未建立多硬件实证结论。目前该协议不消费 SOL 输入，也没有综合总分。

另外，本地诊断层把经验证的事件边接入实测 dispatch 拓扑和性能模型，禁止仅凭时间邻近推断跨 stream 因果，并把诊断结果与正式 timing/score 分离。bootstrap 区间、held-out 模型接受/拒绝和行动质量门，也超出单一 SOL 下界/分数的职责。这里的增量是证据与推断协议，不是 rocprof 对位替换本身；不能据此声称预测已更准确。

来源：本地 [workload_generation.py](../../src/sol_execbench/core/dataset/workload_generation.py)、[generalization/workflow.py](../../src/sol_execbench/core/generalization/workflow.py)、[metrics.py](../../src/sol_execbench/core/generalization/metrics.py)、[跨硬件协议](../user/HARDWARE-GENERALIZATION.md)、[schedule_evidence.py](../../src/sol_execbench/core/bench/performance_model/schedule_evidence.py)、[schedule_feed.py](../../src/sol_execbench/core/bench/performance_model/schedule_feed.py)、[acceptance.py](../../src/sol_execbench/core/bench/performance_model/acceptance.py)。

### 2.7 GPU 所有权和恢复责任是独立的工程增量

本地以 UUID 绑定 L1 ownership / L2 execution，持有所有权直到子进程清理、设备同步、时钟恢复和最终回执完成；容器借用不转移恢复责任，未核实的清理/恢复会阻止后续 owner。SIGTERM/SIGINT 转为清理路径。这比替换 clock API 更进一步，属于共享设备的执行生命周期协议。

应比较的是所有权、借用、恢复责任和持久化失败状态，而不是把上游缩减成“只有一个开关”。官方已有 GPU/DRAM 锁频、读回验证、失败回滚和退出 cleanup trap。该本地协议要求参与者协作，不对可改写共享目录的恶意进程提供安全隔离，也不保证 SIGKILL 后自动恢复。[官方锁频](https://github.com/NVIDIA/SOL-ExecBench/blob/a9fa0804c793d438e70850c33fe34426e66d53dd/src/sol_execbench/core/bench/clock_lock.py#L70)、[官方退出清理](https://github.com/NVIDIA/SOL-ExecBench/blob/a9fa0804c793d438e70850c33fe34426e66d53dd/docker/entrypoint.sh#L29)

来源：[gpu_lock.py](../../src/sol_execbench/core/bench/gpu_lock.py)、[gpu_execution.py](../../src/sol_execbench/core/bench/gpu_execution.py)、[clock_ownership.py](../../src/sol_execbench/core/bench/clock_ownership.py)、[termination.py](../../src/sol_execbench/core/process/termination.py)。

## 3. 尚未吸收或仅部分吸收的 SOLAR 实现

| 官方能力 | 本地状态 | 准确的差距 |
| --- | --- | --- |
| agentic source→executable Affine Loop IR→deterministic einsum | 未吸收 | 缺少 `einsum_lib` 的 Tensor/Dim/kernel、Numba 执行、生成—验证—修正，以及 DSL AST lifting 的完整前端 |
| JAX/Flax 等语言适配和源码转换工作流 | 未吸收 | 本地只有两条固定 PyTorch 路径；官方有相关 agentic 适配、fixture 和工具，但不代表任意程序/所有入口已稳定支持 |
| 多输入、高维 einsum 的 binary schedule 动态规划 | 未吸收该通用算法 | 官方支持 3–8 输入、简单任意 rank 标签的 subset DP；本地搜索限于有序、纯、定形二维矩阵链 |
| estimate 多档模式及有界转换编排 | 未吸收 | 官方有 auto/v0/v1/v2，以及保留失败、限制次数/时间/API/费用的 dispatcher；本地固定选择一条 IR 路径 |
| 用户可直接调用的 Timeloop 导出 | 未吸收该独立入口 | 本地有内部问题序列化和 mapper 执行，但没有同等独立导出命令；这不是分析核心缺失 |
| 更广的 builtin / sampled-density / 稀疏传播 | 部分吸收、部分有意采用不同约束 | 本地是经过审查的结构零和集合需求子集；不能把动态密度观察直接提升为形式下界证据 |
| 论文整体覆盖和实证范围 | 未复现 | 本地 163 workload 的资格回执不能替代 KernelBench、JAX/Flax、机器人和论文算法设计实验 |

### 3.1 前端是最大的架构差距

官方目前公开的 [einsum_lib/core.py](https://github.com/NVlabs/SOLAR/blob/62dab75349315c0844d8fd646631be598270672d/solar/einsum_lib/core.py)、[agent.py](https://github.com/NVlabs/SOLAR/blob/62dab75349315c0844d8fd646631be598270672d/solar/einsum_lib/agent.py)、[einsum_dsl/einsum_converter.py](https://github.com/NVlabs/SOLAR/blob/62dab75349315c0844d8fd646631be598270672d/solar/einsum_dsl/einsum_converter.py) 和 [v2 合同](https://github.com/NVlabs/SOLAR/blob/62dab75349315c0844d8fd646631be598270672d/docs/SOLAR_V2_CONTRACTS.md#L10) 构成这一前端。

本地仍是 torchview→native extended-einsum 和 make_fx→ATen。`graph_expander.py` 对 `enable_agent=True` 明确抛错。已有可执行 native IR 不等于论文的 Affine Loop IR；`ShapeDomain` 是维度约束与逐实例成员判定，不是程序循环、访存和归约的表示；FakeTensor 加速取图也没有补上这个前端。

这是一项有记录的范围选择：stage0 C11 归后续 P3；路线图 L3-d 默认延后，要求覆盖矩阵先显示两条现有路径的系统性缺口。163/163 make_fx readiness 可以解释当前冻结语料的优先级判断，但不能证明前端缺口对新增语言、程序或研究目标都无影响。此次合并结论不改变该实施优先级，也不授权前端开发。

`einsum_dsl` 的缺失属于同一条前端/语义编译缺口，不应解释成“已把等价能力全部外包给 mapper”：DSL 编译将可执行循环和索引归约为语义图；Timeloop/Orojenesis 在已有问题表示上搜索 mapping、tiling 和容量流量。两者处在不同阶段，本地也在 Python 中自行完成语义工作量和访问计量。

来源：[graph_expander.py](../../src/solar/ir/extended_einsum/torchview/graph_expander.py)、[make_fx_backend.py](../../src/solar/graph/make_fx_backend.py)、[shapes/domain.py](../../src/solar/shapes/domain.py)。

### 3.2 收缩搜索有“验证更强、代数域更窄”的交叉关系

官方 [`optimal_dense_binary_schedule`](https://github.com/NVlabs/SOLAR/blob/62dab75349315c0844d8fd646631be598270672d/solar/einsum/ops/base.py#L370) 为 3–8 输入的 dense einsum 做子集动态规划，考虑中间张量尺寸并最小化 `2*MACs + scalar_multiplies`。它输出工作量/调度估计，不是经数值验证的完整程序变换，更不能自动等同于通用 MLA 优化器。[官方解释](https://github.com/NVlabs/SOLAR/blob/62dab75349315c0844d8fd646631be598270672d/docs/EINSUM_GUIDE.md#L214)

本地 [order_domain.py](../../src/solar/analysis/order_domain.py) 只接受不同源操作数、无 effect/alias/shared DAG 的二维矩阵链；[order.py](../../src/solar/order.py) 则对候选重新提取并相对原 reference 验证，保留失败候选、完整分母和预算状态。因此“已实现 search-orders”不表示已经吸收官方的高维多输入算法；反过来，官方的 DP 也不包含本地同等的验证工作流。

官方自动读取邻近 `einsum_graph_reordered.yaml` 只是加载行为，不构成存在完整图重排生产器的证据；本地坚持精确绑定显式选择的图，不应为追求“全部吸收”而复制这种隐式替换。[官方加载代码](https://github.com/NVlabs/SOLAR/blob/62dab75349315c0844d8fd646631be598270672d/solar/analysis/graph_analyzer.py#L1737)

### 3.3 较小的工作流缺口和架构相关模式

官方 [estimate.py](https://github.com/NVlabs/SOLAR/blob/62dab75349315c0844d8fd646631be598270672d/solar/estimate.py#L239) 的 auto 先尝试 mechanical v0，在指定不支持错误时转 v1 API 路径；v2 接受 coding-agent 提供的 IR。另有 [frontend/dispatcher.py](https://github.com/NVlabs/SOLAR/blob/62dab75349315c0844d8fd646631be598270672d/solar/frontend/dispatcher.py#L4) 的有界编排，保留失败并约束尝试、时间、API 和费用。本地未吸收这种编排能力，但不能称上游是无条件通用编译器。自动回退与 fail-closed 并非逻辑矛盾；关键仍是验证、身份、失败保留和结果选择规则。

官方 [totimeloop 命令](https://github.com/NVlabs/SOLAR/blob/62dab75349315c0844d8fd646631be598270672d/solar/cli/totimeloop.py#L16) 将 einsum YAML 导出为 Timeloop YAML。本地 [problem.py](../../src/solar/analysis/orojenesis/problem.py) 有内部序列化，但没有对应独立用户命令。这是接口差异，既不说明本地不能运行 mapper，也不说明官方导出即取得了容量验证。

当前官方 [arch_rates.py](https://github.com/NVlabs/SOLAR/blob/62dab75349315c0844d8fd646631be598270672d/solar/perf/arch_rates.py#L16) 和 [B200 profile](https://github.com/NVlabs/SOLAR/blob/62dab75349315c0844d8fd646631be598270672d/configs/arch/B200.yaml#L44) 提供 dense/sparse peak 选择；本地资源档案按 dtype/累加模式提供单一速率，没有对应的 2:4 资格和双峰值计价模式。它与结构零/反向需求是不同层次，也部分属于架构政策，故不列为排除生态移植后的主要算法缺口。`no_sparse_hardware_throughput_or_kernel_speedup_claim` 仅限定结论，不能单凭该声明推断实现是否存在。

### 3.4 另有 ExecBench 侧未复现的流程

本地没有实现原论文从生产模型自动提取/策划题目的完整流程；当前来源是 pinned AKA 和独立冻结的 LLM_CORE 规则。候选静态审查采用确定性 AST 规则，没有接入论文的 LLM judge。这应列为 ExecBench 的流程差异，不能与 SOLAR 的 agentic IR 转换合并成一个“LLM 缺口”。此项对照的是论文，检查到的 NVIDIA 公开仓库也没有定位出完整 LLM judge 本体，不能假定已有可直接移植的模块。确定性审查的可复现性是取舍，不构成其检测范围更广的证明。[评测合同](../user/EVALUATOR-CONTRACT.md)、[静态审查实现](../../src/sol_execbench/core/bench/reward_hack/static_review.py)

## 4. 已经吸收，不能再列为整体缺失

- 访问区域并集：已吸收，并增加 typed storage/view 绑定和预算；
- unfused / ideal-fused / cache-aware：已有独立计算和可用性状态；
- 固有计数复用和硬件重投影：已有公共 API 和严格失效规则；
- inverse roofline 与参数扫描：已有受限视图/固定容量求解和恢复机制；
- 结构零/反向需求与收缩顺序探索：已有受限实现，但不是官方全部支持域。

这些能力见上述源码，不以旧计划文档中的“未实现”状态为准。也不能反过来把入口存在理解为全域支持：本地 unfused 反向求解仍显式 unsupported，形状域不是全域定理，容量覆盖不是任意 DAG 最优解。这些是本地边界，不都对应一个可直接搬入的上游模块。

多架构问题也需拆开：本地已有假想 profile 重投影和扫描，正式硬件资格仍很窄；上游拥有多个架构 YAML 并不等于每个架构都完成同等级实机验证。本题排除了对位生态移植，因此产品档案数量不作为主要算法差距。

此外，拒绝把 `repr(SymBool)` 当持久化可执行协议，是针对 PyTorch `ShapesSpec` 序列化语义的自有合同选择，不能列成“拒绝吸收官方 SOLAR 的形状协议”。[原始来源核查](solar-upgrade-official-source-check-2026-10-11.md)

## 5. 现有证据允许与不允许的结论

- make_fx 的 163/163 readiness 绑定既有资格链版本；external-reads 清单 128 条已解除阻塞，其中 127 条完成完整回放、1 条大型卷积有内存受限回放缺口。不能将 readiness、静态计量证明、完整 mapper/GPU 验证混成一个指标。[HANDSOFF](../../HANDSOFF.md)
- 新旧 make_fx 图在本地 163 个 workload 上逐字节相同，试点提取阶段有 11–196× 加速；没有与官方当前 Affine 前端做同任务基准，不能据此说整体比官方 SOLAR 快。
- 43 题/163 workload 的 AKA 集和 36 个 LLM_CORE 定义，不能据不同分母声称覆盖超过原版 235 题；本地正式分数是 eager reference 自评的管线结果，不能据此比较优化能力。[README](../../README.md)、[RELEASE/README](../../RELEASE/README.md)
- 诊断 start460/start640 曾被拒绝，start700 处于暂停且未完成生产接受；因果供数接通不等于总体预测准确度通过。[HANDSOFF](../../HANDSOFF.md)
- 本次未跑任何新的测试或硬件实验。这里的“更进一步”指可由源码确认的能力/协议差异，不是已复现的普遍性能或科学优越性。

本仓库更适合定位为：在较受限的程序域上强化执行验证、分析适用性和证据流转，并扩展 kernel-agent 评测研究范围的系统。若要覆盖独立 SOLAR 的主要架构贡献，仍需处理 agentic Affine Loop 前端及其语言适配；若只补分析后端，最具体的缺口之一是高维多输入 einsum 调度算法及其与本地验证合同的整合。
