# E1 / C1：诊断执行器与覆盖审计接入的 CPU 验收

日期：2026-11-08。规划锚点仍为 `8641b9dc78d00250ba1cf75d5e9434f9839e5ef4`；
本次实现位于其后的工作树改动，尚未形成 B1 执行提交。
依据 [E0](solar-m0-judge-preregistration-2026-11-07.md)、
[C0](solar-m0-llm-core-coverage-spec-2026-11-07.md) 和
[迭代计划](solar-next-iteration-plan-2026-11-07.md)。动态任务状态见
[HANDSOFF](../../HANDSOFF.md#solar-evolution)。三个 M0 文件字节保持冻结。

本次范围为 E1/C1 实现和 CPU 检查。没有运行 GPU 工作负载、真实 mapper 或容量探针；
没有生成 E0 的 h0 密钥/commitment/张量或真实 LLM_CORE target view；没有改变正式评分契约、
原 workload checks、默认 IR backend 或 native-op registry。历史 GPU 资格不随这些改动转移。

## 1. E1 实现与边界

入口：`sol-execbench diagnostics judge-development`。

| 接缝 | 实现及约束 |
| --- | --- |
| 开发分母 | [contracts.py](../../src/sol_execbench/core/bench/judge/contracts.py) 固定三任务、w0–w2、12 个输入族、d0/d1，72 个输入 / 288 个候选单元；读取原题前核对六个原始文件 SHA；h0 在分配/写出前拒绝 |
| 输入 | [inputs.py](../../src/sol_execbench/core/bench/judge/inputs.py) 复用 `derive_custom_input_seed`；CPU Torch Generator 产生整数网格，明确实际 BF16/FP32 量化。保存量化字节身份、见证列、G1 指数及适用的排列/符号；工作量界限先于分配 |
| 对照 | [candidates/](../../src/sol_execbench/core/bench/judge/candidates/) 提供十二个真实 Python 源文件，P0/P1/P2/P3 干预保持 E0 定义；经既有静态审查，原数值结果复用 `compare_output_checks` |
| oracle | [binary_oracle.py](../../src/sol_execbench/core/bench/judge/binary_oracle.py) 使用证明不溢出的 int64 或 Python 大整数，直接舍入到目标格式；[interval_oracle.py](../../src/sol_execbench/core/bench/judge/interval_oracle.py) 使用精确有理均值/方差及 80→160 位有向区间，缓存有界 |
| 独立核对 | [oracle_checks.py](../../src/sol_execbench/core/bench/judge/oracle_checks.py) 对每个开发输入的预先固定角点/中点和声明见证列做独立 120 位 Decimal 核对；这是固定点交叉核对，不冒充另一套全元素复算。主 oracle 仍覆盖全输出 |
| 原 evaluator 接入 | [staging.py](../../src/sol_execbench/core/bench/judge/staging.py) 写出独立诊断 Definition/Workload/BenchmarkConfig 与 Solution，保留原 checks；可信 factory 消费 reference service 的调用级私有种子。三个任务均通过现有 ReferenceService 的 CPU 接入检查 |
| 隔离与停止 | [executor.py](../../src/sol_execbench/core/bench/judge/executor.py) 每个开发输入使用独立 CPU 子进程，120s 上限、进程组清理、文件日志、保守地址空间上限 `min(16 GiB, physical/2)`；超时/错误停止后，未到达行仍保留在分母中；执行中源漂移使活动不完整 |
| 报告 | [report.py](../../src/sol_execbench/core/bench/judge/report.py) 分列静态、原 numeric、oracle 误差、强诊断、见证认证和原容差余量；保存 checks、源/运行时/输入/候选身份及实际数组，`performance_authority=false` |

本次 CPU 路径调用的是现行数值比较器，并未运行完整生产 evaluator。报告固定记录
`evaluator_status=not_run_cpu_diagnostic`、`gpu_runtime_integrity=not_run`。
原 numeric pass 不得重命名为 admission/runtime/timing 完整通过；更强诊断失败也不等于作弊。
每个开发输入只做一次 CPU 诊断，尚无 E2 的三进程重复、留出结果或成本配对。

开发期间保留了一次初稿运行 `dev-initial/`。随后核查发现 G2 见证内维的非见证列
仍被随机填充；最终实现将这两行的非见证位置清零，落实“除见证外支持域不相交”及明确结构零。
增加零位置断言，并保存原始初稿结果；不将初稿拼进最终统计。该修正发生在任何 h0 生成之前，
没有改候选、容差、输入族分母或 E0 的机制定义。

## 2. CPU 证据和负结果

证据根：`data/outputs/solar-e1-c1-cpu-20261108/`（未提交）。

最终开发诊断报告及摘要在本文件的“最终留证”段登记。原始数组采用无 pickle 的 NPZ，
保留实际输入、reference、四个候选输出、全输出精确值/区间、贡献绝对值尺度和构造信息。
CPU 总墙钟只用于诊断预算记录，不能替代 E2 的编译、reference/IPC、检查和附加 oracle 成本实验。

初稿和最终运行均发现 LayerNorm 的 P3 与 CPU reference 在 L0/L1 上存在原容差之外的分歧。
最终报告保留逐例原容差余量、oracle/reference 误差、P3/oracle 误差及强诊断结果。
依 E0，这类项不能仅凭“高精度”标签进入合法对照零误拒统计；不得更换种子、删除输入族、
放宽正式容差或直接宣称 GPU 裁判误拒。B1 应登记解释口径，E2 再以实际 reference 与 oracle 界定标签。

## 3. C1 接入与身份

入口：`sol-execbench solar audit-target-view`，默认只读校验；`--execute` 才会调用 worker。
本次没有调用执行入口，也没有拿历史/合成元数据冒充当前设备的实测 target view。

1. [target_view.py](../../src/sol_execbench/core/dataset/target_view.py) 读取现有视图，校验当前 schema、
   source/rule/Definition/workload 摘要、路径范围、完整 decisions、九槽和 UUID 唯一性、
   coverage、hardware/capacity/cohort/distribution 及 canonical view digest。
   生成与读取共用 [corpus.py](../../src/sol_execbench/core/dataset/corpus.py) 的纯身份计算；
   [workload_generation.py](../../src/sol_execbench/core/dataset/workload_generation.py) 的 owner
   复核既有元数据与规则、资源和 checks 一致，不启动生成搜索、容量分配或输入实例化。
2. C0 公共入口固定 source/target 身份、core20/180/360 分母。测试中的玩具视图只能进入内部测试接缝，
   不能通过该入口。报告先建全部行，先 40 smoke，再剩余320；两路径分别记录，jobs=1。
3. [target_coverage.py](../../src/sol_execbench/core/solar_bridge/target_coverage.py) 复用
   `SolarStageAuditRequest → run_solar_stage_worker`，仅 ready 才构造
   `SolarWorkerRequest → run_solar_worker`。没有新增数值转换器或 roofline fallback。
4. 发现旧 stage worker 引用一个缺少 native GPU 生命周期封装的重复入口；现统一调用
   [analyzer.py](../../src/sol_execbench/core/solar_bridge/analyzer.py) 的既有受保护入口，并移除重复实现。
   CPU 检查验证连接和行为，本次未宣称该改动获得真机资格。
5. [coverage_resume.py](../../src/sol_execbench/core/solar_bridge/coverage_resume.py) 要求完整执行源码、
   Python/Torch 语义运行时、view 文件和全部生成身份、policy、mapper provenance/threads、预算完全相同；
   每个工件重新核对 SHA，ready/formal 复用重走现有 receipt/manifest 验证器。
   单行多次尝试有独立 attempt 结构，本版不自动重试；来源变化另开活动。
   遗留 `interrupted_in_progress` 因无法证明已耗预算而拒绝自动恢复。
6. 原始 outcome、reason、stage、工件及分析 metadata 保留；unknown reason 不默认为 unsupported。
   数值/梯度仅投影 receipt 实际字段，alias/mutation 无单独观测时记 `not_observable`；
   external reads、计数和三视图保留正式分析来源，不能由 ready 推算。

成本观测是可选 sidecar，正式 worker IPC/result 和评分 schema 不变：
[observation.py](../../src/solar/observation.py) 记录提取、转换、验证、formal 及每次 mapper 调用墙钟；
[memory_observation.py](../../src/solar/memory_observation.py) 给出 10ms 间隔的进程 RSS 采样峰值、
进程高水位、已初始化设备的 PyTorch allocator 峰值。RSS 采样是下界，不是精确峰值；
allocator 峰值不含直接驱动分配。不可观察字段保留 null 和原因，不补0。

[resource_guard.py](../../src/sol_execbench/core/process/resource_guard.py) 对研究 worker 进程组
（包含 mapper 子进程）监控 RSS 和活动证据量；达到上限先保存原因再终止进程组。
墙钟由原 runner 的有界进程组控制；OOM 没有配额证据时仍属 infrastructure_error。
预算/基础设施失败阻断 smoke 扩展，剩余行保持 not_reached。报告区分矩阵完整性和成本是否实际观察到。
这些观测与资源保护没有在真实 GPU/mapper 活动中验证，须由 B2/C3 验收。

## 4. 定向验证范围

CPU 检查使用 `-n 0`，并在汇总运行时隐藏 CUDA/HIP 设备。新 C1 测试仅使用显式标记的合成
元数据、手写小轴和 fake worker；没有调用真实 LLM_CORE 生成入口、分配探针或 mapper。
含有 `holdout` 角色的合成槽只用于验证调度字段和分母，不是 C2 的真实留出实例。

- 169 项定向测试通过：E1 舍入/区间/输入边界、ReferenceService、子进程与失败保留；C1 reader、
  360 行角色分母、ready/formal 阻断、resume、scope/hash/旧 schema 拒绝；已有 worker/readiness/receipt、
  SOLAR API/管线及 schema 检查回归。命令范围与完整输出见 `cpu-tests.log` 及执行清单。
- Ruff check、全树 format check、ty、coupling、readability、schema、production reachability、
  dataclass policy 检查通过。未提高任何质量基线。
- 新增独立研究 artifact family 归属 judge/bridge domain registry；聚合表和 schema 审计同步注册。
  原评分器、release workflow、Workload/NumericCheck 等正式契约没有改动。

## 5. 使用与后续门禁

只运行开发 CPU 诊断（所有 72 个输入；输出目录必须不存在）：

```bash
uv run sol-execbench diagnostics judge-development --repository . --output data/outputs/judge-dev
```

加 `--case-index 4` 可做显式子集检查，报告不会宣称完整开发分母；加 `--stage-only`
仅准备独立问题/候选文件。h0 不在该入口的合法选择中；E2 的 commitment 和实例化必须另行启动。
E2 可消费已经冻结的纯种子输入构造接缝，不能借启用留出修改构造规则。

已有合格视图的只读检查（本次没有这样的新视图，因此只验证了合成边界）：

```bash
uv run sol-execbench solar audit-target-view \
  --source-manifest problems/LLM_CORE/releases/LLM_CORE_V2/manifest.yaml \
  --descriptor problems/LLM_CORE/targets/configurations/rx9060xt/standard-16gb.yaml \
  --target-view <existing-view>/target-view-manifest.yaml
```

下一依赖是 B1：审查并固定可复现执行提交、依赖/镜像/mapper/资源预算，纳入本次源码差异和
LayerNorm 分歧的判读；然后才能申请/安排 B2 资格刷新。当前 CPU 检查不关闭 B1/B2，
不授权 E2、C2、C3。大卷积 miss、start700 暂停、L3-d 与 gfx942 的既有边界继续保留。

## 最终留证

最终运行目录为 `data/outputs/solar-e1-c1-cpu-20261108/development-final/`。
`development.json` 的 SHA-256 为
`7e66b09b58aa78fd10feb7d3ec14d410c1f5f83566ba822a253f8a9fdfcea65e`；
报告绑定739个执行源文件，canonical 源清单摘要为
`047b6fc082fd7450ca7d3a8f2555b0b90f63c2ccbd73b915401cc007afd7330d`。
执行前后与复核时源码一致，288个引用工件逐一 SHA 核对一致。
运行时为 Python 3.14.3、Torch 2.14.0+rocm10.1.0、每个子进程1线程、设备固定 CPU。

| 项目 | 最终观测 |
| --- | --- |
| 完成性 | `execution_complete=true`；72个输入 / 288个候选单元；72个不同输入 hash、12个候选源 hash；无缺行或 oracle_uncertain |
| 独立 oracle 自检 | 全72输入的540个预定输出位置通过120位独立核对；主 oracle 全输出留存 |
| 静态检查 | 288行均未被静态规则阻断；仍不代表完整 evaluator 准入/运行时通过 |
| CPU 数值/强诊断分类 | passed 176；original_numeric_failed 65；stronger_guarantee_failed 47 |
| 必检见证 | P1 的 G2/R0/L3 与 P2 的 G2/R2/L0，共36/36单元按预登记方向认证通过 |
| P0 | 72/72原 numeric 通过；其中17项 LayerNorm 未达4 ULP强诊断，保留为 reference 精度边界 |
| P3 | 72/72强诊断通过；61/72原 numeric 通过；另11项保留 oracle/reference discrepancy，不计已确认合法对照误拒 |

11项分歧为：L0的w0/w1/w2各d0/d1（6项），L1的w0、w2各d0/d1和w1-d0（5项）。
其逐例余量和误差都在报告中；见证认证分布为169 true、11 false、108不适用。
其中169是认证单元数，与169项 CPU 测试的同数值没有语义关系。

最终开发运行使用可复现的公开开发 nonce（字节 `e1` 重复32次），仅用于本次离线开发活动；
不作为 E2 私有 nonce、h0 密钥或输入唯一性门禁的完整执行证据。
初稿目录 `dev-initial/` 作为已暴露开发历史保留，不与最终数据合并。
测试日志为证据根下 `cpu-tests.log`；质量检查日志、候选源摘要、开发暴露清单和
工件复核结果汇总于 `execution-manifest.json`。全部属于非权威研究证据，不能用于正式评分。
