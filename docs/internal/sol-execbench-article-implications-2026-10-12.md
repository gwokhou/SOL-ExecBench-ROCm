# SOL-ExecBench 榜史文章对 ROCm 项目的启示

日期：2026-10-12。分析对象：NeuralTalk《无问芯穹、Databricks、Qwen3.8 都登过顶：SOL-ExecBench 半年榜史与含金量拆解》（2026-10-11）。正文依据用户提供的原始长截图，已分段阅读全文。原微信网页跳转到受限制的验证码页面，未继续访问。

本地源码基准：`e35d046ee9c64c310ff3be41fe0a78b00bb993b3` 加分析时已有未提交改动。本文是研究和源码审计，不是修复报告；未修改生产实现、题目或现有开发文件，未运行 GPU 实验、提交或发布。工作区仍有并行开发，结论应绑定所引用的代码状态。

## 结论

最值得投入的增量，是把本项目发展成能够回答“候选是否满足声明的语义、性能改善能否复现、SOL 下界在哪些条件下有效”的 AMD 评估基础设施。原生 ROCm 执行与分析是基础价值；可复现的验证器反例、测量可靠性和固定预算下的反馈有效性，可以成为更有说服力的研究贡献。

文章不能证明本项目已经失去价值，也不能证明任意榜单第一没有研究价值。排名本身不足以建立贡献；新的验证方法、可靠的实验协议和能复现的经验规律仍可构成研究成果。

## 原始资料核验

文末 11 个编号去重后是 7 个来源：5 篇论文/文章、一个榜单、一个仓库。推荐阅读卡片不属于该引用列表。已阅读五篇正文及相关仓库材料；榜单 URL 的抓取没有可解析正文，未把截图中的 10 月 10 日排名当作本次实时核验结果。

| 来源 | 对本文有用的证据及边界 |
| --- | --- |
| [NVIDIA 原论文](https://arxiv.org/html/2603.19173v1) | 区分功能 reference、评分 baseline 和 SOL 下界；承认 shape-only 分析的值依赖限制。论文自己已包含部分审计思想，不能都算本项目首创。 |
| [doubleAI WarpSpeed](https://www.doubleai.com/research/warpspeed-approaches-speed-of-light-on-blackwell) | 具体展示输入分布、种子和参数覆盖不足；embedding 梯度案例是在合成 Zipf 语料的小模型训练中暴露问题。不是对全部 235 个 kernel 的真实生产部署验证。 |
| [Recursive](https://www.recursive.com/articles/first-steps-toward-automated-ai-research) | 描述提案、实验、验证和跨任务复用；成绩是作者报告，不能仅凭最终均分归因于某个机制。 |
| [SF Tensor 复盘](https://sf-tensor.com/news/sol-execbench) | 65% 实指 152/235 题前两名相差不超过 0.01 SOL，不是最优性证明。噪声、预算和模型行为主要是作者观察，未在本项目重现。 |
| [RSI 综述 §5.6](https://arxiv.org/html/2609.11873v2) 与 [Hyra 官方结果库](https://github.com/Tencent-Hunyuan/Hyra-results/) | 综述讨论 Hyra 经验管理；官方库将 0.771 标为截至 2026-07-10 的结果。公开 SOL 目录是 10 个展示提交，不能当成全量重放包。 |
| [FlashInfer-Bench 榜单](https://research.nvidia.com/benchmarks/sol-execbench/leaderboard/collection/4/B200) | 子榜与总榜、日期与评测栈必须分开。此次未验证实时排名。 |
| [NVIDIA 仓库](https://github.com/NVIDIA/SOL-ExecBench) | 进一步阅读了具体问题报告，避免仅用公众号的概括推断当前代码。 |

需收紧的叙述：

- 第二名接近第一名可以提示区分度下降，也可能是共享算法、相似搜索或测量条件所致，不能直接推导“物理上没有优化空间”。
- 0.01 SOL 分差与 5% 延迟波动量纲不同，必须通过相同 workload 的评分函数和联合测量分析比较，不能直接比大小。
- “5 月 Reddit 实测”所列引用实际指向 doubleAI 博文。本文采用可核验的博文实验范围，不扩大为全量生产事故。
- [Issue #26](https://github.com/NVIDIA/SOL-ExecBench/issues/26) 只有提交长时间处于 Running 的报告，不能据此确定服务器拥堵原因，更不是正确性漏洞证据。
- “Databricks 约 7 万美元”在截图中是未给出可定位原帖的估算；本文不将它用作预算依据。模型数据污染的行为猜测也不升级为既定事实。

## 1. 优先审计测试数据的辨错能力

### 一个已经定位到本项目的具体盲区

上游 [Issue #25](https://github.com/NVIDIA/SOL-ExecBench/issues/25) 报告：norm 的常数权重/偏置会隐藏参数误用。本地仍存在对应路径：

1. [input_heuristics.py](../../src/sol_execbench/core/bench/input_heuristics.py) 的 `_is_norm_weight` / `_is_norm_bias` 根据名称匹配；`_generate_heuristic_tensor` 分别返回全 1 和全 0。
2. [input_generation.py](../../src/sol_execbench/core/bench/input_generation.py) 对普通 `random` 输入先走这个启发式。显式 `generated` 会走另一条声明式路径。
3. [gpumode_positionwise_ffn/workload.jsonl](../../problems/AMD_AKA/torch2hip/gpumode_positionwise_ffn/workload.jsonl) 的四个 workload 将 `layer_norm_weight` / `layer_norm_bias` 声明为 `random`，其 dtype 是 FP32，会命中上述分支。
4. [reference.py](../../problems/AMD_AKA/torch2hip/gpumode_positionwise_ffn/reference.py) 实际使用这两个参数；[manifest.yaml](../../problems/AMD_AKA/manifest.yaml) 将这道题标为 `scored`。
5. [test_io.py](../../tests/sol_execbench/core/bench/test_io.py) 仍有断言权重全 1、bias 全 0 的测试。

**源码级结论：这个输入族不能在数值上区分正确的 affine 与省略 affine。** 其他输入变动会让整组输入 checksum 改变，所以重复输入检查也不能代替逐参数的语义覆盖。这里没有执行错误候选，不能声称已经完成 CLI/GPU 绕过复现，也没有判定任何历史提交利用了这个盲区。

轻量清点读取所有 authored `problems/AMD_AKA/*/*` JSON，并运行源码中两个纯字符串名称谓词：169 个 workload；426 个 random 输入字段、82 个 generated 字段、105 个 scalar 字段、12 个 custom 字段。上述常数启发式命中 1 道题的 4 个 workload、每个 2 个字段。此清点包含非计分用途的 authored workload，不是发布的 163-workload 分母；没有运行生成器或全量 materialization。

当前清点的 167 个 numeric check 均要求 `required_matched_ratio=1.0`。因此不能直接把上游“放过 1% 元素”的批评照搬到这里。全元素落在容差内，仍不代表输入覆盖足以区分错误算法。

### RMSNorm 是另一个值得优先做的敏感性实验

[l1n36_rmsnorm](../../problems/AMD_AKA/torch2hip/l1n36_rmsnorm/workload.jsonl) 当前用标准随机输入、固定 `eps=1e-5`，`atol=rtol=0.0125`。[上游 Issue #21](https://github.com/NVIDIA/SOL-ExecBench/issues/21) 给出小幅值输入能放大 epsilon 错误的实例。

对常数行，实数算术 `x / sqrt(x² + eps)` 的独立计算为：

| 行内数值 x | 正确 eps=1e-5 | 错误 eps=1e-6 | 绝对差 | 本地容差公式允许值 |
| --- | --- | --- | --- | --- |
| 1 | 0.999995 | 0.9999995 | 0.0000045 | 约 0.025 |
| 0.001 | 0.301511 | 0.707107 | 0.405595 | 约 0.016269 |

这是数学敏感性示例，不是 BF16 执行或完整 checker 测试。它说明有针对性的输入变化比单纯增加同分布随机轮数更有价值。

### 建议方向

建立小型“验证器辨错基准”：为每类语义设计已知错误变体，检查是否拒绝，同时用正确实现检查误拒率。优先覆盖 affine 省略/错索引、epsilon 错误、归约少算、累积精度下降、mask 误用。条件允许时保存最小反例、准确复现命令和输入/候选身份。

实现落点是已有 workload/check/generator 合约。项目已支持 normal、uniform、integer、Bernoulli、constant、simplex 等显式生成器，不必先新建输入框架。容差按运算和误差机制解释；不统一收紧或放宽容差。

修改输入分布会改变测试与可能的计时语义。正式应用需更新 checksum 绑定及发布版本，并重新检查 baseline / candidate / SOL 适用性；不将新规则直接套给旧分数。

## 2. 已有防作弊增量值得保留，但与语义覆盖分开

本次源码核查确认已有：

- [reference_worker.py](../../src/sol_execbench/driver/templates/reference_worker.py) 不导入候选；[ReferenceService](../../src/sol_execbench/core/bench/reference_service.py) 持有 32-byte nonce、参考输出及待验证状态。
- [eval_timing.py](../../src/sol_execbench/core/bench/eval_timing.py) 在每次 warmup / timed invocation 请求新输入、检查 trial 内重复值，并将输出送回可信服务验证；验证不计入候选 event 窗口。
- [formula.py](../../src/sol_execbench/core/scoring/formula.py) 对候选快于 SOL 抛出审计错误，不截成满分。
- 发布验证链绑定 corpus、baseline、candidate 与 SOLAR 工件。

这些设计直接回应上游 [#15 缓存状态](https://github.com/NVIDIA/SOL-ExecBench/issues/15) 与 [#19 计时调用不验证](https://github.com/NVIDIA/SOL-ExecBench/issues/19) 的问题类别。它们不能保证全语义域正确，也不是对恶意原生代码完全隔离的证明。本轮未重跑相关测试。

## 3. 把测量不确定性接到候选选择上

项目不是没有误差条：[timing_evidence.py](../../src/sol_execbench/core/bench/performance_model/timing_evidence.py) 已保存 trial 内原始样本，并构造分层 bootstrap 95% 区间；目前属于诊断 sidecar。正式评分函数与发布聚合不等于已经提供了两候选的配对显著性判定。

建议在既有采样基础上补齐比较协议：冻结候选后独立复测；同设备分块交错 A/B 顺序；绑定相同输入族、时钟、缓存及环境；对配对差值或比值给区间，并预先定义最小有用收益。对“未能分辨差别”和“已证明在给定界限内等效”使用不同结论。把重复搜索中选出的最佳单次成绩当成待验证候选，处理择优造成的偏差。

现有公式 `S = 1 / (1 + (T_k - T_SOL)/(T_b - T_SOL))` 依赖 baseline 和 bound。0.8 不是“使用了 80% 峰值算力”，也不是两种硬件上可直接比较的绝对效率。正式报告应同时保留实际延迟、baseline 身份、SOL 下界、正确性覆盖和分数。

## 4. SOLAR 继续承担条件明确的下界分析

“低于下界”要同时调查语义少算、计时遗漏和下界假设不成立。拒绝给分是正确的审计入口，不能直接归因为作弊。

下界工件应能解释计算量/访存量来自什么语义，哪些 precision、shape、值依赖、alias、mutation 和容量假设成立。对于数据依赖路由或稀疏性，不能把上一次观测到的访问量默认为所有输入都必须付出的成本。当前两条固定 IR 路径及 fail-closed 行为是基础；动态形状应沿[已有升级分析](solar-upgrade-opportunities-2026-10-11.md)推进明确域和实例化。

保持包边界：`solar` 输出下界及适用条件；真实测量、profiler 因果关系、优化建议属于 `sol_execbench`。当前 HIP scope 开发改善的是有限场景下的诊断归属，不能自动证明所有候选的计时范围完整或获得跨 lane 性能预测资格，见[生产 scope 验收](evaluate-hip-scope-acceptance.md)。

## 5. 证明反馈帮助 Agent，而非仅证明能输出反馈

[agent-feedback](../user/agent_feedback_sidecar.md)、profile-summary 和 decision sidecar 已是合适接口。值得验证的贡献是：在相同模型和预算下，这些反馈是否减少无效尝试、缩短获得可靠收益的时间，并提高未参与开发的任务上的成功率。

最小对照可设为同一个模型/提示主体/工具权限/搜索预算，唯一差异是有无结构化诊断反馈；固定 token 上限和 GPU 时间上限，记录实际用量。开发任务用于调整反馈，留出任务冻结后评估；多次独立搜索，统一最终验证器。指标包括首次正确候选成本、达到预定可信收益的成本、held-out 正确率和未分辨性能差异比例。

如果研究经验库，再单独比较无经验、文本经验、绑定代码和失败证据的经验，避免在一个对照中同时更换模型、调度器、验证器和预算。候选生成侧可以探索，裁判规则在一轮比较中必须冻结。

AMD 特征知识应带目标 ISA、工具链和已验证用法，不能只注入“提高 occupancy”一类通用建议。当前资源、指令和 profile 证据可作为输入，但是否提高优化效率仍需上述对照。

## 6. 扩展覆盖时先说明承诺的支持域

固定 shape 的合法专门化与宣称可替代一般算子的实现不同。合同只要求一组 shape 时，域外失败本身不自动构成作弊；若宣称是 drop-in 替代，则必须覆盖额外合法 shape、参数和运行语义。

推荐把现有固定计分集合保持可复现，再逐步增加独立的鲁棒性与使用场景验收。优先选择已有 norm、reduction、softmax、融合 GEMM 任务中的邻近/边界 shape 与输入分布；只在原语义允许时加入 stride、alias、并发、graph replay 等维度。梯度长期效应和跨流行为需要专门协议，不能由单次逐元素比较背书。

当前 AMD AKA 计分分母是 43 题、163 workload，不等于 B200 的 235 题。跨硬件比较应使用共同语义集合，并单列不支持项；不靠静默筛掉难题改善均分。已有[发布记录](../../RELEASE/README.md)是旧 ROCm 7.2 环境下 eager reference 自评约 0.5，证明发布链可工作，不能与外部优化候选约 0.8 比能力，也不能授权当前 ROCm 10.1 工作区的性能声明。

## 建议执行顺序与最小验证设计（未执行）

| 顺序 | 最小工作 | 验收问题 |
| --- | --- | --- |
| P0 | positionwise FFN 的 affine 敏感性审计：四个既有 workload，正确参考与省略/误用 affine 的变体，比较旧输入族与声明式非退化输入族 | 新数据能否拒绝已知语义错误，同时接受正确实现？记录旧数据能否漏检，不预设 CLI 绕过结果。 |
| P0 | RMSNorm 的 epsilon 敏感性：三个既有 shape，单位幅值、小幅值和混合幅值族，正确/错误 epsilon 对照 | 在既定精度与容差下，能否稳定区分语义错误和正常数值差异？ |
| P1 | 冻结少量正确候选，复用原始样本保存，执行独立交错比较 | 分差是否超过本机实际变异和预定有用阈值？如无差异，保留不确定结论。 |
| P1 | 固定模型、预算、任务划分的反馈消融 | sidecar 是否带来可重复的搜索效率或结果改善？ |
| P2 | 按需求增加 shape/domain、梯度、异步或另一硬件验证 | 新增支持域是否有独立正确性、计时和下界适用性证据？ |

P0 可先做有界 CPU 语义/输入敏感性验证；实际 BF16/FP32 GPU 路径、端到端 evaluator 和性能影响需单独验证。先完成小样本再决定扩大，不默认重复全量历史采集。本文没有执行这些 GPU 实验。

## 本轮验证范围

完成截图分段阅读、参考来源追溯、当前源码检查、authored JSON 清点和实数 RMSNorm 敏感性计算。上述结果支持一个明确的输入覆盖盲区及有优先级的改进建议，不支持“全部验证器已审计”“所有候选安全”或“已有优化收益”等更强声明。
