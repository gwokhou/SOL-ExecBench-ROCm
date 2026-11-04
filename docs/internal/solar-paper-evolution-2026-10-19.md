# SOL-ExecBench 与 SOLAR 的论文演化和增量贡献

研究截点：2026 年 10 月 19 日。

这条演化线包含两篇独立论文和一次论文修订：SOL-ExecBench 将 SOLAR 作为基准评测的理论性能分析组件；独立 SOLAR 论文重构了源码到分析表示的前端，并将用途扩展到算法和硬件设计；SOLAR 的 arXiv v2 只补回流程图，没有增加方法或实验。因而，真正需要比较的技术跃迁是 **SOL-ExecBench 中的算子转换流程 → 独立 SOLAR 的可执行 IR 流程**，不能将 v2 当成第三代分析系统。

## 版本与时间

| 本文简称 | 正式论文与版本 | arXiv 标记日期 | 关系 |
| --- | --- | --- | --- |
| 基准内 SOLAR | *SOL-ExecBench: Speed-of-Light Benchmarking for Real-World GPU Kernels Against Hardware Limits*，2603.19173v1 | 2026-03-19 | SOLAR 位于 §4.2，服务于整篇基准论文 |
| 独立 SOLAR v1 | *SOLAR: AI-Powered Speed-of-Light Performance Analysis*，2606.26383v1 | 2026-06-24 | 以分析系统为主体的独立论文 |
| 独立 SOLAR v2 | 同题名，2606.26383v2 | 2026-09-22 | 同一 arXiv 论文的修订稿 |

日期和版本关系见 [SOL-ExecBench v1 元数据](https://arxiv.org/abs/2603.19173v1)、[SOLAR 提交历史](https://arxiv.org/abs/2606.26383)。这里的 v2 是论文版本号，不能据此推断软件发布版本或仓库实现版本。

## SOL-ExecBench 中的 SOLAR

基准论文要回答的是：一个生成的 GPU kernel 距离硬件允许的性能还差多远。SOLAR 提供理论运行时间下界；评测框架先检查正确性，再结合基线、候选 kernel 实测时间与下界计算 SOL score。因此，235 个问题的构建、SOL score 和评测沙箱是 **SOL-ExecBench 的贡献**；其中负责下界推导的部分才是 SOLAR。问题还包含多个具体 shape 的 workloads，问题数与 workload 数不可混用。[SOL-ExecBench v1，§§3、4.2–4.4](https://arxiv.org/html/2603.19173v1)

其流程为：

```text
PyTorch 程序及输入 shape
  → torchview 捕获实际执行路径和算子图
  → 已验证的算子转换函数表
     遇到新算子时，LLM 生成转换函数并用数值比较修正
  → extended einsum 图
  → 结合融合、预取及目标硬件参数推导 SOL
```

此时已经具备以下能力：算子图、扩展 einsum 表示、LLM 自动转换及执行验证、持久化复用已验证转换、图级融合、预取，以及通过 Orojenesis 考虑片上缓冲容量的访存建模。**执行验证、fusion 和 cache-aware analysis 都不是独立论文首次加入。** 核心下界为计算耗时与访存耗时两者的最大值。[SOL-ExecBench v1，§4.2，图 3–4、式 1](https://arxiv.org/html/2603.19173v1#S4.SS2)

从研究定位看，这一阶段的增量是将自动化、可检验的 SOL 推导接入真实 kernel 基准，使评测能衡量剩余性能空间，并将低于估计下界的时间作为审查信号。它不意味着所有下界都具有形式证明；论文已经指出，仅按 shape 分析会遗漏压缩、常量传播等值相关优化。[SOL-ExecBench v1，§§4.2–4.3](https://arxiv.org/html/2603.19173v1#S4.SS2)

## 独立 SOLAR v1 的主要增量

### 从转换算子扩展到翻译源码

独立论文将前端改为由 LLM 把源程序翻译成可执行的 **Affine Loop IR**，先比较原程序与 IR 的输出，再由确定性后端提取 einsum 图。LLM 的工作对象从算子转换函数变为整个源程序到统一 IR 的翻译；确定性后端负责从 IR 提取分析所需结构。[SOLAR v1，§§3.1–3.2](https://arxiv.org/html/2606.26383v1#S3)

```text
PyTorch 或 JAX 源程序
  → LLM 生成可执行 Affine Loop IR
  → 运行原程序和 IR，比较输出，失败则反馈修正
  → 确定性提取 extended einsum 图
  → unfused、fused、cache-aware 分析和设计探索
```

这里的技术工作不只是增加提示词。IR 把受约束的仿射循环 kernel 与可组合的顶层程序分开，配套 named dimensions、dtype 等元数据、Numba 执行支持和 builtins；后端追踪 IR 调用并机械提取维度、归约和数据依赖。语言迁移需要新增模型捕获模块与源语言适配提示，而不是逐个实现完整前端。论文实际展示的跨语言扩展是 JAX/Flax。[SOLAR v1，§§3.1.1–3.2.2](https://arxiv.org/html/2606.26383v1#S3.SS1)

我的判断是，这次主要架构增量在于 **统一可执行 IR 和验证位置的改变**：生成部分有可运行的验证对象，确定性分析有更统一的输入。这是对原有自动转换思想的系统化和泛化，不能简化成“旧版完全靠 LLM 猜 FLOPs，新版才开始验证”。

### 将分析工具发展为算法与硬件设计工具

| 维度 | 基准内 SOLAR 已有 | 独立论文新增或强化的工作 |
| --- | --- | --- |
| 前端 | PyTorch 算子图及算子转换函数表 | 共享可执行 IR；源码翻译；PyTorch 与 JAX/Flax |
| 验证 | 模拟转换后的 einsum，与原算子输出比较 | 比较原程序与 IR 输出，作为后端入口条件 |
| 分析表示 | extended einsum 图 | 系统定义 IR 到图的确定性提取；导出 YAML 供下游工具使用 |
| 融合与缓存 | 已有融合、预取和 Orojenesis | 明确组织为 unfused、fused、cache-aware 三种分析模式并做对比 |
| 算法设计 | 主要服务 kernel 评测 | 用 einsum contraction 顺序探索 MLA 的计算量变化 |
| 硬件设计 | 输入目标硬件参数推导 SOL | 参数扫描、跨平台投影、由延迟目标反求吞吐与带宽需求 |
| 实证范围 | SOL-ExecBench 真实 kernel 基准 | KernelBench、JAX/Flax、机器人模型，以及 SOL-ExecBench 附录分析 |

左列依据 [SOL-ExecBench v1，§4.2](https://arxiv.org/html/2603.19173v1#S4.SS2)；右列依据 [SOLAR v1，§§3–4、附录 A–C](https://arxiv.org/html/2606.26383v1)。表中的“新增”指相对于前一篇论文新增的描述或实验，不代表已经独立审计软件实现的首次出现时间。

Inverse roofline 采用 `所需吞吐 = FLOPs / 目标延迟`、`所需带宽 = Traffic / 目标延迟`。其意义在于把“给定硬件，最快多快”反过来用于“给定延迟，至少需要什么硬件”；论文没有因此改变 roofline 的基本数学模型。[SOLAR v1，§3.3、§4.3](https://arxiv.org/html/2606.26383v1#S3.SS3)

### 新增实验与结论应如何理解

独立论文将覆盖率和分析用途展开为系统实验：KernelBench 评测采用 270 个问题的子集；JAX/Flax 包含 8 个程序；主文展示 3 个机器人模型例子，附录 C 列 7 个配置；附录 B 分析 SOL-ExecBench 的 3,957 个 workloads。[SOLAR v1，§4、附录 A–C](https://arxiv.org/html/2606.26383v1#S4)

几个代表性结果需要区分“分析发现”与“已经实现的加速”：

- KernelBench L3 的 54.6 倍是 PyTorch eager 时间与 fused SOL 的几何平均比值，表示估计的优化空间。
- DeepSeek MLA 的 2.04 倍来自收缩顺序变化后的 FLOPs 减少，不是实测 kernel 加速。
- 机器人 500 Hz 案例中的硬件需求来自 inverse roofline，是模型约束下的必要资源估计，不是部署可达性的充分证明。

这些结果均已出现在六月 v1，不是九月 v2 的新增实验。[SOLAR v1，§§4.1–4.3](https://arxiv.org/html/2606.26383v1#S4)

## SOLAR v2 的实际变化

对两版 arXiv 源码归档逐文件比较，两版各包含 27 个文件，26 个文件字节一致，唯一变化是 `sections/3_method.tex`：将已有但被注释掉的三阶段 TikZ 流程图恢复为可见图，删除误入的 `h` 字符。实验章节、附录、参考文献与所有独立图形资产没有变化。[SOLAR v1 源码](https://arxiv.org/src/2606.26383v1)、[SOLAR v2 源码](https://arxiv.org/src/2606.26383v2)

PDF 第 4 页也验证了这个变化：v1 在 Methodology 第一段显示未解析的图引用 `??`，下一行有孤立的 `h`；v2 补上 Figure 2 和图注，引用恢复正常。后续图号由此顺延，原图 2 变为图 3，原图 12 变为图 13；两版均为 14 页。[SOLAR v1 PDF，第 4 页](https://arxiv.org/pdf/2606.26383v1)、[SOLAR v2 PDF，第 4 页](https://arxiv.org/pdf/2606.26383v2)

因此，v2 的增量是 **修复论文呈现和交叉引用**，不是增加分析算法、IR、语言、数据集、实验或结论。把三阶段 IR 流程归功于 v2，会把六月已经公开的方法错误后移到九月。

## 方法能力与证据边界

独立论文明确承认，有限输入上的数值比较不构成形式等价证明；错误翻译可能恰好通过样本验证。因而“validated”宜理解为通过其数值检查，“zero observed SOL violations”是所报告实验中的观察，不是所有程序和硬件上的可靠性定理。[SOLAR v1，摘要、§3.1.2](https://arxiv.org/html/2606.26383v1#S3.SS1.SSS2)

两阶段都保留了仅依据 shape、无法完整利用数值结构的限制。unfused、fused 和 cache-aware 还采用不同的融合与存储假设，不能把三者当作同一可行实现集合上必然逐级改善的实际时间预测。比如融合减少中间数据搬运，会降低理想运行时间；有限容量使数据重读，则可能提高这个下界。它们分别回答不同优化约束下的性能空间问题。[SOL-ExecBench v1，§4.2](https://arxiv.org/html/2603.19173v1#S4.SS2)、[SOLAR v1，§3.3](https://arxiv.org/html/2606.26383v1#S3.SS3)

从研究增量看，独立 SOLAR 的价值在于把已有的图级 SOL 分析发展为更通用的源码分析工具，并补充跨语言、算法选择和硬件需求的实证。v2 则保持这一研究内容，只修复流程图的缺失。
