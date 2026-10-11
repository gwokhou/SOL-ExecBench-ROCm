# NVIDIA 软件栈升级导致性能退化：公开案例与回测价值

研究日期：2026-10-11。范围：NVIDIA 官方仓库、官方文档、官方论坛与 PyTorch 官方仓库中的第一手记录。本次只做资料研究，没有安装历史软件栈、运行 GPU 实验或独立复现性能数字。背景项目、内部设备标识和实验产物均不披露。

## 结论

存在与既有 AMD 升级调查同类的问题，而且不局限于 GEMM：库版本改变算法分派、代码生成丢失向量化、编译器输入与优化策略发生交互、主机端缓存失效，都有公开记录。最适合作为首个运行回测的是 PyTorch #114911；最适合代码生成回测的是 CUTLASS #3463；最适合检验多因子归因能力的是 CUTLASS #3389。具体算子和 GPU 型号不是筛选核心，**明确的正常／退化版本、干预证据和可复现材料**才是。

本文区分四种证据：原作者测量、维护者确认、公开代码或编译产物、官方修复说明。它们不能相互替代；尤其不能把 issue 关闭解释为底层根因已修复，也不能把只编译的缩减程序当作可执行性能基准。

## 优先候选

| 案例 | 版本变化 | 已公开的关键证据 | 回测用途与边界 |
| --- | --- | --- | --- |
| PyTorch #114911 / cuBLAS | `12.1.0.26` → `12.1.3.1` | 单独降级 cuBLAS 包恢复；缩减到单批次 `bmm`；框架绕过后恢复 | 首选运行回测；可以定位库与分派路径，不能宣称得到 cuBLAS 内部源码 diff |
| CUTLASS #3463 / CuTe DSL | `4.5.3` → `4.7.0`；官方指出回归从 `4.6.0` 引入 | PTX 向量写回变成逐元素写回；维护者确认 `4.6.3`、`4.7.1` 修复 | 首选代码生成回测；公开脚本只编译，运行时退化量仍待独立复现 |
| CUTLASS #3389 / 工具链组合 | 同一 DSL `4.5.2`，内嵌工具链 `12.9.83` → `13.1.66` | PTX 形态、函数名、优化级别的交叉对照；寄存器下降并大量 spill；维护者复现 | 多因子编译归因回测；公开缩减 PTX 不能执行，完整生产复现未公开 |
| cuDNN 算法选择 | `7.6.5` → `8.0.2`，固定 CUDA `10.2` | 缩减脚本、算法日志、autotune 对照；`8.2.1` 官方说明部分层已修复 | 算法选择回测；老环境获取与运行需另行验证 |
| cuBLAS 主机缓存 | CUDA Toolkit `12.0` 引入，`12.0 Update 1` 修复 | 官方明确未使用 cuBLASLt heuristics cache 导致主机开销 | 机制佐证；缺公开最小复现和退化幅度 |
| cuDNN 融合 epilogue | `8.1.0` → `8.1.1`，`8.2.0` 修复 | 官方明确参数化 ReLU 的通用支持造成性能退化 | 机制佐证；缺公开 shape、完整程序和退化幅度 |

表中各行依据见下文对应来源。

## 1. cuBLAS 升级改变执行路径：最适合先做运行回测

PyTorch #114911 的原始 `einsum` 输入为 `(2, 128, 65536)`。报告称某条 cuBLAS `gemv2T_kernel_val` 路径只利用一个 SM，整体慢约 250 倍；在原环境中仅将 `nvidia-cublas-cu12` 从 `12.1.3.1` 降到 `12.1.0.26` 即恢复。这比同时更换整个 PyTorch/CUDA 环境的对照更能支持库版本归因。[原始报告与降级干预](https://github.com/pytorch/pytorch/issues/114911)

维护者把问题缩减为 `m=n=1, k=65535` 的单批次 FP32 `bmm`。A100 上两套框架环境的 `bmm` 为 `14.89 → 209.37 µs`，约 14.1 倍；同形状 `mm` 为 `14.92 → 13.5 µs`。这些不是原始 `einsum` 的 250 倍，不能混为同一个测量。后续原始任务在 PyTorch `2.1.2+cu121` 为 `52402.06 µs`，`2.2.0+cu121` 为 `208.05 µs`。[缩减实验与修复验证](https://github.com/pytorch/pytorch/issues/114911)

PyTorch 的处理是让 batch=1 的路径回退到 `addmm`，绕过退化路径；它不等于公开修复了 cuBLAS 内部算法。[修复 PR #114992](https://github.com/pytorch/pytorch/pull/114992)

**回测目标**：Agent 应检查实际加载的 cuBLAS 包，而不是仅相信 `torch.version.cuda`；提出单独换库和 `bmm/mm` 对照；把结论收敛到库版本及分派路径，并区分框架规避补丁与库内部修复。

## 2. CUTLASS 向量写回退化：代码生成证据非常直接

CuTe DSL `4.5.3 → 4.7.0` 后，报告中的 BF16 tile 写回从 `st.global.v4.b32 × 8` 变为 `st.global.b16 × 64`，同时成对转换变为逐元素转换。作者报告部分受内存限制的内核慢 4–5 倍，但公开缩减脚本使用 fake tensor，仅编译并检查 PTX，未提供这 4–5 倍对应的运行基准。原环境为 GB300，目标 `sm_103a`。[CUTLASS #3463](https://github.com/NVIDIA/cutlass/issues/3463)

维护者在 issue 末尾确认 `4.6.3` 和 `4.7.1` 已修复。官方 `4.8.0` 发布说明进一步明确：这是 `4.6.0` 引入的、动态 stride 张量的 `cute.autovec_copy` 生成逐元素而非向量指令的回归。[维护者确认](https://github.com/NVIDIA/cutlass/issues/3463#issuecomment-5468954863)、[官方发布说明](https://github.com/NVIDIA/cutlass/releases/tag/v4.8.0)

**回测目标**：Agent 应从升级后的 PTX 差异发现向量化丢失，再通过正常、退化、修复版本验证。它与既有 AMD 案例中的写回指令宽度问题在诊断层面相似，但不能据此宣布两者共享源码原因。原对照同时更换了 DSL 与内嵌 NVVM，不能把 NVVM 单独判为根因；本次也未取得具体修复 diff。

## 3. CUTLASS 工具链交互：不能只找一个“坏版本”

CUTLASS #3389 固定 DSL `4.5.2`，更换工具链依赖，内嵌编译工具从 `12.9.83` 变为 `13.1.66`。作者报告生产内核 `1.114 → 1.46 ms`，约增加 31%；寄存器从 168 降至 80，并出现大量 spill。另一个 standalone 测量是 `1.119 → 1.69 ms`，属于不同实验，不能与前者拼成一组数据。[原始报告](https://github.com/NVIDIA/cutlass/issues/3389)

作者的交叉实验指向 **13.1 产生的 PTX 形态 × 函数名包含 `cutlass` × `ptxas -O3`** 的交互；改名、降低优化级别或使用 12.9 产生的 PTX 均能缓解相应异常。NVIDIA 参与者确认了编译复现。公开缩减 PTX 已去掉运行语义，只能用于编译检查；完整生产源码与运行复现私下提供给维护者。[实验矩阵与维护者反馈](https://github.com/NVIDIA/cutlass/issues/3389)

后续 `4.6.2` / `4.7` 提供函数名前缀控制，并修正单独传递 `--ptxas-options=-O2` 的行为。这是可用的缓解与控制接口，不能据 issue 关闭宣称 ptxas 内部原因已修复。[结案说明](https://github.com/NVIDIA/cutlass/issues/3389)

**回测目标**：Agent 应把“编译器版本导致”继续拆成前端产物、后端编译器、优化级别、符号名等变量；区分可执行复现与仅编译复现；在缺少公开运行程序时停止宣称完整性能闭环。

## 4. cuDNN 算法选择：有公开复现，也有官方修复链接

论坛报告固定 CUDA `10.2`、TensorFlow `2.3` 与 GTX 1080 Ti，对比 cuDNN `7.6.5` 和 `8.0.2`。缩减卷积模型在关闭 autotune 时从 `30.2` 降到 `26.9 it/s`，即吞吐下降约 10.9%，换算每次迭代耗时增加约 12.3%。开启 autotune 后分别为 `41.7`、`39.4 it/s`；不能把它写成缩减模型的差距完全消失。[公开脚本、结果与日志](https://forums.developer.nvidia.com/t/cudnn8-regression-in-algorithm-selection-heuristics/153667)

API 日志展示了同层算法变化，例如 FP32 输入 `[1,256,25,38]`、卷积核 `[256,256,3,3]` 的前向卷积由 WINOGRAD 转为 GEMM。作者也指出 TensorFlow 在 cuDNN 8 使用了不同查询 API，这是需要进一步隔离的因素。cuDNN `8.2.1` 的官方修复说明直接链接该论坛，确认其中部分层在 Volta 和 Pascal 上已修复。[原始日志](https://forums.developer.nvidia.com/t/cudnn8-regression-in-algorithm-selection-heuristics/153667)、[cuDNN 8.2.1 修复记录](https://archive.docs.nvidia.com/deeplearning/cudnn/archives/cudnn-890/release-notes/index.html)

**回测目标**：识别启发式选择与最佳可用算法之间的差异，并检查框架查询 API 的变化。官方确认的是相关特定层，不能扩展为所有模型或所有平台完全恢复。

## 5. 两项官方机制记录

**主机缓存失效。** CUDA Toolkit `12.0 Update 1` 的 cuBLAS 修复记录明确说明：`12.0` 开始未使用 cuBLASLt heuristics cache，导致额外主机端开销；更新版降低了这部分开销。资料没有给出 shape、硬件和具体幅度。它直接说明“GPU 算子升级变慢”也可能来自 CPU 端算法选择工作，不能仅看 kernel 时间。[cuBLAS 12.0 Update 1 发布说明](https://docs.nvidia.com/cuda/archive/12.1.0/cuda-toolkit-release-notes/index.html#cublas-release-12-0-update-1)

**融合尾处理扩展。** cuDNN 官方明确记载 `8.1.1` 相对 `8.1.0` 的卷积融合 ReLU epilogue 退化，原因是推广对参数化 ReLU 的支持，`8.2.0` 已修复。未公开 shape、幅度及源码差异。它证明“增加通用功能使原有路径变慢”这一机制确实发生过，但还不是现成运行回测。[cuDNN 官方修复记录](https://archive.docs.nvidia.com/deeplearning/cudnn/archives/cudnn-890/release-notes/index.html)

## 回测集如何使用这些材料

建议按下面的顺序建设，避免只有若干 issue 摘要而没有可评估任务。

1. **运行回测**：先封装 #114911；固定驱动、框架和 GPU，保存实际加载库与包哈希，再对照正常／退化 cuBLAS。另设框架补丁对照，检查正确性和性能恢复。
2. **代码生成回测**：用 #3463 的公开编译脚本，验证正常／退化／修复版本的 PTX；运行性能是单独的验收项，需要增加真正执行并验证输出的程序。
3. **交互归因回测**：用 #3389 的公开 PTX 矩阵检验编译归因；没有获得可执行最小程序前，评分只覆盖编译资源与代码生成结论。
4. **算法选择回测**：加入 cuDNN 案例，考察查询 API、heuristics 与 autotune 的区分能力。

以上是后续设计建议，均未执行。每项运行测量都应采用两侧一致、可核验的锁频策略，保留完整调用耗时与 GPU 事件耗时，控制 warmup、缓存和编译阶段。公开历史耗时不能直接并入本地性能结果。

若要评价 Agent 的独立归因能力，应只给它初始报错、复现入口和软件版本，将修复说明、根因讨论及后续评论留作评分参考；直接把完整 issue 喂给 Agent，再让它复述答案，只能证明资料总结能力。公开案例仍可能存在模型训练数据泄漏，所以应称为开发回放集，并保留后续新发生、未公开结论的案例用于独立验证。

cuBLAS/cuDNN 的公开接口与发行包可以定位到库版本、触发条件、算法和规避方案，但不应承诺总能获得内部实现的 commit diff。产品应允许在证据到达相应边界时给出明确的部分结论。

## 筛选时排除的近似结果

H100 上 BF16/FP16 Conv3d 比 FP32 慢很多的近期论坛报告虽然标题写着 regression，但没有旧版本对照，只比较数据类型，尚不足以证明“升级引入”。因此没有放进上述升级回测候选。[原始报告](https://forums.developer.nvidia.com/t/cudnn-bug-report-conv3d-performance-regression-with-bfloat16-float16-on-h100/355210)

同理，跨 GPU 型号的性能差距、自研内核优于 cuBLAS、只列出优化收益而没有正常／退化版本的材料，都不能直接作为升级退化的因果案例。

## 尚未结案的近期候选

CUTLASS #3707 于 2026-10-01 提交，查询时仍 open。报告 FA4 block-sparse backward 在 `4.5.3 → 4.6.0.dev0` 后，document mask 从 `1.279 → 2.329 ms`，sliding window 从 `0.858 → 1.045 ms`，约增加 82% 与 22%。提供了复现脚本、九份 PTX 和原始材料归档；固定 ptxas `13.0` 后，spill 仍随前端 PTX 改变。[原始 issue](https://github.com/NVIDIA/cutlass/issues/3707)

NVIDIA 参与者已经复现，并解释为特定 PTX 形态触发 ptxas 的异常变换；给出的缓解包括 O2、移除函数名中的 `cutlass` 前缀及对 FA4 作补丁。后续说明受影响内核的 O2 SASS 与在 O3 下禁用该特定优化策略时一致，但仍建议总体默认 O3。[维护者复现与缓解](https://github.com/NVIDIA/cutlass/issues/3707#issuecomment-6031477648)、[SASS 与优化策略说明](https://github.com/NVIDIA/cutlass/issues/3707#issuecomment-6050971926)

这已经具有维护者确认的机制证据，仍不能称作正式通用修复，或简单把 NVVM 单独判为根因。原对照的 quack/FFI 也改变，同一 b20 源码还需 fmax 兼容补丁；作者进一步做了同源控制，而 `4.8` 栏又涉及 FA4 源码变化，不能当成干净的单变量对照。该案可以检验 Agent 能否继续查询后续评论，发现已确认的机制和缓解，而不滞留在最初报告的猜想上。本次未运行其材料。[实验控制与状态](https://github.com/NVIDIA/cutlass/issues/3707)
