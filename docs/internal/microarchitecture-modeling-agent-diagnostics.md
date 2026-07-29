# 从形式化 Roofline 下界到架构感知诊断

## 摘要

论文
[Microbenchmark-Driven Analytical Performance Modeling Across Modern GPU
Architectures（arXiv:2605.04178v1）](https://arxiv.org/html/2605.04178v1)
的核心启发，不是简单地用一个更复杂的公式替换 Roofline，而是建立三层
性能认知：

```text
形式化下界 T_SOL
        ≤ 可达到的最优执行时间

架构感知预测 T_pred
        ≈ 某个具体实现的预期执行时间

实测时间 T_k
        = 最终性能事实
```

对 SOL-ExecBench-ROCm 来说，最重要的是保持 `T_SOL` 的可审计下界性质，
同时新增微基准驱动的诊断预测层。这样既不破坏正式计分，又可以区分下界
松弛、微架构模型缺项，以及编译器和工作量计数偏差，并将这些归因转化为
GPU Kernel Agent 可执行的优化反馈。

## 一、论文对本项目的启发与建议

### 1.1 应当如何理解论文结果

论文报告朴素 Roofline 在现代 GPU 上可能超过 95% 的执行时间预测误差，
而架构感知模型在 B200 和 MI300A 微基准上取得了显著更低的误差。

不过，这个结论需要结合三点理解：

1. 经典 Roofline 更接近理想性能上界或时间下界，并非精确 runtime
   predictor。用它与实际执行时间比较，出现较大偏差本来就在预期之内。
2. MI300A 最亮眼的约 0.09% MAE 使用了按 workload/tile 校准的乘数；
   未校准模型约为 5–8% MAE。
3. 模型跨平台迁移到 H200、MI250X 后，应用级误差明显升高。论文还发现，
   从源码推导的 FLOP/Byte 与 profiler 统计最多可相差约 1000 倍。

因此，现代 GPU 性能误差通常来自三个独立层次：

- 形式化下界没有表达 launch、cache、occupancy 等实际损耗；
- 微架构预测模型没有表达访问模式、调度和流水线；
- workload characterization 与编译后实际执行不一致。

### 1.2 本项目当前所处的位置

项目现有 SOLAR 已不再是纯粹的单一 FLOP Roofline：

- 计算侧按 MFMA、VALU、SFU、reduction 等资源分别统计时间；
- 独立资源允许重叠，最终取最长资源时间；
- Orojenesis 提供容量约束和 tile-aware 的正式下界；
- 架构 profile 已记录 LDS、L1、L2、L3、VRAM 等层次；
- 静态 evidence、Decision sidecar 和 Profile Summary sidecar 已具备诊断基础。

但正式内存时间仍主要使用：

```text
T_memory = audited_bytes / single_global_bandwidth
```

相关实现位于 `src/solar/analysis/formal_analysis.py`。RX 9060 XT profile
虽然已有内存层次容量，但目前只有 VRAM 带宽，见
`src/solar/rocm/profiles/RX_9060_XT.yaml`。

### 1.3 核心建议：正式下界与诊断预测分离

建议维护两条独立模型路径：

| 模型 | 目的 | 允许使用的数据 |
| --- | --- | --- |
| `T_SOL` | 计分所需的形式化下界 | 理论峰值、不可避免运算和流量、容量证明、Orojenesis |
| `T_pred` | 预测具体实现的实际时间 | 持续带宽、launch 延迟、cache、occupancy、ISA、profiler、经验校准 |

不能为了提高预测准确率，将实测持续吞吐、平均 launch latency、候选实现
occupancy 或经验乘数直接放进 `T_SOL`。项目计分契约要求：

```text
T_k >= T_SOL
```

否则属于审计失败，见 `docs/SCORING-V3.md`。

### 1.4 建议建立 gfx1200 专用微基准参数包

论文研究的是 CDNA3/MI300A，而项目正式目标是 gfx1200/RDNA4，不能直接
复制其公式。尤其项目已经明确：RDNA4 动态寄存器分配使静态 occupancy
推导不可靠，相关限制见
`src/sol_execbench/core/bench/decision/derivation.py`。

gfx1200 参数包建议覆盖：

- kernel launch、barrier、同步延迟；
- L1/L2/L3/VRAM 延迟和持续带宽；
- working-set size sweep；
- copy、transpose、stride、gather、atomic 等访问类型；
- VALU/WMMA 按 precision 和 tile 的持续吞吐；
- LDS 带宽及 bank conflict；
- 动态 VGPR、spill、active waves 与 tile size；
- compute/memory overlap。

参数证据应绑定 GPU 身份、ROCm/编译器版本、时钟和功耗策略，并采用
tuning/held-out 分离。

### 1.5 同时保存三种工作量描述

每个 workload 应尽量保留：

1. SOLAR IR 推导的语义运算量和不可避免流量；
2. 编译后 ISA 的指令、资源和 footprint；
3. profiler 观察到的 dispatch、流量、cache 和执行计数。

三者差异应形成显式的 `characterization_gap`，不能通过修改硬件吞吐参数
将差异悄悄吸收到模型中。

## 二、可行的诊断落地例子

### 2.1 通用判别方法

为每个 workload 计算：

- `T_pred(IR)`：使用语义图推导的 FLOP、Byte 和资源计数；
- `T_pred(HW)`：使用实际 dispatch、ISA 和 profiler 计数；
- `T_measured`：实际执行时间；
- `T_frontier`：当前已知最优执行时间；
- `T_SOL`：正式下界。

可以构造三个指标：

```text
L = T_frontier / T_SOL
C = T_pred(HW) / T_pred(IR)
R = T_measured / T_pred(HW)
```

其中：

- `L` 表示形式化下界松弛程度；
- `C` 表示语义工作量与硬件实际工作量之间的 characterization gap；
- `R` 表示排除计数差异后的微架构模型残差。

基本判别规则是：

- `L` 大、`C≈1`、`R≈1`：`T_SOL` 合法但比较松；
- `C` 明显偏离 1：编译器分解、融合、padding 或计数口径问题；
- `C≈1`、`R` 明显偏离 1：微架构模型缺项。

### 2.2 小 Sigmoid：识别 launch-bound 下界松弛

`problems/AMD_AKA/torch2hip/gpumode_sigmoid` 的 `w1` 只有 65,536 个
FP32 元素。最少一次读和一次写约 0.5 MiB，按 320 GB/s 标称带宽
计算，纯内存时间下界约为 1.64 微秒。

如果观察到：

- 只有一个 kernel dispatch；
- 实际读写量接近最低流量；
- 不同小尺寸上的 `T_measured - T_SOL` 近似固定；
- 大尺寸上固定差值的相对占比下降；

则主要是 launch 和未饱和固定成本。`T_SOL` 较松，但不是形式化错误。

如果实际出现乘法、sigmoid、再乘法三个 kernel，则是编译器未融合。
该 workload 的参考表达式确实包含这三个逻辑操作。

### 2.3 Transpose：识别访问模式模型缺项

`problems/AMD_AKA/torch2hip/gpumode_transpose` 的 `w1` 和 `w2`
都包含 262,144 个 FP32 元素，因此最低读写字节数相同。

但其布局分别为：

- `8×16×32×64`；
- `2×64×128×16`。

如果实际流量都接近一读一写，但有效带宽差异显著，则单带宽模型缺少：

- coalescing；
- cache-line 利用率；
- transaction 数；
- tile 形状；
- LDS bank conflict。

如果其中一个产生额外 buffer 或额外 dispatch，则应归因于编译器或
library transpose 算法，而不是带宽参数。

### 2.4 RMSNorm：识别 reduction、LDS 和 barrier

`problems/AMD_AKA/torch2hip/l1n36_rmsnorm` 的 `w1` 是 `64×1024`，
`w2` 是 `512×128`。二者都有 65,536 个元素，基础流量相近，但
reduction 宽度相差 8 倍。

可能的诊断结果：

- 单 kernel、流量相近，但时间差与 LDS/barrier stall 相关：模型缺少
  reduction tree 和 wave 调度；
- 一个尺寸生成 partial reduction + finalize，另一个为单 kernel：
  编译器 lowering 差异；
- 出现 scratch spill：代码生成或寄存器分配问题；
- IR 假定一次读入，而实现重复读取完整行：候选实现流量与理论流量
  不一致。

### 2.5 Irregular Matmul：区分模型与编译器路径

`problems/AMD_AKA/torch2hip/l1n8_matmul_irregular` 包含
`1823×781×511` 等非 tile 整数倍形状。

可按以下方式归因：

- ISA 含预期 WMMA，但误差随 `M/N/K mod tile_size` 增大：模型缺少
  边缘 tile 和 wave 利用率；
- ISA 没有 WMMA，转而使用 VALU 或 FP32 conversion：编译器/lowering
  问题；
- profiler 的 issued operations 接近向上取整后的 tile 数，并显著
  大于 `2MNK`：有效 FLOP 与硬件执行 FLOP 口径不同；
- `T_pred(HW)` 能预测实测而 `T_pred(IR)` 不能：
  characterization/compiler gap。

这些 padding 运算通常不应进入 `T_SOL`，因为理想实现未必必须采用相同
padding 策略。

### 2.6 Cross Entropy：区分长 reduction 与多阶段实现

`problems/AMD_AKA/torch2hip/l1n95_cross_entropy` 的 `w3` 是
`16384×8192`，`w4` 是 `8192×16384`。两者 logits 数量相同，
但 reduction 宽度不同。

- 单 fused kernel、流量相近，但 `C=16384` 更慢：模型缺少长 reduction
  的 occupancy、同步和 cache 行为；
- 一个尺寸 fused、另一个产生 max、exp/sum、gather、final reduce：
  库算法选择或编译器融合问题；
- SOLAR 只计一次 logits 读取，而 profiler 显示多次扫描：正式下界
  可能仍合理，但候选预测必须使用实际多阶段流量。

### 2.7 MiniGPT Block：检查复杂图的工作量计数

`problems/AMD_AKA/torch2hip/l3n44_mingpt_block` 同时包含 LayerNorm、
QKV projection、causal attention、残差和 MLP。

其中：

- `B=32,S=256`；
- `B=16,S=512`。

二者 `B×S` 相同，因此线性 projection 工作量接近；但 attention score
的 `B×S²` 增加约一倍。

如果 attention 子层计数没有随 `B×S²` 增长，应检查：

- shape propagation；
- causal mask；
- broadcast；
- attention FLOP/Byte accounting。

如果 IR 计数正确，但 profiler 显示大量 reshape、contiguous 和 mask
materialization，则是 eager/compiler 分解。如果已经采用 fused attention，
但预测仍随 `S` 系统性偏离，则模型可能缺少 softmax reduction、LDS 容量
或 cache working-set 转折。

## 三、诊断落地对 GPU Kernel Agent 的实际意义

### 3.1 核心价值是性能信用分配

GPU Kernel Agent 通常只能看到：

```text
修改代码 → 编译 → 运行 → runtime/score 改变
```

但它不知道变化来自：

- kernel 设计；
- 编译器是否生成预期 ISA；
- library 是否切换算法；
- profiler/IR 计数口径；
- 性能模型误差；
- 不可达的形式化目标。

完成诊断归因后，反馈可以变成：

| 归因 | Agent 下一步 |
| --- | --- |
| `T_SOL` 松、launch-bound | 停止局部调参，尝试融合或结束搜索 |
| 未生成预期 WMMA | 修复 dtype、layout、alignment 或 intrinsic lowering |
| 实际流量高于理论流量 | 消除临时 tensor、contiguous 和重复扫描 |
| 流量正常但 transaction 异常 | 优化 coalescing、tile 和 vector width |
| LDS/barrier 主导 | 修改 reduction tree 和 wave 分工 |
| 模型证据不足或矛盾 | 重采 counter，不立即生成新代码 |
| 模型与计数均可信 | 将残差归因于实际 kernel 设计 |

### 3.2 避免追逐不可达目标

对于 launch-bound 小 kernel，Agent 可能持续尝试：

- 更大 block；
- 更多 vectorization；
- 更高 occupancy；
- 更激进近似；
- 更复杂的寄存器缓存。

但 runtime 已经接近实际 launch floor。识别出下界松弛后，可以改用：

- empirical frontier；
- 同类空 kernel 的 launch floor；
- 多轮 improvement 小于噪声阈值；

作为停止条件，把 GPU 时间和搜索预算留给真正有提升空间的 workload。

### 3.3 避免错误诊断诱导性能回归

如果模型错误地把 transpose 判定为“已达到显存带宽”，Agent 可能放弃
LDS tiled transpose；如果错误地把 kernel 判为 VGPR pressure，Agent
可能缩小 tile，反而降低数据复用。

可归因诊断能够告诉 Agent：

- 当前结论是否由硬件 counter 支持；
- 是容量、流量还是访问效率问题；
- 哪类优化变量值得继续搜索。

### 3.4 更快解决编译器没有兑现设计的问题

Agent 生成了合理的 WMMA kernel，不代表编译器最终一定生成 WMMA。

ISA 诊断可以产生直接反馈：

```text
expected: FP16 WMMA
observed: FP32 VALU + conversion
action: fix dtype/layout/lowering before tile search
```

这能避免 Agent 在错误的硬件路径上继续调整 block size、prefetch 和
occupancy。

### 3.5 改善训练数据和经验复用

未经归因的训练记录只有：

```text
candidate B is 38% faster than candidate A
```

归因后可以记录：

```text
change: use rocWMMA intrinsic
compiler effect: WMMA instruction appeared
traffic effect: unchanged
runtime effect: -38%
confidence: high
```

后者更适合用于：

- 强化学习 reward；
- preference pair；
- 成功轨迹蒸馏；
- kernel pattern memory；
- 跨 workload 优化规则；
- backend/tool routing。

它还能防止将编译器偶然行为、计数口径变化或模型误差错误归功于无关代码
修改。

### 3.6 提升固定搜索预算下的端到端效果

诊断体系不会凭空提高硬件峰值，也不能替代实际 runtime。其现实收益主要
是：

1. 减少无效候选和无意义 profiling；
2. 更快选择正确的优化维度；
3. 提高每次运行反馈的信息量；
4. 防止错误 reward 污染 Agent 经验；
5. 改善停止、换策略和换 backend 的决策；
6. 提高固定候选数或固定 GPU 时间下的 best-of-N 性能；
7. 提高优化策略跨 shape、跨 workload 的迁移能力。

最终目标是把 Kernel Agent 的闭环从：

```text
随机生成 → 计时 → 再随机生成
```

升级为：

```text
提出硬件假设
  → 生成实现
  → 验证 ISA、流量和 runtime
  → 归因误差
  → 选择正确优化动作
  → 判断继续、换路径或停止
```

因此，这项工作的核心价值不是单纯让 Roofline 数字更准确，而是让 GPU
Kernel Agent 获得可靠、可操作、可迁移的性能反馈。
