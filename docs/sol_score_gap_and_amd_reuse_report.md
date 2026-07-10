# SOL Score 与 NVIDIA 实现差异及可复用基础设施评估（ROCm 视角）

更新时间：2026-07-10

本报告用于回答两个问题：
1) 当前仓库实现的 SOL Score 与 NVIDIA 实现差距在哪；
2) 在 `github.com/AMD-AGI` 或 `github.com/ROCm` 下有哪些基础设施可以复用。

## 结论先行

最关键结论是：**SOL Score 的数学公式本体与 NVIDIA 实现对齐**，但“可验证性/可复用评分流程”与 NVIDIA 公开语义尚不一致，主要差异集中在：

- t_SOL 的来源模型（`SOLAR` 可比但非等价）；
- 基线（`t_b`）来源与可追溯性；
- 聚合策略与失败记账；
- 评分产出是否有官方化/可发表的 `official_score` 载荷。

换句话说：**不是公式错了，而是“评分语义闭环”还没完成 NVIDIA 公共语义等价闭环。**

## 与 NVIDIA SOL ExecBench 实现的差距

### 1. 公式实现基本一致（gap 最小）

- 公式实现：`1 / (1 + (T_k - T_SOL) / (T_b - T_SOL)` 与 NVIDIA 的公开定义一致。
- 本项目位置：`src/sol_execbench/sol_score.py`。
- 实际边界处理：当 `T_b <= T_SOL` 时返回 `1.0`（若 `T_k <= T_SOL`）否则 `0.0`，这与现网实现模式一致。

### 2. “公式一致”不等于“指标口径一致”

1) `t_SOL` 口径不同  
- NVIDIA 路线是基于 SOLAR 图/算子分析链条产出可复现实验中的 SOlar 下界（当前官方也有已知局限，如融合模型采用 whole-graph 近似等）；  
- 当前仓库是 AMD-native 推导链路（`amd_sol/v2_math.py`），以每算子 FLOP/byte 与硬件模型的 roofline/归约规则近似下界，且存在 `unsupported` 降级。

2) `t_b` 的可追溯基线差异  
- NVIDIA 侧依赖“可复现实验 baseline（预先建立）”；  
- 当前仓库有基线工件与 `scoring_baseline` 概念，但官方化基线发布与完整覆盖尚未完全闭环。

3) 失败与未评分的处理不同  
- NVIDIA 公开公式语义下，未打分通常仍按明确分母策略计入（如失败记为 0）并保持闭环可解释；  
- 当前仓库的 AMD-native 套件侧重“有完整输入才计入 mean”，因此缺失/失败会被排除，导致结果更接近“可用样本均值”，而非统一可比较论文/榜单口径。

4) 公式层不等于官方评分产出  
- 现在有 `official_score` 模块（`src/sol_execbench/core/scoring/official_score.py`）以及完整的 gate 设计，但文档与代码都指出**尚未接入 CLI/runner/sidecar 写出路径**；目前主输出仍是 `amd_native_score` 派生侧链。  
- 官方闭环一旦接入，关键新增项是统一的聚合策略（aggregation policy）、可复核基线源枚举、以及与执行迹一致的 provenance。

### 3. 架构范围差距

ROCm 端目前更偏向“RDNA4/CDNA3 可复现实验范围”，并保留了对 NVIDIA 路径语义的兼容形态，但未主张与 NVIDIA leaderboard 结果直接可比。  
`AMD SOL` 当前文档说明“可复用/derived”与“论文级口径”是分层的，不能直接等同于官方 leaderboard semantics。

## 可以复用的基础设施（AMD-AGI / ROCm）

建议按“可直接复用 / 有条件复用 / 先抽象再复用”分级。

### 可直接复用（价值高）

- **ROCm/rocm-libraries/shared/origami**  
  - 价值：GEMM 的确定性解析式最优配置/解析时延可直接用于局部 `t_SOL` 近似。  
  - 复用点：把当前 `amd_sol` 的某些 op 家族与 Origami 输出对齐，用于高置信 op 的 bound 计算。  

- **TraceLens（AMD-AGI）**  
  - 价值：trace 采集与算子图解析（roofline / bound）体系可复用；已有与 Origami 对接接口；可复用 MI 架构参数/基准抽象。  
  - 复用点：作为 `t_SOL` 与覆盖率审计输入，替代当前部分内建 trace 处理链路。  

- **rocprofiler-compute（ROCm/rocm-systems/projects/rocprofiler-compute）**  
  - 价值：硬件计数、roofline 微基准、系统级指标和基准对比；能校准硬件模型（尤其是 peak/带宽/功耗相关）。  
  - 复用点：把软件预测模型（Origami/TraceLens/amd_sol）与实测 profile 结果闭环。

- **Magpie / APEX（AMD-AGI）**  
  - 价值：评测流程编排、结果持久化、正确性与时延对比、反作弊与 trace 管线经验。  
  - 复用点：无需从零搭环境与数据追踪，直接复用 isolation、sidecar、审计思路。

### 有条件复用（需要接口对齐）

- **ROCm libraries 与 AITER/MIOpen/CK/rocBLAS/hipBLASLt**  
  - 价值：用于稳定 baseline source（`t_b`）与对照解实现。  
  - 条件：需固化版本、编译选项、时钟和输入形状，使每次 `scoring_baseline` 可复现。

- **hipBench（ROCm）**  
  - 价值：原生 HIP kernel 基准工具链，适合做小规模对照实验。  
  - 条件：项目状态不同于成熟生产基准，当前需验证成熟度与覆盖。

### 不建议直接复用（适配成本高）

- **NVIDIA-specific 的全流程 benchmark 依赖栈**（例如原生 NCCL/CUPTI/CUDA timing）  
  - 本质上与 ROCm 运行时不同，仍可借鉴策略，但实现层不能直接复用。

## 建议的最小迁移路径

1) 保留当前公式实现不变；把“分歧点”集中在可追溯的入口上：  
   - 统一采集 `trace`、`timing`、`baseline`、`sol_bound` 的来源与 evidence id；
   - 对每个 workload 产出官方化证据（official evidence）并记录 block reason。

2) 将 `t_SOL` 变成分层生成链：  
   - 基线默认 `NVIDIA SOLAR` 公开链路 + AMD-specific fallback；  
   - 对 GEMM 先用 Origami，其他 op 可先走 TraceLens/amd_sol，明确声明覆盖与降级。

3) 在官方聚合前先统一分母策略：  
   - 明确失败/未评分是 0 计还是剔除；  
   - 与 NVIDIA 口径一致时在文档中固定该策略并上架到官方 Score evidence。

4) 先小步复用：先把 `baseline artifact` 与 `官方评分 sidecar` 接入；再逐步扩展到更多 `op_family`。

## 与现有本地文档映射

- `docs/original_parity.md`：记录当前对上游兼容边界；
- `docs/EVALUATOR-CONTRACT.md`：定义官方评分证据门槛；
- `docs/internal/RDNA4-DENOMINATOR-POLICY.md`：当前分母/覆盖策略现状；
- `docs/original_parity.md` 与 `docs/analysis.md` 可作为后续报告挂接点。

## 备注

本文档是**工程落地评估报告**，不是 leaderboard 等价声明。后续若要主张与 NVIDIA 公开结果可比，需要在证据链、分母策略与可复现基线发布节奏上达到可复核闭环。
