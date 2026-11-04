# NVIDIA correctness gate 数值覆盖审计与本项目补齐建议

日期：2026-10-12。仅分析；没有修改实现、题目或评分，没有运行候选、GPU 或测试套件。

上游远端 HEAD 经 `git ls-remote` 及只读浅克隆核验为
`a9fa0804c793d438e70850c33fe34426e66d53dd`。本地基准为
`e35d046ee9c64c310ff3be41fe0a78b00bb993b3` 加已有工作树修改。
范围是公开评估器、仓库样例/单元测试及本项目源码；未审计线上服务部署、全部
Hugging Face 题目、自定义输入或非公开校准流程。因此“通用协议没有要求”不等于
“每一道题都没有”，源码默认容差也不代表全部榜单题目的实际容差。

## 结论

原版具备多轮差分正确性检查，但没有把结构化数值压力覆盖、输入相关误差界、
oracle 的数值可信度作为通用验收协议。这足以支持“gate 不能保证同等数值鲁棒性”
的判断，不足以证明某个榜单提交违规或估计排名失真的比例。

本项目已加强执行完整性、全元素检查和语义检查，但仍有同类数值覆盖缺口。
建议先建立与原得分分开报告的数值诊断 cohort，再决定是否升级正式契约。

## 1. 上游实际检查链

| 环节 | 源码行为 | 证据 |
| --- | --- | --- |
| 轮次 | 每 workload 10 轮，每轮调用 gen_inputs 和数值比较；常量、safetensors 或确定性 factory 不保证每轮值不同 | [eval_driver.py 413–575](https://github.com/NVIDIA/SOL-ExecBench/blob/a9fa0804c793d438e70850c33fe34426e66d53dd/src/sol_execbench/driver/templates/eval_driver.py#L413-L575) |
| 输入 | 普通 FP32/FP16/BF16 为 randn；FP8 先截断到 [-2,2]；另有名称启发式、custom factory、safetensors | [io.py 110–138](https://github.com/NVIDIA/SOL-ExecBench/blob/a9fa0804c793d438e70850c33fe34426e66d53dd/src/sol_execbench/core/bench/io.py#L110-L138)、[412–480](https://github.com/NVIDIA/SOL-ExecBench/blob/a9fa0804c793d438e70850c33fe34426e66d53dd/src/sol_execbench/core/bench/io.py#L412-L480) |
| 判据 | 先转 FP32，再计算逐元素 abs(error) <= atol + rtol * abs(reference)，检查匹配比例及可选最大绝对误差上限 | [correctness.py 101–149](https://github.com/NVIDIA/SOL-ExecBench/blob/a9fa0804c793d438e70850c33fe34426e66d53dd/src/sol_execbench/core/bench/correctness.py#L101-L149) |
| 默认值 | atol=rtol=.01，matched ratio=.99，cap=None；允许 workload 覆盖 | [workload.py 81–99](https://github.com/NVIDIA/SOL-ExecBench/blob/a9fa0804c793d438e70850c33fe34426e66d53dd/src/sol_execbench/core/data/workload.py#L81-L99) |
| 防退化 | NaN/Inf 拒绝、显式配对 -Inf 例外、非零 reference 对全零 candidate 拒绝；第0轮检查 shape/dtype | [correctness.py 37–88](https://github.com/NVIDIA/SOL-ExecBench/blob/a9fa0804c793d438e70850c33fe34426e66d53dd/src/sol_execbench/core/bench/correctness.py#L37-L88)、[eval_driver.py 509–544](https://github.com/NVIDIA/SOL-ExecBench/blob/a9fa0804c793d438e70850c33fe34426e66d53dd/src/sol_execbench/driver/templates/eval_driver.py#L509-L544) |

checker 已有近零、混合大小、outlier、cap 等单元测试。这证明比较器针对给定输出
按规则工作，不证明所有被评估 kernel 都会收到对应的难输入。

## 2. 缺少哪些数值测试

以下缺口指通用生成与验收协议；特定任务可以通过 custom factory 单独覆盖。
论文对应内容见 [How to grade the accuracy of the BLAS §3–4](https://arxiv.org/html/2609.12307)。

| 缺少的系统性覆盖 | 漏检机制 | 建议诊断输入与判据 |
| --- | --- | --- |
| 宽指数范围和内维成对缩放 | 单位尺度随机数很少检验低精度计算能否保留大小量混合贡献；FP8 默认截断更窄 | 域内指数扫描；成对 AD 与 D^-1 B；结合循环移位和随机置换，借鉴 Test 2 系列 |
| 互补稀疏结构产生精确零 | 普通随机矩阵几乎不会主动形成特定点积逐项为零 | 零行/列和互补支持；独立检查指定位置，借鉴 Test 1a/1b |
| 在大数背景中嵌入微小非零输出 | 全零防护只拒绝整张输出全零；部分小输出被清零仍可能落入 atol | 互补支持中插入可表示的小贡献，借鉴 Test 1c；保证目标值在目标 dtype 中仍非零 |
| 强消去和小残差 | 同分布随机不系统控制条件数；相对最终近零结果的误差会误导 | 大小量、正负项、归约次序置换；按 sum(abs(products)) 或 sum(abs(x)) 解释误差 |
| 行列尺度失衡与等价变换 | 单个容差不能展示行列间精度差异 | 外部行列缩放及内维共同置换；分别对各变换结果验算，不要求浮点输出逐位相同 |
| 运算长度与误差增长 | 固定 workload 容差没有表达随 K/归约长度增长的数值保证 | 首先覆盖既定 shapes 的长度；诊断各长度误差与理论尺度，新增 shape 另行声明支持域 |
| 算子参数敏感区 | 合理默认值可能让错误实现与正确实现近似重合 | norm 的非平凡 affine、variance 接近 eps、小幅值；softmax 的大共同偏移/极端差值等属于算子专门扩展，不是 BLAS 原文分级 |

上游 [io.py 281–285](https://github.com/NVIDIA/SOL-ExecBench/blob/a9fa0804c793d438e70850c33fe34426e66d53dd/src/sol_execbench/core/bench/io.py#L281-L285)
对识别到的 norm weight/bias 生成全1/全0，是具体语义覆盖盲区。
它和 BLAS 精度退让不同，但同样说明更多 seed 不能替代有区分力的输入族。

不能直接复制 FP64 的 2^±500 范围到 BF16/FP8；必须先验证输入、变换和目标输出
均符合任务的可表示范围。异常值/下溢/溢出需单独约定；论文 §3 自身也有这些边界。

## 3. 生成难输入仍不够：判据与 oracle 的缺口

### 3.1 输出相似度不是 BLAS 精度保证

上游比较器只收到 output/reference/tolerance，拿不到 A/B，无法计算
`S = |A| @ |B|`。论文 A 界涉及 `|C_hat - AB| <= f_A(n) * epsilon * S`；
B/C 使用其他输入尺度。不能从一次 allclose pass 推出任何等级。

例如 reference=[1,1e-4]，candidate=[1,0]，atol=.01 时两个元素均可通过；
也不会触发整张输出全零保护。这是容差公式的解析示例，不是实际候选复现。
它符合该容差契约，却不能据此推断微小量保真。

候选若只承诺应用容差，就不能因不满足更强 A 界而追认违规。B/C 不是“错误”；
强消去下 A 界也不保证相对最终近零结果的误差很小。

### 3.2 尾部错误约束可选

上游测试 [test_max_error_cap_no_cap](https://github.com/NVIDIA/SOL-ExecBench/blob/a9fa0804c793d438e70850c33fe34426e66d53dd/tests/sol_execbench/core/bench/test_correctness.py#L476-L484)
明确断言：在放宽匹配比例且没有 cap 时，一个元素的大错可通过。
这不是隐藏实现缺陷，而是契约选择；全部235题如何设置还需数据集清点。
论文构造的少量关键位置若直接混入总体比例，可能失去辨错效果。诊断应单列
这些位置及各输入族，不把它们平均掉。

### 3.3 FP32 比较瓶颈

上游和本地 [correctness.py](../../src/sol_execbench/core/bench/correctness.py)
均在减法前转成 FP32。本地 [output_checks.py](../../src/sol_execbench/core/bench/output_checks.py)
的 normalized_max 也如此。FP64 的 1 与 1+2^-30 转 FP32 后相同，即使声明
atol=1e-12、rtol=0，差异也会消失。这是源码与浮点表示推导，未运行实验；
不是证明当前低精度榜单受到该问题影响，但它阻止直接接入 FP64 数值评级。
本地 normal/uniform 声明式生成器也先在 FP32 生成，再转目标 dtype。

### 3.4 reference 一致性与数学准确度分开

上游 driver 执行题目 reference，通用层没有要求独立高精度 oracle。
本地独立 reference 进程解决候选污染问题，但仍执行原 reference 计算语义，
不能自动排除 reference 自身的系统误差。

本地 [aka_calibrate_tolerances.py](../../scripts/internal/aka_calibrate_tolerances.py)
重复同一 reference 并测运行间差异；[aka_tolerance.py](../../src/sol_execbench/core/dataset/aka_tolerance.py)
再加 dtype floor 与安全系数。确定性但不准确的实现仍可有零重复波动。
这套标定不能代替对数学真值的准确度检验。
上游公开代码只消费已给容差，本轮未找到公开数值标定流水线；不能据此否认
[SOL 原论文 §4.4](https://arxiv.org/html/2603.19173v1)所述离线校准存在。

oracle 应对候选实际收到的、已量化的输入求值，并尊重题目规定的中间舍入。
小规模可用精确/多精度算法；规模较大时可用已验证高精度参考或带误差预算的参考。
FP64 并非自动等于真值，误差界也需要计入 oracle 不确定性。混合精度还须分开
输入量化、累加和输出舍入，不能按输出 dtype 机械套经典同精度界。

## 4. 本项目已具备的基础与接入位置

- [workload.py](../../src/sol_execbench/core/data/workload.py) 的 NumericCheck、
  CodeDistanceCheck 强制全元素通过；还支持 exact、tie-aware routing。
  上游默认允许1%不匹配的批评不能照搬本项目。
- [custom_inputs.py](../../src/sol_execbench/core/bench/custom_inputs.py) 已支持联合
  输入生成、seed隔离、shape/dtype检查和来源记录，可生成相关 A/B，没必要另造输入通道。
- [reference_service.py](../../src/sol_execbench/core/bench/reference_service.py) 已有
  独立参考、私有 nonce 派生 seed、计时调用验证；继续复用。
- [output_checks.py](../../src/sol_execbench/core/bench/output_checks.py) 的 normalized_max
  分母是 max(abs(reference))+epsilon，不是论文的输入尺度误差界。
- [input_heuristics.py](../../src/sol_execbench/core/bench/input_heuristics.py) 仍有
  norm weight/bias 全1/全0路径；positionwise FFN 的四个 workload 命中，见
  [前序审计](sol-execbench-article-implications-2026-10-12.md)。bare weight/bias
  名称的 l1n40_layernorm 不命中，不能混为一谈。

第一阶段建议另存诊断结果，保留原正式 checks 与分数。当前每个输出由一个闭合
typed check 覆盖，不能简单给同一输出再塞一个 BLAS check。若将来升为正式门禁，
需明确输出/输入关系、误差预算及版本契约，按仓库规范原子升级，而非静默改变旧分数。

## 5. 最小验证设计（建议，未执行）

| 既有任务 | 既有范围 | 首批压力族与 oracle |
| --- | --- | --- |
| [l1n2_standard_matmul](../../problems/AMD_AKA/torch2hip/l1n2_standard_matmul/workload.jsonl) | BF16；(M,K,N)=(128,256,64),(256,128,512),(64,512,128)；atol=rtol=.0125 | 普通随机对照、域内指数范围、互补稀疏/微小非零、缩放置换；从已量化 BF16 输入构造高精度 oracle，显式考虑输出舍入 |
| [l1n47_sum_reduction](../../problems/AMD_AKA/torch2hip/l1n47_sum_reduction/workload.jsonl) | FP32；长度64/256/4096；atol=rtol=1.25e-5 | 全正对照、正负消去、小量贡献、顺序置换；精确或高精度和，误差尺度 sum(abs(x)) |
| [l1n40_layernorm](../../problems/AMD_AKA/torch2hip/l1n40_layernorm/workload.jsonl) | FP32；长度128/256/1024，eps=1e-5；atol=rtol=1.25e-5 | 大共同偏移+小可表示方差、近常数行、方差接近eps、非平凡affine；稳定高精度语义oracle，不贴BLAS等级 |

建议首轮每任务3个既有 shape，每shape 4个适用输入族，每族3个种子/变换实例，
每实现每任务36个 case，共108个 case；这是辨错烟雾验证预算，不是统计覆盖证明。
先采用验证过的正常实现、降低精度/丢小量等故意弱化实现、可靠自适应 fallback
三类对照，确认能发现目标缺陷且不误拒合法浮点重排。少算项等语义突变需单列，
不能与合法弱精度算法合并称作弊。

每个 case 保存输入域与family、实际输入/hash、候选与oracle身份、精度模式、
原规则结果、误差尺度、关键位置误差与反例。先验证数值诊断，再在相同输入族上
测性能；若要归因速度来源，需要只改变精度策略的配对干预。

验收顺序：

1. 比较器/oracle精度及生成器范围自检，避免裁判自身抹掉差异。
2. 结构化输入能否检出预设弱点，正常实现与合法fallback是否误拒。
3. 冻结候选盲测；分开报告原契约违反、更强精度保证不成立、域外失败及oracle不确定。
4. 仅在取得证据后讨论正式门禁/版本更新，不把诊断通过称为全域证明。

相对于上游，联合输入接入和严格全元素检查是已有工程能力；潜在的新贡献是
数值机制明确的压力族、可信误差预算与可复现 gate 反例。它们目前仍是分析方案，
没有实测有效性或性能收益证据。
