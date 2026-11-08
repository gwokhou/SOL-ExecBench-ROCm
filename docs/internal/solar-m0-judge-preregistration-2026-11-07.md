# M0 / E0：裁判辨错实验预登记

制定窗口：2026-11-07—08；冻结日期：2026-11-08（Asia/Shanghai）。
研究规格标识：`M0-E0-20261107-r1`（沿用规划日期的文档标识，不是生产 schema）。
状态：**设计冻结，未实现、未生成实例、未运行候选**。
源码/环境锚点与文件级核查范围见 [B0](solar-m0-baseline-evidence-2026-11-07.md)。
本文把[迭代计划](solar-next-iteration-plan-2026-11-07.md)的 E0 补全为可交给 E1 的规格。

## 1. 问题、假设与固定分母

研究问题：现行裁判能否检出已知原契约违反，是否误拒合法对照；独立诊断能否发现原容差
未承诺的细粒度数值性质；增加诊断的运行成本是多少。不开新排行榜，不改 workload checks。

四个候选策略组，每组包含 GEMM、sum、LayerNorm 各一个实现，共 **12 个待实现候选**。
每策略组 3 任务 × 3 shape × 4 输入族 × 3 实例 = 108 个逻辑 case；
全活动 **432 个候选—case 单元**，其中开发 288、留出 144。
每任务的单个候选实际消费 36 个 case，不是每个单任务实现都消费 108 个。
每单元 3 个独立进程重复，共 **1296 个正确性评测单元**；重复不扩大样本数。

| 任务 / 固定 UUID 后缀 | 轴（顺序 w0,w1,w2） | 输入/输出与原 checks |
| --- | --- | --- |
| `aka-l1n2_standard_matmul-w{0,1,2}` | `(M,K,N)=(128,256,64),(256,128,512),(64,512,128)` | A/B/output BF16；atol=rtol=.0125 |
| `aka-l1n47_sum_reduction-w{0,1,2}` | `(M,N)=(128,256),(1024,64),(256,4096)` | x/output FP32，最后一维 sum，keepdim；atol=rtol=1.25e-5 |
| `aka-l1n40_layernorm-w{0,1,2}` | `(B,N)=(128,256),(64,1024),(512,128)` | x/weight/bias/output FP32，eps 固定为实际输入 FP32 的 1e-5；atol=rtol=1.25e-5 |

共同规则：全元素匹配比例 1、max_error_cap=null、allow_negative_inf=false；保留原 shape/dtype、
sanity、静态审查、隔离与计时完整性规则。修改输入分布须生成独立诊断 Workload，绑定原 UUID/SHA，
不覆盖原文件，不把诊断通过作为原 corpus 的官方分数。

冻结文件均位于 `problems/AMD_AKA/torch2hip/`：

| 文件 | SHA-256 |
| --- | --- |
| `l1n2_standard_matmul/definition.json` | `32983314f87a12778e973d2557d33ae06c8fabf27459a8b7d4edc4cda17e4742` |
| `l1n2_standard_matmul/workload.jsonl` | `1884b69eb45cbc18a53d3bc746fbe0ccaaa5102a8964daab189f22ee973a0849` |
| `l1n47_sum_reduction/definition.json` | `69e4a427d37168da498c1cd9b81b63df9a1637880b2c71ba79aeea9135055136` |
| `l1n47_sum_reduction/workload.jsonl` | `8b1bd83967d61a7e6a5a15be0ec58703e2beefef96b709939d9276c231623ef6` |
| `l1n40_layernorm/definition.json` | `949f4ca1d7c6214a02427209bdd68786a8dc9eb724e52bd9e0ead57bbcdca557` |
| `l1n40_layernorm/workload.jsonl` | `18096048cf9462ed546db2c3b895eae85c7488943f40a530a485dd610b93aafa` |

## 2. 输入构造域与实例派生

所有构造先产生有限二进制有理数，再 round-to-nearest-ties-to-even 到实际输入 dtype；
oracle 只读取这些量化后的值。除下表明确指定的小量外，不涉及 subnormal、NaN/Inf 或溢出。
行/列合法置换、整行符号和域内二次幂缩放用于实例差异；变换后重新计算 oracle，
不要求浮点重排结果逐位等于变换前输出。

`Q` 是整数均匀网格 {-16,…,16}/16；随机抽样只决定符号、网格与置换，不改变 family 语义。
`j` 为零起始特征索引；结构见证位置随置换一起移动并记录，不能事后挑最大错误点。

| Family | 固定构造；w0–w2 均应用 | 必须保存的见证 |
| --- | --- | --- |
| G0 / ordinary | A、B 从 Q 独立取值，然后 A 第0行、B 第0列清零并置 A[0,0]=B[0,0]=1，固定 C[0,0]=1 锚点 | 全输出与量化输入；不是无界正态分布 |
| G1 / scale | 从 G0 构造 `A'=AD, B'=D^-1 B`；D 对角指数在 -8…8 均衡循环，按实例置换 | 每个内维的指数和量化后乘积尺度 |
| G2 / small-nonzero | 除见证外 A 支持前 K/2、B 支持后 K/2，且相互不交；保留 k=K-1 的 `A[:,k]=1, B[k,0]=1` 锚点，以及 k=K-2 的 `A[:,k]=1, B[k,1]=2^-20` 小量；其余这两列贡献为零 | 锚点输出1、小量输出2^-20、其余指定零位置；均在 BF16 可表示 |
| G3 / cancellation | A 内维取1；B 前 K-2 项按成对 `(+2^8,-2^8)` 排列，最后两项为 `1/16,0`；实例改变共同内维置换和输出列符号 | 精确残差 ±1/16；逐输出 `sum(abs(products))` |
| R0 / positive | x 取 `1+q/32`，q 为0…31；最后一个元素固定1 | 最后元素遗漏造成的单位误差 |
| R1 / cancellation | 依次重复 `(2^12,1,-2^12)`，不足三项的尾部为0 | 精确和及 `sum(abs(x))` |
| R2 / small-contribution | 偶数位1、奇数位2^-20，所有现有 N 为偶数 | 总和 `N/2*(1+2^-20)`，小量贡献单列 |
| R3 / ordering | R1 的同一多重集，按实例确定的循环位移/排列重排 | 排列和每个输出精确和；不把顺序变化视为新语义 |
| L0 / offset | 每行 `2^12 + (-1)^j*2^-8`，weight=1,bias=0 | 实际量化均值、总体方差；禁止输入舍入后变成常数 |
| L1 / near-constant | 每行 `1 + (-1)^j*2^-20`，weight=1,bias=0 | 非零小方差及 eps 占比 |
| L2 / epsilon | 每行 `(-1)^j*2^-8`，weight=1,bias=0 | 方差2^-16与实际 eps 的关系 |
| L3 / affine | x 从 Q 取值，第0行固定全零；weight 按 `(1/2,2,-1,3/2)` 循环，bias 按 `(-1/2,1/4,1,-1/4)` 循环；各实例共同置换特征 | 第0行正确输出为bias，忽略affine为0；其余行检验非平凡affine |

每族三个实例编号 d0、d1、h0；d0/d1 开发，h0 留出。逻辑 ID 为
`spec/task/workload_uuid/family/instance`，重复编号独立记录，不嵌入分母 ID。
G2 的内维见证 K-1/K-2 不参与置换，实例只变行/输出列及其余支持元素，确保 P1 必检点仍被遗漏；
R0 的尾元素固定1也不参与置换。所有其他变换不得破坏表内固定条件。
开发实例固定 base_seed=1701/1702；h0 的密钥在 E1 完成候选与判据冻结后、E2 前由评测侧一次生成，
先登记 SHA-256 commitment，再实例化 h0，不搜索种子或失败后重抽。本轮不生成该密钥/实例。
h0 使用32字节随机密钥，`base_seed` 为 HMAC-SHA256(key, UTF-8逻辑ID) 的前8字节大端整数
与 `0x7fffffffffffffff` 的按位与；该值再进入现有调用级派生，不作为候选可见 metadata。

每次调用继续使用 [derive_custom_input_seed](../../src/sol_execbench/core/bench/custom_inputs.py)
的 workload/round/run_nonce/variation 派生与可信 worker 私有 nonce；h0 密钥只进入评测侧派生。
随机变换必须保持上述见证。输入 hash 唯一性、实际输入留证、参考输出隔离均保留。
候选不得看到实例角色、密钥或 oracle 输出。种子、生成器实现和字节序/遍历顺序在 E1 代码中钉桩，
仅实现这些规则；发现定义歧义须在任何 h0 生成前追加勘误，不可边看留出边解释规格。
基础随机数使用固定版本的 CPU Torch Generator、派生种子、整数均匀抽样后映射 Q，
按 A→B 或 x 的 row-major 顺序生成；结构覆盖在随机填充之后，置换在最后。
未另指定的支持域内元素取 Q；零支持域不抽值。结构性约束优先于通用缩放/符号/置换。
这只是冻结输入族内的实例留出，不代表未见过的算子/形状/输入族泛化。

## 3. 候选干预与预期

候选实现 SHA 尚不存在，属于 E1 的交付；本节冻结干预机制。每任务恰好四个实现，
不添加只对已看到的 h0 特判的版本。P0/P3 是待验证合法对照，不能仅凭命名认定合法。

| 组 | GEMM | sum | LayerNorm | 预登记预期 |
| --- | --- | --- | --- | --- |
| P0 / reference-equivalent | 与 Definition 相同的 matmul | 同一 sum/keepdim | 同一 F.layer_norm 与 affine | 原 gate 应通过；独立 oracle 不预设 reference 完全准确 |
| P1 / semantic mutant | 丢掉最后一个内维项 | 丢掉最后一个元素 | 忽略 weight/bias | G2 锚点、R0、L3 是必检原契约违例见证；其他族逐例报告，不预设全失败 |
| P2 / numerical weakening | 将计算输出中 `0<abs(y)<2^-12` 的元素清零，保留其余元素 | 累加前把 `abs(x)<2^-12` 的输入清零 | 用 FP32 `mean(x*x)-mean(x)^2` 替代稳定方差，其他语义不改 | G2/R2 预期体现“原数值容差内，但更强保真失败”；L0 预期暴露不稳定方差；须由 oracle 认证后入对应分母 |
| P3 / legal alternative | 从 BF16 输入做 FP64 累加、最后舍入 BF16 | 精确二进制和或 FP64 累加、最后舍入 FP32 | 稳定双遍 FP64 mean/variance，真实 eps/affine，最后舍入 FP32 | 预期通过原 gate；若 reference 自身误差引发不一致，标 oracle/reference disagreement，不能先贴“误拒”标签 |

原 gate 的 admission/静态审查、运行时完整性、数值结果分列。若 P2 被现行精度策略静态拒绝，
记录为该策略检出；不得绕过它后声称“原 gate 通过”。G2/R2 的 pass 预期专指原数值容差，
不是预判整个候选符合全部执行政策。没有进入数值阶段就记 not_reached。

所有候选输出均保持原 shape/dtype；异常、非有限值、整张清零等保持现行 sanity 拒绝。
不得通过屏蔽保护让弱化实现运行。必要的离线输出比较只是独立诊断，不替代生产裁判 verdict。

## 4. Oracle、标签与比较精度

三层结果分别存储：原 evaluator verdict；相对于实际 reference 的原数值判据；
相对于独立 oracle 的更强诊断。原契约使用实际 reference，不能拿另一套数学语义替换其标签。

- GEMM/sum：有限二进制输入表示为整数乘二次幂，使用不会溢出的精确整数乘加得到全输出，
  再按输出 dtype 舍入；同时记录未舍入精确值。不得用 int64 未证明安全的累加器。
- LayerNorm：用实际 FP32 x/weight/bias/eps，精确有理 mean/总体 variance，再用有向舍入区间
  包住 sqrt、除法和 affine；初始至少80位十进制精度，若目标 FP32 舍入单元不能唯一决定，
  只允许提升一次到160位，否则 oracle_uncertain。提升由区间宽度触发，不由候选误差触发。
- 对解析见证另作独立推导；E1 在开发集上核对精确整数与独立多精度实现。
  比较不经过通用 FP32 强制降位；用 FP64 的1与1+2^-30、精确零和2^-20 等自检点确认差异未被抹掉。
- 原 reference 与 oracle 不一致时单列 reference/oracle discrepancy；当前容差下的 oracle 标签
  只有在 reference 不确定性已被界定、决策有严格间隔时才能用于“漏检/误拒”统计。

更强诊断冻结为：G2 指定小量位置必须保留非零且零位置不生造非零；R2 的小量贡献必须保留；
L0–L3 对照区间算得的正确舍入 FP32 值，逐元素误差不超过4 ULP（零处按最小 subnormal 间距）。
4 ULP 是本实验预登记的诊断目标，不是原 Definition 的新增保证；不将 LayerNorm 贴 BLAS 等级。
G0/G1/G3、R0/R1/R3 另报告精确误差及输入尺度归一误差，不事后设阈值制造通过/失败。

E1 必须生成“见证认证表”：每个必检点的确切原容差余量、oracle 区间、诊断阈值和预期分类。
若构造并不具有预设辨错力，记录设计失败并在 h0 开启前修订规格；不能删除该族或用新种子掩盖。
M0 冻结的是可验证假设，不把未执行的 oracle 自检写成通过。

## 5. 执行顺序、预算与成本对照

后续执行顺序固定为：E1 开发集与 oracle 自检 → 候选/生成器/比较器源 SHA 冻结 → B1/B2 →
评测侧 h0 commitment → E2 开发和留出三重复 → 成本实验。h0 一经看到，所有修复转下一活动。

正确性活动使用当前 BenchmarkConfig 支持的诊断计数：warmup=0、iterations=1、trials=1、
min_measurement_time_seconds=null、lock_clocks=true、benchmark_reference=false；
使用原 gate，不从这一调用取得官方性能结论。最多1296个单元，不增加重试来投票。

成本分母固定为三个任务的 **w0 × 四族 × d0 × {P0,P3}**，共24候选—case块；
每块3重复、两种模式（原检查；原检查+诊断），最多144个 session。
沿用10 warmup、50 iterations、3 trials、min_measurement_time_seconds=null，保留 reference 成本记录。
对应候选 warmup+timed 调用上限为23,040；协议额外的资格/验证调用由 E1 逐入口列明、B1 固定，
不得隐式增加自适应采样。该计数不是总设备 dispatch 数。
两模式在同一逻辑块内交替顺序，按重复编号反转；输入受各自私有 nonce 派生，
匹配的是固定分布/逻辑块，不伪称输入字节相同。若需逐字节配对，须先证明接缝支持且不破坏唯一性。

资源预算是上限而非耗时预测：E2 GPU占用累计最多6小时（正确性和成本合计），CPU oracle最多12小时；
正确性单元墙钟120s、成本 session 300s、单次 oracle 120s。一个 GPU 活动，jobs=1；
本轮无 mapper。每 worker host 内存上限为 `min(16 GiB, B1确认物理内存的1/2)`，输出证据总量上限20 GiB。
设备分配仍遵守现行 gate/容量约束，不能靠降低输入或跳过检查满足预算。
这些是研究活动的更紧上限，不改生产默认值；B1 可降低，不可在见到结果后提高。
达到预算保留原分母并标 INCOMPLETE；不要求为满足该上限牺牲正确性。

成本分别报告编译/准备、reference/IPC、现行检查、附加 oracle、总墙钟和峰值内存，
按24块给出原始重复、配对差/比值及分位数。不设速度收益门槛，不因候选被拒绝而补选更快候选。

## 6. 验收、产物与变更规则

每个记录必须有逻辑 ID/重复/角色、实际输入 hash与生成器身份、候选/执行树身份、
原 checks、admission/runtime/numeric verdict、oracle 与参考身份/误差预算、关键位置、
独立诊断分类、预算/成本、原始证据位置。输入、输出和异常记录留在未提交 data/outputs。

- **实验有效**：432个逻辑单元可对账；三重复齐全或明确 INCOMPLETE；见证/oracle自检通过；
  无事后改分母/阈值、无 h0 反馈调参。固定缺陷未检出也可以是有效实验的负结论。
- **裁判达标**：通过独立认证的原契约必检违例全部检出；已确认合法的 P0/P3 零误拒；
  诊断见证符合预登记分类；重复一致。把不确定项从统计分母单列，不用删行制造100%。
- **停止条件**：输入域失败、oracle不可判、协议错误、OOM/timeout、身份漂移分别记录；
  不通过多数投票、不恢复旧nonce、不手工写 timing JSON。

预期产物：pre-registration 摘要、exposure ledger、witness-certification、candidate manifest、
432单元/1296重复对账、分阶段混淆表、反例包和成本表。它们目前均未生成。
E1 的代码 SHA、h0 commitment、B1/B2 的环境/准入信息是预先声明的后续填充槽，
不是允许更改任务、输入族、候选干预或成功标准的空白项。
任何实质修订生成新规格版本，说明原因、发生时点及已暴露分母；旧规格和负结果保留。
