# SOLAR 演进：阶段 0 能力、假设与基线

冻结日期：2026-10-20（Asia/Shanghai）。本文第 1–4 节保留当时的设计输入与文档冻结记录。2026-10-23 已完成 BASE-01 的 CPU 前置工作，见[改进前基线报告](solar-base01-cpu-2026-10-23.md)；当前源码另见[2026-10-26 工作树复采](solar-base01-cpu-2026-10-26.md)。

2026-10-20 这一轮仅修改文档、读取源码及既有证据、计算文件摘要，没有运行 SOLAR/模型测试、mapper 或 GPU 实验。当时状态为：**文档基线已冻结，运行基线 `not_run`**；该历史状态由后续 BASE-01 补齐，原冻结值不覆盖改写。

- [验收样本清单](solar-evolution-acceptance.md)：固定样本身份、输入、oracle、预期边界和分母。
- [阶段 1–3 任务规格](solar-evolution-tasks.md)：依赖、修改范围、交付物及退出条件。
- [统一待办入口](../../HANDSOFF.md#solar-evolution)：只在该入口维护本项仓库级推进状态。
- [当前 SOLAR 边界](../SOLAR-BOUNDARY.md)、[双路径比较合同](../user/CROSS-PATH-COMPARISON.md)：现有生产行为的权威定义。本文中的拟议视图、复用入口和扫描功能尚未实现。

## 1. 固定比较对象

| 对象 | 身份 | 使用范围 |
| --- | --- | --- |
| 本项目 | HEAD `e35d046ee9c64c310ff3be41fe0a78b00bb993b3` 加当前工作树 | 下述能力经源码核对；SOLAR 核心四个目录无未提交改动，外层存在其他工作，不能把整棵工作树当作干净 HEAD |
| 官方 master | [d7e792df82382feced17f15c340c21afb5c6b867](https://github.com/NVlabs/SOLAR/tree/d7e792df82382feced17f15c340c21afb5c6b867) | 默认分支的静态访问区域、转换和性能模型 |
| 官方 release 分支 | [2e381641ee70ea29634108e7cb3cedd545359961](https://github.com/NVlabs/SOLAR/tree/2e381641ee70ea29634108e7cb3cedd545359961) | `release/llm4arch-sync-20261011` 的批处理、重投影、dtype 和稀疏增量；不与 master 混写 |
| 独立论文 | [SOLAR，arXiv:2606.26383v2](https://arxiv.org/html/2606.26383v2) | 方法目标，不作为公开代码已实现的证明；v1/v2 关系的背景核查记录为 `solar-paper-evolution-2026-10-19.md`，属于独立研究工作，不随 BASE-01 提交 |

官方结论承接上述固定提交的源码审计，不宣称跟踪了之后的上游 HEAD。实现任务开始时只需核对拟移植文件的差异和来源，不自动扩大到新的分支或重做全部研究。

## 2. 能力清单与复用决定

这里的“直接复用”指继续使用本项目现有实现及其已限定接口，不表示无需为新调用路径验证。“适配移植”保留上游来源和许可，重新接入本项目类型、语义及证据合同。“自建”可以调用现有后端，但新增能力本身没有完整可搬用实现。

| ID | 能力 | 本项目状态与边界 | 官方可取部分／处理决定 | 落点 |
| --- | --- | --- | --- | --- |
| C01 | 提取与语义 IR | 两条固定路径：torchview→extended einsum、make_fx→ATen；不自动回退。[定义](../../src/solar/ir/contracts.py) | 直接复用本项目；官方也不是完整 Affine Loop IR 前端 | 阶段 0–3 基础 |
| C02 | 转换验证 | 主流程强制回放，含输出、mutation、alias 和梯度检查；有限样本，不是全输入等价证明。[实现](../../src/solar/verification/verify.py) | 直接复用；官方验证包装依赖未随所查提交提供的 verifier，LLM handler 不能替代它 | 所有阶段 |
| C03 | dtype／资源计量 | 逐 tensor 字节及 AMD 多资源工作量；部分精度归属依赖 profile。[计量](../../src/solar/analysis/graph_accounting.py) | 直接复用。官方 release 的 dtype 计价不构成本项目新增能力；不迁入只用 tensor-core 计算时间的简化 | S2-01、S2-03 |
| C04 | 有限容量分析 | 已有 pinned runner、曲线、容量点、受限链／区域及证据审计；不是任意 DAG 的全局最优解。[入口](../../src/solar/analysis/orojenesis_evidence.py) | 直接复用已支持范围。官方主要有 Timeloop 导出，不能立项为“从官方引入完整 Orojenesis” | S2-03、S3-02 |
| C05 | 静态访问区域并集 | 当前主要按输入标识取最大读量，缺少完整的底层 tensor 区域并集 | 适配官方 [access_regions](https://github.com/NVlabs/SOLAR/blob/d7e792df82382feced17f15c340c21afb5c6b867/solar/analysis/access_regions.py)；借鉴几何算法，不照搬字符串属性解析 | S1-01–03 |
| C06 | 受验证工件复用／重投影 | 没有面向设计探索的独立公共请求；现有分析不能整体视为硬件无关 | 适配官方 [reprice 流程](https://github.com/NVlabs/SOLAR/blob/2e381641ee70ea29634108e7cb3cedd545359961/scripts/reprice_execbench.py)，自建复用与失效合同 | S2-01–02 |
| C07 | 三视图时间分析 | 有不同字节计量及正式下界，缺少论文意义的三视图报告 | 自建公式与报告。官方三组字段均用全图 roofline，fused 与 prefetched 相同，见[已知差距](https://github.com/NVlabs/SOLAR/blob/d7e792df82382feced17f15c340c21afb5c6b867/docs/SOL_GUIDE.md#L227) | S2-03–04 |
| C08 | 反向硬件需求／敏感性 | 尚无完整入口 | 自建；官方批量 reprice 是基础，不是 inverse roofline 或硬件参数网格 | S3-01–03 |
| C09 | 稀疏／按需计算 | 有相关操作语义，无通用有效密度传播 | 阶段 4 受限适配。官方 release 的阈值、log_softmax 及零初始化假设须另审，阶段 1–3 不顺带加入 | 后续 P2 |
| C10 | 收缩顺序搜索 | 现有链／区域分析不改变代数结合顺序 | 阶段 5 自建；候选验证、成本评价可复用 | 后续 P2 |
| C11 | Affine Loop IR／LLM 修复／JAX | 缺少这些完整前端能力 | 阶段 6 自建为主，由实际覆盖缺口驱动 | 后续 P3 |
| C12 | 发布与双路径审计 | 已有内容绑定、路径比较和外层正式资格；比较不择优取结果 | 直接复用。本轮不变更正式分数、设备资格或发布策略 | 所有阶段 |

## 3. 分析假设台账

下列 ID 是文档引用号，不是已发布 wire identifier；实施时由对应 canonical registry 定义真实字段。

| ID | 固定假设／声明 | 对实施的约束 |
| --- | --- | --- |
| A01 | 分析对象是带源码、输入约束、具体 shape/dtype、effects 和验证策略的程序实例 | 不从单一 trace 推出所有 shape 或所有输入值成立；shape 扫描先逐实例提取和验证 |
| A02 | 原图工作量与允许代数／稀疏优化后的工作量是不同分析对象 | 阶段 1–3 只完善原图与既定融合假设的计量；不得自动应用阶段 4–5 变换 |
| A03 | 单个约定模型中的时间下界不等于实际可达运行时间 | 不包含实际调度、启动、功耗／温度变化等完整模型；候选或基线实测时间不输入 SOLAR |
| A04 | 拟议 unfused 视图固定为算子串行、每算子内资源与 DRAM 理想重叠 | 定义 `c(v)=max_r Σ_m W(v,r,m)/P(r,m)`，`T_U=Σ_v max(c(v),B_U(v)/BW)`；不是允许跨算子并发时的无条件下界 |
| A05 | 拟议 ideal fused 视图采用明确的理想融合与存储假设 | 报告全图资源下界和允许消除后的 I/O；不能消除 mutation、别名可观察行为或必须物化的边界。全图资源重叠下界不声称有可实现的融合调度 |
| A06 | cache-aware 视图采用受支持图形、固定工具链与明确容量域 | 缺证据时 unavailable；局部结果不得标成完整 cache-aware。现有“全局 compulsory I/O + 最大额外量”保守组合，不无依据地改成求和 |
| A07 | 三视图比较只在相同程序、工作量、资源及适用约束下解释 | 不规定三个数在任意输入上有统一排序；同一 fused 模型下增加有限容量约束不应降低其必要流量 |
| A08 | 静态外部读取统计的是同一底层 tensor 的唯一访问元素集合 | 不同输入不能仅因 shape 相同合并；不可证明的跨输入存储别名不猜测；保留 dtype 字节宽度。读取区域与输出写入／初始化责任分别处理 |
| A09 | 验证成功不自动证明资源公式或区域模型正确 | 语义回放、独立计量 oracle、工具证据和设备实测分栏记录 |
| A10 | 探索配置与正式 profile 资格不同 | 假想吞吐／带宽／缓存情景不得继承原 profile 的发布资格；不得通过放宽既有 audit gate 实现探索功能 |
| A11 | 反向查询求模型下的必要资源约束 | 共用资源各模式按 `Σ_m W(r,m)/P(r,m) <= T_target`，带宽按 `B/BW <= T_target`；容量改变时 B 也可能改变。unfused 求和模型必须使用对应约束，不能套全图公式 |
| A12 | 功能和验收策略先冻结，错误与失败始终保留 | 不删失败 workload、更换路径、放宽容差或挑有利视图以通过验收；旧原始证据不覆盖改写 |

模块归属保持现有方向：`solar` 管语义、计量、架构情景与分析工件；`sol_execbench.core.solar_bridge` 是外层唯一适配入口；benchmark 枚举、baseline/candidate 时间、评分及 profiler 解释由外层负责。新 schema 涉及的 producer、reader、测试、CLI 和文档必须原子更新，只接受当前 canonical 版本。

## 4. 当前源码与输入基线

### 4.1 文件身份

| 项目 | SHA-256 |
| --- | --- |
| [请求／结果合同](../../src/solar/contracts.py) | `8e2452efe4d01becd2d24e19c3cf34fe2ab23e8aef7cbfcdf89552f3ea5666bc` |
| [计量实现](../../src/solar/analysis/graph_accounting.py) | `d74950334e1ce8d32339cb2184a92983a66b2f6618b494925b008b8443142c74` |
| [下界组合](../../src/solar/analysis/formal_analysis.py) | `505c738323bd281390200f576ad76084ab65e395c9e7581e03f87b0f6f358add` |
| [容量证据](../../src/solar/analysis/orojenesis_evidence.py) | `847602b235cffd5d81ed303d5d3372be4b40966627530ed81c2ec8223c3b7e5b` |
| [SOLAR schema registry](../../src/solar/schema_versions.py) | `01dc3f55fb1c0f68e7ed299b22b74f2f7bd27454c9514c6893c6225cb30794be` |
| [RX profile 文件](../../src/solar/rocm/profiles/RX_9060_XT.yaml) | `3ab62e1ddc7f6e6d607a4483677fc4e85d69cc6b0e5d08adfd65dcdd3c8fb4d3` |
| [packaged resource audit 文件](../../src/solar/rocm/audits/rx9060xt_resource_peaks_v3.json) | `f822fcc4f0019c12da9e46dbfb42f2b4f5f56b65cf2813c9ab7c0582c3083e48` |
| [AKA manifest](../../problems/AMD_AKA/manifest.yaml) | `8f057f7890016239456e137e639e4538df22733ecf49721a919f8f6877acc857` |
| [pyproject](../../pyproject.toml) | `bbd082f08bbc3b6a398e4402fa23c929edf0c0aa90c5ec297841ab1c67e95212` |
| [lock](../../uv.lock) | `50668e6d0243eb80691d935424dedd2e2b4cf9c4b9d62937cd74dd71f2b83d37` |
| [Dockerfile](../../docker/Dockerfile) | `7c741e959e7c85955327769ea0c09d0772bf47ef7fd8e975fa893dc2842157fd` |

以上均为原始文件字节摘要；不要与 API 的规范化 profile 内容摘要混用。

额外冻结两组集合摘要：取 `git ls-files --cached --others --exclude-standard -z` 的唯一文件路径，按仓库相对 POSIX 路径排序；对每个存在的文件生成 UTF-8 `路径 + NUL + 文件SHA256小写十六进制 + LF`，拼接后再做 SHA-256。不跟随未纳入列表的 ignored data。

| 集合 | 文件数 | 集合 SHA-256 |
| --- | --- | --- |
| `src/solar/`、`tests/solar/`、`src/sol_execbench/core/solar_bridge/`、`tests/sol_execbench/core/solar_bridge/` | 218 | `112a1649ce4cc6df1552cfbb2b2b35cc65639a68f95a60c443f0f3be66d4ef4e` |
| 上述 git 文件列表中所有非 `.md` 文件，含当前其他未提交工作 | 1169 | `6d17f7b908fc3f18404699faefac9638b545cf8ffb86d51e488aa02956f74ad0` |

集合摘要用于发现漂移，不是可恢复的完整环境包。`BASE-01` 必须记录执行时精确源码／工作树、依赖和镜像身份；若与这里不同，追加差异说明，不改写本次冻结值来冒充同一基线。

### 4.2 保留证据与缺口

| ID | 本轮实际核验 | 可以支持的结论 | 不能支持的结论 |
| --- | --- | --- | --- |
| B01 | 2026-10-11 P0 回执、JUnit、profile/audit 摘要及 static/canary/full gate 文件摘要相互匹配；JUnit 为 671 tests、0 failure/error/skip | 历史专项记录仍在，文件绑定一致 | 不是在当前 HEAD 重跑；未重新执行 validator，也未递归核验全部原始采集文件 |
| B02 | 旧栈升级回执的 full_cpu artifact 摘要与本地 JUnit 一致；2550 tests 中 51 skipped、无 failure/error，即 2499 passed | 保留的历史 CPU 回归计数可复核 | 不证明记录之后所有工作树变化或本轮新规划已验证 |
| B03 | `data/outputs/solar-cross-path-focus-cycle2-c84869e/path-comparison.json` 当前不存在 | [原比较文档](../user/CROSS-PATH-COMPARISON.md)保留其 41 workloads／82 路径运行及差异说明 | 原文的 27/14 状态不能作为当前数值基线；不能将 41 workloads 写成计量全部相等 |
| B04 | [发布标记](../../RELEASE/release-bundle.json)文件存在，原始文件摘要为 `1f006e3aec3e97140e7dec5abc4bc9909b0ab781df954c5f2723fd04754bcd83` | 当前有历史发布标记 | 本轮未递归验收 release bundle；不作为新分析公式或当前源码的发布资格 |
| B05 | 本轮样本仅完成定义／输入文件读取与身份冻结 | 样本分母和预计检查项可复核 | 当前语义执行、数值结果、mapper 结果、GPU 资格均为 `not_run` |

B01 路径前缀为 `data/outputs/solar-p0-rocm101-20261011-r2/`：`p0-validation.json` SHA-256 为 `7ec1ec6679c1c5c1a39347821667fe5a0d602f081d9c09b5ea27a5aed5df075d`；`solar-regression.xml` 为 `10dbc681e42f6be8a1c360161ca163587d3e9e2bea55f084b009a08663e4e8bd`。其证据仍限于 `instruction_and_runtime_corroboration_only`，不是无节流峰值证明，见[原专项说明](solar-p0-rocm101-completion-2026-10-11.md)。这里的阶段 0 与该旧文的资源审计“P0”是不同工作。

B02 路径前缀为 `data/outputs/stack-upgrade-followup-20261011/`：`validation-receipt.json` SHA-256 为 `0153af5a1157efaaad2522c34e134133ee12c5f93e386c39425df8f3f3489faa`；`cpu-accepted.xml` 为 `30aac35d93071bdaf674d0f5ef3e8f60a180938a764977b33eacf851c516beea`。本轮只核对该 CPU artifact 绑定，未审计回执中的其他实验。

### 4.3 阶段 0 退出状态

| 交付 | 本轮状态 |
| --- | --- |
| 能力清单、来源、复用类型、阶段归属 | 文档已冻结 |
| 分析假设、模块边界、结果资格 | 文档已冻结；拟议接口尚未发布 |
| 源码／配置／样本文件身份 | 已读取并计算摘要 |
| 历史证据、缺失工件及适用范围 | 已分栏记录 |
| 新样本执行、当前行为基线、GPU 验收 | `not_run`；由 `BASE-01` 和后续限定验收任务获取 |
| 阶段 1–3 实现与发布 | 未开始；任务规格已拆分，不构成本轮实施授权 |

文档完成不关闭运行验收缺口。第一批实现不得用新增结果覆盖 `BASE-01` 的旧结果目录，也不得将缺少工件解释为旧实现失败。

### 4.4 本轮文档检查

- 已做独立静态检查：75 份当前文档、371 个本地链接，未发现断链、退役引用或缺失必需引用；本轮 5 份文档另查锚点、空白及围栏。能力引用对照源码中的静态 registry 字面量，没有调用 evaluator。
- 11 个源码／配置文件摘要、两组集合摘要、R 组 8 个 UUID 及 16 个输入文件摘要均与文档一致；本轮范围外 1244 个已登记文件保持不变。
- `git diff --check` 对修改的既有导航／待办文档通过；新增文档另做静态空白检查。
- 完整的 `scripts/check_current_docs.py` 在宿主和项目 Python 中都因缺少 `pydantic` 而未能启动。上述独立静态检查不冒充该入口运行通过；没有安装依赖、运行模型测试或进行 GPU 实验。

## 5. BASE-01 状态更新（2026-10-23）

已补齐可执行样本和独立 oracle，并在与 `uv.lock` 一致的固定镜像中完成 CPU 采集：32 个 F 槽位全部尝试，30 个验证／计量完成、2 个转换拒绝；8 条计量差异原样保留。相关回归为 736 passed，完整文档检查也在该镜像内通过。生产源码保持不变；R 组、真实 mapper、GPU 资格及阶段 1–3 均未执行。详见[结果、失败历史和证据边界](solar-base01-cpu-2026-10-23.md)。

## 6. 当前工作树复采（2026-10-26）

在相同锁定镜像中重新采集：32 槽位全部尝试，30 个验证／计量完成、2 个转换拒绝；F04 的 4 个差异保留，F08 的计数变化来自本轮开始前已有的生产改动。相关 CPU 检查为 821 passed、1 failed，失败已定位到既存提取包装器与对象身份断言的冲突；84 项 BASE-01 支撑测试全部通过。按更新后的 AGENTS.md 补齐了结果边界校验和异常留存测试。本轮未修改生产源码或推进阶段 1–3，详见[独立复采报告](solar-base01-cpu-2026-10-26.md)。
