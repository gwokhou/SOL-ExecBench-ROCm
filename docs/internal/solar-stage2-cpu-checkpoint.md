# SOLAR 阶段 2 CPU 实施检查点

日期：2026-10-26。**阶段 2 的 CPU／fixture 范围已完成**。真实 mapper 和 GPU 资格不在本次完成声明内；阶段 3–5 继续按完整规格实施。完整目标与唯一进度入口仍是 [HANDSOFF](../../HANDSOFF.md#solar-evolution)，原任务要求不缩减。

## 已有实现与剩余边界

| 任务 | 本检查点已实现 | 尚需完成 |
| --- | --- | --- |
| S2-01 | `IntrinsicGraph` 将固有计数、访问集合与 profile 分开；重复投影不加载图或重新计数。资源归属、精度检查、容量审计和速率计价重新执行。源摘要来自实际加载字节，移除邻近重排图的隐式选择 | 计数规则取整个分析实现的保守源码指纹；不提供跨进程固有计数缓存，重开会话重新计数。相应失效和逐槽位验收见下文 |
| S2-02 | `load_verified_program` 读取完整绑定工件，检查调用方指定的 manifest/source/path/execution 身份、当前 schema、全部文件摘要、独立访问证明、精确验证策略和完整用例结果。加载不执行 reference、取图或 pickle | 已通过 `ReuseSession` 直接接入；受限域及最终逐槽位验收已记录 |
| S2-03 | unfused／ideal fused／cache-aware 已写入 analysis schema v6；M01=16/10、M02=13。共享资源内累加各模式耗时；unfused 使用稳定求和；缺容量覆盖时 cache-aware 没有数值时间或流量 | F03-D 两路径现已使用实际链描述符，检查 64/4096 B 模拟曲线、缺容量点和损坏证据。该曲线用于覆盖／组合验收，不是对实际最优流量的证明；真实 mapper 未运行 |
| S2-04 | 完整分析生产者、正式 bound 读取、路径比较读取和性能描述读取使用当前三视图合同；正式 SOL 字段组合不变 | 公共请求／结果、`solar reprice`、有界子进程、完整分析与复用三视图展示、路径比较 schema v2 已接通；32 个槽位的完整对照已完成，见下方报告 |

M04 已通过公开入口检查：F01/F05 两路径的完整分析与复用结果在计数、流量、视图和时间上相同，调用监测禁止 reference、提取、图文件重载以及重复计数。名义速率加倍时可用视图时间减半；profile 改变时不继承正式 audit。资源归属和容量审计每次重新执行，容量、mapper 身份、timeout、access budget 都进入相应身份。返回结果为独立副本，调用者修改输出不会污染会话。

新结果为独立派生工件，并非新的转换证明。`reuse.json` 绑定来源 manifest、固有快照、当前规则、请求与实际 profile、mapper 工具身份／配置、完整文件清单。结果恢复必须提供 record 与 projection 两个摘要；缺件、额外文件和半写入目录不接受。CLI 的外层时限已用真实 CPU 替身进程验证嵌套 mapper 清理；当前没有已完成扫描点缓存，阶段 3 的扫描身份仍须纳入全部外层调度预算。

默认复用为模型分析，不重新取得 profile audit。原 profile 的 score eligibility 可保留；正式 publication eligibility 还要求显式重新通过 audit 且来源本身已具备资格。profile 变化或显式 hypothetical 时两种资格均为 false，旧 audit 字段被移除。

## 新增规则的落实

- 文件读取在检查字节上限后才解析，固定完整清单并拒绝额外文件、缺件、重复路径和符号链接。只保留分析需要的内容；摘要验证过的 bytes 直接进入解析器。
- YAML 同时限制解析节点、嵌套和展开后的值数量，正常共享别名仍可用；深层嵌套、递归别名和指数展开被拒绝。巨大 nominal shape 的短文本仍被接受。
- 验证用例先检查实际 seed×pattern 数量，再构造列表；复用入口在读取 manifest 前完成此检查。输入大整数本身不会被误判为大量计算。
- 当前验证器为 `solar.verification.ir.v8`，执行记录增加 Python 实现／版本、Torch 版本和语义解释器源码摘要。字段缺失或变化需要重新验证，不接纳旧成功标志。指纹生成不枚举 GPU，文件数和总字节也有上限。
- 损坏或矛盾的合同抛出明确错误；缺容量证明保留 unavailable，不以 fallback 改为成功。正式资格仍走既有发布链。

工件加载器仍由 pipeline 所有，公开复用入口为 `solar.reuse.ReuseSession`；请求合同为 `solar.reuse_contracts`。验证执行身份明确绑定所列运行时字段；这不是完整操作系统／驱动镜像认证。所有预算在相应源码所有者处定义，未提高质量阈值。

## 本地证据

证据均在未提交的 `data/outputs/solar-stages1-5-20261026/`：

- `intrinsic-projection/check-02`：969 项相关 CPU 回归通过；类型／lint 修正和稳定求和在 `check-03` 经 55 项测试及全部质量检查通过。
- `reuse-boundaries/check-02`：1027 项相关 CPU 回归通过。检查清单误含已删除的旧 `verification/semantic_values.py`，因此该次 Ruff／format／ty 失败，原始记录保留。
- `reuse-boundaries/check-03`：90 项边界测试通过；测试中的字典类型收窄仍需修正。
- `reuse-boundaries/check-04`：92 项受影响测试及 Ruff、format、ty、coupling、readability、文档检查全部通过。源码在每次检查期间保持一致。
- `reuse-boundaries/check-05`：96 项受影响测试及全部质量检查通过；补充拒绝非法梯度结果和缺失／矛盾的执行后端字段。
- `reuse-boundaries/scope-guard.json`：相对本轮合同检查起点，429 个范围外的既有源码／支撑文件摘要未变。

锁定镜像和依赖锁沿用阶段 1，无网络、无 GPU 挂载；真实 mapper、R 组和 GPU 实验未运行。两轮 32 槽位采集均保留，最终结果为 `projection-api/attempt-s2-02`，见[逐槽位报告](solar-stage2-cpu-2026-10-26.md)；此前 `attempt-s1` 只证明其自身源码检查点。所有失败检查、日志、JUnit、环境和源文件摘要保留；不覆盖原始 BASE-01。

## 公共入口检查

`projection-api/check-06` 的 1120 项选择中，1118 项通过，两个旧 CLI fixture 尚缺三视图读取的测试替身。修正后 最终 `check-09` 的 197 项受影响检查以及 Ruff／format／ty／coupling／readability／文档检查全部通过。此前环境守卫拒绝、JSON 指数数值往返、类型收窄、跨包依赖和同名测试文件问题均保留各次日志，没有提高质量阈值。

API／CLI 用法：

```python
from solar.reuse import ReuseSession
from solar.reuse_contracts import ProjectionRequest, ReuseJob

# source 是 VerifiedProgramRequest：来自已接受运行的 manifest/source/execution
# 摘要、固定 IR path、完整 VerificationPolicy 和工件目录。
session = ReuseSession(source)
projection = ProjectionRequest(architecture=profile, output_dir=output_dir)
result = session.project(projection)

# 对另一未使用的输出目录，可把同一请求交给隔离 CLI。
job = ReuseJob(source=source, projection=other_projection)
request_file.write_text(job.model_dump_json(indent=2))
```

```bash
uv run sol-execbench solar reprice job.json --timeout 900
```

请求文件中路径按进程工作目录解释，建议使用绝对路径。摘要必须来自已接受的来源；给任意文件临时计算摘要并不能建立外部信任。重开会话要重新读取／校验和计数，已有会话内的投影复用固有计数；不会加载 pickle，也不会运行 GPU。

## 进程清理收口

`process-cleanup/check-01` 的 222 项回归通过；仅一项 Ruff 风格检查需修正。最终 `check-02` 的进程／桥接测试和全部质量检查通过。真实 CPU 替身覆盖外层超时、忽略 SIGTERM 的 mapper 及后代、正常 leader 退出后仍存活的子进程、无响应／无发布，以及非法超时在启动前拒绝。worker 的 SIGTERM 会展开清理上下文；mapper 在回收 leader 前清理整个独立进程组，保留有界脱敏日志。SIGKILL、主动脱离会话的任意程序不属于可展开清理保证。没有运行真实 mapper。

`projection-api/closeout.json` 摘要为 `91d3354925356007ef6ee448a25e195f3d44752c1df40a4d0c7d677d0d3b2044`，绑定 1447 个证据文件。该采集属于进程修复前的计量源码检查点；进程修复由上述单独回归覆盖，没有把历史采集标成最终源码重跑。
