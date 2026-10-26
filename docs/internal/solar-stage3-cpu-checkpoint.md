# SOLAR 阶段 3 CPU 实施检查点

日期：2026-10-26。**阶段 3 的 CPU／fixture 范围已完成**：反向查询、预算扫描、恢复和报告已通过 1223 项相关 CPU 回归，最终边界增补另通过 316 项受影响测试及全部质量检查。唯一状态入口仍为 [HANDSOFF](../../HANDSOFF.md#solar-evolution)。阶段 4–5 未开始，真实 mapper、R 组和 GPU 实验未运行。

## 支持域与合同

`solar.explore.query_resources(InverseRequest)` 接受有摘要绑定的完整复用结果。固定程序、IR 路径、视图和容量后，返回每个资源的 `Σ_m W(r,m)/P(r,m) <= T_target` 约束；DRAM 项为 `B/BW <= T_target`。同一资源的各模式先累加时间，独立资源分别报告。单模式可给出最低必要速率，多模式只给约束和共同倍率，不虚构各模式的唯一速率分配。

首批支持 ideal fused 和完整 cache-aware；unfused 返回 `unsupported/unfused_inverse_unsupported`，缺证视图返回 unavailable。非法时间、速率、缺资源工作、矛盾的正向结果和未知资源组是错误，不改写成 unsupported。指定资源组扩大时，报告未变资源是否仍阻止目标，以及按共同倍率反代的模型时间。结果始终 `necessary_only=true`、`performance_authority=false`，没有设备可达性或硬件推荐含义。

公开扫描请求为 `solar.scan_contracts.ScanRequest`，数学网格合同由 `solar.analysis.scan_contracts.ScanDefinition` 所有。`solar.explore.scan_profiles` 每次建立一个 `ReuseSession`，同一尝试只计数一次；各点重做资源归属、容量审计及速率计价。所有点，包括原速率的基线点，都显式属于 hypothetical，不能继承旧 profile audit、分数或发布资格。

当前 exploration 家族 schema 为 v1。请求和终态报告读取先检查版本，再解释业务字段；旧 schema、布尔／浮点伪版本、缺失摘要及受损复用工件不接受。原正式分析和发布资格链保持不变。

## 网格、预算和恢复

- 轴支持指定资源组的共同倍率、DRAM 带宽倍率、指定 cache 层的容量字节数。倍率轴第一项必须为 1，容量轴第一项必须等于已知基线容量；不支持通过这些轴改变 shape 或源程序。
- 最多 8 个轴、64 个点，在构造笛卡尔积前核对实际基数；重复轴／坐标、相交资源组和未知资源拒绝。巨大容量数值本身不等于大量计算，低基数的巨大容量坐标可通过规划。
- 容量更新重新进入现有容量选择／证明逻辑，且同步 profile 的容量字段。缺 mapper、缺曲线点或不完整覆盖保持失败或 unavailable；不把固定旧流量外推成新的容量证明。
- 调度串行。默认每点 30 s、每次尝试总预算 300 s；准备、恢复和计算消耗调度预算。CLI 另有总预算加 10 s 清理余量的外层进程时限，记录实际 mapper 线程数和可见 CPU 数。核心 API 使用 Linux 主线程定时器，拒绝覆盖调用者已有的活动定时器；它不是任意本机代码的硬中断保证。
- 完成、缺视图、失败、输入无效、超时及未运行都有明确终态。异常不变量向外传播，仍尽可能留下本次尝试状态；不会把内部错误变为一个可用估算。进程清理沿用阶段 2 的 CPU 实测边界，SIGKILL 或主动脱离所属进程组的任意工具不在展开清理保证内。
- 恢复必须使用新输出目录和旧 `scan.json` 摘要。源、路径、策略、完整 profile、网格、视图、预算、规则实现、mapper 身份、调度实现或线程策略变化均改变身份。完成点逐一重新核验其完整派生工件和数值；失败／未运行点在新目录重新尝试。旧目录不覆盖，恢复记录引用旧完成点，因此须保留被引用的目录。

`scan.json` 保留全部计划点、三视图、所选估算、状态、工件摘要及前一尝试引用。缺失数值不会以 0 代替。报告显示固定模型下的瓶颈、相对基线变化与单轴敏感性；可选 Pareto 集只使用事先声明的多个最小化目标，单位为秒、倍率或字节。相同目标值的点按计划顺序保留，不声称求得芯片成本、面积、功耗或商业性价比最优值。目标时间检查只表示没有被当前模型下界排除。

## 入口

```bash
uv run sol-execbench solar requirements inverse-request.json
uv run sol-execbench solar scan scan-request.json
```

请求文件由对应的 Pydantic 请求对象 `model_dump_json()` 生成。`InverseRequest.source` 是 `ProjectionReference`，包含目录、record/projection 摘要和读取预算；`ScanRequest.job` 是现有 `ReuseJob`，其 `projection.output_dir` 指定本次尝试的新目录。外层 CLI 只通过 `core/solar_bridge` 访问 SOLAR。

扫描默认不请求 mapper。显式配置 mapper 属于另一种工具情景；本次只运行合成曲线测试替身，未执行实际工具。Python API 的固定 profile 示例：

```python
from solar.analysis.scan_contracts import AxisKind, ScanAxis
from solar.analysis.views import AnalysisView
from solar.explore import scan_profiles
from solar.scan_contracts import ScanRequest

# job 来自已接受来源的 ReuseJob，output_dir 必须尚不存在。
request = ScanRequest(
    job=job,
    view=AnalysisView.IDEAL_FUSED,
    axes=(
        ScanAxis(kind=AxisKind.RESOURCE_SCALE, resources=("mfma",), values=(1, 2)),
        ScanAxis(kind=AxisKind.BANDWIDTH_SCALE, values=(1, 2)),
    ),
)
result = scan_profiles(request)
```

## 验收与证据

| 规格 | 当前 CPU 证据 | 边界 |
| --- | --- | --- |
| M06 | 独立有理数 oracle：120 ops、80 B、目标 2 s → 60 ops/s、40 B/s；两种全图视图反代一致 | 必要条件 |
| M07 | 同资源模式时间 10+3，目标 10，倍率 1.3；未扩展资源仍可能使目标不满足 | 不使用模式时间最大值替代求和 |
| M08 | 非法时间／速率、未知资源、缺失工作和不一致视图拒绝；unfused 及缺容量单列 | 无无穷大“成功”结果 |
| M09 | 独立四点期望 2/2/2/1 s，无遗漏／重复；实际 F01/F05 两路径另检查一次计数和四次计价 | 新增点单独计算分母 |
| M10 | 部分失败后恢复与新运行一致、完整点不再投影、预算／profile／工件漂移拒绝、总预算、点超时、缺 mapper、容量缺证、异常不变量保留；CPU 替身验证 deadline 展开 mapper 清理 | 合成容量曲线不是真实 mapper 最优流量证明 |

未提交证据目录为 `data/outputs/solar-stages1-5-20261026/`。`inverse/check-01` 与 `scan/check-01/02` 的失败保留；它们揭示文档／类型检查问题、初始分层循环、测试策略缺项，以及 cache 来源字段的往返错误。后者已修复为保留 `None`，并增加 profile 往返回归，没有放松身份比较。

`scan/check-03`：306 项受影响测试及全部质量检查通过。`scan/check-04`：1223 项相关 CPU 回归及全部质量检查通过。[逐槽位记录](solar-stage3-cpu-2026-10-26.md)已完成：32 个来源槽位中 20 个合格，80 个参数点、20 次反向查询、80 个恢复点全部完成。F/R 分母仍为 32/16；R 的 16 个槽位保持 `not_run`。采集源码摘要为 `64b378d8b78fd1619404183eef3dc53006bb4ac9213754a9e65f9a8a0989765b`。其后的增补仅改测试；最终生产源码与采集时逐文件相同。`check-05/pytest-artifacts` 额外保留失败、恢复及中断测试的完整运行工件。原始 BASE-01 的 627 个绑定仍有效。
