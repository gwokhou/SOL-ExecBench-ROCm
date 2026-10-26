# SOLAR S4：受限结构零与需求传播

本页说明实现契约和证据边界；唯一进度入口仍是 [HANDSOFF](../../HANDSOFF.md#solar-evolution)。验收规格见 [S4-01–03](solar-stages1-5-implementation.md#2-s4受限稀疏与按需计算p2)。本次只使用锁定 CPU 镜像，不运行 GPU、真实 mapper 或 R 工作负载，不恢复暂停的发布资格工作。

## 数值域与原图

`ProjectionRequest.restricted` 是可选情景。默认 `numerical_domain="unrestricted_ieee"` 返回 `not_applicable`，保留原始 dense 分析。只有调用者显式选择 `finite_no_overflow_ignore_signed_zero`，才允许消去有结构证明的零乘项。这表示调用者声明所有操作数和中间量有限且不溢出，并接受不观察有符号零和浮点异常；不是从验证样本推导出的输入保证。非零收缩项不重排。

`0 * inf` 和 `0 * nan` 的反例保留为测试。输入样本中恰好出现的零不会成为结构零。当前可传播的零来自静态 `tril` / `triu`、已证明的 view、部分逐元素零保持规则，以及两个输入的静态收缩。`log_softmax` 不传播零；copy、mutation、atomic、opaque、动态索引、未覆盖 dtype 或操作是完整需求边界，并保留原因和原成本。

两个前端分别捕获、验证、回放：Extended 原生能力表新增 `tril` / `triu`，执行器直接调用对应公共 Torch 操作。对角线来自结构化参数解码，不读取显示字符串，不转用 make_fx。dense 外部读取使用三角区域证明；完整 mask 写入和既有整数选择工作保留。矩形、不同 diagonal、批次、空域和全域由独立坐标枚举检查。区域数量取决于倾斜边界，而非名义矩阵面积；超大但全空／全满的区域可常量成本接受。

## 需求传播和模型

分析先构造有预算的逻辑坐标关系，再反向合并所有消费者和返回输出的需求。支持静态 slice/select/getitem 与其他已证明的 view、受限逐元素广播，以及两个输入、无重复标签或省略号的固定形状收缩；native `mm` / 二维 `matmul` / 三维 `bmm` 进入同一坐标关系。未支持的形状或收缩不冒充已证明的优化。

F06-D/Q 原图 MAC 分别为 48/70，受限三角情景均为 30。每个算子另报输入读取、输出写入、按需输出、零初始化和资源工作。mask 仍完整物化；返回的内部 view 在实际生产者上合并输出存储，不把重叠 view 当成两份写入。共享输出阻止不合法的上游裁剪。外部读按逻辑 base 合并。直接返回外部输入或其别名仍报告不可用，不假定一次不存在的复制。未知边界仍可给出保留成本的计算对比，但外部读未证明时，相应完整时间视图不可用。

资源价格复用同一 profile 的资源归属和吞吐率，独立资源取最大值，共享模式相加。改写情景单独报告 unfused 和 ideal-fused；cache-aware 必须重新取得变换域适用的容量证明，目前明确不可用。完全没有结构选择的情景直接保留原图视图，不枚举大型 dense 张量。所有结果都是模型工作量／时间，不是测得的 kernel 加速，也不宣称稀疏硬件吞吐率。

预算分别限制坐标单元、收缩项和时间，先保留预算再分配坐标或项。预算耗尽记录 `budget_exhausted`，不写零成本或偷偷改用 dense 的受限估计。非法参数和内部不变量错误仍抛错。内部使用已经验证的 IR／access 契约，不重新加载参考程序。相同情景在一个复用会话内缓存计数，缓存键包含数值域与全部预算。

## 入口、身份与工件

公共入口仍为 `ReuseSession.project` 和 `solar reprice REQUEST`。请求在 `projection` 内增加：

```json
"restricted": {
  "numerical_domain": "finite_no_overflow_ignore_signed_zero"
}
```

输出的原 `solar-analysis.yaml` 保持 dense 含义；独立的 `restricted-computation.json` 使用 canonical `solar_demand` schema，绑定图摘要、约束、状态、成本、假设和资格。`reuse.json` 的完整清单绑定该工件；读取检查当前 schema、摘要、来源和约束，缺失／损坏不可恢复为成功。

`solar scan` 与 `solar requirements` 的请求增加 `computation="restricted"` 时选择受限情景，默认仍为 `dense`。受限扫描还必须提供 `projection.restricted`。扫描身份包含选择、数值约束、预算和规则摘要；旧点不能跨这些身份恢复。没有可用受限情景的查询明确不可用。CLI/桥接展示 dense 和 restricted 两组信息，并经隔离 CPU worker 执行。

复用记录只要包含受限情景，就不继承 `sol_score_eligible` 或 `publication_eligible`；受限结果自身这些字段恒为 false，`performance_authority=false`。原图的随机／边界／零样本和梯度验证只证明原 IR；受限域的输入条件仍明确是调用者声明。

## 验证证据

证据保存在未提交的 `data/outputs/solar-stages1-5-20261026/demand/`。前期检查发现的结构化 diagonal 解码、空三角读取集合、返回 view 的存储并集，以及测试／类型／格式问题均已修复；失败记录保留。`check-07` 通过 1364 项相关 CPU 测试；随后只修正测试风格，`check-08` 通过 67 项受影响测试及 Ruff、format、ty、coupling、readability 和文档检查。最终生产源码与完整回归及采集一致。

[逐槽位报告](solar-stage4-cpu-2026-10-26.md) 记录 32 个 F 槽位：24 个公共分析和受限计数完成，8 个 F07/F08 源证明不足而未进入复用；20 个受限 ideal-fused 视图可用，4 个 F04 reduction 边界保留不可用。24 个默认 IEEE 请求均明确拒绝零消去。F06 两路径均保持 dense 48/70 MAC，并给出受限 30/30 MAC；另有 8 个扫描点、8 个恢复点和 4 个逆向查询。F02 的 4 个收集脚本失败在独立目录补采成功，36 次进程尝试全部保留。采集前后源码相同，原始 BASE-01 的 627 项绑定和 417 个范围外既有源码文件未变。

这些证据完成 S4-01–03 的 CPU／fixture 验收。静态 view 仍受现有 access 规则约束；未证明的步长、别名或操作保留边界。没有实测 GPU 加速、真实 mapper 或发布资格结论。
