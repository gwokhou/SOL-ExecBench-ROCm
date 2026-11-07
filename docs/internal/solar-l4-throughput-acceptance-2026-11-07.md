# L4 吞吐工程线剩余三项：预固定验收规格

制定日期：2026-11-07（实施前预固定）。前置：L4 第 1 项（stage 测量
报告）已由 L3-a 交付；本文闭合第 2、3、4 项。路线图原则："按测量推进，
不预设方案"——每一项的结论由既有测量数据驱动，允许"数据判无需变更"
作为合法闭合。

## 0. 测量依据（已有证据，预固定引用）

- **L3-a 测量报告**（`data/outputs/solar-l3a-20261105/measurement/`）：
  R16 分母上 mapper 合计 30 次 8.2s、采样峰值 27MB——单 mapper 全核
  策略在常规槽位上不构成瓶颈。
- **L3-c 覆盖矩阵**（`data/outputs/solar-l3c-20261106/`）：conv 家族
  mapper 在全核（~2900% CPU）下仍耗尽 7200s 生产预算（5/5）——
  全核对单 mapper 搜索是**必要**而非浪费；降低线程只会更早触顶。
- **批量防护已存在**：`formal_mapper_job_limit` 已被 release runner
  （jobs 上限校验）与 `solar_bridge/scan.py`（身份绑定）消费。

## 1. L4-2 mapper 线程数（预固定）

**决策（测量驱动）**：默认保持全核（conv 数据证明必要；常规槽位
无感）。**可配置化**：`orojenesis_mapper_thread_count` 增加显式环境
覆盖 `SOLAR_OROJENESIS_MAPPER_THREADS`（正整数；未设置/非法值时
回退全核并保持现有语义；0、负数、非整数拒绝回退默认，不部分采纳）。
`scan_execution_identity` 等身份绑定随配置自然反映新值。

Gate T1：默认无环境变量时行为与现值逐位一致；合法覆盖生效并进入
身份哈希；非法值回退默认（不抛错、不静默采纳半值）；单测覆盖三分支。

## 2. L4-3 并发路径核查（预固定）

审计范围（生产路径）：release runner 并行提交、qualification CLI
`--jobs`、`solar_bridge/scan.py`、`run_cross_path_focus`（compare-paths）。

**已知缺口**：qualification 的 jobs 只校验 >0，与 runner 的
resource_policy 上限校验不一致（同一 jobs 值 runner 拒绝而 qualify
CLI 接受）。**修复**：将 jobs 上限校验提升为 `resource_policy` 公共
函数 `validate_formal_job_count`，两处共同消费；qualify 路径
jobs=1 保持无条件通过，jobs>1 走同一 CPU 预算校验。

Gate T2：审计表四路径各有"有界依据"结论；qualification jobs>上限
被拒、jobs=1 与合法值通过；runner 行为零变化（既有测试通过）；
compare-paths 若为串行循环则记录"串行即有界"。

## 3. L4-4 有界队列与解释器并行（预固定）

- **盘点**：生产路径全部任务提交点逐个列出提交机制与在途上限；
  release runner 的滑动窗口（active 字典 + 完成后补提交）按构造有界。
- **Executor.map(buffersize=)**：生产路径**无** `Executor.map` 调用点
  （盘点结论）；该子项以"无目标 + 盘点证据"闭合。
- **解释器并行**：显式拒绝并钉桩——torchview recorder 临时替换全局
  `torch.nn.Module.__call__` 与 torch 创建函数，同解释器并发取图不
  安全；取图保持进程隔离。若已有测试钉桩则引用，无则补一条注释性
  测试引用 recorder 源码事实。

Gate T3：盘点表落证据目录；无生产 `Executor.map`（rg 证据）；解释器
并行拒绝有测试或源码引用钉桩。

## 4. 总 gate 与证据

- 全量测试套件通过 + ruff/ty 全绿。
- 证据：`data/outputs/solar-l4-throughput-20261107/{audit.json,
  thread-config-probes.json, l4-summary.json}`。
- HANDSOFF + roadmap L4 状态更新 + 提交。

## 5. 边界

- 不改 mapper 二进制、Orojenesis 超时、扫描/顺序搜索语义。
- 不引入新的并发框架；解释器并行维持拒绝。
