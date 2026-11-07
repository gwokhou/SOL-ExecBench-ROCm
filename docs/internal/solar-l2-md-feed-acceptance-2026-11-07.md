# L2 M-d 生产供数管线：预固定验收规格

制定日期：2026-11-07（实施前预固定）。前置：L2 受限域闭合
（`39329249`，其规格 §7.4 将本线登记为遗留：跨 artifact 标识桥 +
显式 opt-in）。

## 0. 语义设计（预固定）

1. **显式 opt-in**：`PerformanceDiagnosticBuildRequest` 增
   `schedule_scope_path: Path | None = None`；缺省 None 时构建产物与
   现状**逐字节一致**（能力与资格分离；普通 profile 不自动升级）。
2. **标识桥**（scope archive ↔ counter evidence）：scope.json 的
   `dispatches[].dispatch_id`（int）与 counter CSV `Dispatch_Id`（同一
   rocprofv3 分发号空间的字符串形式）经 `str()` 对齐；边端点必须
   ⊆ counter 侧已知 dispatch_id 集合，任何未知端点 → 整体 fail-closed
   （reason `scope_dispatch_identity_unresolved`），不做部分喂入。
3. **边提取**：仅取 `links` 中 **双端均为 dispatch** 的跨 lane 事件边
   （`predecessor_dispatch_id` 与 `successor_dispatch_id` 均非空，且两
   dispatch 的 lane 标识不同）；copy 前驱边（predecessor 为
   copy_correlation）不构成 dispatch 边。scope 要求
   `schedule_complete == true` 且 `copy_domain` 合格，否则
   `scope_schedule_incomplete` fail-closed。scope.json 的
   `artifact_sha256` 绑定逐文件校验。
4. **喂入**：成功时 `cross_lane_verified=True` +
   `event_edges=(predecessor,successor)` 对（str）进入
   `_WorkloadDiagnosticInputs`；`predict_hw` 的既有消费路径零改动。
   管线状态 reason 记入 sidecar（如 `schedule_feed_edges_verified`）。
5. **CLI**：`sol-execbench diagnostics performance`（现名以代码为准）
   增 `--schedule-scope PATH` 透传。

## 1. Gate（预固定）

- **M1 桥单元**：以 L2 真实捕获（`solar-l2-schedule-20261106/
  full-captures/*`）的 scope.json 真实字节驱动——含双 dispatch 端点
  link 的案例提取出正确边；`schedule_complete=false` / 未知端点 /
  copy-only link / 坏 sha 各自 fail-closed 且 reason 正确。
- **M2 构建集成**：opt-in 时 `cross_lane_verified`/`event_edges` 到达
  predict_hw（schedule evidence v3 发布 `event_wait` 边）；无 opt-in
  时与改动前构建产物逐键相等（默认不变性回归）。
- **M3 CLI**：`--schedule-scope` 到达 builder（plumbing 测试）。
- **M4 GPU 双采集**：同一 workload 的 scope 采集（L2 harness 案例形
  态，含 dispatch↔dispatch 跨 lane 边）+ counter 采集，经生产 CLI 构
  建 → sidecar 的 schedule evidence 含 verified `event_wait` 边且
  `cross_lane_event_dependencies=verified`。
- **M5 负例**：同一证据不带 opt-in → 产物与带改动主干的无标志构建
  一致（零静默升级）。

## 2. 证据

`data/outputs/solar-l2-md-feed-20261107/{unit/, gpu-dual/, summary.json}`。

## 3. 边界

- 不改 predict_hw / build_schedule_evidence / scope 采集语义。
- performance_authority 与 adversarial_attestation 保持字面 false；
  M-d 输出是诊断解释，不是性能授权（承 L2 规格条款）。
- M4 若同一进程无法同时开 counter 与 scope 会话，按生产形态采用
  "同 workload 两次受控重放" 的双采集（两空间的分发号可比性正是本
  桥的验证对象），在证据中记录采集形态。
