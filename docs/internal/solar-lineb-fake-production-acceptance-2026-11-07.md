# 线B fake 取图生产化：预固定验收规格

制定日期：2026-11-07（实施前预固定，分支 `lineb-fake`）。前置：L3-a
试点 adopt=true（D1–D4 全过，O2 文档全等 8/8 槽，提取墙钟 11–196×）。

## 0. 变更（预固定）

生产 `make_fx_backend.extract_operator_graph` 切换为 FakeTensorMode 静态
取图（试点验证的实现：observe 与 trace 均在 fake mode 下，签名来自
fake 输出的 shape/dtype）。试点模块 `make_fx_fake.py` 删除（生产接管，
按惯例原子清除废弃路径）；其直接引用者同提交原子更新：
- `tests/solar/test_make_fx_fake.py` → 改为生产 backend 的行为测试
  （含数据依赖拒绝负例，与试点语义一致）；
- `scripts/internal/solar/l3a_worker_pilot.py` 的 fake 变体 → 调生产
  backend（历史 campaign 工具，语义随生产接管）。
torchview 默认路径零改动。

## 1. Gate（预固定）

1. **B1 全语料图等价**：163 个 scored workload × make_fx：新 backend
   （fake）产物 `operator_graph.yaml` 与 pristine-HEAD backend（real）
   产物 **逐字节 sha256 相等**（双 worktree 提取，CPU，逐槽隔离）。
   163/163 必须全等；任何不等即失败（不得放宽为投影等价）。
2. **B2 单元**：数据依赖分支拒绝、梯度捕获、序列化往返等试点测试
   全部在 backend 上通过；`tests/solar`+`tests/scripts` 全绿。
3. **B3 GPU 校验**：链窗口外执行 ≥3 个代表槽（含 R06 动态索引）在
   cuda:0 上 backend 提取 + 转换 + 验证回放通过。
4. **B4 收益记录**：提取墙钟对比（抽样 ≥8 槽）记录进证据，不设阈值
   （收益已在 L3-a 证明，此处仅复核方向不回退）。

## 2. 证据

`data/outputs/solar-lineb-fake-20261107/{equality-matrix.jsonl,
gpu-validation/, lineb-summary.json}`（主树提交）。

## 3. 边界

- 不改 torchview、转换、验证、access 语义；fake 仅替换 make_fx 取图
  内部实现，接口签名不变。
- 线C 链运行期间主树 src 冻结——本线全部开发在独立 worktree，合并
  在链完成后进行。
