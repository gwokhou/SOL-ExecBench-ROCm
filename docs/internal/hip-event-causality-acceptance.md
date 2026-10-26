# HIP event 到 dispatch 的关联验收

后续实际 PyTorch eager 的完整性、隐式 dispatch 与范围绑定结果，见
[PyTorch 范围验收及第 3 项关闭边界](hip-torch-scope-acceptance.md)。本页保留最初的关联验收记录。

本次实现限定在诊断 sidecar。目标是从一次采集中识别有依据的
`producer dispatch → RECORD → WAIT → consumer dispatch`，并逐项报告无法关联的等待。
它不启用跨 stream 耗时预测，不把屏障持续时间解释为可删除的同步开销。

## 关联依据

GPU HIP event 的 correlation ID 绑定同一 capture 内的主机 API；校验成功返回码、
event handle、线程、stream 和设备/queue。主机提交先后选择 WAIT 前最后一次 RECORD；
kernel dispatch 再通过各自的 launch correlation ID 绑定到同一 stream 中观测到的提交。
结果保留四个主机调用 ID、两个 dispatch ID、event handle、设备/queue/stream、
capture、进程及源文件 SHA-256。correlation ID 只作身份，不作排序键。

支持范围是同进程、同设备、单主机线程提交、显式非默认 stream、成功的普通
record/wait/launch。扇出和已完成事件的重录均应正确关联；destroy 切断旧记录。
不依靠 GPU 完成时间寻找最近事件，也不把独立 stream 的时间邻近转换成依赖。

HIP 文档规定，事件记录覆盖此前状态，仍在记录时再次记录的结果未定义；
因此观测到未完成重录时必须拒绝对应关联。
[事件语义](https://rocm.docs.amd.com/projects/HIP/en/latest/reference/hip_runtime_api/modules/event_management.html)
与 [stream wait 语义](https://rocm.docs.amd.com/projects/HIP/en/latest/reference/hip_runtime_api/modules/stream_management.html)
是解释提交关系的依据；真实运行验收另行记录，不由文档代替。

## 验收条件

| 层次 | 必须满足 |
|---|---|
| CPU 正例 | 单次等待、扇出、重录、capture/进程隔离；反转文件排列不改变边；GPU 屏障完成先后不决定 generation |
| CPU 反例 | 缺 host/barrier/dispatch、重复身份、失败返回码、错误 handle/agent/queue、默认 stream、多线程、调用重叠、destroy 后未 record、未完成重录、graph、超限或未结束 capture 均不得伪造对应边 |
| 生产接入 | profile summary 使用当前 schema；仅成功采集提供关联；保留未解决项和所有证据限制；反馈不提供同步删除建议 |
| GPU 正例 | 独立命名的 producer/consumer kernel 构成答案；reuse、destroy/recreate 各有三条指定边，独立 kernel 不入边；数值全部正确 |
| GPU 反例 | independent 无 event 边；unrecorded wait 无 dispatch 边且明确未解决 |
| 重复性 | 四种场景各三次独立进程，交替执行顺序；每次独立保存原始 trace 和解析结果 |
| 运行环境 | 目标 gfx1200、已核验 STABLE_PEAK、现有宿主所有权/容器清理机制；结束后恢复并核验 AUTO；不混入性能对照 |

## 证据边界

即使所有观测 WAIT 都能关联，结果仍是 `partial`：普通 trace 本身不能独立证明
没有丢记录，也未绑定候选/workload 的完整 dispatch 范围。若同一事件的中间一次
record 在 host 与 GPU 两侧同时缺失，仅靠剩余记录无法发现；不能宣称完备性。
本次只验收在上述范围和条件下的正向诊断关联，不验收完整调度 DAG 或性能预测。

rocpd 继续提供 GPU 屏障观察；当前格式没有这里必需的主机成功返回值，因此因果关联
只读 JSON。默认 opt-in rocpd 诊断采集同时保留 JSON，不跨格式拼接，也不重复相加。
IPC、HIP Graph、默认/PTDS stream、多线程竞争、跨设备、WAIT 前的 stream 销毁或设备重置、其他 launch 形式和未知记录
保持未解决或拒绝。部分依赖已知不等于其他依赖不存在。

运行入口为 `scripts/internal/upgrade/event_causality.py`，HIP fixture 为同目录下的
`hip_event_causality_probe.cpp`。原始结果单独归档到 `data/outputs/hip-event-causality-20261011/`。

## 本次验收结果（2026-10-11）

[GPU 验收回执](../../data/outputs/hip-event-causality-20261011/attempt-20261011-213120/receipt.json)
保存镜像 digest、采集时的代码及二进制 hash、目标 UUID、时钟 lease 和每次 trace 的 hash。
镜像为本地已核验的 ROCm 10.1.0 / PyTorch 2.14.0 验证镜像
`sha256:0ec2e6cba29e673ebe5ad83b7049e5044647fc340174d1c855379efaf36409dd`。
实际 fixture 是原生 HIP 程序，本次不外推到任意 PyTorch workload。

| 场景 | 运行次数 | 每次的关联结果 | 结论 |
|---|---:|---|---|
| 同一 event 完成后重录 | 3 | 第一代 producer → 两个 consumer；第二代 producer → 第二代 consumer | 三条指定边全部匹配 |
| event 销毁后重建 | 3 | 同样三条指定边 | 未误接旧生命周期 |
| independent | 3 | 零条边 | 未把时间邻近视为依赖 |
| unrecorded wait | 3 | 零条边、一个未解决 WAIT | 未伪造 producer |

所有 fixture 数值检查通过。重录/重建场景中的独立 kernel 均未入边，共确认 18 条指定边。
宿主 guard 只修改目标 UUID；另一 AMD GPU 保持 AUTO。结束时先确认容器退出，再恢复
目标 AUTO，回执确认 lease 为 `restored` 且可重新取得 GPU 锁。

保留了两次先前尝试：`attempt-20261011-212940` 因脚本误用本地取锁接口，在提交 GPU
工作前超时；改为现有 `acquire_evaluation_gpu_lock` 后恢复。`attempt-20261011-213036`
已取得三条正确边，但答案比较器未处理真实符号的 `.kd` 后缀而拒绝。两次均已核验 AUTO；
它们不计入 12 次成功验收，也没有删除或混入其他性能对照。

后续 CPU 收尾补充了 decoded-JSON 往返读取和缺失 destroy 身份/stream 生命周期的拒绝。
最终解析器对这 12 份原始 trace 离线重放，必须与采集时保存的关联结果逐项一致；
[最终重放回执](../../data/outputs/hip-event-causality-20261011/replay-final/receipt.json)
确认 12 份结果逐项一致：18 条边、3 个未解决 WAIT；同时验证完整 profile summary 的
JSON 往返读取和反馈仍为 informational/unknown。回执附最终代码 hash 及源文件快照。
CPU 重放不增加 GPU 采集次数，也不生成性能证据。

[CPU 回归日志](../../data/outputs/hip-event-causality-20261011/checks/focused.log)记录
607 项相关测试通过；[静态检查回执](../../data/outputs/hip-event-causality-20261011/checks/static-results.json)
记录 Ruff、格式、ty、coupling、readability、schema、文档、production reachability 和
Python reuse 检查全部通过。这是本次改动的相关测试集，并非整个仓库全量测试。

本阶段可关闭“限定范围的 event → dispatch 诊断关联”。完整跨 lane 调度资格保持未关闭：
下一道验收门槛是可独立核验的 trace 完整性，以及 capture dispatch 到 workload/candidate
完整执行范围的绑定。当前不通过继续猜测 GPU 时间戳来绕过这两个门槛。


后续进展：已在[独立账本与范围验收](hip-trace-scope-acceptance.md)中关闭受控原生执行器的
两项缺口；通用采集仍保留上述边界。新的显式 `hipEventSynchronize` 证据可确认重录前完成，
避免将滞后的 event 观察结束时间误判为未完成重录。原有 12 份结果重放保持不变。
