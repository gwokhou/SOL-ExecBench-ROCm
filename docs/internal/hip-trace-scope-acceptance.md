# HIP trace 完整性与 workload 范围验收

后续进展：已将 SDK 提交/完成核对接入实际 PyTorch eager workload，并覆盖内部复制
helper；本页保留原生执行器的验收范围。见 [PyTorch 范围验收](hip-torch-scope-acceptance.md)。

本轮先验收受控、单进程、单主机线程、单 GPU 的原生 HIP 执行器。
“完整”限定为该执行器公开且经过审计的 launch / event / synchronize 记录集合；
它不是所有 ROCm 内部活动、内存访问或任意第三方 workload 的完备证明。

## 可验证条件（实验前固定）

1. 执行器在 profiler 之外写调用账本，包含 run ID、workload ID、候选二进制 SHA-256、
   PID/TID、GPU UUID，每次 launch/RECORD/WAIT/destroy/设备同步的顺序、参数和返回码。
   账本有正常结束标志及数值验证结果；重录前增加同一 event 的成功 `hipEventSynchronize`。账本来源须绑定经过审计的执行器和实际运行文件，
   不能由解析器从 trace 反推，也不能接受候选程序自行声称已完整记录。
2. 采集前由控制器生成唯一 run ID，固定 workload、执行命令、输入场景与二进制 hash；
   执行后要求正常退出。trace 的命令、进程、设备身份必须与这份预期和账本一致。
3. 账本和 trace 的上述 host 调用逐项相等；每个 launch 恰好对应一个完成的 dispatch，
   每个普通 RECORD/WAIT 恰好对应一个 GPU 屏障。总数相等不足以通过：还核对 correlation、
   stream、event handle、kernel 名称及提交顺序，拒绝多余项、重复项、未知 launch 和额外线程。
4. 全部 dispatch 必须在成功的起始/结束设备同步之间提交并完成，且全部绑定到同一 workload
   和候选；不允许先过滤 helper、未知 kernel 或额外 queue 再宣称完整。
5. 负例必须覆盖同时删除 host 与 GPU RECORD、同时删除 launch 与 dispatch、调换调用、
   重复身份、额外 dispatch、错误 run/PID/GPU/候选身份、缺终结同步、缺账本结束、失败返回码、
   修改原始文件及错误 schema。所有负例必须拒绝资格，不能保留“已通过”字段。
6. GPU 正例沿用已知 kernel 答案：reuse、recreate、independent 各三次独立进程；
   STABLE_PEAK 下采集，复用现有宿主 guard，结束后核验 AUTO。未 RECORD 的 WAIT 在本轮
   严格资格中拒绝，因为不能要求其一定产生 GPU 屏障；它仍可作为上一阶段的 partial 诊断。

## 资格的停止边界

通过后可以说明：对这个经过审计的受控执行器，本次相关记录集合与独立账本一致，
所有 dispatch 已绑定到指定 workload/candidate。普通 JSON 的 event 关联仍维持 partial，
不会因另一次受控实验通过而自动升级。此次也不启用性能预测或同步删除建议。

账本和 profiler 仍共享 HIP runtime，不能排除共同实现缺陷；hash 是绑定而非可信来源证明。
是否信任执行器由上层控制器的源码/二进制审计决定。未经相同审计的 PyTorch、HIP Graph、
多线程、子进程、库内隐式提交等不获此资格。未映射的 memory-copy 等活动也不成为完整调度边。

SDK [buffered services](https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/api-reference/buffered_services.html)
提供 lossless buffer、drop count 和显式 flush。这些机制可支持后续通用采集器，
但当前普通 JSON 不暴露可独立核验的全链路 loss/flush 回执，所以本轮使用独立调用账本核对，
不以退出码、fini_time 或连续 dispatch ID 单独证明无丢失。

原始证据单独归档于 `data/outputs/hip-trace-scope-20261011/`，不进入现有性能对照。

## 实测中修正的字段假设

第一次采集暴露了 SDK 符号表包含零号占位符，以及 HIP UUID 文本与 SDK GPU ID
小端字节的编码差异。适配后仍按完整身份核对，不忽略设备或实际 dispatch 的 kernel。

第二次采集中，event 记录的 end_timestamp 比末尾设备同步返回晚约 150 微秒，而
全部 kernel dispatch 均在同步返回前结束。这证明此环境不能把 event 记录结束时间直接
充当精确的 GPU 完成边界；具体延迟来源尚未确认。

验收继续要求全部 dispatch 的提交及执行落在两次设备同步之间。event 的 host 提交必须
位于同一范围，GPU 屏障记录必须与账本逐项对应且存在于正常结束的 capture 中；不要求
其观察时间早于设备同步返回。重录通过对同一 event 的成功 `hipEventSynchronize` 证明
前一代完成，避免用一个可能滞后的观察时间判定未完成重录。此规则有成功/失败同步反例。

SDK 对 HIP event 完成后才发出记录、以及某些 WAIT 不产生屏障的说明见
[HIP event API 定义](https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/develop/_doxygen/rocprofiler-sdk/html/fwd_8h_source.html)。
这不证明 event 的记录区间就是可归因的阻塞时间；本轮不作这样的解释。

## 验收结果（2026-10-11）

[宿主与 GPU 回执](../../data/outputs/hip-trace-scope-20261011/attempt-20261011-220719/receipt.json)
确认 9 次独立进程全部通过，全部使用目标 gfx1200 的 STABLE_PEAK；容器退出后恢复并核验 AUTO，
另一 AMD GPU 保持 AUTO，时钟所有权回执为 restored，GPU 锁可再次取得。

| workload | 次数 | 每次全部 dispatch | 每次 event 依赖 | 数值结果 |
|---|---:|---:|---:|---|
| reuse | 3 | 6 | 3 | 全部正确 |
| recreate | 3 | 6 | 3 | 全部正确 |
| independent | 3 | 1 | 0 | 全部正确 |

39 个 dispatch 全部绑定到所属 run ID、workload、候选二进制 hash、进程和 GPU；
18 条事件依赖符合独立 kernel 名称答案。采集前的 `controller-intent.json`、独立的
`journal.json`、原始 JSON/rocpd、采集退出状态、文件 hash 绑定和 `scope.json` 分别保存，
不是从一份 trace 生成多份互相引用的“独立证据”。

[最终离线验收回执](../../data/outputs/hip-trace-scope-20261011/replay-and-faults-final/receipt.json)
覆盖 9 份正例逐项重放、16 类真实 trace 副本破坏拒绝，以及上一阶段 12 份关联结果保持不变。
副本测试重新计算其输入 hash 后仍须因内容不一致而拒绝，排除了“只靠 hash 改变就通过负例”的
循环验证；另有两项专门验证保留 hash 不匹配。原始证据始终保持原样。

两个失败尝试也已保留：`attempt-20261011-215928`（格式适配）和
`attempt-20261011-220218`（event 观察时间不满足错误的设备同步边界假设）。
两次退出后均核验 AUTO，不计入 9 次成功验收；后一次促成了上述更准确的完成证据规则。

首次旧 fixture 的 trace 含 6 个显式 kernel 之外的 2 个内存辅助 dispatch。这不是过滤的理由。
本轮使用映射主机内存，使经过审计的闭合程序拥有明确的全部提交集合；正例不包含隐式库 kernel。
额外 helper 的副本反例验证了出现这类提交时资格会被拒绝。

## 接入与后续边界

生产侧入口为 `qualify_hip_trace_scope`，位于
[hip_scope.py](../../src/sol_execbench/core/bench/rocm_profiler/hip_scope.py)。
`sol_execbench.hip_trace_journal.v1` 是独立账本契约，
`sol_execbench.hip_trace_scope.v1` 是限定范围的资格回执。控制器必须提供预先固定的
`HIPScopeExpectation` 和执行后保留的两个文件 hash；校验失败抛出稳定原因码，不能返回部分合格回执。
源码 hash 绑定调用方对执行器的审计；校验函数自身不会证明任意二进制已被可靠地插桩。

本阶段关闭的是**受控原生执行器的相关 trace 记录集合完整性和全部 dispatch 范围绑定**。
普通 profile summary 的 event 关联继续保留 partial；它没有独立账本时不能自动升级。
范围回执明确 `performance_authority=false`、`schedule_complete=false`：
设备同步、内存操作等其他依赖域尚未转换成完整调度图，不能直接用于跨 stream 预测。

通用 workload 的下一步需要在可信执行边界生成覆盖库内提交、线程和子进程的账本，或让 SDK
采集器提供可核验的回调、drop/flush/finalization 回执，再按同样的一一对应与否决规则验收。
候选自己输出一份“完整账本”、同源 JSON/rocpd 数量相同、或正常退出，均不能替代这些证据。


[相关 CPU 回归](../../data/outputs/hip-trace-scope-20261011/checks/focused.log)为 635 项通过；
[静态检查](../../data/outputs/hip-trace-scope-20261011/checks/static-results.json)中 Ruff、格式、ty、
coupling、readability、schema、当前文档、production reachability 和 Python reuse 全部通过。
这不是仓库全量测试。[汇总回执](../../data/outputs/hip-trace-scope-20261011/report.json)
绑定上述回执与最终源码快照；实验二进制及两个失败尝试保留在同一独立目录。
