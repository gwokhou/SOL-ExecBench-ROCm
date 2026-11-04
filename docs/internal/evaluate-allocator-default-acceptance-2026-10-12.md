# 生产关闭 expandable segments：修复与验收

2026-10-12。生产子进程现在统一使用 `PYTORCH_ALLOC_CONF=expandable_segments:False`。
普通 evaluate、参考 worker、编译和 scope 重放共用环境构造入口，父进程即使设置 True，
也不会重新开启该模式；旧变量别名同样不会传入。正确性阈值、十轮数值检查、逐次计时
输出验证和 scope 拒绝规则均未修改。

这是对已复现故障的生产规避，**没有修复底层 HIP/VMM 实现**。原因及责任边界见
[数值失败排查](fp16-evaluate-vmm-diagnosis-2026-10-12.md)和
[API 契约与上游反馈核查](hip-vmm-contract-assessment-2026-10-12.md)。

后续[扩大代表性验收](evaluate-allocator-expanded-acceptance-2026-10-12.md)已通过：
34 个 workload 覆盖 FP16/BF16/FP32 matmul、softmax、RMSNorm、行求和以及四种 dtype 的
反复分配与双向传输。两种 seed、普通 evaluate 与 scope 共 136 条正例 trace、68 份合格 scope；
10 组错误候选和 50 个离线证据负例全部拒绝。仍不放宽数值标准，也不扩大完整调度或性能资格。
下文保留首次验收原始范围，不把两批数据混为一次实验。

## 验收条件和结果

验收直接运行仓库公共 evaluate 命令，不使用修改过的 driver 副本。
保留原始三个 FP16 workload，`M/K/N` 分别为 `1823/781/511`、`359/127/211`、
`1024/333/717`。两个正确候选同名，分别执行 `A @ B` 和 `copy_(A)` 后再 `A @ B`；
第三个候选返回全零，用于检查错误结果仍会被拒绝。启动器故意继承 True 设置，
检验生产环境清理是否生效。

| 公共 CLI 场景 | 正常评估 | scope 结果 |
| --- | --- | --- |
| matmul，`--profile none` | 3/3 PASSED | 不申请 scope 资格 |
| matmul，`--profile rocprofv3-scope` | 3/3 PASSED | 3/3 qualified，每次 1 个范围内 dispatch |
| copy + matmul，同上 | 3/3 PASSED | 3/3 qualified，每次 2 个 dispatch，其中 1 个为库内隐式 copy |
| 全零错误候选，同上 | 3/3 INCORRECT_NUMERICAL | 3/3 rejected，CLI 返回非零 |

九条正确候选 trace 的十轮检查最大绝对误差均为 0，逐次计时输出验证也全部通过。
错误候选以 `scope_canonical_evaluation_not_passed` 拒绝。六份合格结果分别绑定
三个不同 workload 摘要和两个不同候选源码摘要，未因候选同名而混用。
本次没有新增跨流 event 正例；已有事件资格边界保持不变。

## 时钟、环境与独立归档

目标为 gfx1200，BDF `0000:05:00.0`，ROCR UUID `GPU-a30fa29c1cca1511`。
使用既有验证镜像
`sha256:0ec2e6cba29e673ebe5ad83b7049e5044647fc340174d1c855379efaf36409dd`，
Python 3.14.4、PyTorch 2.14.0+rocm10.1.0、HIP 7.16.26385。
这不是未经修改的官方镜像认证。

整个 GPU 窗口由现有 host guard 持有目标卡的 STABLE_PEAK 租约，普通评估与 scope
重放均在该窗口内。结束时先确认容器消失，再恢复并核验 AUTO；owner 状态为 restored，
无 release blocker，排他锁可重新取得，另一张 AMD GPU 保持 AUTO。异常恢复机制
沿用生产 guard，本轮没有另行注入异常退出。

同一窗口补查了 HIP VMM 设备能力：返回支持值 1、最小粒度 4096 字节，补齐此前纯 HIP
复现未显式记录的前提。支持能力不等于实现没有缺陷。

原始命令、候选、trace、scope 三路证据、源码摘要和时钟回执独立保存于
[`attempt-20261012-114510`](../../data/outputs/evaluate-allocator-default-20261012/attempt-20261012-114510/receipt.json)。
总验收见 [report.json](../../data/outputs/evaluate-allocator-default-20261012/report.json)。
本次只验收正确性和有限 scope 资格，短时运行产生的 timing 文件不用于升级收益比较，
不并入已有性能对照。

## 回归测试和静态检查

新增三种环境回归：无外部配置、当前变量继承 True、旧别名继承 True；scope 启动测试
也验证实际传给执行器的环境。修改生产代码前，这四项均失败；修改后通过。
同步更新编译、普通评估和 profile 评估的环境断言。

相关 CPU 回归为 **875 passed、1 skipped**；跳过的是需要 Triton/GPU 的参考 JIT 测试，
本轮 CPU 测试容器未挂 GPU。实际 GPU 验收使用上表独立运行。
Ruff、格式、ty、耦合、可读性、schema、文档链接、生产可达性和复用检查均通过。
首次 ty 命令漏配测试 helper 的导入路径，产生导入错误；补齐 PYTHONPATH 后通过，
首次日志与重跑回执均保留。没有运行全仓库 GPU 测试。

## 仍不宣称什么

- 本机、这组输入上的生产规避有效，不代表 VMM 在所有 AMD 平台都会损坏数据。
- scope 继续保留 `performance_authority=false`、`schedule_complete=false` 和
  `adversarial_attestation=false`；不会因本次通过而扩大为完整调度或对抗性安全证明。
- 当前证据支持按 AMD HIP/VMM 路径的实现缺陷处理，但没有确定责任组件或提交，
  也没有证明与上游相似报告同源。
- 再开放 expandable segments 前，需要单独通过覆盖分配、释放、重映射及双向传输的
  数据完整性资格检查；不能用放宽容差或关闭验证替代。
