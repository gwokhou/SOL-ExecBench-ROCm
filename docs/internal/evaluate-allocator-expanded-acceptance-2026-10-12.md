# 关闭 expandable segments：扩大生产验收

2026-10-12。本轮扩大的是已有生产规避的验收范围，生产实现与正确性标准均未修改。
默认仍为 `PYTORCH_ALLOC_CONF=expandable_segments:False`，沿用公共 evaluate、
独立参考 worker、逐次计时输出检查和 scope 资格检查。前一轮三个不规则 FP16 尺寸的
结果见[首次验收](evaluate-allocator-default-acceptance-2026-10-12.md)。

## 为什么选这些场景

此次用四类输入精度、规则与不规则尺寸、不同计算形式，以及反复分配和双向传输，
检查规避是否只对原始矩阵乘用例有效。不是把有限通过外推到所有 allocator 或 GPU。

| 分组 | 唯一 workload 数 | 覆盖内容 | 正确性检查 |
| --- | ---: | --- | --- |
| FP16 matmul | 4 | 原始三个不规则尺寸，加 `64×64×64` | 原始 `atol=rtol=0.00125` |
| BF16 matmul | 3 | `M/K/N=64/64/64`、`359/127/211`、`257/333/129` | 沿用 FP16 的 `0.00125`，未使用更宽的 BF16 容差 |
| FP32 matmul | 3 | 同上三个尺寸 | 原 FP32 square matmul 的 `0.0000125` |
| FP32 softmax | 3 | `128×128`、`1×131072`、`17×127` | 原始 `0.0000125` |
| BF16 RMSNorm | 3 | `128×256`、`17×333`，后者另测标准差 `0.0001` 的输入 | 原始 `0.0125`，`eps=0.00001` |
| FP32 行求和 | 2 | `128×256`、`17×333` | 原始 `0.0000125` |
| FP16、BF16、FP32、uint8 传输，各一组 | 16，共四组 | 每种 dtype 各四种长度，保留存活副本、分配/释放、CPU↔GPU 往返 | 两个输出均逐字节 ExactCheck |
| **合计** | **34** | **10 组** | 数值检查均要求全部元素通过 |

原始三个 FP16 尺寸保持 `M/K/N=1823/781/511`、`359/127/211`、`1024/333/717`。
新尺寸复用相应原始检查；新增 dtype 使用独立定义，未把它们伪装成原始语料。
验收后另核对保留用例的 axes 与 checks，以及 FP32 matmul 的检查来源。

每个正例分别用 seed 200、201 运行 `--profile none` 和 `--profile rocprofv3-scope`。
每个 scope 结果绑定 workload、候选源码和执行身份。每组另有一个故意错误的候选，
使用相同候选名但不同源码；它必须数值失败，scope 也必须拒绝。

## 传输检查怎样避免参考一起出错

每种 dtype 测试 `45593`、`65536`、`524289`、`2097153` 个元素。
其中 `524289` 个 FP32 元素和 `2097153` 个 uint8 元素略大于 2 MiB，
让有限集合包含分配大小边界附近的情况；没有据此识别分配器内部真实分段，也没有制造 OOM。

可信参考 worker 先在 CPU 生成循环整数模式，并在前 64 个元素加入每次调用的新值，
私下保留 CPU 真值。输入上传后，参考先检查 GPU 输入回传是否仍等于 CPU 真值，
再由真值生成预期输出。候选只收到 GPU 输入，拿不到该私有变量。
这适用于本轮串行、单进程的可信参考入口，不构成对恶意候选的独立安全隔离证明。

候选先 clone 输入，保留这个存活副本；随后执行 12 轮大小变化的临时分配、已知值填充、
D2H/H2D 和释放。临时长度依次为 `L-1`、`L`、`L+1`、`L+4096`，重复三次。
每轮检查往返后的已知填充值。最后再对输入做 CPU↔GPU 往返，返回存活副本和往返结果。
两个输出均以 uint8 存储视图送入现有 ExactCheck，因此浮点输入也按原始字节比较，
没有靠数值容差掩盖传输损坏。负例保留第一个正确输出，只把第二个输出置零，检查器必须拒绝。

这里仍有边界：CPU 真值在进入 GPU 输出通道后也使用相同硬件和传输实现。
此设计降低参考从受损 GPU 输入生成真值的风险，但不能排除所有共同路径故障。
输入只使用有限整数模式和低幅值测试，没有覆盖所有 NaN、Inf、次正规数或随机位模式。

## 生产路径与时钟条件

调用的是公共 CLI 和当前生产 driver，没有修改 driver 副本。
启动器故意把 `PYTORCH_ALLOC_CONF` 及两个旧别名都设置为 True；
候选和参考代码加载时均断言实际环境为 False、旧别名不存在。
这样检查到的是子进程收到的环境，不只是启动器自述。

每条正例保留十轮正确性检查，并检查两轮 trial 各 8 次计时调用的输出；warmup 为 2。
框架仍要求每次调用使用不同输入，未绕过计时协议。eager 正例和参考通常执行相同数学算法，
因此主要检验生产执行、隔离、分配和传输；零误差也不能证明共同使用的数学库没有算法错误。

目标为 gfx1200 / RX 9060 XT，BDF `0000:05:00.0`，
ROCR UUID `GPU-a30fa29c1cca1511`，SMI UUID `a3007590-0000-1000-800f-a29c1cca1511`。
镜像固定为 `sha256:0ec2e6cba29e673ebe5ad83b7049e5044647fc340174d1c855379efaf36409dd`，
Python 3.14.4、PyTorch 2.14.0+rocm10.1.0、HIP 7.16.26385。
这是既有本地验证镜像，不是未经修改的官方镜像认证。

目标卡由既有 host guard 持有 STABLE_PEAK 租约；配置内 `lock_clocks=false`
表示外层 owner 管理时钟，不表示未锁频。另一张 AMD GPU 不参与实验，保持 AUTO。
每个窗口结束后先确认容器消失，再恢复并核验 AUTO、owner restored 和排他锁可重新取得。
时钟恢复结果只证明清理成功，不作为 AUTO 下的性能测量。

## 结果与证据

**本轮代表性验收通过。** 34 个正例 workload × 两个 seed × 两种评估模式，
形成 136 条通过的 canonical trace；其中 68 次 scope 申请全部合格。
十轮正确性检查共 2000 项：720 项数值检查、1280 项逐字节检查，全部通过。
计算组的十轮最大绝对误差为 0，所有正例的逐次计时输出验证均通过。

| 验收项 | 结果 | 能支持的判断 |
| --- | ---: | --- |
| 普通 evaluate 正例 | 68/68 PASSED | 所测计算与传输路径在生产规避下通过 |
| 带 scope 的 canonical 正例 | 68/68 PASSED | 收集资格证据时仍通过相同正确性标准 |
| workload/candidate 绑定的 scope | 68/68 qualified | 捕获范围内共 1436 个 dispatch，包括 224 个标记为库内隐式的 dispatch |
| 故意错误候选 | 10/10 INCORRECT_NUMERICAL，CLI 非零 | 错误结果仍被拒绝 |
| 错误候选的 scope | 10/10 rejected | 拒绝原因均为 `scope_canonical_evaluation_not_passed` |
| 无 GPU 的原始 scope 重放 | 68/68 与原结果一致 | 留存产物可以用当前资格检查器复核 |
| 语义破坏负例 | 50/50 拒绝 | 缺证据、丢记录或身份错误不能靠重新计算 hash 获得资格 |

50 个证据负例来自每组一份真实捕获，分别删除 ledger、设置非零 drop count、取消 finalized、
改错 intent 摘要、删除一个 SDK export dispatch。修改后重新计算 artifact 摘要，再调用
生产资格检查器；因此检查的是产物语义与交叉一致性，不只是 hash 不匹配。
这些是 CPU 离线负例，不是额外 50 次 GPU 运行或对实际采集丢包率的估计。

每组跨两个 seed 的合格结果绑定相同正例候选源码，各 workload 摘要互不混用；
同名错误候选的源码摘要与正例不同。库内隐式 dispatch 的计数只涵盖被检查器识别并捕获的
kernel dispatch，不代表全部 DMA 传输或内存依赖已经获得资格。

总结果独立保存于 [report.json](../../data/outputs/evaluate-allocator-expanded-20261012/report.json)。
源码和文档快照保存在该归档的 `final-source/`，
[manifest.json](../../data/outputs/evaluate-allocator-expanded-20261012/manifest.json)列出各文件的 SHA-256；
这是本地留存与一致性检查，不是外部签名或安全认证。
接受的计算结果分别来自
[首个窗口](../../data/outputs/evaluate-allocator-expanded-20261012/attempt-20261012-190025/receipt.json)和
[最终窗口](../../data/outputs/evaluate-allocator-expanded-20261012/attempt-20261012-191007/receipt.json)；
首个窗口的整体状态仍为 failed，不把其中局部成功改写成整轮通过。
另一个失败窗口的[AUTO 回执](../../data/outputs/evaluate-allocator-expanded-20261012/attempt-20261012-190641/receipt.json)
也保留。[离线重放及负例回执](../../data/outputs/evaluate-allocator-expanded-20261012/attempt-20261012-191007/offline/receipt.json)
记录全部判定原因。

本轮 CPU 预检通过 34 个问题契约与 48 个传输探针，后者为每个传输 workload 三次不同输入；
后审计核对了实际 trace、原始检查来源、续跑 provenance 和生产源码摘要。
文档链接检查与 `git diff --check` 通过。此次没有修改生产 Python 实现，
没有重复首次验收中的 875 项 CPU 回归；该历史结果仍按首次验收范围解释。

## 准备失败与重复计数边界

首个窗口的 12 个计算组命令已通过，产生 36 条正例 trace、18 份 scope 资格；
随后传输定义因对浮点输出直接使用 ExactCheck，被问题契约拒绝，尚未执行传输候选。
现有检查要求整数或布尔输出，因此改成 uint8 存储视图，仍保留实际浮点输入。

第二个窗口中，固定传输输入违反每次计时调用必须使用不同输入值的要求；
首个用例被判为 REWARD_HACK，其后三个出现未完成上一轮验证的连带 RUNTIME_ERROR。
这不是本轮观察到的数据损坏。之后使用每次调用更新的随机前缀和可信 CPU 真值重新验收，
没有放宽框架契约。两次失败均保留原始日志和恢复回执，失败传输结果不算通过。

已经成功的 12 个命令没有重跑。继续运行前，核对生产 Python 源码以及这些分组的定义、
workload、候选和配置摘要；它们的结果复制到最终目录，并明确标注原始窗口。
最终统计按一次逻辑结果计数，不能把目录副本当作新实验。
归档中早期生成的固定模式 safetensors 是未使用的准备材料；最终传输输入来自自定义生成入口。

## 这次通过后仍保留哪些门槛

- 只支持本机、这个镜像和这 34 个 workload 的有限生产规避资格，不证明 HIP/VMM 已修复。
- 环境变量控制默认 PyTorch allocator，不能禁止自定义候选或第三方库自行调用 HIP/HSA VMM。
- scope 仍为单进程、单提交线程 eager 执行；内存依赖、Graph、多线程、多 GPU、训练和 OOM 压力均未取得资格。
- `performance_authority`、`schedule_complete`、`adversarial_attestation` 继续为 false。
  本轮产生的时间数据只服务于执行协议检查，不并入升级收益或 GEMM 性能对照。
- 没有新增异常退出故障注入，也没有重复全仓库 GPU 测试；失败窗口的恢复是实际观察，
  不能替代对所有信号和断电场景的证明。

恢复 expandable segments 仍需单独通过可触发冻结基线与修复版的数据完整性验收，
再评估容量、碎片或性能收益。[上游阻塞分析](vmm-upstream-blocking-assessment-2026-10-12.md)
中的恢复启用门槛不因本轮普通分配通过而降低。
