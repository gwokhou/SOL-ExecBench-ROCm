# VMM 问题的上游状态与推进边界

核查日期：2026-10-12。本文区分上游公开状态、本机发布的实验记录，以及本库的工程准入判断。仅做分析，本次核查没有新增 GPU 实验或修改生产实现。

**本库不需要等该 issue 关闭才能推进升级主线。直接受阻的是重新启用 expandable segments / 可疑 VMM 地址复用路径及其收益验收；普通分配下的评估、限定 scope、SOLAR 与 GEMM 修复验收可以继续。** 这个判断来自已有生产规避验收，不是对整个 ROCm 栈的安全保证。

## 上游公开状态

通过 GitHub REST API 核查了 [TheRock #8744](https://github.com/ROCm/TheRock/issues/8744)、[用户指定评论](https://github.com/ROCm/TheRock/issues/8744#issuecomment-6121181907)、完整评论列表和 issue 时间线。截至本次读取，issue 为 `open`，标签为 `status: triage`，已指派 `lucbruni-amd`，没有里程碑。时间线没有给出关联修复提交或关闭事件，评论列表只有用户指定的这一条，没有维护者公开回复。指派和 triage 表示进入调查流程，不等于 AMD 已独立复现或确认根因。

原始 issue 报告的是 Radeon AI PRO R9700（gfx1201）、HIP 7.14、Linux 7.1.13 环境下，同一虚拟地址在 VMM unmap/release 后重映射时，其他存活缓冲区出现按页计的数据异常。作者还报告了接近 OOM 时的异常；换用新虚拟地址的有限对照未观察到错误。这些是报告者的观察，没有公开的维护者根因确认。[来源：issue 正文](https://github.com/ROCm/TheRock/issues/8744)

## 指定评论增加了什么证据

指定评论作者为 `gwokhou`，API 的 `author_association` 为 `NONE`。这是本机调查结果的公开发布，不能把它与本机记录相互引用后当成两份独立证据，更不能当成 AMD 独立复现。评论创建于 2026-10-12 17:09，更新于 17:15（北京时间）。[来源：评论 API](https://api.github.com/repos/ROCm/TheRock/issues/comments/6121181907)

评论明确增加了 gfx1200 / RX 9060 XT 的复现记录，环境为 SDK 包版本 10.1.0、HIP 7.16.26385、Linux 7.0.0-38-generic。它报告的是 GPU 上检查到的数据不一致，计数通过独立 coherent host-mapped 通道返回；与原始 gfx1201 的存活缓冲区整页覆盖，受害模式并不相同。[来源：指定评论](https://github.com/ROCm/TheRock/issues/8744#issuecomment-6121181907)

基本生命周期实验的每行都是三个独立进程，每进程 700 次迭代：

| 生命周期 | HIP VMM 错误迭代 | 直接 HSA VMM 错误迭代 |
| --- | ---: | ---: |
| 同 VA，每次 release/create handle | 9/2100 | 17/2100 |
| 新 VA，每次 release/create handle | 0/2100 | 0/2100 |
| 同 VA，保留 handle 后 unmap/map | 0/2100 | 0/2100 |

同 VA 新 handle 与新 VA 对照的 VA 预留大小一致；保留 handle 组的预留范围更小，不能视为完全相同条件下只改变 handle 生命周期。新 API handle 也不能证明物理页不同。直接 HSA 组绕过了 HIP/CLR 的 VMM 对象管理，但仍使用 HIP kernel 提交、同步和 coherent 计数通道，不能说已经排除了整个 HIP 执行路径。[来源：指定评论](https://github.com/ROCm/TheRock/issues/8744#issuecomment-6121181907)

最新评论中的有 trace 对照进一步区分了目标设备与另一设备的 replay：

| 原 UNMAP 返回至 allocation free 之间的干预 | 错误迭代 |
| --- | ---: |
| 冻结 baseline，无 replay | 20/2100 |
| replay 选择目标设备 | 0/2100 |
| replay 选择另一设备 | 22/2100 |
| 另一设备 replay，helper 总耗时至少 1 ms | 13/2100 |

目标设备组的 2097 次稳态迭代均记录到 primary VM/device 的 direct PASID flush，在原 free 前返回；另一设备组均走序列匹配的快速返回路径，没有 direct flush。这使目标 VM 的 flush 相关路径成为优先调查方向，但所有 replay 组仍包含辅助对象同步；还没有 flush-only 与 sync-only 对照。另一设备耗时对照也不能证明锁竞争和等待条件等价。[来源：指定评论](https://github.com/ROCm/TheRock/issues/8744#issuecomment-6121181907)

## 仍然不能从上游页面推出什么

评论明确保留以下边界：没有确认 gfx1200 与 gfx1201 共享根因；没有责任组件、源码语句、commit、首次受影响版本、硬件 TLB/cache 单元或可靠修复；没有匹配旧栈对照，因此不能据此定性为这次升级引入的回归；尚未交叉运行两份复现器，也未覆盖原始 issue 的 OOM 分支。[来源：指定评论](https://github.com/ROCm/TheRock/issues/8744#issuecomment-6121181907)

trace 覆盖的是指定软件事件。14700 个 generation 中 clear-fence completion 只为 14697 个建立了资格，三个保持未知。软件 flush 成功返回也不是独立硬件应答。数据不一致不能区分物理覆盖、读写地址翻译异常和缓存可见性问题。有限实验中的零错误不能证明安全，更不能作为部署故障率或性能收益。[来源：指定评论](https://github.com/ROCm/TheRock/issues/8744#issuecomment-6121181907)

公开复现包可在不依赖 PyTorch 或修改 runtime 的情况下运行基本 HIP/HSA 生命周期对照；它没有运行延迟或 replay 干预，也没有 patched runtime。包内规范化记录只能复核导出的计数和 hash，不能独立重建未公开的完整原始 trace 身份。可携带 runner 仅记录调用者声明的时钟策略，仍需调用者在外部管理并核验时钟。[来源：评论所附复现包](https://gist.github.com/gwokhou/53e27a97162e0ac47c06625ac2c5a14b)

上述事实表明上游问题仍处于调查阶段；本库是否准许某条运行路径，应依据本库的当前配置和实际验收，不能简单等同于 issue 是否关闭。
## 本库具体被阻塞什么

生产[环境构造器](../../src/sol_execbench/core/process/environment.py)已固定
`expandable_segments:False`，不继承外部 True 或旧别名。
[独立验收](evaluate-allocator-default-acceptance-2026-10-12.md)覆盖原始三个不规则 FP16
workload：普通 evaluate 全部通过，两个候选六项 scope 全部合格，错误候选三个尺寸全部拒绝。
这使生产规避与底层修复成为两个独立交付目标。

后续已执行[扩大生产规避验收](evaluate-allocator-expanded-acceptance-2026-10-12.md)：
34 个 workload、两种 seed、普通 evaluate 和 scope 共 136 条正例 trace 全部通过，
68 份 scope 合格，10 组错误候选与 50 个离线证据负例全部拒绝。
覆盖新增精度、算子、反复分配和 CPU↔GPU 传输，所有 GPU 窗口保持 STABLE_PEAK 并核验恢复 AUTO。
这是普通分配路径的新增本机证据，不是上游修复确认；重新开放 VMM 的门槛保持不变。

| 工作或声明 | 当前阻塞程度 | 原因与处理 |
| --- | --- | --- |
| 重新开启 expandable segments；推广同 VA、新 handle 的 VMM 路径 | 直接阻塞 | 数据完整性已有反例，不能作为生产默认，也不能靠等待补丁或一次零错误解除 |
| 评估 expandable segments 的碎片改善、容量收益或速度收益 | 正式收益验收阻塞 | 错误计算或损坏参考会污染计时与容量结果；可以设计方法，数据正确后再授权比较 |
| 宣称具体 AMD 提交负责、底层已经修复、gfx1200/gfx1201 同源或升级引入回归 | 结论尚未闭合 | 评论明确保留组件、旧栈、交叉复现和独立刷新干预缺口；issue 被关闭也不能代替这些证据 |
| 以开启 VMM 的失败运行作为候选排名或参考真值 | 对应结果不可准入 | 保留诊断证据；若确需该 cohort 的性能结论，应在可验收配置下单独重采，不把全部历史证据一并废弃 |
| 任意模式、任意 workload 的完整调度资格 | 本来就未关闭 | 内存域、Graph、多线程和对抗性信任边界仍缺证据；VMM 修复不会自动补齐它们 |

这里的“阻塞”针对具体调用路径和声明。生产环境变量只控制默认 PyTorch 分配器，
不能阻止候选、第三方库或自定义 allocator 自己调用 HIP/HSA VMM；新路径仍要核查实际分配行为。
反过来，也没有证据说明所有普通 hipMalloc、所有内存池、所有 Graph 或整个 ROCm 10.1 都受影响。

## 解除之前可以继续的工作

| 工作线 | 可以交付什么 | 验收边界 |
| --- | --- | --- |
| 生产 evaluate | 保持规避，扩大代表性 correctness 验收，覆盖不同 dtype、分配/释放顺序、重复调用和双向传输 | 新 shape/算子仍逐项验收，不把现有三个尺寸当全库保证；不放宽容差或关闭逐次检查 |
| GEMM 与复合矩阵乘修复 | 保持语义地优化重复配置维护，验证边界写回、小 tile、默认算法选择和跨尺寸效果 | 新旧对照使用同样的关闭 VMM 配置与 STABLE_PEAK；候选/参考必须通过，不能混用 VMM 诊断数据 |
| 第 3 项的诊断子目标 | 扩展可信 eager 执行器、原生编译产物身份绑定、有限 event 边、缺证据拒绝和 artifact 回归 | [现有 scope 验收](evaluate-hip-scope-acceptance.md)只授予有限 capture 资格；完整调度与内存依赖继续保持未关闭 |
| SOLAR 核心增量 | ShapesSpec 到自有形状域契约、静态 fake 取图试点、ATen 差分、标量/断言表示与实例化检查 | CPU/元信息阶段可先做；真实 GPU reference/IR 回放仍走关闭 VMM 的独立验收。FX/fake 取图不是 HIP Graph capture |
| 资源与观测基础 | 维护已完成的 P0 证据，继续确有缺口的 PMC/ISA/资源探针资格工作 | 当前资源探针源码使用 hipMalloc；[P0](solar-p0-rocm101-completion-2026-10-11.md)的原适用范围不因本 issue 自动失效。PMC 有自身的资格条件，VMM 修复不自动授予计数有效性 |
| Python 与 CPU 工程 | 分阶段耗时分析、有界队列、mapper 配额、哈希/解析和 schema 回归 | 是否投入优化由测得的瓶颈决定，与 VMM 缺陷无直接依赖 |
| 底层诊断和上游协作材料 | 固定复现包、保留模块/源码身份、完善每条因果边的证据和残差，准备修复版验收计划 | 下一项有区分力的本机实验是匹配内核下拆开同目标请求的刷新与辅助同步；需另行安排，不在本次分析中启动 |

GEMM 工作的具体待验收项见[性能因果报告](rocm101-gemm-causal-technical-report.md)：
当前机制排查已经达到能指导修复的停止边界，完整尾处理功能、默认选择与跨尺寸收益仍属
修复验收。VMM 的数据完整性问题不把这些已有条件干预自动推翻，也不允许跳过新结果的数值验证。

Graph、kernel replay、Origami/CK 等研究可先做 CPU 契约、编译和不依赖可疑分配器的限定试点；
是否准入实际 GPU 路径取决于各自 allocation、snapshot、正确性和完整性验收，不能给它们
统一贴上“被 VMM 阻塞”或“完全不受影响”。

## 建议推进顺序

1. **先稳住生产评估。** 现有规避保留，把下一批新增 workload 的数值、传输与 scope 资格作为准入检查；
   实际 GPU 运行继续使用目标 UUID 的 STABLE_PEAK，结束由 owner 恢复并核验 AUTO，证据独立归档。
2. **继续升级价值主线。** 优先做 GEMM 修复的完整语义和默认候选验收，或较小的静态 fake/图覆盖差分试点。
   两者都能形成具体交付，不要求先证明 VMM 的内部硬件机制。
3. **把 VMM 根因与恢复启用留作独立研究线。** 已公开的设备选择和延时层无需泛泛重复。
   若下一项内核干预会改变修复选择，再推进；否则保留当前因果树与未闭合边，等待可测试修复。

此顺序是本次工程建议，不是已经启动的新实验，也不为后台监控、上游发言或修改内核提供授权。

## 怎样才算解除阻塞

生产普通分配路径已经有局部验收，并不需要等待 issue 关闭。重新开放 VMM 至少需要：

1. 明确目标 GPU、宿主内核/amdgpu、ROCr/HIP、镜像与候选配置；拿到可审阅、可复现的修复变化。
   修复版本必须包括真正生效的组件，不能假设只更新容器就能修复宿主驱动问题。
2. 在同一机器验证能触发的冻结基线与修复版，预先规定进程数、循环、判定与记录边界；
   覆盖同 VA/新 handle、活跃 buffer、释放/重映射、无目标 D2H 和双向传输。
   若宣称覆盖上游同一故障，还需两份复现器交叉验证；OOM 覆盖也不能由非 OOM 结果代替。
3. 原始生产 workload、扩大后的代表性集合、参考传输、逐次计时输出和 scope 正负例均按原标准通过。
   有限零错误只支持所测范围的资格，不证明整个 VMM 路径永远正确。
4. 数据完整性通过后，再独立评估碎片、容量和性能收益，判断是否值得重新开启；没有收益也可以保留普通分配默认。

“上游确认根因”“本机修复验收”“恢复生产启用”是三个不同状态，应分别跟踪。
即使 issue 以 workaround 关闭，也不能自动解除本库的 VMM gate；反之，本库的普通分配工作可以继续交付。
