# VMM 数据损坏：预期限制还是实现缺陷？

核查日期：2026-10-12。依据是本机已归档的[纯 HIP 实验](fp16-evaluate-vmm-diagnosis-2026-10-12.md)、AMD API 文档和上游原始报告；另在同一镜像与设备补查了 VMM 支持能力，见[生产验收](evaluate-allocator-default-acceptance-2026-10-12.md)。

**静默改坏有效数据不属于文档允许的正常行为。现有证据支持将它按当前 AMD HIP/VMM 路径的实现缺陷处理，而不是接受为开启 expandable segments 的正常代价。** 这仍不是“AMD 已确认本例根因”，也没有证明责任落在某个 runtime、内核模块、固件或硬件单元。

同日进一步确认：无目标 D2H 的 GPU 内部校验仍能发现错误；直接 HSA VMM 的同地址/新 handle
组合为 17/2100 次错误，新地址与同 handle 对照各 0/2100。具体方法和源码边界见
[完整技术报告](hip-vmm-data-corruption-report-2026-10-12.md)。

同日源码干预进展：原 ROCr 库的构建清单与内嵌 revision 均绑定到 `7da43026e4…`，
准确版本重建基线仍复现。释放原 KFD allocation 前等待 1 ms 为 5/2100 次错误，
释放返回后等待为 0/2100；实际 MAP/UNMAP 等待和 KFD free ioctl 均成功返回。
这缩小了时序边界，仍没有证明某个组件、提交或微架构单元负责。详见完整报告第 6、7 节。

内核绑定进展：运行与磁盘 amdgpu Build ID 一致，已取得 Ubuntu `linux 7.0.0-38.38`
准确源码包。源码同时存在 TLB 完成 fence 和 TTM BOOKKEEP fence 等待，不能从异步释放设计
直接推断漏等待。首批内核 trace 已采集，基线仍有 11/2100 次损坏；观察到 CPU 页表更新、
共享已置位 fence 和旧资源退役重叠，但尚未覆盖直接 PASID flush 与清零 fence 的完成关系。
首批的告警开销和 BO 标签残差保留不变。第二批 trace 无告警，仍有 12/2100 次损坏：
2100 次清零 fence signal 返回均早于资源释放入口；循环内失效序号增加、`last_flushed` 为 0，
所覆盖的直接 PASID flush 调用为零。不能据此排除其他失效机制。MAP 后额外映射和约 58 µs
纯延时各为 0/2100。进一步把映射放到释放前，本轮 0/2100，而同位置等待 1 ms 仍有
11/2100；这分离了两组操作，却未拆分映射中的页表/HDP/刷新，也未取得 helper 的实际刷新 trace。
最新追踪已逐轮确认：完整映射每轮有两次同 VM/PASID 刷新位于 UNMAP 与原 free 之间。
后续跳过映射步骤的 KFD 重发仍为 0/2100，额外 GEM_VA 更新则有 13/2100；优先解释已收敛到
KFD 同步/刷新尾部。最新追踪已核验 4194 个稳态重发轮次的实际参数、零映射步骤和释放前
同 VM/PASID 刷新；resume 组没有新增所覆盖的页表/HDP。追踪批重发及组合各 0/2100，
额外更新组仍有 4/2100；各批不合并分母。同步对象为辅助 GTT 对象，sync/锁与刷新尚未拆开。
最新清零任务的 3/12600 条完成边未捕获，保留未知。
又完成双卡设备选择对照：辅助对象预先映射两张卡，handle与主要同步/锁路径相同，
重发选择目标卡两批各0/2100，选择另一卡分别29/2100、13/2100；另一卡请求再加1ms仍16/2100。
新的独立追踪已验收内部 device/VM 调用：目标组 2097 个稳态轮次执行释放前目标刷新，
另一设备及加等待组共 4194 轮进入不同 VM/设备，因序号相等直接返回，没有 direct flush。
追踪批目标组 0/2100，另两组分别 22/2100、13/2100；同步辅助对象的动作仍在。
这进一步限制了辅助同步/主要共享锁加等待已足够的解释；尚非同一目标请求的纯刷新干预。
清零完成仅验收 14697/14700，3 条缺口逐代保留；没有责任语句或提交归属结论。
见完整报告第 6.7–6.14 节及完整因果树。

## API 使用是否越过边界

逐项检查了[本机纯 HIP 源码](../../data/outputs/fp16-evaluate-numerics-20261012/attempt-20261012-112436/vmm_regression.cpp)。它不依赖 PyTorch、矩阵乘、参考服务或跨进程内存共享。

| 检查项 | 本机探针与判断 |
| --- | --- |
| 设备能力 | 原探针未查支持属性；本轮已在同一设备和镜像补查：`hipDeviceAttributeVirtualMemoryManagementSupported=1`，架构 `gfx1200`，最小粒度 4096 字节。见[能力回执](../../data/outputs/evaluate-allocator-default-20261012/attempt-20261012-114510/vmm-capability.json)。 |
| 粒度和大小 | 先查询最小粒度，再把 2 MiB 向上对齐。reserve/create/map/access/unmap 使用同一大小；各次复制均未越界。 |
| 映射参数 | device 0、Pinned 类型、map offset 0、flags 0，设置同设备读写权限；每轮先解除旧映射，再释放 handle。 |
| 生命周期与同步 | 填值、前见证、D2H、后见证之间均同步；最后一次见证完成后才 unmap/release。未见仍有工作访问已释放映射的代码。 |
| CPU 缓冲区 | `malloc` 得到 pageable 内存，但这不是本接口禁止的用法。HIP 对 `hipMemcpyAsync` 明确规定：未锁页的主机缓冲区会使复制同步执行。探针还显式等待 stream 完成。 |
| 虚拟别名 | 每个 handle 只由用户映射到一个 VA。普通 `hipMalloc` 见证缓冲区是独立分配，不是同一物理 handle 的第二个映射。 |

AMD 的 [VMM 指南](https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_runtime_api/memory_management/virtual_memory.html)要求先查支持能力、遵守粒度并设置权限，也给出了 kernel 写入后复制到 CPU 的用例。[VMM API 参考](https://rocm.docs.amd.com/projects/HIP/en/latest/reference/hip_runtime_api/modules/memory_management/virtual_memory_reference.html)规定上述 offset、flags 和映射参数；[复制 API 参考](https://rocm.docs.amd.com/projects/HIP/en/latest/reference/hip_runtime_api/modules/memory_management.html)说明 pageable 缓冲区的同步退化行为。

需要区分一个真实的 RDNA 限制：AMD 文档警告，同一物理内存通过两个不同虚拟地址访问时，L1 可能不一致，并建议该用法采用 volatile 指针。本例没有构造这种双重虚拟别名。把“解除映射后重用地址”直接等同于该限制，没有文档依据。若 runtime 内部遗留映射或失效处理有问题，应继续按实现问题排查，而不能要求普通调用者猜测隐藏的别名。[AMD 虚拟别名说明](https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_runtime_api/memory_management/virtual_memory.html#virtual-aliases)

因此，结合补齐的设备能力记录，代码审查没有发现足以解释现象的已知 API 前置条件违规。这是对公开契约的核对，不等于完成编译器、runtime 和内核实现审计。

## “实验性”意味着什么

PyTorch 将 `expandable_segments` 标为实验性、默认关闭，目标是减少分配大小变化产生的碎片；文档没有把数据损坏列为可接受结果。实验性说明仍有兼容性和成熟度风险，不会使错误结果成为正确行为。[PyTorch 分配器文档](https://docs.pytorch.org/docs/main/notes/cuda.html#optimizing-memory-usage-with-pytorch-alloc-conf)

不要拿跨进程共享限制解释本例：参考服务先复制到 CPU，再传送 safetensors 字节，并未导出 GPU allocation handle。纯 HIP 复现进一步去掉了 IPC。PyTorch 当前源码同时保留旧的 IPC 限制注释和较新的 handle 导入实现，支持范围随平台和版本变化；不能只引用注释便断言所有 expandable IPC 都不支持。[PyTorch 分配器源码](https://github.com/pytorch/pytorch/blob/main/c10/cuda/CUDACachingAllocator.cpp)

## 上游已有相似反馈，但尚未证明同源

| 原始来源 | 截至核查日能支持的判断 |
| --- | --- |
| [ROCm/TheRock #8744](https://github.com/ROCm/TheRock/issues/8744) | 2026-10-05 的用户报告：gfx1201、HIP 7.14，纯 HIP 在复用已解除映射的 VA 后损坏存活缓冲区；使用新 VA 的对照通过。AMD 账号已分派调查人员并标为 triage。尚无公开根因确认或修复提交，是最接近本例的线索。 |
| [PyTorch #195202](https://github.com/pytorch/pytorch/issues/195202) | 维护者 jeffdaily 要求区分 `.item()` 的 CPU 视图与 GPU 内容错误；随后用户给出 GPU 侧检查并链接上述纯 HIP 报告。这支持我们保留独立 GPU 见证的实验设计，不能替代本机证据。 |
| [ROCm/legacy-rocm-build #6603](https://github.com/ROCm/legacy-rocm-build/issues/6603) | 用户报告某些配置使有效 allocation 读回零；其最终触发条件包含 `max_split_size_mb` 与 expandable segments 的组合，并撤回早期“expandable 单独触发”和 SDMA/KFD 归因。本机无需这个组合，不能直接合并为同一故障。 |

上游用户报告是观察证据；AMD 分派和 triage 是受理证据。两者都不等于 AMD 已复现或承认某个确定缺陷。这里也不能把用户对 TLB/MES 的怀疑升级成本机根因。

## 对生产处理的含义

关闭 expandable segments 是规避已复现故障的合理措施；保留原正确性标准和逐次验证。代码库此前无条件启用实验分配模式，是我们能直接修正的触发条件。底层静默数据损坏则应作为 HIP/VMM 缺陷线索保留。

当前可发表的措辞是：“当前 gfx1200 验证环境中，VMM 路径出现违反数据完整性预期的行为，纯 HIP 可复现；生产关闭该路径后恢复通过。具体底层缺陷及其与上游报告的关系尚未确认。”要进一步宣称 AMD 某个版本引入了回归、某提交负责或已被上游修复，还分别需要旧栈对照、源码二分或修复版复验。
