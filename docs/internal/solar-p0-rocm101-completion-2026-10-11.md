# SOLAR P0：ROCm 10.1 资源审计刷新

完成日期：2026-10-11。本次仅执行已选定的 P0；动态形状、fake 取图、算子覆盖、profiler 功能、并发和 Origami 均未实施。

## 结果

正式 `RX_9060_XT` profile 已更新为 `rx9060xt-amd-resource-v6`，绑定新栈生成并通过现有 SOLAR 校验器的资源审计：

- [正式 profile](../../src/solar/rocm/profiles/RX_9060_XT.yaml)
- [正式审计](../../src/solar/rocm/audits/rx9060xt_resource_peaks_v3.json)
- 审计 SHA-256：`f822fcc4f0019c12da9e46dbfb42f2b4f5f56b65cf2813c9ab7c0582c3083e48`
- 精确 schema 保持 `solar.resource_peak_calibration.v4`；文件名中的 v3 是既有历史命名，不是 reader 的 schema 版本。

GPU 为 AMD Radeon RX 9060 XT / `gfx1200`，ROCR UUID 为
`GPU-a30fa29c1cca1511`，BDF 为 `0000:05:00.0`。使用已有验证镜像
`sha256:0ec2e6cba29e673ebe5ad83b7049e5044647fc340174d1c855379efaf36409dd`，
其中 Python 为 3.14.4、Torch 为 `2.14.0+rocm10.1.0`，HIP 为
`7.16.26385`，HIP compiler 为 `7.16.26385-0000000`，AMD SMI 为 27.1.0。
HIP 组件编号与 ROCm 总发行版编号不同。

此发行组合也见 [AMD 安装说明](https://rocm.docs.amd.com/projects/ai-ecosystem/en/latest/frameworks/pytorch/install.html)；本次资格结论来自上述本地实际运行和原始证据，而非安装页。

采集源码基准为 `52cf050d10c53ce9fb3904122d86af525cd38cb5`。现有冻结 producer、HIP 探针和正式审计验证器均未改动；只更新生成的审计、profile 的证据绑定/说明、对应测试和文档。

## 资格链与测量

沿用 `run_qualified_rdna4_resource_peak_calibration.py`，没有直接绕过 gate 调用 producer：

| 阶段 | 本次结果 |
| --- | --- |
| qualify-static | 10 个探针编译通过；不挂载 GPU |
| qualify-canary | 4 个代表探针通过 |
| qualify-full | 10 个探针全部通过 |
| configuration selection | 每个可调候选 3 个新进程，保持既定反向/正向顺序 |
| held-out | 冻结配置后，每探针 7 个新进程，每进程 7 个样本，共 490 个保留样本 |
| resource audit validator | SHA-256、schema、raw statistics、遥测、ISA、compiler/runtime 和完整覆盖全部通过 |

覆盖 5 种精度：FP32、FP16、BF16、FP8、INT8。9 个必需模式全部覆盖：
VALU FP32/FP16/BF16，WMMA FP16/BF16/FP8/INT8，以及 SFU、reduction。
另有 VRAM stream-copy 探针。7 项指令检查全部通过，包括 BF16 VALU 的 FP32 fallback 证明。

FP32 选择 16 个 accumulators；四个 WMMA 探针均选择每 reported WGP 128 waves。
旧审计 FP8/INT8 的 32 waves 没有沿用。tuning 与独立样本没有混用。

已有升级 smoke tests 和 profiler pilot 不是这 9 个资源模式的完整替代，故本次重新取得了专项证据；没有重复启动其他升级实验。

## 证据边界

所有新采集运行于目标卡已核验的 STABLE_PEAK 窗口。控制流程持有共享 GPU 锁，只管理该卡的时钟，异常和正常退出都由现有 lease/容器清理机制处理。完成后控制回执记录 `peak_verified=true`、`auto_verified=true`、`collection_completed=true`，另行检查也确认 AUTO 且没有遗留目标 GPU 进程。

140 个 held-out 前后遥测快照中，138 个为 `THROTTLED`，2 个为 `UNTHROTTLED`。没有过滤这些观测，也没有为取得更好状态而重采。因此继续保持：

```yaml
required_unthrottled: false
evidence_scope: instruction_and_runtime_corroboration_only
```

本次不是“已经达到无节流理论峰值”的证据。理论吞吐、资源上限、内存容量、带宽及精度 alias 均保持原值；实测结果不替换名义上限，也没有形成新栈相对旧栈的性能提升结论。

精度支持说明现在直接引用 hash-bound HIP runtime probes，去掉旧栈 rocWMMA/PyTorch 私有矩阵 API 已验证的沿用声明。FP8 runtime probe 使用 OCP E4M3；E5M2 仍属 ISA 来源支持。INT4 仍没有本 profile 的 qualified runtime probe，因此继续 exempt / source-only；不再借用“旧版本没有 API”解释当前栈。

这份专项审计不更新历史 RDNA4 综合工程 gate、发布 baseline、候选结果或官方分数。它也不新增通用的 runtime-stack 比对机制：审计自身记录具体适用栈，测试要求 packaged audit 与项目锁定 Torch 一致；不能把 profile 成功加载解释为任意其他运行环境都已合格。

## 验证与保留记录

本轮新运行的 CPU 回归：`tests/solar`、`tests/sol_execbench/core/solar_bridge` 和 `test_large_batch_gpu_entrypoints.py`，**671 passed / 0 skipped**。已有正式校验器在替换 profile 前独立验证新审计通过。

对应测试更新了 profile revision、实际选中的 tuning configuration，并增加 packaged audit 与项目 Torch 依赖一致、HIP/compiler 身份一致、源码 revision 不是缺失占位值的检查。

Ruff、格式、ty、耦合、可读性、production reachability、文档、schema 和复用检查全部通过，最终结果保留在下面的 `checks.json`。没有修改生产 Python 算法或冻结校准实现。

主要未提交原始证据位于：

- [第二轮目录](../../data/outputs/solar-p0-rocm101-20261011-r2/)
- [最终 P0 验证回执](../../data/outputs/solar-p0-rocm101-20261011-r2/p0-validation.json)
- [原始新审计](../../data/outputs/solar-p0-rocm101-20261011-r2/resource-audit.json)
- [采集入口及固定参数](../../data/outputs/solar-p0-rocm101-20261011-r2/acquire.py)
- [static gate](../../data/outputs/solar-p0-rocm101-20261011-r2/qualification/static/gate.json)、[canary gate](../../data/outputs/solar-p0-rocm101-20261011-r2/qualification/canary/gate.json)、[full gate](../../data/outputs/solar-p0-rocm101-20261011-r2/qualification/full/gate.json)
- [CPU JUnit](../../data/outputs/solar-p0-rocm101-20261011-r2/solar-regression.xml)、[质量检查](../../data/outputs/solar-p0-rocm101-20261011-r2/checks.json)

第一次尝试的隔离容器缺 ISA 缓存，且 Git safe-directory 导致 source revision 为 `unknown`；未产出正式审计，退出恢复 AUTO。首轮日志和 gate 保留在 `data/outputs/solar-p0-rocm101-20261011/`，没有被修补或冒充成功证据。

随后有界下载并校验了仓库锁定的官方 ISA 归档，在无 GPU 环境完成 helper 构建及 ISA 预检；Git safe-directory 只通过容器进程环境设置。第二轮使用新目录、完整 qualification 链、正确源码 revision 和只读离线 ISA 缓存。旧正式 profile/audit 另存于第二轮 `prior-profile/`，可审阅和回退。

本次没有安装或替换主机 ROCm、Torch 或驱动，没有提交或推送 Git 变更。
