# B1：执行版本、环境与 CPU 基线冻结

日期：2026-11-08。状态：**B1 完成；B2 尚未启动**。
依据[迭代计划](solar-next-iteration-plan-2026-11-07.md)、原 B0/E0/C0 冻结件与
[E1/C1 CPU 验收](solar-e1-c1-cpu-acceptance-2026-11-08.md)。
本次范围为本地提交冻结、依赖环境校准、只读身份验证和 CPU 检查；没有运行 GPU、
真实 mapper、容量探针，没有生成 h0 或真实 LLM_CORE target view，也没有发布正式分数。

## 1. 执行源码与证据绑定

| 身份 | 冻结值 |
| --- | --- |
| 规划锚点 | `8641b9dc78d00250ba1cf75d5e9434f9839e5ef4` |
| 执行提交 | `160b9988c0f1c6f28be514651918419a39f0d438`；本地 DCO 签署提交，未推送 |
| 变更范围 | E1/C1 的生产接入、测试、M0/迭代文档与已批准的 E0 计数勘误；69个文件。正式评分、原题 checks、依赖锁、默认 IR 与 native-op registry 不变 |
| 执行源清单 | 739个文件；canonical SHA-256 `047b6fc082fd7450ca7d3a8f2555b0b90f63c2ccbd73b915401cc007afd7330d`，与 E1 最终开发运行一致 |
| 相对锚点的二进制 diff | `data/outputs/solar-b1-freeze-20261108/execution-source.patch`，SHA-256 `f80c022175c0eab7131d9467a54219dc549bce4d45b183c8b7c6372b68e98dae` |
| 独立执行 checkout | `data/outputs/solar-b1-freeze-20261108/execution-source/`，detached HEAD 固定上述提交；台账文档的后续提交不移动它 |
| 清洁性 | 现行 `verify_release_source_state` 验证通过；release-relevant 路径及测试均与执行提交一致，无额外源码 |

证据根为 `data/outputs/solar-b1-freeze-20261108/`，不提交运行工件。
`freeze-manifest.json` 连接源码、环境、测试日志、mapper、预算及各文件摘要；
它是 B1 研究登记，不是手写的正式 release environment、qualification receipt 或评分输入。
该清单 SHA-256 为 `1ceb04b24674daf7ed19090044761e3be29eb6ce879cfd679eef79d5d2f8280c`。
仓库另有三篇此前的研究文档仍保持原字节、未纳入此次提交。
后续必须使用上述 detached checkout；不要把继续更新台账的分支 HEAD 当成执行 SHA。

## 2. 依赖和容器

| 项目 | 观察和选择 |
| --- | --- |
| `pyproject.toml` | SHA-256 `32a7241257ee2788ef96141258dc5e4a3a40c8e8ada4df93edd828450b84d61b` |
| `uv.lock` | SHA-256 `50668e6d0243eb80691d935424dedd2e2b4cf9c4b9d62937cd74dd71f2b83d37` |
| 宿主 CPU 环境 | CPython 3.14.3、Torch 2.14.0+rocm10.1.0、HIP 7.16.26385；已安装且属于锁文件的生效包版本均匹配 |
| 后续执行容器 | `sol-execbench:solar-b1-20261108`；内容 ID `sha256:1b92e7b1a59987aaef4449e7e6eec0fdb138de209b16e09e214f0dc5e17ddaaa`。tag 仅作索引，准入核对完整 ID |
| 容器 CPU 环境 | CPython 3.14.4、Torch 2.14.0+rocm10.1.0、HIP 7.16.26385；生效包与锁文件无版本冲突；`pip check` 通过 |
| 语义实现 | 宿主与容器均为 `d26c9feb84c803d0ea2d0a5ddf67a30cd8db064c4b83eed2df331af83ffe9da7`；Python patch 版本不同，不能互换语义回执 |

基础工程镜像保留：`sol-execbench:engineering-89cf9c1-20261031`，
ID `sha256:f197e6c23fa43b23badfd3e684eb18a4f48dde4184ec35e18eb56a0b61952340`。
初次按 distribution 遍历所得清单未考虑 system-site-packages 的遮蔽，列出4个差异；
改用实际 metadata 导入优先级复核后，networkx/ninja/pygments 的生效版本原已匹配，
只有 packaging 26.0 需校准为26.3。派生镜像同时补齐锁定的 pytest-xdist 3.8.0、execnet 2.1.2，
以遵守仓库测试插件要求；没有绕过 `required_plugins` 或修改锁文件。

构建配方、6个wheel及版本/摘要选择在证据根 `image-context/`；安装使用
`--no-index --no-deps --require-hashes`，构建 RUN 无网络。5个wheel验证现有锁文件摘要；
networkx 的 AMD 精确URL在原锁中无hash，本次补记下载工件 SHA-256
`d47fbf302e7d9cbbb9e2555a0d267983d2aa476bac30e90dfbe5669bd57f3762`，不冒充原有锁摘要。
构建前核对基础tag内容ID，保留构建日志；原有同名基础镜像没有覆盖。
Docker的本地内容ID不能据此假定为可从远端拉取的manifest；首次digest解析失败的日志保留，
最终构建使用经内容核对的本地基础tag。复建前必须重复该核对，不能只相信tag。

容器 CPU 查询/测试均无网络、无设备透传、只读源码，覆盖默认时钟入口后直接运行Python，
GPU未初始化。它们不能替代B2的原生设备/时钟/ownership/容器门禁。
执行时源码必须由上述冻结checkout只读挂入 `/sol-execbench`，并使用 `/venv/bin/python`；
镜像内原有代码不是本次执行源码。image ID、挂载源或运行时有变动须重新冻结。

## 3. Mapper、设备选择与预算

mapper仅做文件/provenance校验，调用现行 `validate_toolchain_identity`，未启动二进制。
home固定为
`<external-crosssol-workspace>/out/paper-a-local-20261005-v8/infrastructure-amd-v8/prerequisites/external/amd-orojenesis`。
Timeloop源提交 `97d52178bf9a9c209bf79be96b87c164bcd35625`；
binary SHA-256 `18591892b1ecec3264ec729b0e457ec9f22422993f656ece40dba809c032d77a`；
provenance SHA-256 `99efa80834e837cfe14eef9d039812397ac1a0145f93134d49ae8017fe7ff460`。
后续在容器以相同绝对路径只读挂载提供，原allowlist、source/build验证与正式bound要求不变。

设备选择沿用 C0 的 RX 9060 XT、gfx1200、standard、32 CU、16GiB。
预期UUID为 `<redacted-gpu-uuid>`，BDF `<redacted-pci-bdf>`，
来源是 `data/outputs/solar-l1-release-20261105/r16-terminal-ledger.json` 的历史记录。
本次没有重新枚举设备；B2必须实际确认UUID/配置、空闲与时钟状态，C2再取得容量证据。
预期选择不是当前 observed_device，也不保证仍有可用GPU窗口。

宿主当前 affinity 为逻辑CPU 0–31；生产mapper政策解析32线程。
后续显式固定 `SOLAR_OROJENESIS_MAPPER_THREADS=32`、jobs=1；CPU可见性变化时停止核对，
不能让默认值改变后沿用旧身份。物理内存32,163,278,848 bytes，
单worker上限为 `min(16GiB, physical/2)` = **16,081,639,424 bytes**。
CPU容器身份查询使用2CPU/2GiB，定向回归使用4CPU/8GiB；这些是本次检查的限制，
不是mapper执行配额或性能基准。

| 活动 | 固定分母 / 次数 | 超时与活动停止上限 |
| --- | --- | --- |
| B2 | 代表回归及 AKA static 163 → canary 112 → full 163；资格路径 make_fx，jobs=1；大卷积miss不重试 | readiness单worker3600s；必要formal单worker14400s、mapper单次7200s；本轮资格活动6小时、20GiB证据，停止后保留未到达项 |
| E2正确性 | 432逻辑单元×3=1296 session；每session最多11次候选调用 | 单元120s、单次oracle120s |
| E2成本 | 24块×3×2=144 session；候选预热/计时25,920次；含检查共27,360次 | session300s；E2正确性+成本共6小时GPU、12小时CPU oracle、20GiB证据；jobs=1 |
| C2 | 36条决策；合格core20项/180槽；require_complete_profile=true | 容量活动30分钟；容量类与具体view由C2门禁取得 |
| C3 | 两固定路径360行，先40 smoke再320；最多360 readiness+360 formal，默认每行一次 | readiness3600s、formal14400s、mapper7200s；smoke2小时且包含在总12小时中，50GiB证据；jobs=1 |

B2总活动停止上限是本次为资格刷新选定的研究限制，不改生产默认值。
E0/C0原停止预算保持不变；预热计数按用户批准的
[r2-counts勘误](solar-e0-cost-count-erratum-2026-11-08.md)绑定。
正确性与成本合计最多41,616次候选、68,976次reference调用；包括现行10轮正确性检查、
timing准备和逐计时重验，不是dispatch总数。成本活动固定benchmark_reference=true，
正确性活动false；无counter replay、无自动加样/投票重试。

## 4. CPU 验证与历史证据口径

- 宿主相关CPU回归 **288 passed / 39.87s**；同一组在冻结容器内 **288 passed / 43.74s**。
  这是两种环境各自的288项，不拼成576个独立场景。日志为 `cpu-tests.log`、`container-cpu-tests.log`。
  范围覆盖E1、全bridge单元、资源guard、源状态、release CPU合同/验证器、SOLAR API/readiness/管线、schema。
  测试使用 `-n 0`、隐藏HIP/CUDA设备并排除硬件/C++标记；未选择会生成真实LLM_CORE轴点的生成套件。
- Ruff、format、ty、coupling、readability、schema、production reachability、dataclass policy、
  diff空白检查通过；质量基线没有提高。提交hook的Ruff与DCO检查也通过。
- E1最终报告的执行源摘要与提交逐文件相同，288个引用工件摘要仍匹配。
  复用已完成的72开发输入/288候选单元证据，不重新抽样以改善结果。
  11项LayerNorm P3与CPU reference的分歧以及P0的17项强诊断失败均保留；
  不能预先标为GPU裁判误拒，E2必须以实际reference/oracle标签分别判读。
- 原B0/E0/C0文件字节不变。旧AKA 163 ready、127项G4及大卷积miss仍属原版本证据；
  fake-GPU直接工件/旧workspace环境文件缺口、外部176路径严格对应未证实等B0结论继续保留，
  不用本次CPU日志补成历史GPU证明。
- 当前只读owner校验确认AKA static/canary/full选择仍为163/112/163；LLM_CORE的36个Definition
  与36条规则通过来源校验，生成workload数为0。资格runner摘要仍为
  `8588ecbee62b4d8293a27fa139b9c975e3f6f8e08f96707667c67f591f14b3fb`；
  没有运行qualify-static或写出新的资格gate。

## 5. B1 出口

B1关闭的是“可复现执行源+环境选择+CPU基线+预算”门，未关闭B2硬件资格。
下一步是按原规则进行B2准入/代表回归/资格刷新；仍需后续GPU执行范围与可用窗口。
本次不启动E2、C2、C3，不生成h0 commitment，不改变start700暂停、L3-d或gfx942边界。
若B2观察到运行时、镜像、设备配置、资源配额或源码不符合本冻结，先回到B1登记差异；
不得在采集期间边改源码边沿用旧资格。
