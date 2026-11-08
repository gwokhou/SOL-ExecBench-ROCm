# M0 / B0：基线证据清点

核查窗口：2026-11-07—08；清点完成日期：2026-11-08（Asia/Shanghai）。
状态：**B0 只读清点完成；当前版本资格未刷新**。文件日期沿用本轮规划日期。
对应[迭代计划](solar-next-iteration-plan-2026-11-07.md)；实施状态仍由
[HANDSOFF](../../HANDSOFF.md#solar-evolution) 维护。
本次只读取源码、Git 元数据、已存在的 JSON/YAML/Markdown 和包元数据，计算文件摘要、核对引用。
没有导入 Torch/项目运行入口、执行验证器、运行测试、mapper、设备枚举、容量探针或 GPU 工作。

## 1. 锚点和身份规则

HEAD：`8641b9dc78d00250ba1cf75d5e9434f9839e5ef4`，main。
`git status --short -- docker problems pyproject.toml scripts src uv.lock` 无输出。
已有文档修改和未跟踪研究文档保留；本清点不是整个工作区 clean 的声明。
没有刷新远端；main 与 origin/main 的关系只是本地引用状态。

| 身份层 | 实际 owner / 验证入口 | 对复用的约束 |
| --- | --- | --- |
| 发布执行源码 | [release_environment.py](../../src/sol_execbench/core/scoring/release_environment.py) / `verify_release_source_state` | 精确 commit + 指定生产路径清洁性；docs 不在路径集合，但新 commit 仍改变 source revision |
| 资格链 | [release_solar_qualification.py](../../src/sol_execbench/core/scoring/release_solar_qualification.py) / `_verify_gate` | task/stage/scope、subject、runner、configuration、source、parent hash、item IDs 都要匹配 |
| readiness 复用 | [corpus_readiness.py](../../src/sol_execbench/core/solar_bridge/corpus_readiness.py) / `verify_readiness_receipt` | 图、access、reference、policy、cases、execution 的绑定；文件存在或字节匹配不单独授予复用资格 |
| 语义运行时 | [execution.py](../../src/solar/verification/execution.py) / `require_current_semantic_runtime` | verification/ir/graph 与指定文件摘要，加 Python/Torch 版本；当前 SHA 不等于完整 runtime identity |
| mapper | [identity.py](../../src/solar/analysis/orojenesis/identity.py) / `validate_toolchain_identity` | allowlist 二进制、source/build provenance 必须均匹配；没有 checkout 回退 |
| 硬件/容量 | [formal_device.py](../../src/sol_execbench/core/solar_bridge/formal_device.py)、[corpus_models.py](../../src/sol_execbench/core/dataset/corpus_models.py) | 实际配置、设备观察与容量证据分开；配置模板不是当前设备测量 |

资格 runner 的当前 canonical preimage hash 为
`8588ecbee62b4d8293a27fa139b9c975e3f6f8e08f96707667c67f591f14b3fb`；
旧三份 gate 为 `15760906624f056ae7c9cd19344cc3bdbf48bf2e3666afacfae704d72a12afb6`。
本次按源码的 JSON 排序/分隔方式重算，未调用生产资格验证器。

当前语义源码映射共 98 个文件，摘要为
`d26c9feb84c803d0ea2d0a5ddf67a30cd8db064c4b83eed2df331af83ffe9da7`。
旧 full 的 163 份 attestation 全部记录
`b1ad416b7307f39deb8eeca091639bf56d99143381398f4ff1dc7b8ab39ae2c2`。
**源码身份已不匹配，不能把旧 receipts 作为当前执行回执复用**；不需要运行 GPU 才能确认这一点。

## 2. 内容冻结索引

以下均为本次读取原始字节所得 SHA-256；不是生产模型的 canonical profile/descriptor digest。
生成视图中的结构化 digest 仍须使用 owning source 的定义。

| 文件（仓库相对路径） | SHA-256 |
| --- | --- |
| `uv.lock` | `50668e6d0243eb80691d935424dedd2e2b4cf9c4b9d62937cd74dd71f2b83d37` |
| `pyproject.toml` | `32a7241257ee2788ef96141258dc5e4a3a40c8e8ada4df93edd828450b84d61b` |
| `problems/AMD_AKA/manifest.yaml` | `24cee562796ada61a9f026a25ac55c3f2d33af69007487cc2f4eaa3a975bb872` |
| `problems/LLM_CORE/releases/LLM_CORE_V2/manifest.yaml` | `8de8b6367380fe3200907bf6f62d93dcc9b77b3040df8201e29ef03a568169e7` |
| `problems/LLM_CORE/targets/configurations/rx9060xt/standard-16gb.yaml` | `df0d88a8a4b96cab7e1b08bbd9c8ca8d0644cdc70b95cc468e3c15b3c6805b0c` |
| `src/solar/rocm/profiles/RX_9060_XT.yaml` | `3ab62e1ddc7f6e6d607a4483677fc4e85d69cc6b0e5d08adfd65dcdd3c8fb4d3` |
| `src/solar/rocm/audits/rx9060xt_resource_peaks_v3.json` | `f822fcc4f0019c12da9e46dbfb42f2b4f5f56b65cf2813c9ab7c0582c3083e48` |
| `src/sol_execbench/core/scoring/release_solar_qualification.py` | `441f998c6a1916b03a35690adb3e2f0c51c60150d402324bd4a6e5fcdcc905f6` |
| `src/solar/analysis/orojenesis/configuration.py` | `48ec1ee87968c8731824cb2c15ed21dc0ee3304cb01dfaae74654d368050151e` |
| `src/sol_execbench/core/solar_bridge/resource_policy.py` | `d5e7bb0864ab98e8d600b78ad8034b4160cd88ac487b3829b09ef330da708309` |

LLM_CORE manifest 引用的 36 个 Definition + 36 条规则逐文件摘要核对 **72/72 匹配**。
以 release 根目录相对路径为键、实际文件 SHA 为值，compact/sorted JSON 映射摘要为
`d656adb1cb7a78789cf4155ffe0c4cf0a964ada767a9e667f83d539de8eb3dfd`。
该映射摘要只用于本清点，不新增生产协议。

## 3. 历史证据逐项处置

“可保留”指可作为历史记录/规格输入；不等于当前版本已通过生产复用验证器。
“须刷新”是未来 B1/B2 的任务；“缺证据”是本次定向检查未找到充分工件，不推断历史工作未发生。

| 能力 / owner | 已读取证据及范围 | 处置与后续 |
| --- | --- | --- |
| 当前合并 CPU / 基线维护 | HEAD 提交说明的 4319+25+149 项记录；无本轮执行日志 | 可保留历史说明；B1 运行相关检查，不把计数当当前验收 |
| make_fx 资格 / scoring | 三份 gate 均 source=`39329249b2e9bda967aec7e601deb3a319502250`；163/112/163，authority=false | **须刷新**：source 和 runner 已漂移；B2 使用新执行树重建 workspace/链 |
| full readiness / bridge | 163 行唯一；三阶段引用的 489 个工件全部存在、SHA 匹配；163 份语义身份均旧 | 历史字节完整性可保留；不能直接恢复为当前 ready |
| fake 图等价 / IR | equality matrix 163 唯一项均 equal=true 且 real/fake SHA 相同 | 可保留对账记录；矩阵本身未含源树身份，来源由 HANDSOFF 绑定；B2 回归生产后集成树 |
| fake GPU 3 槽 / IR | HANDSOFF 有 3/3 记录；声明路径下没有 `gpu-validation/` 或 `lineb-summary.json`，只有 equality 两文件 | **缺直接工件**；B1 可补索引，B2 代表回归仍必需；不伪造摘要 |
| 外部读取 / analysis | batch8 reinventory 128 项均空阻塞；batch8 G4 14 项中 13 analyzed、1 memory-bound miss | 可保留发现；127/128 全线计数来自 HANDSOFF 的跨批汇总，本次未逐个重审 127 份证明 |
| 大卷积回放 / verification | `l2n52_conv_activation_batchnorm-w4`，B=64,H=W=128；报告显式 unfold 约23.6GB、OOM 日志摘录 | **缺端到端回放证据**，本轮默认不重试；禁止把静态证明当完整 replay |
| M-d / performance_model | M4 JSON：13 dispatches、verified edge (11,12)、cross_lane_verified=true | 可保留历史摘要；摘要未绑定当前 source，不能由此授予发布权威 |
| L4 / resource_policy | summary 在 `995c020a5f77f408eb73d6678db7e555511c9795` 标三 gate 通过；probe 默认32/override4 | 可保留；不把旧 CPU 可见核数当当前配额，B1 重新冻结配置 |
| AKA L3-c / bridge | `7e069bd273029b53e3e0749d84b5a23e15af3f43`，326 行；summary 显示旧形式层失败 | 旧覆盖基线，不当作当前形式分析计数，更不外推 LLM_CORE |
| 资源峰值 / platform | profile 内引用的 audit SHA 匹配；audit source=`52cf050d10c53ce9fb3904122d86af525cd38cb5` | 可保留审计佐证；不等于当前时钟/配置/峰值重新测量 |
| 早期阶段1–5 / 基线维护 | 外部 CHECKPOINT-0528、0553 存在并已读；0553 写两路径 canary 2/2、当时 R16 尚有阻塞 | 保留外部地址，不复制；与 main 176 路径整合的严格 preimage 对应仍未证明 |

三级 gate 引用的 preflight、两份 matrix 和两份 summary 的长度/SHA 均匹配；
canary/full parent SHA 分别等于上一级 gate 文件 SHA。此为文件级核对，**不是生产 schema/资格验收**。
`solar-l3e-external-reads-20261106/l3e-summary.json` 是 batch1 时点总结，
不能用其文件名中的 overall 代替 batch8 终态。

下表根目录均为 `data/outputs/`，只读、不改写：

| 证据相对根的路径 | SHA-256 |
| --- | --- |
| `solar-qualification-mfx-20261107/static/gate.json` | `b95845ce9519fd6da441342e38b9ccc6acac7ea09df6fdc22ea5669741ba3dca` |
| `solar-qualification-mfx-20261107/canary/gate.json` | `b7ff3fc2339a53734664cc8f74ff1791ee2b6d9d62698ebd06b64fbb2fd022a5` |
| `solar-qualification-mfx-20261107/full/gate.json` | `fcb49d4af4d81669a2c47d8857b58f15ec33cce6dfb5d4c152da4dbc3f8c184b` |
| `solar-qualification-mfx-20261107/full/readiness/matrix.jsonl` | `a0eac779d1e8c884896c5e3a3627a09f7d0b4e97ac8566fd2850d8f60d377751` |
| `solar-qualification-mfx-20261107/full/readiness/summary.json` | `ad83d17a169fceaf25a35b415362320a97bd75616026438a10f6782434a79485` |
| `solar-lineb-fake-20261107/equality-matrix.jsonl` | `50ee3e7e9a156f2728a468ea55bc545a263ab37037653b0c67c85aa6c4db8a7c` |
| `solar-lineb-fake-20261107/lineb-equality-summary.json` | `08fb3455058c5481ad00b534e8e9e9da1c7ebad0f27fc77e0367ae02641352b4` |
| `solar-l3e-external-reads-20261106/batch8-reinventory.json` | `83ef7675f5465514b3a7a7aae02324b87b8a8eb4da64fe9671814bb5630f99be` |
| `solar-l3e-external-reads-20261106/batch8-g4-summary.json` | `9315bca967593e09605b41861172a1cf0f3bf42da671779fccbc5e6198949ca9` |
| `solar-l2-md-feed-20261107/m4-dual-collection.json` | `97327a75a721c3ccb5ee25d23723e545b1381ca20488a1d5add37ce55cec7377` |
| `solar-l4-throughput-20261107/l4-summary.json` | `244d86aaf567683b02cd48d5bbcfdc52925030c2a690130b20f0753b2931c5cc` |
| `solar-l3c-20261106/l3c-summary.json` | `3a20a4cc9075a0fc7844aefea658904c5fb1a5b1e315b742efbe78860d380890` |
| `solar-l3c-20261106/coverage-matrix.jsonl` | `8275c7d9ee57734ba4f03e0f7bdaae1dba809ede85dacc8d8a95104f1aa1e814` |

外部根：`<external-solar-workspace>/data/outputs/solar-resume-20261026/mapper-followup/gpu-proposal/`。
`CHECKPOINT-0528.md` SHA=`fe065d8873fa4a0b13a643e94243701d143ae03c2c5898b4d8230b4d13c909b6`；
`CHECKPOINT-0553.md` SHA=`7bd325f1977f098838ad5bb2e57f96815ef0b1ec47a84780f76071f173ebdc40`。
其中旧设备 UUID `<redacted-gpu-uuid>` 仅为历史观察。

## 4. 环境、mapper 和未闭合身份

- `.venv/pyvenv.cfg` 声明 CPython 3.14；安装包 METADATA 和 `torch/version.py`
  声明 Torch `2.14.0+rocm10.1.0`、ROCm `10.1.0`、HIP `7.16.26385`。
  没有启动该解释器查询运行时；旧回执的 Python `3.14.3` 不冒充当前精确 patch 版本。
- 当前 shell 的 `SOLAR_OROJENESIS_HOME` 未设置。已读 policy 固定 Timeloop
  commit `97d52178bf9a9c209bf79be96b87c164bcd35625`，allowlist binary SHA
  `18591892b1ecec3264ec729b0e457ec9f22422993f656ece40dba809c032d77a`。
  本次未定位/执行未来活动选用的二进制与 provenance；这是 **B1 待绑定项**，不表示机器没有安装 mapper。
- 旧 workspace `solar-workspace-20261107/baseline/plan.json` SHA
  `7073d139a9127440fef54f6801cc74d532edcd5bd9f25de903cf392a06bf3aee`，source 仍为 `39329249…`。
  其 `environment_path=baseline/environment.json` 在该 workspace 中不存在；本次没有得到该活动镜像 digest。
  不将旧 workspace 当成可原样恢复的当前发布环境。
- 当前设备 UUID、空闲状态、时钟、可见显存、容量类、实际镜像 digest、CPU affinity 未查询。
  B1 记录选择，B2/C2 按现行门禁观察核验；本次只固定目标模板，不声称它已成为 observed_device。
- 正式计价仍要求 `capacity_constrained_tile_aware_v1`；qualification timing 仍
  `performance_authority=false`。没有改 bound、评分、device、ownership 或资源策略。

## 5. B0 出口

B0 完成的含义是：版本/合同 owner 已定位，现存证据有明确读取层级和摘要，
不能复用与缺少的证据已有责任任务。**不是所有缺口均已补齐，也不是当前 GPU 基线已合格。**

交给 B1：冻结新 source/runner/runtime/镜像/mapper/资源配置，补齐必要工件索引；
交给 B2：新身份下的代表回归和资格刷新；保留大卷积 miss、torchview 缺口与外部历史范围。
E0/C0 可用本索引作为预登记输入，无需为了规划补跑任何实验。
