# BASE-01：当前 SOLAR 的 CPU 运行基线

执行日期：2026-10-23（Asia/Shanghai）。`BASE-01` 的 CPU 范围已完成：可执行样本、独立 oracle、当前行为记录和相关回归均已落地。**这是修改生产计量之前的基线，不是阶段 1–3 的功能验收。** 本轮没有修改任何 `src/` 文件，没有运行 GPU 实验或真实 mapper，没有推进阶段 1–3。

输入规格见[验收清单](solar-evolution-acceptance.md)，原冻结值见[阶段 0](solar-evolution-stage0.md)，任务边界见[BASE-01](solar-evolution-tasks.md)。仓库推进状态仍由 [HANDSOFF](../../HANDSOFF.md#solar-evolution) 维护。

## 1. 环境和来源

| 项目 | 本轮固定值／边界 |
| --- | --- |
| HEAD | `e35d046ee9c64c310ff3be41fe0a78b00bb993b3`；工作树有既存改动，实际源码按文件摘要绑定，未简写成干净 HEAD |
| 镜像 | `sha256:0ec2e6cba29e673ebe5ad83b7049e5044647fc340174d1c855379efaf36409dd`；使用已有镜像，没有重建或安装依赖 |
| 锁文件 | `50668e6d0243eb80691d935424dedd2e2b4cf9c4b9d62937cd74dd71f2b83d37`；镜像内和仓库 `uv.lock` 一致，12 个关键已安装依赖逐项匹配锁文件 |
| Python／PyTorch | Python 3.14.4；PyTorch `2.14.0+rocm10.1.0`，所有 F 输入与语义回放使用 `cpu` |
| 测试／质量工具 | pytest 9.1.1、pytest-xdist 3.8.0、Ruff 0.16.0、ty 0.0.64；完整安装清单保留在环境记录中 |
| 容器限制 | 无网络、无 `/dev/kfd`／`/dev/dri`／NVIDIA 设备，非特权、移除全部额外 capabilities；源码只读，仅证据目录可写 |
| 执行预算 | 单槽位 180 秒；每条输出流最多保留 1 MiB 尾部；逐槽位独立进程并清理进程组；Torch 单线程，容器限 4 CPU／8 GiB |
| 字节码 | 最终运行设置独立 `PYTHONPYCACHEPREFIX` 且禁止写字节码，避免使用工作区既有缓存 |
| 计价条件 | RX 9060 XT 现有 profile、`precision=fp32`、无 Orojenesis；计数观察不要求新的设备 audit，所有本轮结果明确 `publication_eligible=false` |

这里的 CPU 指提取和语义回放的执行设备；计价仍使用上述 GPU 架构 profile，并不是 CPU roofline，也没有测量 CPU 或 GPU 性能。原 profile 中保留的历史 audit 字段不赋予本轮设备资格。

## 2. 新增支撑

- [F 可执行样本](../../scripts/internal/solar/base01_samples.py)：16 个具体实例，固定 seed、输入 dtype、独立存储及两项既定验证策略例外。
- [独立 oracle](../../scripts/internal/solar/base01_oracles.py)：小域整数集合枚举、分数算术及明确的合同输入；不导入生产图、计量、区域并集或反向查询算法。
- [单槽位观察器](../../scripts/internal/solar/base01_worker.py)：复用现有转换审计与计量入口，保留失败阶段和原始工件，语义验证失败时不继续计量。
- [CPU 采集入口](../../scripts/internal/solar/run_base01_cpu.py)与[身份记录](../../scripts/internal/solar/base01_records.py)：锁定环境、源码与输入摘要、全部槽位、超时及原始日志；输出目录已存在则拒绝覆盖。
- [样本测试](../../tests/solar/test_base01_samples.py)、[oracle 测试](../../tests/solar/test_base01_oracles.py)、[采集器测试](../../tests/solar/test_base01_runner.py)：共 69 项，覆盖独立参考计算、存储／策略、数学目标、身份和失败留存。

新增的 JSON 是本次维护任务的运行记录，未注册或发布新的 SOLAR 公共协议；阶段 2 的复用接口及 schema 仍未实现。

## 3. 分母与结果

| 组 | 本轮结果 | 仍不能宣称 |
| --- | --- | --- |
| F | 16 实例 × 2 路径 = 32 槽位，全部尝试；30 个语义验证通过且计量完成，2 个转换拒绝；另保留 8 条计量比较差异 | 不是 32/32 成功，不是 30 个结果全部满足数学目标 |
| G | 10 个独立几何 oracle 均符合冻结目标；拒绝、未知、effects／copy 边界有明确输入规格 | 生产区域并集能力仍为 `not_implemented` |
| M | 10 个规格已落地；6 个有可执行数值／定义域 oracle，4 个是合同 fixture（M03/M04/M05/M10） | 合同 fixture 完整不等于未来接口通过；全部新增目标能力仍为 `not_implemented` |
| R | 8 个 UUID、16 个输入文件摘要重新匹配；两条路径共 16 个槽位均保留 `not_run` | 未执行真实 workload，不是全部 corpus 或旧 41-workload 验收 |
| 现有回归与新增支撑 | 736 passed，0 failed/error/skipped，其中 667 项既有 SOLAR／桥接层测试、69 项新增测试 | 不代表全仓库测试、GPU 资格或真实 mapper 验收 |

F 组结果如下。除 F06 的路径差异外，两条路径的下列总量一致；D/Q 是两个固定实例，不是两轮随机抽样。

| 样本族 | 语义与计量状态 | 关键观察 |
| --- | --- | --- |
| F01 MatMul | 两路径 D/Q 均完成 | 60／385 MAC；188／668 B，符合独立目标 |
| F02 add＋sum | 两路径 D/Q 均完成 | `resource_work` 中 VALU=15／77、reduction=12／70，非 MAC 工作未丢失；不能用旧 `other_ops=0` 字段误判 |
| F03 左结合链 | 两路径 D/Q 均完成 | 84／165 MAC；未改变结合顺序，未产生容量证明 |
| F04 重叠切片 | 两路径 D/Q 均完成，但读取计量有差异 | D 实测分析字段 264 B，数学目标 392 B；Q 为 392 B，目标 648 B；共 4 个路径实例差异 |
| F05 混合 dtype | 两路径 D/Q 均完成 | 224／147 B；两种转换各 32／21 元素，VALU=32／21，符合目标 |
| F06 tril＋MatMul | make_fx D/Q 完成；torchview D/Q 转换拒绝 | make_fx 保留 dense 48／70 MAC；torchview 为 `native_operation_unsupported`：`'tril' is not registered` |
| F07 动态索引 | 两路径 D/Q 均完成 | 索引保持显式输入；当前 `fused_bytes` 为 608／676，按整输入统计。本轮不把这些字段认证为精确的动态访问集合下界 |
| F08 原地修改＋别名 | 两路径 D/Q 均通过语义回放，但修改工作量未被计入 | `resource_work={}`、`num_orphaned_layers=2`；独立目标要求 VALU fp32=8／12。共 4 个路径实例差异；梯度检查按原规格关闭 |

F04 的结果与既有“按输入标识取最大读取”行为一致。F08 表明语义等价与计量完备性是不同检查：原地修改和 alias 回放通过，并不保证修改操作已计入成本。这些问题已成为后续 S1-01／S1-03 的具体回归输入，本轮没有修正生产实现。F06 的不支持状态同样保留，没有自动切换路径或替换程序。

## 4. 尝试历史与证据

所有原始结果位于未提交的 `data/outputs/solar-base01-cpu-20261023/`；没有覆盖旧专项或阶段 0 的证据。

- `attempt-01/`：第一次 32 槽位采集，30 完成、2 转换拒绝，最初记录 4 个切片字节差异。
- `checks-01/`：714 passed、20 failed。失败均发生于假 mapper 的可执行权限检查；容器默认 `/tmp` 挂载为 `noexec`，即使 fixture 权限为 0755，`os.access(..., X_OK)` 仍为 false。
- `attempt-02/`：最终基线。保持镜像和生产源码不变，隔离 Python 缓存，补充原地修改的独立资源计数比较；32 个槽位的语义状态、失败原因、输入摘要和已有总量与第一次完全相同，新增报告 4 个 F08 差异。
- `checks-02/`：临时目录显式允许执行测试 fixture，并隔离旧字节码后，736 项通过；比第一次多的 2 项是新增的 D/Q 原地修改工作量 oracle 检查。未运行真实 mapper。
- `containers-01.json`／`containers-02.json`：真实容器镜像 ID、挂载、设备隔离和退出状态；`attempt-comparison.json` 保留 32 槽位逐项比较。
- `before-files.json`／`before-status.txt`／`head.txt`：本轮开始前工作树；`acceptance-spec-at-collection.md` 保存采集时规格原文，避免之后的文档状态更新改变历史来源。

最终采集的 `source-before.json` 与 `source-after.json` 完全相等。源码和样本在采集期间未变；完成采集后仅更新状态文档，冻结规格和原始摘要保留。完整性闭合与本轮改动范围见根目录 `closeout.json`。

| 最终工件 | SHA-256 |
| --- | --- |
| `attempt-02/summary.json` | `a914d1142aa15b822b7acb1c40e61acdbaa30d4913b808023e948507e7893ff2` |
| `attempt-02/source-before.json` | `6dfd56a0714a6ead012066755bbdc01a6e8452d6e2802cc789c678997c2e45bb` |
| `attempt-02/environment.json` | `6ead140fe630e727b4478e425ae2576864c67d95b95168c933274eee641ea34d` |
| `checks-02/cpu-regression.xml` | `003f89ca2c2da679d76e36d45aa5a8656b9fbd9c03af396d28fc5721a84b5616` |

Ruff、格式、ty、coupling、readability 以及完整文档检查均已通过对应范围。本次通过的完整文档检查在固定镜像内完成；无需修复或改动宿主的 Python 环境。

## 5. 重现方式与退出边界

在仓库根目录创建一个新的证据父目录，再用以下命令采集。不要复用已有 `run` 目录；入口会拒绝覆盖。命令中的镜像已固定，**不添加设备挂载或 `--privileged`**。`/tmp` 的执行权限只用于 CPU 测试 fixture，真实 mapper 不属于 BASE-01 运行。

```bash
rtk proxy mkdir -p data/outputs/solar-base01-cpu-replay
rtk proxy docker run --rm --user 1000:1000 --network none --read-only \
  --cap-drop ALL --security-opt no-new-privileges \
  --pids-limit 256 --cpus 4 --memory 8g \
  --tmpfs /tmp:rw,exec,nosuid,nodev,size=2g \
  --mount "type=bind,source=$PWD,target=/workspace,readonly" \
  --mount "type=bind,source=$PWD/data/outputs/solar-base01-cpu-replay,target=/evidence" \
  --workdir /workspace \
  --env PYTHONPATH=/workspace:/workspace/src \
  --env PYTHONDONTWRITEBYTECODE=1 \
  --env PYTHONPYCACHEPREFIX=/tmp/base01-python-cache \
  --env OMP_NUM_THREADS=1 --env MKL_NUM_THREADS=1 \
  --env MPLCONFIGDIR=/tmp/matplotlib --env XDG_CACHE_HOME=/tmp/cache \
  --entrypoint /venv/bin/python \
  sha256:0ec2e6cba29e673ebe5ad83b7049e5044647fc340174d1c855379efaf36409dd \
  -m scripts.internal.solar.run_base01_cpu --output /evidence/run
```

运行入口的退出码表示槽位采集及身份完整性，不表示数学目标全部满足；必须检查 `summary.json` 的失败与差异。CPU 回归的确切命令和日志保留在 `checks-02/checks.json`，需要临时可写且可执行的目录。

BASE-01 的 CPU 前置条件现已满足。阶段 1–3 仍未开始；真实 R 执行、mapper 容量证据、GPU 资格和发布均未开展，也没有从本轮 CPU 结果继承任何正式资格。
