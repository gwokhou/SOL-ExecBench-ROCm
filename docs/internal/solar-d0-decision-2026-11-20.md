# Solar M3 / D0 三线决策综合与下一轮立项报告（2026-11-20）

> [!IMPORTANT]
> 本报告正式闭合 [下一轮迭代计划](solar-next-iteration-plan-2026-11-07.md) 的收官里程碑 **`M3 / D0`**（纯 CPU 核验与文档决策，未启动新的 GPU 实验）：
> - **跨线证据索引核验完成**：由 d0_synthesis.py（`scripts/internal/solar/d0_synthesis.py`，已归档于 tag `v0.1`） 对 **`47` 份封存运行收据**与 **`14` 份 M0–M2 不可变报告**完成逐文件 SHA-256 校验，并在未提交目录 `` `data/outputs/solar-next-iteration/20261120-d0/` `` 原子产出四分索引包（`d0-manifest.json` SHA-256 `18dbca28ba6a9cb4a2672d640eaef8d6ae7ed83760b29f07dcd20c349b6557d4`）。
> - **官方契约与关键基线不变**：正式评分契约、已发布官方分数 `gfx1200-official-score-v2`（`0.497818327`）、默认 `IRBackend` 与原生算子注册表在本轮均未更改；**`L3-d` 新前端继续暂缓**（`0/12` C3 缺口候选）。
> - **下一轮立项选定（严格遵守 `<= 3` 项硬上限，推荐执行顺序 `F2 -> F1 -> F3`）**：
>   1. **`F2`（P0，顺序 1）**：修复 `solar_bridge` / `OrojenesisRunner` 嵌套进程组（Nested-PGID）超时写盘竞态，并将 `7,200 s` `timeloop-mapper` 超时明确归类为显式超时码（纯 CPU 修复，前置解锁 C3 无漂移 `--resume`）。
>   2. **`F1`（P1，顺序 2）**：提出并原子升级 LayerNorm 正式评测参考契约至 `centered_fp64_v1`（基于 `3,669` CPU 调用、`156/156` 开发集 GPU 与 `186/186` 独立 `h0` 留出 GPU 全通证据）。
>   3. **`F3`（P1，顺序 3）**：在 `make_fx_aten` 主线上攻克排名第 1 的 `incomplete_tile_aware_bound`（`12` 定义 / `63` 行，含 `local_global`、`batched_matmul`、`batched_bias` 跨尺度退化）与占总墙钟 `93.2%` 的 `formal:mapper`（`40,272.21 s`）BMM 瓶颈，并同步闭合 `sliding_window`（`arange` 规则）、`rope_*`（索引读取证明）与 `attention_sink`（`effect_boundary`）三项确定性缺口。

相关索引：
- [下一轮迭代计划（2026-11-07）](solar-next-iteration-plan-2026-11-07.md)
- [SOLAR 演化路线图（2026-11-05）](solar-evolution-roadmap-2026-11-05.md)
- [HANDSOFF](../../HANDSOFF.md#solar-evolution)
- [文档索引](../README.md)

---

## 1. 三线终态与“主张—证据—边界”总对账

全部 `13` 条核心主张在 `` `data/outputs/solar-next-iteration/20261120-d0/decision/claims-matrix.json` `` 中以精确指纹绑定；任何局部结论均受下表边界条款约束，禁止跨基线 SHA 或跨工作负载角色外推。

| ID | 所属线 | 终态判定 | 核心主张与量化实证 | 绑定主证据与关键指纹（SHA-256 前 8 位） | 不外推边界 |
|---|---|---|---|---|---|
| `B-1` | B 线 | `PASSED` | 提交 `043a6820`（与 `622c6ba3` 同 739 源文件体系）完成原生准入、`8/8` 代表槽（`6 ready + 2` 精确拒绝）与 `AKA make_fx` 三级链（`163 -> 112/112 -> 163/163`），无 GPU 回放审计通过 | [B2 验收](solar-b2-acceptance-2026-11-08.md)；`solar-b2-20261108/qualification/full/gate.json`（`329eac6a...`）、`final-manifest.json`（`0c670dcd...`） | `performance_authority=false`；不包含 `torchview` 全语料 `42` 缺口，也不自动推导 `LLM_CORE` 就绪 |
| `B-2` | B 线 | `UNPROVED_RETAINED` | 历史 AKA 外部读取证停在 `127/128` 完整 G4；大卷积 `l2n52_conv_activation_batchnorm-w4`（`~23.6 GB` unfold OOM）保留为未证实缺口 | [M0 基线清点](solar-m0-baseline-evidence-2026-11-07.md)（`84422daf...`） | 不得以 `163 ready` 冒充 `163` 完整形式容量分析 |
| `B-3` | B 线 | `PASSED` | 提交 `b45e6f9a` 仅改 `hardware.py`（WGP*2 CU 与受界显存保留对齐），保持 `standard-16gb.yaml` 逐字节不变并在 `C2/C3` 前通过 `8/8` 代表槽 | `` `solar-c2-c3-20261119-r2/preflight/source-diff-vs-prior.json` ``（`6dc37055...`）、`` `representatives-verdict.json` ``（`b5348aca...`） | 为 `C2/C3` 提供最小准入刷新；未重跑 AKA 全 163 链 |
| `E-1` | E 线 | `PASSED`（实验有效） | 后继 E2 完整跑完预登记全分母：`432/432` 逻辑单元、`1,296/1,296` 正确性会话、`144/144` 成本会话、`72/72` 离线回放；GPU 墙钟 `2h 59m 59s`、证据 `12.37 GiB`，独立审计复现 `36,477` 份输入 | [E2 收尾结论](solar-e2-successor-results-2026-11-19.md)（`42669c1a...`）；`` `independent-evidence-audit.json` ``（`f9a14b60...`）、`` `terminal-verification.json` ``（`5d2b067e...`） | 仅证明无缺行与审计一致，不等于裁判达标 |
| `E-2` | E 线 | **`CRITERION_NOT_MET`**（裁判未达 E0 标准） | 原生产裁判检出 `81/81` 必检 P1 违例、`558` 个已认证合法对照 `0` 确认数值误拒（另有 `90` 个 LayerNorm 不确定对照、`3` 次计时 `REWARD_HACK`），但 **`15` 个 LayerNorm L1 单元未通过 3 次重复一致性** | `` `correctness.json` ``（`543c6168...`）、`` `counterexample-and-witness-index.json` ``（`77705be6...`） | 保留原 E2 负结论与正式契约；`3` 个已用 E2 `h0` 承诺永久退役 |
| `E-3` | E 线 | `PASSED`（开销实测） | 在 `66/72` 可比较配对测得额外诊断墙钟比值中位数：GEMM `1.54x`（`+4.44 s`）、求和 `1.16x`（`+0.98 s`）、LayerNorm `2.96x`（`+12.50 s`，范围 `2.88–18.84x`）；另 `6` 对 LayerNorm P3 提前拒判保留空值 | `` `cost.json` ``（`3330f27e...`） | 反映冷会话缓存下诊断开销，不作官方评分计时或加速比 |
| `E-4` | E 线 | `PASSED`（研究修复验证） | Opt-in `centered_fp64_v1` 通过 CPU `3,669` 次留存调用、开发集 `156/156` 会话，以及独立 `h0` 留出 `186/186` 会话（`168/168` 正对照、`24/24` 三类重复一致性组、`18/18` P1/P2 负对照全部拒绝） | [CPU 收据](solar-e2-layernorm-reference-cpu-2026-11-19.md)（`b90bbc41...`）、[开发 GPU](solar-e2-layernorm-gpu-results-2026-11-19.md)、[留出 GPU](solar-e2-layernorm-holdout-results-2026-11-19.md)（`cdfbd3ef...`） | 已在三种形状研究协议下闭合；进入官方评审需通过立项 **`F1`** 原子升版 |
| `C-1` | C 线 | `PASSED`（C2 完成） | RX 9060 XT 显存标定为 `mem-12gib`（`15.30 GB` 稳定连续可分配 / `17.10 GB` 可见），在 `r1` 与 `r2` 冻结完全一致的 `20` 算子 / `180` 工作负载 / `360` 行目标视图（`36` 条准入决策） | `` `capacity-evidence.json` ``（`c05a4fd7...`）、`` `c2-summary.json` ``（`0ad8eadd...`）、`workload_view_digest = 65b39113...` | 仅覆盖 `rx9060xt/standard-16gb` 与 `profile=core`，非核心算子与跨配置未测 |
| `C-2` | C 线 | **`BOUNDED_INCOMPLETE`**（12h 有界终态） | `r2` 在 `12.0 h`（`43,200.38 s`）自动停于预算上限，证据体积 `243.92 MiB`（vs `50 GiB`）：已执行 `168/360`（`40 passed`、`111 failed`、`16 unproved`、`1 budget_exhausted`、`0 infrastructure_error`），`192` 行 `not_reached`；`8/20` 定义全 9 槽闭合、`12/20` 仅 `smoke` | [C2/C3 终态报告](solar-c2-c3-results-2026-11-20.md)（`a0da91f1...`）；`` `target-coverage.json` ``（`86dd98e5...`）、`` `c3-resume-closeout.json` ``（`9ec4b4fe...`） | 不可写成完整 `LLM_CORE` 覆盖完成；后续扩展归立项 **`F3`** |
| `C-3` | C 线 | `PASSED`（覆盖差分洞察） | `0/8` 完整 `make_fx_aten` 定义具 9 槽均一性（`local_global`、`batched_bias`、`batched_matmul` 在非 smoke 槽退化）；`formal:mapper` 占用 **`40,272.21 s`（93.2%）**；双路径共同通过仅 `4` 组 `smoke` `norm_activation` | `` `c3-summary.json` ``（`ba77a820...`） | 证明 `smoke` 不可外推至全槽位；`96` 对双路径行仍 `not_reached` |
| `C-4` | C 线 | `PASSED`（审计与根因） | 无设备终态审计验证全部 `168` 行语义收据且非超时瞬态 `0` 漂移，并准确定位 `row-167` 嵌套进程组（`PGID_2` `timeloop-mapper` 晚刷盘 `+3.189 s`）导致原目录 `--resume` 受阻；`r1` smoke（`851acbfe...`）与 `r2` 保留件均逐字节未变 | `` `c3-final-audit.json` ``（`91fc0ebd...`）、`` `c3-exposure-ledger.json` ``（`e6969c96...`） | 修复此竞态前保持禁止直接续跑带超时漂移的目录，由立项 **`F2`** 首要解决 |
| `D-1` | 跨线 | `INVARIANT_PRESERVED` | C3 全部 `12` 类去重缺口的 `l3d_frontend_candidate` 均为 `false`；本轮结束时正式契约、注册表、默认 `IRBackend` 与 `start700` 暂停状态均保持未动 | `` `c3-summary.json` ``（`ba77a820...`）、d0_synthesis.py（`scripts/internal/solar/d0_synthesis.py`，已归档于 tag `v0.1`） | `L3-d` 暂缓维持到出现跨双路径不可消解的系统性前端表达缺口为止 |

---

## 2. 留出承诺与暴露隔离总账（防止样本泄露）

| 活动 / 线 | 隔离载体与精确身份 | 已暴露分母（永久丧失盲测资格） | 仍未暴露分母（后续可保留条件） | 后续使用规则 |
|---|---|---|---|---|
| 原 E2（`622c6ba3`，20 GiB 停） | `` `solar-e2-20261108-r2/experiment/holdout-commitment.json` ``（`579db95d...`） | 含 `325` 个正确性会话（含 GEMM `h0`） | SUM / LayerNorm 留出未触及但同一 `h0` 承诺已整体结束 | **整体退役**，不得以同一承诺续跑 |
| 首版后继 E2（PID 停于 500） | `` `solar-e2-successor-frozen-20261118/successor-confirmation/holdout-commitment.json` ``（`5ed7a186...`） | `500` 个正确性会话，活动整体冻结失效 | 同一活动整体退役 | **整体退役**，不得复用 |
| 修正版后继 E2（完整 432 单元） | `` `solar-e2-successor-reaping-frozen-20261118/successor-confirmation/holdout-commitment.json` ``（`1d0681d7...`） | 全量 `432` 单元（含所有 `h0` 会话） | 无（已完整揭示） | **永久历史基准**，仅作基线对比 |
| LayerNorm `centered_fp64_v1` `h0` 确认 | `` `solar-e2-layernorm-gpu-holdout-20261119/holdout-confirmation/holdout-commitment.json` ``（`36385b8b...`） | 全部 `186` 个 `h0` 会话已执行并解密复核 | 无（`h0` 密钥已释放） | 为 `F1` 提案充分证据；未来改版后需另签新实例验证 |
| `C3` `r1` 冒烟 & `r2` 12h 扫描 | `` `c3-exposure-ledger.json` ``（`e6969c96...`，对应视图 `65b39113...`） | 共 **`84` 个唯一 UUID / `168` 行**：`smoke` `20` UUID（`40` 行）、`development` `32` UUID（`64` 行）、**`holdout` `32` UUID（`64` 行）** | 共 **`96` 个唯一 UUID / `192` 行**：`development` `48` UUID（`96` 行）、**`holdout` `48` UUID（`96` 行）** | 已暴露的 `32` 个 `holdout` UUID 不得再作未暴露确认集；`F3` 训练与调试严格限定在 `smoke` + `development`，保留上述 **`48` 个未到达 `holdout` UUID** |

---

## 3. 对 `L3-d` 新前端与显存上限项的负面决策

依据 [M0 LLM_CORE 覆盖率规范](solar-m0-llm-core-coverage-spec-2026-11-07.md) §7 与 [下一轮迭代计划](solar-next-iteration-plan-2026-11-07.md) §6 的硬判据，本轮 **明确否决现在启动 `L3-d`（Affine Loop IR / LLM / JAX 前端）**：
1. **双路径均非共同提取缺失**：`make_fx_aten` 在已执行的 `84` 行就绪扫描中 `graph_extraction = 84/84 passed`、`ir_conversion = 84/84 passed`，说明图提取与 IR 转换并非共性瓶颈；`12` 类去重缺口中 `l3d_frontend_candidate = false` 占比 **`12/12`（100%）**。
2. **形式层与硬件内存边界须分治**：排名第 1 的 `incomplete_tile_aware_bound`（`12` 定义 / `63` 行）与 `93.2%` 墙钟占用来自 **Timeloop/Orojenesis 形式映射与 tile-aware 边界构造**；`16` 行 `attention` `capacity-*` `conversion_not_proven_device_oom` 与 `4` 行 `torchview` eager OOM 来自 **RX 9060 XT 15.92 GiB 物理显存边界**，均无法通过增加前端方言消除。

---

## 4. 下一轮立项选定（严格 `3` 项上限，推荐顺序 `F2 -> F1 -> F3`）

排序原则：**先用纯 CPU 基础设施加固解阻塞（`F2`），再落地证据已齐备的裁判契约升级（`F1`），最后集中投入攻克规模最大的 SOLAR `LLM_CORE` 主线阻塞与 BMM 搜索耗时（`F3`）**。

### 优先级 1（执行顺序 1）：`F2` — `solar_bridge` / `OrojenesisRunner` 嵌套进程组超时回收与映射超时码修正
- **优先级与性质**：`P0` / 纯 CPU 代码与单元测试修复（不消耗 GPU，不改变任何评分契约或已封存 `r1`/`r2` 工件）。
- **真实消费者**：后续所有长时间 `C3` 续跑战役及自动化 `target-coverage --resume`。
- **具体归因与修改范围**：
  - [runner.py](../../src/sol_execbench/core/solar_bridge/runner.py) 与 [runner.py（Orojenesis）](../../src/solar/analysis/orojenesis/runner.py)：消除 `row-167` 实证的 `PGID_1`（worker）强杀后孤立 `PGID_2`（`timeloop-mapper`）继续刷盘 `.formal.*/timeloop-mapper.oaves.csv` 长达 `+3.189 s` 引发 [target_coverage.py](../../src/sol_execbench/core/solar_bridge/target_coverage.py) `coverage resume artifact drift` 的竞态。
  - 将 `timeloop-mapper` 触达 `7,200 s` `TimeoutExpired`（`row-082`/`083`）明确映射为 `BUDGET_EXHAUSTED` 下的显式 `mapper_timeout` 理由码，而非当前 `toolchain_unavailable`（`FAILED`）。
- **验收标准**：构造嵌套子进程超时仍在写盘与 `7,200 s` mapper 超时两类确定性 CPU 夹具，确认返回前子进程树全回收且从原目录 `--resume` 零漂移通过。

### 优先级 2（执行顺序 2）：`F1` — `centered_fp64_v1` 升格为 LayerNorm 正式评测参考的契约提案与原子更新
- **优先级与性质**：`P1` / 裁判契约原子升级（跨版本升级并附新留出 GPU 确认，保持 `gfx1200-official-score-v2` 历史发布不可变）。
- **真实消费者**：正式 SOL-ExecBench LayerNorm 数值正确性与计时一致性门禁。
- **依据与修改范围**：直接消费 `E-4` 已验实证（CPU `3,669` 次调用 + dev `156/156` + `h0` `186/186`），在 `src/sol_execbench/core/bench/judge/`、`src/sol_execbench/core/integrity/` 与 `problems/AMD_AKA/torch2hip/l1n40_layernorm/` 上提出并落实受版本管理的参考升级方案。
- **验收标准**：`check_schema_versions.py` 与全量裁判单测通过；若需上机确认，须另签全新预登记实例集，严禁复用已退役 E2 与 `36385b8b...` `h0` 承诺。

### 优先级 3（执行顺序 3）：`F3` — `make_fx_aten` `LLM_CORE` 形式边界退化、93.2% BMM 映射耗时与 3 项确定性规则闭合
- **优先级与性质**：`P1` / `SOLAR` 形式分析与覆盖深度主线攻坚（须在 `F2` 合入后再启动长时 GPU/CPU 验证）。
- **真实消费者**：`LLM_CORE` `core` 配置下的 `make_fx_aten` 正式覆盖率与扫描吞吐。
- **聚焦范围**（限定主线 `make_fx_aten`，不散焦到二次对比路径 `torchview`）：
  1. 根因修复 `incomplete_tile_aware_bound`（`12` 定义 / `63` 行），优先解释为何同一 `make_fx_aten` 定义会在 `smoke`/`latency` 通过而在非 smoke 尺度退化（`local_global`、`batched_matmul`、`batched_bias`）。
  2. 测量并约束 4D BMM 在 Orojenesis `linear-pruned` 下的搜索预算与剪枝策略（消除 `40,272.21 s` 占 `93.2%` 墙钟的极端长尾）。
  3. 同步闭合 `make_fx_aten` 上已精确定位的 3 项确定性规则/证明缺口：`sliding_window.v2` 的 `arange` `amd_resource_v3` 规则、`rope_interleaved.v2` / `rope_yarn.v2` 索引读取证明、`attention_sink.v2` 跨作用域写后读 `effect_boundary` 证明。
- **验收与隔离约束**：开发回归仅使用 `smoke`（20 UUID）与 `development`（*-low）工作负载；已暴露的 `32` 个 `holdout` UUID 标记为仅可做已知回归不可作盲测，**保留未到达的 `48` 个 `holdout` UUID 用于最终修复后验收**。

### 明确延后清单（不占当前 3 项名额）

| 延后项 | 处置 | 原因与再触发条件 |
|---|---|---|
| `L3-d` 新前端（Affine Loop IR / LLM / JAX） | `DEFERRED` | C3 缺口中 `0/12` 属共同前端提取/表达缺失；出现独立跨路径系统性前端缺口后才启动 |
| 16 GiB 显存 OOM 项与 `l2n52` 大卷积回放 | `RETAINED_OUT_OF_SCOPE` | `16` 行 `attention` `capacity-*`、`4` 行 `torchview` eager OOM 及 `l2n52`（`~23.6 GB`）属物理显存上限 |
| `torchview_extended_einsum` 次要适配器缺口 | `DEFERRED_SECONDARY` | 仅作独立对比路径（`expand_as`、`sliding_window` 输出映射、`attention_sink` 数值梯度差异）；待 `make_fx_aten` 主线 `F3` 收敛后再排期 |
| `L3-b` 阶段 2 符号化 Orojenesis、多输入 einsum DP、`L5` `gfx942`、`start700` 重发 | `DEFERRED` | 需独立新任务章程或专用硬件窗口，不并入本轮出口 |

---

## 5. `D0` 四分索引目录产出与校验摘要

未提交索引根目录为 `` `data/outputs/solar-next-iteration/20261120-d0/` ``，由 `.venv/bin/python -m scripts.internal.solar.d0_synthesis` 原子写入并在 `tests/scripts/test_d0_synthesis.py` 覆盖边界测试：

| 索引文件（相对 `` `data/outputs/solar-next-iteration/20261120-d0/` ``） | 大小 (B) | SHA-256 |
|---|---:|---|
| `baseline/index.json` | 8,710 | `b8dd5165bd8e9b77cce86ca81c8e6f69ca33c95abd338af6c9a098aad19f8dab` |
| `judge/index.json` | 9,701 | `e1ef2948388a07d2f6f89518522422c5a46f9b6bbdc2febb0df1d5878c8c9062` |
| `llm-core/index.json` | 9,006 | `79691dbc595084a68ef335ffcbd3b4bbf74912b29b95c10c6168f3bfcde5acfb` |
| `decision/claims-matrix.json` | 8,225 | `e4aac20d74e73274097c018d424ed4f32a4204cd14e6267bec070e9fb5ca2405` |
| `decision/selected-followups.json` | 5,527 | `39c25cdb90c2032e8ee835a221ece4331255f7b483a6549d8301f97f34b48b21` |
| `decision/d0-manifest.json` | 797 | `18dbca28ba6a9cb4a2672d640eaef8d6ae7ed83760b29f07dcd20c349b6557d4` |
