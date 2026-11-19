# 历史文件清理决策

日期：2026-11-08。用户已授权批量分析并在必要时物理删除历史文件。
清理前提交为 `1c6404e5e4e9adb822c899011a67dc2a26cb697f`。
本次扫描 1,535 个受版本控制的文本文件，建立 132 个内部文档/脚本的引用清单，
再对候选逐项核查源码调用、测试、文档、CI 与替代入口。
这不是对所有历史数据或每条研究结论的重新验收。

## 已退役文件

共删除 7 个受版本控制的文件（4 份文档、3 个脚本，650 行）。
删除前逐字节匹配上述 Git 提交；未删除未提交内容。

| 路径 | 决策依据与保留入口 |
| --- | --- |
| `docs/internal/python_architecture_migration.md` | 迁移已完成；仍有效的 slotted/keyword-only dataclass、显式序列化和 immutable registry 规则并入[开发指南](../user/DEVELOPMENT.md) |
| `docs/internal/execbench-rocm101-python314-torch214-research.md` | 升级前建议；旧栈和“事件因果尚未接入”等状态已过期，无当前引用；现行入口为 [ROCm 指南](../user/rocm.md)、[scope 验收](hip-torch-scope-acceptance.md)与 HANDSOFF |
| `docs/internal/solar-rocm101-python314-torch214-research.md` | 已被[升级后分析](solar-upgrade-opportunities-2026-10-11.md)和[当前路线图](solar-evolution-roadmap-2026-11-05.md)取代；唯一引用已更新 |
| `docs/internal/solar-l3e-batch2-recon-2026-11-06.md` | 实施前勘察的开放问题已由[批次 2 冻结验收规格](solar-l3e-batch2-acceptance-2026-11-06.md)解决并取代；规格、后续验收台账和证据保留 |
| `scripts/internal/solar/l3e_batch1_campaign.py` | 仅服务已关闭批次 1 的固定翻转集和 G4 门，无外部调用方；[原验收规格](solar-l3e-external-reads-acceptance-2026-11-06.md)、当前 external-reads 测试和证据保留 |
| `scripts/internal/solar/lineb_equality.py` | 一次性双工作树比较；硬编码 pristine 路径在当前目录与工作树重合，已不能作为现行比较入口；[fake 生产化规格](solar-lineb-fake-production-acceptance-2026-11-07.md)、生产 backend 测试与历史等价证据保留 |
| `scripts/internal/solar/lineb_extract_slot.py` | 唯一调用方为上一行已退役的脚本；随调用方一并删除 |

恢复任意文件时使用上表完整路径替换 `<path>`：

```bash
rtk proxy git show 1c6404e5e4e9adb822c899011a67dc2a26cb697f:<path>
```

该提交是本地恢复锚点，不假设已推送。删除清单的原始 SHA-256、引用方和理由
保存在忽略目录 `data/outputs/history-cleanup-20261108/retirement-manifest.json`；
Git 中的旧实现用于历史复现，不能直接继承为当前执行入口或资格证据。

## 保留与修正

- BASE-01 的原始/复采报告绑定改进前行为、失败历史及证据摘要，继续保留。
  `base01_samples`、`base01_oracles`、records 和 workers 仍被当前测试使用。
- 阶段 1–5 的 CPU/mapper 报告、各批次预固定规格、P0 资源证据、论文版本比较、
  R accounting 说明和未闭合研究均未因年代或零引用而删除。
- native trace-scope 等 GPU 辅助入口没有足够的退役依据，继续保留；未执行。
- [文档索引](../README.md)区分当前入口和历史证据，修正 AKA 文件改名后的断链。
  mapper 报告的六个证据链接改为经存在性核对的外部根及相对路径；不复制原始工件。
- 升级机会文档增加历史快照标记；upgrade 脚本指南的过期 same-lane-only 说明
  更新为 v3 的受限、显式 opt-in 证据边界。

M0 三份冻结规格、独立 E0 算术勘误、B1 的 38 个绑定工件与独立执行副本，
以及三个原有未提交研究文档均保留原字节。未清理生成数据、缓存、Docker 镜像、
外部工作树或历史诊断存储；旧数据退役计划不能仅凭年龄或 CAS 导入情况执行。

## 验证范围

本次只运行 CPU 文档/架构检查及相关测试，不运行 GPU、真实 mapper、容量探针，
不生成留出实例。生产源码、依赖、正式评分合同和质量基线未修改。
当前 B1 执行基线仍为 `160b9988c0f1c6f28be514651918419a39f0d438`；
清理不改变 B2/E2/C2/C3 的未执行状态。验证结果登记在本节及 HANDSOFF。

- 文档检查通过；清理前的 8 处既有断链已修正或随过时文件移除。
- Ruff check/format、ty、coupling、readability、schema、production reachability、
  dataclass policy 与 `git diff --check` 通过，未调整检查基线。
- 相关 CPU 测试 **143 passed、1 failed**。唯一失败为
  `test_active_cuda_nvidia_residue_is_classified`：
  `src/sol_execbench/core/bench/judge/executor.py:185` 中清空
  `CUDA_VISIBLE_DEVICES` 的 CPU 隔离用法没有对应分类。
  在未修改的 B1 执行副本上单独重跑，复现同一行的失败；这是一项既存审计规则缺口，
  未通过改生产逻辑、删除测试或放宽规则掩盖。原 B1 的 288 项记录不是这组测试，仍保持原样。
- 首次直接调用 `pytest` 时仓库根目录不在模块搜索路径中，BASE-01 三个测试模块
  收集失败；改用 `.venv/bin/python -m pytest -n 0` 后获得上述实际结果。
  两次日志/JUnit 与 B1 单项复核记录均保留在清理证据目录。

后续用户单独授权修复上述审计缺口并推进 B2。该修复为本地提交 `043a6820`，
宿主与固定容器各 18 项检查通过；本节原 143/1 结果不改写。
新执行身份与 B2 停止点见[修订冻结](solar-b1-refreeze-2026-11-08.md)和
[B2 检查点](solar-b2-checkpoint-2026-11-08.md)。
