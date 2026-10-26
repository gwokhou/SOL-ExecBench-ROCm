# SOLAR 阶段 1 CPU 验收映射

检查日期：2026-10-26。本文件对应 S1-01–03；任务进展由 [HANDSOFF](../../HANDSOFF.md#solar-evolution) 维护。冻结数学目标仍以 [F/G/M/R 清单](solar-evolution-acceptance.md) 为准，BASE-01 中的 `not_evaluated` 是当时的观测，不回写为本轮通过。

## 访问合同与 G 组

| 场景 | 生产行为及边界 | 已执行 CPU 覆盖 |
| --- | --- | --- |
| G01–03 | 同 base 重复、重叠、带洞区域分别为 64、96、64 元素 | `test_access_regions::test_frozen_interval_oracles`；小域坐标枚举同时检查顺序、重复及单调性 |
| G04 | transpose 后映射回原 base，条带并集 28 | `test_access_regions::test_transpose_and_split_keep_base_coordinates_and_output_slots`；`test_access_graph::test_transpose_then_slice_is_in_original_base_coordinates` 双路径提取 |
| G05 | 只读消费的 split slot，共 32 元素 | 同上几何测试；`test_access_graph::test_only_consumed_partition_slots_contribute` 双路径 |
| G06 | 独立 base 各读 32，共 64；未证明共享存储拒绝 | `test_access_regions::test_g06_independent_bases_keep_both_frozen_footprints`；`test_access_graph::test_different_inputs_are_not_merged_by_matching_shape`；`test_input_storage` 在 clone 前检查别名及布局漂移 |
| G07 | 动态索引不可获得静态精确读取；数值计量保留为 diagnostic | `test_access_graph::test_dynamic_indices_do_not_acquire_a_static_exact_map`；`test_external_reads::test_unknown_access_and_mutation_remain_diagnostic`；独立 oracle 保留相同 shape 对应 3/4 个位置的反例 |
| G08 | 负索引和空域分别 12/0；非单位步长明确 unavailable | `test_access_regions::test_empty_negative_scalar_and_nonunit_boundaries`；`test_access_graph::test_nonunit_slice_retains_an_explicit_unavailable_reason` |
| G09 | 越界、伪造元素数、冲突 base 元数据拒绝；3 区域超出 2 区域预算时下界为 8，而非包围盒 40 | `test_access_regions` 的 invalid boxes、bad metadata、incompatible bases 及冻结预算场景；`test_access_contract::test_g09_declared_64_elements_cannot_fit_a_32_element_domain` |
| G10 | mutation/atomic/opaque effect 保持活跃；copy 阻断透明追踪；不跨 epoch 合并 | `test_effect_liveness`；`test_access_graph::test_copy_boundary_is_not_a_transparent_view`；`test_access_regions` 的 epoch 和非连续 reshape 拒绝 |

上述文件均位于 `tests/solar/`。S1 的访问证明限于已支持算子图的静态输入域，不是代数最小工作量、按需消零或完整缓存流量。未知操作不会通过零读取获得精确资格。

`access.json` 绑定 executable IR 与 operator graph 两份摘要，并在加载时检查；派生映射、源 shape/dtype 和独立存储条件在信任边界验证。转换证明及正式工件清单必须包含访问工件。篡改图、operator、访问映射、存储、版本或删除 operator 均拒绝。只有区域、分割单元或时间预算耗尽可标记保守下界；动态访问、copy、effects 等未证明原因不得冒充该下界。

## F 组及证据边界

`accounting-integration/attempt-s1` 保留锁定镜像中的 32 个 F 槽位：30 个语义与计量完成、2 个既有 torchview `tril` 转换拒绝；完成槽位无冻结数学差异。F04 的读取并集为 96/160 元素，加输出后 392/648 B；F08 为 8/12 VALU，访问仍未证明。相对原始 BASE-01，仅 F04/F08 数值变化；完整差异及 494 个工件绑定见该目录 `comparison.json`。

新增规则审查证据位于 `data/outputs/solar-stages1-5-20261026/contract-closeout/`。`check-01` 的 946 项相关 CPU 回归及 Ruff、format、ty、coupling、readability、文档检查全部通过。其后仅增加冻结 G06/G09 测试；`check-02` 的 59 项相关测试及同一组质量检查全部通过。两次检查均保存环境、源摘要、命令、日志和 JUnit。

运行使用固定镜像 `sha256:0ec2e6cba29e673ebe5ad83b7049e5044647fc340174d1c855379efaf36409dd`，无网络、无 GPU 挂载；依赖锁未改。真实 mapper、R 组和 GPU 实验未运行。CPU 完成不提供性能、容量模型全域或发布资格；旧 BASE-01 及其失败记录保持不变。
