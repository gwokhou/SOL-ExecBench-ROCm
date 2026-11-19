# LayerNorm 研究参考开发集 GPU 验收结果

2026-11-19。**`centered_fp64_v1` 通过 gfx1200 开发集整链验收。** 运行前独立冻结了参考与 P0 源码、P1/P2/P3 候选、原数值检查、研究实现、排程、镜像、Docker init 和目标设备。使用 RX 9060 XT，ROCm 10.1.0，PyTorch 2.14.0+rocm10.1.0。冻结 SHA-256 为 `142b2502b7c5693da11d16b3b6bef417e28706e8e74f11c25d6b5bad301a9efd`。 [验收规格](solar-e2-layernorm-gpu-protocol-2026-11-19.md) · [冻结包](../../data/outputs/solar-e2-layernorm-gpu-dev-20261119/freeze.json) · [宿主启动计划](../../data/outputs/solar-e2-layernorm-gpu-dev-20261119/host-plan.json)

## 完整分母

| 验收项 | 结果 | 分母与证据 |
| --- | --- | --- |
| P0/P3 整链正对照 | 120/120 通过 | P0、P3 各 60 个会话，每个会话完成 10 次正确性和 1 次可信计时验真，参考与候选逐次均满足独立 oracle 的 4 ULP |
| 同字节重复 | 24/24 组通过 | 72 个独立会话，每组 3 次。每个调用位置的输入、参考和候选输出逐位一致 |
| 不同特征置换 | 24/24 组通过 | 72 个会话，其中 24 个与同字节组共享锚点。每个调用位置的三种输入字节均不同，逆置换后输入、oracle、参考和候选输出逐位一致 |
| P1/L3 语义负对照 | 18/18 预期拒绝 | 原判定均为 `INCORRECT_NUMERICAL`，忽略 affine 的见证通过认证 |
| P2/L0 方差负对照 | 18/18 预期拒绝 | 原判定均为 `INCORRECT_NUMERICAL`，FP32 方差不稳定的见证通过认证 |

三种形状为 `(128,256)`、`(64,1024)`、`(512,128)`。156/156 个会话均有终态，无异常失败、中断、未到达会话或不确定会话。原始裁判判定保留 120 个 `PASSED` 和 36 个 `INCORRECT_NUMERICAL`。全部原始调用工件共 1,356 份，其中 1,236 次正确性和 120 次计时验真。36 个负对照均在第一轮正确性判定结束，其后的 324 个正确性和 36 个计时位置保留为未到达。 [逐槽与逐调用审计](../../data/outputs/solar-e2-layernorm-gpu-dev-20261119/audit.json)

GPU 宿主窗口为 21 分 21 秒。开发证据目录实际分配约 1.45 GiB，oracle 累计用时 5 分 12 秒 CPU，临时缓存采样峰值 232 KiB。完成后时钟恢复 `AUTO`，GPU 活跃进程数为零，两级锁释放。无设备审计退出码为 0，复核了原始调用 SHA-256、开发输入派生、逐次容差和一致性分组。110 项相关 CPU 测试通过。 [GPU 交还](../../data/outputs/solar-e2-layernorm-gpu-dev-20261119/development/gpu-closeout.json) · [审计执行记录](../../data/outputs/solar-e2-layernorm-gpu-dev-20261119/host-run/audit.json)

本轮只使用 d0/d1 开发输入。原 E2 的全分母、15 个不一致单元、90 个不确定对照、计时拒绝及未达标结论继续留存。L1/P2 的真实越界见 CPU 历史对照，本轮复核的 P2 负对照范围为 L0。正式评分契约沿用原版本，C2/C3 仍待启动。 [CPU 修复验收](solar-e2-layernorm-reference-cpu-2026-11-19.md) · [原 E2 结论](solar-e2-successor-results-2026-11-19.md)
