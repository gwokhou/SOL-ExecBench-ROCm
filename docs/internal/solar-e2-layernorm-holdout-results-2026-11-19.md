# LayerNorm 研究参考独立留出 GPU 确认结果

2026-11-19。**`centered_fp64_v1` 通过 gfx1200 独立留出（`h0`）整链确认。** 本次活动在冻结身份（SHA-256 `de52da67655a2cf9e6d091f1df79fc5037cd5ab1d12999de096319511b049a4d`）与用户显式 `--open-holdout` 授权下完成，使用 RX 9060 XT、ROCm 10.1.0、PyTorch 2.14.0+rocm10.1.0 与 `docker --init`。运行前生成并封存一次性 `holdout-commitment.json`（SHA-256 `36385b8b851e63b2febaa9051714ff728341bc25a418bfb57e222f9c07fa3ab2`，`instances_materialized_at_commitment=0`），在 `online-summary.json` 落盘后再释放 32 字节密钥供无设备审计复核（`audit.json` SHA-256 `cdfbd3efb03f3ddda9255aeb8e7762da032bc376664e9f9655309a32b26ffdaf`）。 [确认规格](solar-e2-layernorm-holdout-protocol-2026-11-19.md) · [冻结包](../../data/outputs/solar-e2-layernorm-gpu-holdout-20261119/freeze.json) · [宿主启动计划](../../data/outputs/solar-e2-layernorm-gpu-holdout-20261119/host-plan.json) · [审计结果](../../data/outputs/solar-e2-layernorm-gpu-holdout-20261119/audit.json)

## 完整分母

| 验收项 | 结果 | 分母与证据 |
| --- | --- | --- |
| P0/P3 整链正对照 | 168/168 通过 | P0、P3 各 84 个会话，每个会话完成 10 次正确性与 1 次可信计时验真（`PASSED`），参考与候选逐次均满足独立 oracle 的 4 ULP |
| 同字节重复（`fixed`） | 24/24 组通过 | 72 个独立会话，每组 3 次；全部调用位置输入、逆置换后参考与候选输出逐位一致 |
| 特征置换（`permuted`） | 24/24 组通过 | 72 个会话（复用 `repeat=2` 锚点并新增 48 个会话，特征维平移 `0,1,17`）；原始输入互异，逆置换后输入与 oracle 逐位一致；描述性逆置换后参考与候选输出亦 24/24 组逐位一致 |
| 独立抽样重复（`independent_draw`） | 24/24 组通过 | 72 个会话（复用 `repeat=2` 锚点并新增 48 个会话，`repeat=5,6` 各使用独立 `uuid/nonce`）；跨组全部 `3×11` 份输入互异，三次独立抽样全部通过且零重复分歧 |
| P1/L3 语义负对照 | 9/9 预期拒绝 | 原判定均为 `INCORRECT_NUMERICAL`，忽略 affine 的见证通过认证，同一单元内 3 次重复输入互异（6/6 负对照单元通过互异性检查） |
| P2/L0 方差负对照 | 9/9 预期拒绝 | 原判定均为 `INCORRECT_NUMERICAL`，FP32 方差不稳定的见证通过认证，同一单元内 3 次重复输入互异 |

三种形状保持 `(128,256)`、`(64,1024)`、`(512,128)`。186/186 个会话全部正常完成并由审计判定为 `passed`（`acceptance_passed=true`），无异常失败、中断、未到达或不确定会话；原裁判判定保留 168 个 `PASSED` 与 18 个 `INCORRECT_NUMERICAL`。共记录 1,866 份原始调用归档（1,698 次正确性与 168 次计时验真）；18 个负对照在首轮正确性判定早拒，其后 162 个正确性位置与 18 个计时位置按设计保留未到达。

GPU 宿主包装器墙钟为 `1,624.67` 秒（27 分 05 秒；在线容器内部运行 `1,613.25` 秒），累计 oracle CPU 时间为 `421.78` 秒（7 分 02 秒），实际分配证据为 `2,121,461,760` 字节（约 `1.98 GiB`，逻辑大小 `2,105,743,682` 字节），会话临时缓存采样峰值为 `237,568` 字节（`232 KiB`）。GPU 交还回执（`gpu-closeout.json` SHA-256 `3e90f02f30805eb6cec1e66b1c3390570ed96962ba961c27747ce83749452372`）记录时钟恢复 `AMDSMI_DEV_PERF_LEVEL_AUTO`、GPU 活跃进程数为 0、L1/L2 锁释放、`clean=true`、`returncode=0`。无设备独立审计耗时 `9.05` 秒、退出码 `0`，复算核验了 `base_seed(case, key)` 派生、全部输入、原始归档 SHA-256 及各分组一致性。 [GPU 交还](../../data/outputs/solar-e2-layernorm-gpu-holdout-20261119/holdout-confirmation/gpu-closeout.json) · [在线摘要](../../data/outputs/solar-e2-layernorm-gpu-holdout-20261119/holdout-confirmation/online-summary.json)

原 E2 的 432 单元 / 1,296 正确性会话、15 个不一致 LayerNorm 单元、90 个不确定对照与未达标结论均完整留存在历史记录中；正式评分契约保持不变（是否纳入正式合同待 `D0` 后另行提案）；`C2/C3` 仍未启动。 [CPU 修复验收](solar-e2-layernorm-reference-cpu-2026-11-19.md) · [开发集 GPU 结果](solar-e2-layernorm-gpu-results-2026-11-19.md) · [原 E2 收尾](solar-e2-successor-results-2026-11-19.md)
