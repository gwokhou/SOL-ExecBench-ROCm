# LayerNorm 研究参考独立留出 GPU 确认规格

2026-11-19。为 opt-in 研究参考 `centered_fp64_v1` 预登记独立留出确认活动（`layernorm-centered-fp64-holdout-v1`）。执行环境固定为 RX 9060 XT（gfx1200）、ROCm 10.1.0、PyTorch 2.14.0+rocm10.1.0、绑定 `docker --init` 与只读 `capsule` 覆盖层，沿用 E2 相同三种 FP32 形状（`(128,256)`、`(64,1024)`、`(512,128)`）、L0–L3、实际 FP32 epsilon、affine、原全元素数值容差及独立 4 ULP 诊断。全部正负对照仅使用 `JudgeInstance.HOLDOUT`（`h0`）；在明确授权并进入在线执行前不生成任何密钥、承诺或输入实例。历史 E2 结论与正式评分契约保持原样。 [开发集规格](solar-e2-layernorm-gpu-protocol-2026-11-19.md) · [开发集 GPU 结果](solar-e2-layernorm-gpu-results-2026-11-19.md) · [原 E2 收尾](solar-e2-successor-results-2026-11-19.md)

## 分母

| 对照 | 固定排程 | 探测范围 |
| --- | ---: | --- |
| P0/P3 同字节重复（`fixed`） | 3 形状 × 4 输入族 × 2 策略 × 3 会话 = 72 | 同一调用位置在三个独立进程使用相同 `h0` 输入字节（`repeat` `0,1,2`，`shift=0`） |
| P0/P3 特征置换（`permuted`） | 3 形状 × 4 输入族 × 2 策略 × 3 置换 = 72 | 复用 `repeat=2` 锚点并新增 48 会话（`repeat` `3,4`，`shift` `1,17`） |
| P0/P3 独立抽样重复（`independent_draw`） | 3 形状 × 4 输入族 × 2 策略 × 3 独立抽样 = 72 | 复用 `repeat=2` 锚点并新增 48 会话（`repeat` `5,6`，`shift=0`），各用独立 `(uuid, nonce)` 检验 E0 跨抽样重复一致性 |
| P1/L3 语义负对照 | 3 形状 × `h0` × 3 独立会话 = 9 | 忽略 affine 的必检违例 |
| P2/L0 方差负对照 | 3 形状 × `h0` × 3 独立会话 = 9 | FP32 不稳定方差见证 |

合计 **186** 个独立会话（最多观测 `168 × 11 + 18 × 1 = 1,866` 次调用；324 个早拒后正确性位置与 18 个计时位置按设计保留未到达）。单次 32 字节随机密钥仅在在线容器获得容器托管锁后生成，写入 `holdout-commitment.json`（`instances_materialized_at_commitment=0`），并通过 `verify_worker_commitment` 核验后方可供 `base_seed(case, key)` 派生使用；密钥仅在 `online-summary.json` 写出后于关闭阶段释放至 `holdout-key-after-online.bin` 供无设备审计复算。

## 验收

- **168 个正对照会话**原裁判均为 `PASSED`，完成 10 次正确性与 1 次可信计时校验（`warmup=0`、`iterations=1`、`trials=1`），参考与候选逐次分别满足独立 oracle 的 4 ULP，且单会话内所有输入 SHA-256 互异。
- **24/24 `fixed` 组**：三次同字节重复的输入、逆置换后参考输出及逆置换后候选输出在每个调用位置均逐位一致。
- **24/24 `permuted` 组**：三个原始输入 SHA-256 在每个调用位置均互异，逆置换后输入与 oracle 逐位一致，且三者均通过原容差与 4 ULP；逆置换后输出是否逐位一致作为描述性字段留存。
- **24/24 `independent_draw` 组**：跨三次独立抽样的全部 33 份输入（`3 × 11`）SHA-256 互异，且三次会话均通过原容差与 4 ULP，无重复分歧。
- **9/9 `P1/L3` 与 9/9 `P2/L0` 负对照**：均在首轮正确性阶段以 `INCORRECT_NUMERICAL` 拒绝，`witness_certified=True`、`reference_4ulp=True`、`candidate_4ulp=False`，且同一单元内 3 次重复的输入 SHA-256 互异。
- **交还与独立审计**：原生 GPU 交还回执 `clean=true`、`returncode=0`；无设备容器审计退出码为 0，完整核验留出承诺、32 字节事后密钥、`base_seed` 重算、全部归档 SHA-256 与逐组结果。

## 预算与停机条件

GPU 串行执行，沿用 `120 秒/会话`、`120 秒/会话` oracle CPU 上限与 `256 MiB` 会话临时缓存；总原生 GPU 窗口上限 `3,600 秒`（1 小时，按开发集实测外推 186 会话约 26–27 分钟），证据容量上限 `6 GiB`（按开发集实测外推保留约 `1.95 GiB`，含活跃预留约 `2.97 GiB`）。任何身份漂移、承诺校验失败、预算耗尽或运行时/参考异常立即终止活动且不重试，完整留存已生成工件、未到达槽与异常原因。
