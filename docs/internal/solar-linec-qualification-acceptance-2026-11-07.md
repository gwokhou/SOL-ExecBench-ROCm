# 线C 全分母资格链（make_fx 分母）：预固定验收规格

制定日期：2026-11-07（执行前预固定）。前置：L1 curated 闭合、L3-c 矩阵
（make_fx 就绪 163/163）、线A 收官（external-reads 分母清零）。

## 0. 范围（预固定）

在冻结语料与 L1 基线 workspace 上，以生产 CLI
`sol-execbench solar qualify-static|qualify-canary|qualify-full`
按序运行**make_fx_aten 后端**的完整资格链。torchview 链不在本批（其
42 项就绪缺口已由 L3-c 矩阵记录；是否补跑另立）。

## 1. 输入与协议（预固定）

- workspace：L1 释放 workspace（`data/outputs/solar-l1-release-20261105/
  workspace/`，plan.json 绑定 manifest sha 不变）；manifest：
  `problems/AMD_AKA/manifest.yaml`（sha `24cee562…`）；mapper：pinned
  可信 home（sha `18591892…`）。
- 隔离 qualification root：
  `data/outputs/solar-qualification-mfx-20261107/`；device `cuda:0`
  （gfx1200，GPU 锁协议 env 与既有活动一致）；backend
  `make_fx_aten`；per-item 超时 3600s。
- 三级 gate 哈希链（static→canary→full）由生产代码自行绑定与校验。

## 2. Gate（预固定）

1. **C1 static**：CLI exit 0，static gate 生成且 stage/denominator 与
   全 163 scored 一致。
2. **C2 canary**：CLI exit 0，canary readiness 矩阵中其风险抽样分母
   全 ready（make_fx 就绪覆盖已被 L3-c 全量证明，预期无失败）。
3. **C3 full**：CLI exit 0，full readiness 矩阵 **163/163 ready**。
4. **C4 连续性**：三级 gate 的 parent 哈希链完整；任何一级失败即停，
   失败证据原样保留，不得改分母或放宽重跑。
5. **C5 登记**：HANDSOFF + roadmap L1 延迟项销账（全分母链于 make_fx
   分母闭合）；corpus 发布资格声明按链语义记录。

## 3. 证据

`data/outputs/solar-qualification-mfx-20261107/{static,canary,full}/`
（生产产物原样）。

## 4. 边界

- 只运行生产链，不改资格代码、分母选择或 gate 语义。
- 已知风险（预声明）：l2n52 家族验证在 GPU 通过（L3-c 实证）；若个别
  槽因瞬时环境失败，按 fail-closed 记录并以同一分母重试该槽
  （resume 语义），不做任何剔除。
