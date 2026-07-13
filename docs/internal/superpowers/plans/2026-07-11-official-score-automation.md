# Official Score 自动化与统一口径 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 自动生成 `official_score_evidence.v1`，并以固定 suite 分母将所有被阻断 workload 按零分聚合，保留可审计的 authority 与 provenance 边界。

**Architecture:** 保持 `AmdNativeScore` 为诊断性派生输入；在 `official_score.py` 固化唯一聚合策略并由 `OfficialScoreSuiteEvidence` 进行零分记账。dataset runner 收集已经生成的 AMD-native score，验证 official 请求的 baseline/coverage 输入后写出单一 official sidecar；现有独立 `official-score` CLI 复用同一策略和 schema，不能再接受任意自由文本策略。

**Tech Stack:** Python 3.12、dataclasses、Click、argparse、JSON、Pytest。

## Global Constraints

- 唯一有效 policy 字面量：`fixed_suite_denominator_zero_for_blocked`。
- 每个 workload 的 `score: null` 和 `status: "blocked"` 仍表示没有 official authority；仅 suite 聚合将其贡献视为 `0.0`。
- official runner 输出必须提供 `sol_execbench.scoring_baseline.v1`，不得回退到 `reference_latency`。
- 不改变 AMD SOL/SOLAR bound、硬件校准或 `amd_native_score` 的 derived authority 边界。
- 只在 `--official-score-report` 显式请求时产生 official sidecar；原有 AMD-native-only 调用保持兼容。
- 所有新增 JSON 保持 `schema_version="sol_execbench.official_score_evidence.v1"`，并保留 workload 的 trace、timing、baseline、SOL bound、hardware-model 与 AMD-score 引用。

---

## File Structure

- Modify: `src/sol_execbench/core/scoring/official_score.py` — 唯一 policy 校验、固定分母零分 suite 聚合与稳定序列化字段。
- Modify: `tests/sol_execbench/core/evidence/test_official_score_evidence.py` — gate 与聚合口径的单元测试。
- Modify: `src/sol_execbench/core/dataset/runner_scoring.py` — 从完整 `AmdNativeScore` 集合构建并原子写出 official sidecar 的可复用桥接函数。
- Modify: `scripts/run_dataset.py` — official-output 参数校验、baseline 加载、运行后 sidecar 生成与 execution-closure provenance 引用。
- Modify: `tests/sol_execbench/core/dataset/test_run_dataset_amd_score.py` — dataset derived/all phase 的 official 端到端测试。
- Modify: `src/sol_execbench/cli/commands/official_score.py` — 现有离线 emitter 仅接受固定 policy，并输出与 runner 相同的聚合结构。
- Modify: `tests/sol_execbench/cli/commands/test_official_score_cli.py` — 固定 policy、阻断零分和无效 policy CLI 测试。
- Modify: `tests/sol_execbench/fixtures/confirmed_evidence/confirmed-pass.bundle.json`, `missing-score.bundle.json`, `missing-baseline.bundle.json`, `placeholder-baseline.bundle.json`, `profiler-partial.bundle.json`, `diagnostic-only-sidecar.bundle.json` — 将 fixture 中过时的聚合策略和 suite 计数/分数更新为统一 schema。
- Modify: `docs/user/EVALUATOR-CONTRACT.md` — official 输出、固定分母和自动化状态。
- Modify: `docs/internal/RDNA4-DENOMINATOR-POLICY.md` — 官方 score 与现有 RDNA4 validation denominator 的关系和零分规则。
- Modify: `README.md` — dataset runner 与 standalone official-score 的安全用法。

### Task 1: 固化 official 聚合策略和零分分母

**Files:**
- Modify: `src/sol_execbench/core/scoring/official_score.py:45-195`
- Test: `tests/sol_execbench/core/evidence/test_official_score_evidence.py:75-210`

**Interfaces:**
- Consumes: `Iterable[AmdNativeScore]` 与每项现有 `score_authority`/`blocker_reason_codes`。
- Produces: `OFFICIAL_AGGREGATION_POLICY: str`，`validate_official_aggregation_policy(policy: str | None) -> str | None`，以及带 `total_workload_count`、`blocked_count`、`zero_scored_count` 的 `OfficialScoreSuiteEvidence.to_dict()`。

- [ ] **Step 1: 写出失败的固定分母聚合测试**

在 `test_official_score_evidence.py` 中定义 policy 常量并将已有成功案例改为使用它；新增混合 suite 用例：一个得分 `0.75`，一个 `reference_latency` blocker。断言 `score == mean_score == 0.375`、总数 2、`scored_count == 1`、`blocked_count == zero_scored_count == unscored_count == 1`，以及 suite authority 为 false。

```python
def test_suite_uses_fixed_denominator_and_zero_for_blocked_scores():
    report = build_official_score_suite_evidence(
        (_amd_score(score=0.75), _amd_score(score=0.9, baseline_source="reference_latency")),
        aggregation_policy=OFFICIAL_AGGREGATION_POLICY,
    )

    payload = report.to_dict()
    assert payload["score"] == 0.375
    assert payload["total_workload_count"] == 2
    assert payload["scored_count"] == 1
    assert payload["blocked_count"] == 1
    assert payload["zero_scored_count"] == 1
    assert payload["score_authority"] is False
```

- [ ] **Step 2: 运行测试确认现状失败**

Run: `uv run pytest tests/sol_execbench/core/evidence/test_official_score_evidence.py -q`

Expected: FAIL，因为 policy 常量/新字段不存在，且当前均值会错误地为 `0.75`。

- [ ] **Step 3: 实现唯一 policy 和 suite 贡献值**

在 `official_score.py` 中添加 policy 常量和严格校验；删除自由文本“只要非空”的语义。为 suite 增加 `total_workload_count`、`blocked_count`、`zero_scored_count` 与 `score_contributions`（或等价私有 helper）。`mean_score` 只对 `score_authority` 项取 score、其余项取 `0.0`；空 suite 返回 `None`。序列化新计数且保留 `unscored_count` 作为 `blocked_count` 的兼容别名。

```python
OFFICIAL_AGGREGATION_POLICY = "fixed_suite_denominator_zero_for_blocked"

def validate_official_aggregation_policy(policy: str | None) -> str | None:
    normalized = policy.strip() if policy else ""
    return normalized if normalized == OFFICIAL_AGGREGATION_POLICY else None

@property
def mean_score(self) -> float | None:
    if not self.scores:
        return None
    return statistics.mean(
        score.score if score.score_authority and score.score is not None else 0.0
        for score in self.scores
    )
```

让 `_official_score_blockers()` 对 `None` policy 保持 `missing_aggregation_policy`；在 `official_score_from_amd_native_score()` 和 suite builder 中改用新 validator。

- [ ] **Step 4: 补齐空 suite、全阻断和无效 policy 测试**

增加三个明确用例：全阻断 suite 得 `0.0` 且无 authority；空 suite 得 `null`、总数 0；任意旧 policy（例如 `"mean of per-workload SOL scores"`）产生 `missing_aggregation_policy`，不能绕过 gate。

```python
def test_all_blocked_suite_is_zero_not_null():
    report = build_official_score_suite_evidence(
        (_amd_score(baseline_source="reference_latency"),),
        aggregation_policy=OFFICIAL_AGGREGATION_POLICY,
    )
    assert report.to_dict()["score"] == 0.0
```

- [ ] **Step 5: 运行评分单元测试**

Run: `uv run pytest tests/sol_execbench/core/evidence/test_official_score_evidence.py -q`

Expected: PASS。

- [ ] **Step 6: 提交 Task 1**

```bash
git add src/sol_execbench/core/scoring/official_score.py tests/sol_execbench/core/evidence/test_official_score_evidence.py
git commit -s -m "feat: fix official score denominator policy"
```

### Task 2: 提供 runner 可复用的 official sidecar 写入器

**Files:**
- Modify: `src/sol_execbench/core/dataset/runner_scoring.py:1-87`
- Test: `tests/sol_execbench/core/dataset/test_run_dataset_amd_score.py:1-180`

**Interfaces:**
- Consumes: `list[AmdNativeScore]`, `aggregation_policy: str`, `report_path: Path`，可选 `BaselineCoverageReport`。
- Produces: `write_official_score_report(report_path: Path, amd_scores: list[AmdNativeScore], *, aggregation_policy: str, coverage_report: BaselineCoverageReport | None = None, source_score_ref: str | None = None) -> None`，写入 canonical JSON。

- [ ] **Step 1: 写出写入器的失败测试**

在 dataset test 中使用两个 `AmdNativeScore` fixture 调用新函数到 `tmp_path / "official-score.json"`。验证文件存在、schema、唯一 policy、`score=0.375`、计数和每项 `input_refs["amd_native_score"]`。

```python
write_official_score_report(
    report_path,
    [_official_ready_score(0.75), _official_blocked_score()],
    aggregation_policy=OFFICIAL_AGGREGATION_POLICY,
    source_score_ref="reports/amd-score.json",
)
payload = json.loads(report_path.read_text())
assert payload["score"] == 0.375
assert payload["scores"][0]["input_refs"]["amd_native_score"] == "reports/amd-score.json"
```

- [ ] **Step 2: 运行测试确认失败**

Run: `uv run pytest tests/sol_execbench/core/dataset/test_run_dataset_amd_score.py -q`

Expected: FAIL，提示 `write_official_score_report` 尚未导出或定义。

- [ ] **Step 3: 实现 bridge 和安全写入**

在 `runner_scoring.py` 导入 `build_official_score_suite_evidence` 和 policy validator。拒绝无效 policy，构建每个 UUID 到同一 AMD score report ref 的映射，调用 suite builder，以 `json.dumps(..., indent=2, sort_keys=True) + "\n"` 写入。先创建父目录；写入临时同目录文件后以 `Path.replace()` 原子替换，避免留下半份 official artifact。

```python
suite = build_official_score_suite_evidence(
    amd_scores,
    aggregation_policy=aggregation_policy,
    source_score_refs_by_workload_uuid={
        score.workload_uuid: source_score_ref for score in amd_scores
    } if source_score_ref else None,
    coverage_report=coverage_report,
)
```

- [ ] **Step 4: 运行 bridge 测试**

Run: `uv run pytest tests/sol_execbench/core/dataset/test_run_dataset_amd_score.py -q`

Expected: PASS。

- [ ] **Step 5: 提交 Task 2**

```bash
git add src/sol_execbench/core/dataset/runner_scoring.py tests/sol_execbench/core/dataset/test_run_dataset_amd_score.py
git commit -s -m "feat: add official score sidecar writer"
```

### Task 3: 将 official 输出接入 dataset runner

**Files:**
- Modify: `scripts/run_dataset.py:82-120, 1724-2000, 2251-3228`
- Modify: `src/sol_execbench/core/dataset/runner_scoring.py:14-19`
- Test: `tests/sol_execbench/core/dataset/test_run_dataset_amd_score.py:603-950`

**Interfaces:**
- Consumes: 新 CLI 参数 `--official-score-report PATH`、`--official-aggregation-policy fixed_suite_denominator_zero_for_blocked`、现有 `--amd-score-report PATH`、`--scoring-baseline PATH`。
- Produces: 完整 dataset run 后的 official sidecar；execution closure provenance 下的 `derived_evidence.official_score_report` 和 policy。

- [ ] **Step 1: 写出 runner 参数契约失败测试**

新增三个 `run_dataset.main()` 测试：

```python
# (a) official report + valid scoring baseline + derived trace writes sidecar
# (b) --official-score-report without --scoring-baseline exits through ap.error
# (c) --official-score-report without --amd-score-report exits through ap.error
```

成功测试使用已有 matmul fixture、一个包含对应 `(definition, workload_uuid)` 的 scoring baseline，并传入：

```python
"--phase", "derived",
"--amd-score-report", str(amd_report_path),
"--official-score-report", str(official_report_path),
"--official-aggregation-policy", OFFICIAL_AGGREGATION_POLICY,
"--scoring-baseline", str(baseline_path),
```

断言 official payload 引用 AMD report、固定分母字段存在、任何 provisional bound 仍通过 blocker 产生 `score=0.0` 而非 authority。

- [ ] **Step 2: 运行测试确认失败**

Run: `uv run pytest tests/sol_execbench/core/dataset/test_run_dataset_amd_score.py -q`

Expected: FAIL，argparse 尚不认识 official 参数。

- [ ] **Step 3: 增加参数和启动期校验**

在 `scripts/run_dataset.py` 添加路径和 policy 参数。只要提供 official output，就要求 `--phase` 为 `all` 或 `derived`、同时有 AMD score report 与 scoring baseline，且 policy 等于唯一常量；否则使用 `ap.error()`。保留无 official 参数的旧行为。

```python
ap.add_argument("--official-score-report", type=Path, default=None,
                help="Write official_score_evidence.v1 after derived scoring.")
ap.add_argument("--official-aggregation-policy", default=None,
                help=f"Required policy: {OFFICIAL_AGGREGATION_POLICY}.")

if args.official_score_report is not None and args.scoring_baseline is None:
    ap.error("--official-score-report requires --scoring-baseline")
```

- [ ] **Step 4: 在所有 derived 完成点写出 official sidecar**

在 serial、pipeline post-trace 和 `--phase derived` 的共同终点（现有最后一次 `write_amd_score_report()` 后）调用 Task 2 writer。以输出目录相对路径作为 source score ref；把 official report 和 policy加入 `provenance["derived_evidence"]`，从而使 execution closure 可审计。不要在每个 problem 内写 suite sidecar，避免 partial suite 覆盖最终文件。

```python
if run_derived_phase and args.official_score_report is not None:
    write_official_score_report(
        args.official_score_report.resolve(), amd_scores,
        aggregation_policy=args.official_aggregation_policy,
        source_score_ref=_relative_ref(args.amd_score_report.resolve(), output_dir),
    )
```

- [ ] **Step 5: 运行 focused dataset 测试**

Run: `uv run pytest tests/sol_execbench/core/dataset/test_run_dataset_amd_score.py -q`

Expected: PASS。

- [ ] **Step 6: 提交 Task 3**

```bash
git add scripts/run_dataset.py src/sol_execbench/core/dataset/runner_scoring.py tests/sol_execbench/core/dataset/test_run_dataset_amd_score.py
git commit -s -m "feat: automate official score dataset sidecars"
```

### Task 4: 将独立 official-score CLI 迁移到统一 policy

**Files:**
- Modify: `src/sol_execbench/cli/commands/official_score.py:1-123`
- Modify: `tests/sol_execbench/cli/commands/test_official_score_cli.py:105-245`
- Modify: `tests/sol_execbench/fixtures/confirmed_evidence/confirmed-pass.bundle.json`, `missing-score.bundle.json`, `missing-baseline.bundle.json`, `placeholder-baseline.bundle.json`, `profiler-partial.bundle.json`, `diagnostic-only-sidecar.bundle.json`
- Modify: `tests/sol_execbench/core/evidence/test_confirmed_evidence_fixtures.py:1-100`

**Interfaces:**
- Consumes: `--aggregation-policy fixed_suite_denominator_zero_for_blocked`、AMD score JSON 和 measured registry。
- Produces: 与 runner 字段完全一致的 `official_score_evidence.v1` JSON；既有 coverage blocker 保留。

- [ ] **Step 1: 写出 CLI policy 与零分失败测试**

把 `test_official_score_cli.py` 的所有合法调用替换为常量；新增混合 score report 测试，断言一个 placeholder baseline 使 suite score 从 `0.75` 变为 `0.375`。新增一个传旧 policy 的 invocation，断言非零 exit 且帮助文本列出唯一 policy。

```python
result = CliRunner().invoke(cli, [
    "score", "official", "--amd-native-score", str(report_path),
    "--measured-registry", str(registry_path),
    "--aggregation-policy", "mean of per-workload SOL scores",
])
assert result.exit_code != 0
assert OFFICIAL_AGGREGATION_POLICY in result.output
```

- [ ] **Step 2: 运行 CLI 测试确认失败**

Run: `uv run pytest tests/sol_execbench/cli/commands/test_official_score_cli.py -q`

Expected: FAIL，因为当前 CLI 接受任意非空 policy 且报告仍排除 blocked 项。

- [ ] **Step 3: 实现严格 Click policy 校验与 fixture 更新**

在 CLI 中以 `click.Choice([OFFICIAL_AGGREGATION_POLICY], case_sensitive=True)` 定义 option，默认不设值、仍要求用户显式声明。删除帮助中旧的自由文本例子。更新 confirmed-evidence fixtures：policy 字段改为唯一值；混合/blocked suite 的 score、计数和 authority 与 Task 1 的序列化保持一致。

- [ ] **Step 4: 运行 CLI 和 fixture 测试**

Run: `uv run pytest tests/sol_execbench/cli/commands/test_official_score_cli.py tests/sol_execbench/core/evidence/test_confirmed_evidence_fixtures.py -q`

Expected: PASS。

- [ ] **Step 5: 提交 Task 4**

```bash
git add src/sol_execbench/cli/commands/official_score.py tests/sol_execbench/cli/commands/test_official_score_cli.py tests/sol_execbench/fixtures/confirmed_evidence tests/sol_execbench/core/evidence/test_confirmed_evidence_fixtures.py
git commit -s -m "feat: enforce unified official score policy"
```

### Task 5: 更新 authority 与操作文档

**Files:**
- Modify: `docs/user/EVALUATOR-CONTRACT.md:76-126`
- Modify: `docs/internal/RDNA4-DENOMINATOR-POLICY.md:1-96`
- Modify: `README.md:80-150`
- Test: `tests/sol_execbench/test_contract.py:287-390`

**Interfaces:**
- Consumes: 最终 JSON schema、runner 参数和 policy 常量。
- Produces: 公开的 policy 定义、CLI 示例与“不升级 provisional/degraded bound authority”的明确边界。

- [ ] **Step 1: 写出 contract 文档断言**

在 `test_contract.py` 增加断言，确保 contract/相关文档中同时出现 `fixed_suite_denominator_zero_for_blocked`、`official_score_evidence.v1` 与 `0` 分阻断规则，且不再宣称 official gate “not yet auto-emitted”。

```python
assert "fixed_suite_denominator_zero_for_blocked" in Path("docs/user/EVALUATOR-CONTRACT.md").read_text()
assert "blocked workloads contribute 0" in Path("docs/user/EVALUATOR-CONTRACT.md").read_text()
```

- [ ] **Step 2: 运行测试确认失败**

Run: `uv run pytest tests/sol_execbench/test_contract.py -q`

Expected: FAIL，因为文档仍记录 runner 没有自动写出 official artifact。

- [ ] **Step 3: 更新三份文档**

在 evaluator contract 中说明 fixed denominator、workload null/blocked 与 suite zero contribution 的区别，并将 integration status 改为 dataset runner 可自动写出。RDNA4 policy 增补：其 235 计数不等于自动拥有 official authority，只有请求 official output 且通过 gate 的完整 benchmark 才能宣称 authority；blocked 项仍在 official score 分母中计 0。README 增加一条完整命令：

```bash
uv run scripts/run_dataset.py data/SOL-ExecBench/benchmark --phase derived \
  --amd-score-report out/amd-score.json \
  --scoring-baseline baselines/gfx1200.json \
  --official-score-report out/official-score.json \
  --official-aggregation-policy fixed_suite_denominator_zero_for_blocked
```

- [ ] **Step 4: 运行 contract 测试**

Run: `uv run pytest tests/sol_execbench/test_contract.py -q`

Expected: PASS。

- [ ] **Step 5: 提交 Task 5**

```bash
git add docs/user/EVALUATOR-CONTRACT.md docs/internal/RDNA4-DENOMINATOR-POLICY.md README.md tests/sol_execbench/test_contract.py
git commit -s -m "docs: document official score automation"
```

### Task 6: 全量回归与工件接口验证

**Files:**
- Verify only: 本计划涉及的源代码、fixture 和文档。

**Interfaces:**
- Consumes: Tasks 1-5 的实现。
- Produces: 经测试与静态检查确认的 official automation 变更集。

- [ ] **Step 1: 运行所有直接受影响的测试**

Run: `uv run pytest tests/sol_execbench/core/evidence/test_official_score_evidence.py tests/sol_execbench/cli/commands/test_official_score_cli.py tests/sol_execbench/core/dataset/test_run_dataset_amd_score.py tests/sol_execbench/core/evidence/test_confirmed_evidence_fixtures.py tests/sol_execbench/test_contract.py -q`

Expected: PASS。

- [ ] **Step 2: 运行 lint 与格式检查**

Run: `uv run --with ruff ruff check . && uv run --with ruff ruff format --check .`

Expected: 两个命令均以 exit code 0 结束。

- [ ] **Step 3: 手工验证 emitted sidecar 的 authority 边界**

使用 Task 3 fixture 的 official JSON，确认：`schema_version` 正确、`aggregation_policy` 为唯一值、`score_authority` 在任一 blocked 项时为 false、`score` 仍是完整分母均值、每项保留 blocker 和 input refs。确保没有文档或输出将默认 gfx1200 的 degraded/provisional bound 标成 leaderboard authority。
