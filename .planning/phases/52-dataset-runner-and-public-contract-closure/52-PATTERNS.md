# Phase 52: Dataset Runner And Public Contract Closure - Pattern Map

**Mapped:** 2026-05-23
**Files analyzed:** 8 likely new/modified files
**Analogs found:** 8 / 8

## File Classification

| New/Modified File | Role | Data Flow | Closest Analog | Match Quality |
|-------------------|------|-----------|----------------|---------------|
| `scripts/run_dataset.py` | utility / runner | batch, file-I/O, request-response subprocess | `scripts/run_dataset.py` AMD score and timing evidence helpers | exact |
| `src/sol_execbench/core/scoring/amd_score.py` | service / model | transform, batch | `src/sol_execbench/core/scoring/amd_score.py` score guard/report helpers | exact |
| `src/sol_execbench/core/scoring/solar_derivation.py` | model / parser | transform, file-I/O sidecar | `src/sol_execbench/core/scoring/solar_derivation.py` sidecar dataclasses and strict parser | exact |
| `tests/sol_execbench/test_run_dataset_amd_score.py` | test | file-I/O, batch | same file's AMD score report and sidecar tests | exact |
| `tests/sol_execbench/test_public_contract_guardrails.py` | test | public contract guardrail | v1.10 SOLAR and derived artifact guardrails in same file | exact |
| `tests/sol_execbench/test_v1_9_validation_closure.py` | test | documentation / claim guardrail | v1.9 docs and claim guardrails | role-match |
| `tests/sol_execbench/test_solar_derivation_contract.py` | test | contract fixture validation | fixture scope and no-execution tests | role-match |
| `docs/analysis.md` and `docs/internal/solar_derivation_contract.md` | documentation | public/internal contract | existing AMD score and SOLAR sidecar sections | exact |

## Pattern Assignments

### `scripts/run_dataset.py` (utility / runner, batch + file-I/O)

**Analog:** `scripts/run_dataset.py`

**Import pattern** (lines 49-63):
```python
from sol_execbench.core.data.definition import Definition
from sol_execbench.core.data.trace import Trace
from sol_execbench.core.data.workload import Workload
from sol_execbench.core.scoring.amd_score import (
    AmdNativeScore,
    build_amd_native_suite_report,
    score_amd_native_trace_workload,
)
from sol_execbench.core.scoring.baseline_artifact import (
    ScoringBaselineArtifact,
    load_scoring_baseline_artifact,
)
```

**Opt-in report/sidecar CLI option pattern** (lines 620-642):
```python
ap.add_argument(
    "--amd-score-report",
    type=Path,
    default=None,
    help="Optional path for a derived AMD-native suite score JSON report.",
)
ap.add_argument(
    "--amd-sol-bound-dir",
    type=Path,
    default=None,
    help=(
        "Optional directory for derived AMD SOL bound v2 sidecars when "
        "--amd-score-report is enabled."
    ),
)
```

**Per-workload derived sidecar pattern** (lines 461-505):
```python
def build_amd_score_reports_for_problem(
    *,
    definition_payload: dict,
    workload_path: Path,
    traces_payload: list[dict],
    trace_ref: str,
    baseline_artifact: ScoringBaselineArtifact | None = None,
    sol_bound_artifact_dir: Path | None = None,
) -> list[AmdNativeScore]:
    definition = Definition(**definition_payload)
    workloads = {
        workload.uuid: workload
        for workload in (
            Workload(**json.loads(line))
            for line in workload_path.read_text().splitlines()
            if line.strip()
        )
    }
    traces = [Trace(**trace) for trace in traces_payload]
    ...
    if artifact is not None and sol_bound_artifact_dir is not None:
        sol_bound_artifact_dir.mkdir(parents=True, exist_ok=True)
        sidecar_path = (
            sol_bound_artifact_dir
            / f"{definition.name}.{trace.workload.uuid}.amd-sol-v2.json"
        )
        sidecar_path.write_text(json.dumps(artifact.to_dict(), indent=2))
        sol_bound_ref = str(sidecar_path)
```

**Suite report write pattern** (lines 865-879):
```python
if args.amd_score_report is not None:
    report_path = args.amd_score_report.resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report = build_amd_native_suite_report(
        amd_scores,
        baseline_summary={
            "problems": len(summaries),
            "scores": len(amd_scores),
            "baseline_entries": (
                len(scoring_baseline.entries) if scoring_baseline else 0
            ),
        },
    )
    report_path.write_text(json.dumps(report.to_dict(), indent=2))
```

**Apply to Phase 52:** Add any SOLAR derivation runner integration as an optional sidecar/report path next to `--amd-score-report`, not as primary `sol-execbench` behavior. Keep canonical `traces.json` writes unchanged. Use workload UUID keyed maps, mirroring existing `workloads` lookup and `trace.workload.uuid` refs.

### `src/sol_execbench/core/scoring/amd_score.py` (service / model, transform + batch)

**Analog:** `src/sol_execbench/core/scoring/amd_score.py`

**Report model pattern** (lines 61-92, 97-152):
```python
@dataclass(frozen=True)
class AmdNativeScore:
    definition: str
    workload_uuid: str
    measured_latency_ms: float | None
    baseline_latency_ms: float | None
    sol_bound_ms: float | None
    score: float | None
    claim_level: str
    warnings: tuple[str, ...]
    baseline_source: str
    evidence_refs: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            ...
            "claim_level": self.claim_level,
            "warnings": list(self.warnings),
            "baseline_source": self.baseline_source,
            "supported": self.supported,
            "evidence_refs": dict(self.evidence_refs),
        }
```

```python
@dataclass(frozen=True)
class AmdNativeSuiteReport:
    scores: tuple[AmdNativeScore, ...]
    baseline_summary: dict[str, int] | None = None
    schema_version: str = AMD_SCORE_SCHEMA_VERSION
    derived: bool = True
    canonical_output: str = CANONICAL_BENCHMARK_OUTPUT

    @property
    def evidence_summary(self) -> dict[str, int]:
        summary = {
            "trace": 0,
            "timing": 0,
            "sol_bound": 0,
            "baseline": 0,
            "hardware_model": 0,
        }
```

**Score guard pattern** (lines 156-215, 397-421):
```python
def score_amd_native_workload(
    artifact: AmdSolBoundArtifact | AmdSolBoundV2Artifact,
    *,
    measured_latency_ms: float | None,
    baseline_latency_ms: float | None,
    ...
    solar_derivation: SolarScoreGuard | None = None,
) -> AmdNativeScore:
    sol_bound_ms = _artifact_sol_bound_ms(artifact)
    warnings = _warnings_for_artifact(artifact)
    solar_aggregate = _solar_aggregate_status(solar_derivation)
    if solar_aggregate is not None:
        warnings.extend(_warnings_for_solar_aggregate(solar_aggregate))
        warnings = _unique(warnings)
    ...
    return AmdNativeScore(
        ...
        claim_level=AMD_SCORE_CLAIM_LEVEL,
        warnings=tuple(warnings),
        baseline_source=baseline_source,
        evidence_refs=evidence_refs,
    )
```

```python
def _solar_aggregate_status(
    solar_derivation: SolarScoreGuard | None,
) -> SolarAggregateStatus | None:
    if solar_derivation is None:
        return None
    if isinstance(solar_derivation, SolarAggregateStatus):
        return solar_derivation
    payload = solar_derivation.to_dict()["aggregate_status"]
    return SolarAggregateStatus(...)
```

**Public evidence ref key pattern** (lines 436-448):
```python
def _evidence_refs(
    *,
    trace_ref: str | None = None,
    timing_evidence_ref: str | None,
    sol_bound_ref: str | None,
    baseline_ref: str | None = None,
    hardware_model_ref: str | None = None,
) -> dict[str, str]:
    refs: dict[str, str] = {}
    if trace_ref:
        refs["trace"] = trace_ref
    if timing_evidence_ref:
        refs["timing"] = timing_evidence_ref
    if sol_bound_ref:
        refs["sol_bound"] = sol_bound_ref
    if baseline_ref:
        refs["baseline"] = baseline_ref
    if hardware_model_ref:
        refs["hardware_model"] = hardware_model_ref
```

**Apply to Phase 52:** Preserve `AMD_SCORE_CLAIM_LEVEL == "amd-native-derived"`. Do not add public `evidence_refs` keys for internal SOLAR fields unless a derived-report-only metadata block is explicitly separate and guarded. Missing SOLAR sidecar data should mean no `solar_derivation` guard; explicit `aggregate_status.status == "unscored"` should suppress score and add unscored warnings.

### `src/sol_execbench/core/scoring/solar_derivation.py` (model / parser, transform + sidecar file-I/O)

**Analog:** `src/sol_execbench/core/scoring/solar_derivation.py`

**Sidecar shape pattern** (lines 307-386):
```python
@dataclass(frozen=True)
class SolarCoverageSummary:
    family_counts: dict[str, int]
    status_counts: dict[str, int]
    families: tuple[SolarFamilyCoverage, ...]
    missing_patterns: tuple[SolarCoveragePattern, ...]
    unsupported_patterns: tuple[SolarCoveragePattern, ...]
    degraded_node_ids: tuple[str, ...]
    unsupported_node_ids: tuple[str, ...]
    estimated_node_ids: tuple[str, ...]
    provenance: tuple[SolarCoverageSourceRef, ...]
```

```python
@dataclass(frozen=True)
class SolarDerivationEvidence:
    definition: str
    workload_uuid: str
    groups: tuple[SolarSemanticGroupEvidence, ...]
    tensors: tuple[SolarTensorEvidence, ...]
    warnings: tuple[str, ...]
    source_boundary: dict[str, bool]
    schema_version: str = SOLAR_DERIVATION_SCHEMA_VERSION
    derived: bool = True

    def to_dict(self) -> dict[str, object]:
        coverage_summary = _coverage_for_groups(self.groups)
        aggregate_status = _aggregate_status_for_groups(self.groups, self.warnings)
        return {
            "schema_version": self.schema_version,
            "derived": self.derived,
            "definition": self.definition,
            "workload_uuid": self.workload_uuid,
            "groups": [group.to_dict() for group in self.groups],
            "tensors": [tensor.to_dict() for tensor in self.tensors],
            "warnings": list(self.warnings),
            "source_boundary": dict(self.source_boundary),
            "coverage_summary": coverage_summary.to_dict(),
            "aggregate_status": aggregate_status.to_dict(),
        }
```

**Derivation boundary pattern** (lines 390-426, 2341-2346):
```python
def build_solar_derivation_evidence(
    definition: Definition,
    workload: Workload,
) -> SolarDerivationEvidence:
    graph = build_bound_graph(definition, workload)
    estimates = estimate_bound_work(graph)
    return derive_solar_derivation_evidence(definition, workload, graph, estimates)
```

```python
def _default_source_boundary() -> dict[str, bool]:
    return {
        "canonical_trace_jsonl": False,
        "public_schema": False,
        "candidate_solution_execution": False,
    }
```

**Strict parser pattern** (lines 566-630, 928-938, 1048-1058):
```python
def solar_derivation_from_dict(payload: dict[str, Any]) -> SolarDerivationEvidence:
    if not isinstance(payload, dict):
        raise ValueError("SOLAR derivation evidence payload must be an object")
    legacy_keys = {...}
    phase51_keys = legacy_keys | {"coverage_summary", "aggregate_status"}
    raw_keys = set(payload)
    if raw_keys == legacy_keys:
        has_phase51_fields = False
    elif raw_keys == phase51_keys:
        has_phase51_fields = True
    else:
        _require_exact_keys(payload, phase51_keys, source="SOLAR derivation evidence")
        has_phase51_fields = True
    ...
    if has_phase51_fields:
        provided_coverage = _coverage_summary_from_dict(...)
        provided_aggregate = _aggregate_status_from_dict(...)
        expected_coverage = _coverage_for_groups(groups)
        if provided_coverage.to_dict() != expected_coverage.to_dict():
            raise ValueError("coverage_summary does not match semantic groups")
```

**Apply to Phase 52:** If the runner writes SOLAR sidecars, use `build_solar_derivation_evidence(definition, workload).to_dict()` and name files deterministically by `definition.name` and `workload.uuid`. If the runner reads sidecars, parse through `solar_derivation_from_dict()` and key parsed evidence by `workload_uuid`; do not trust loose JSON shape.

### `tests/sol_execbench/test_run_dataset_amd_score.py` (test, batch + file-I/O)

**Analog:** `tests/sol_execbench/test_run_dataset_amd_score.py`

**Import runner helper by path pattern** (lines 13-23):
```python
REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_DATASET_PATH = REPO_ROOT / "scripts" / "run_dataset.py"
spec = importlib.util.spec_from_file_location("run_dataset", RUN_DATASET_PATH)
assert spec is not None
run_dataset = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(run_dataset)
build_amd_score_reports_for_problem = run_dataset.build_amd_score_reports_for_problem
```

**Derived report assertion pattern** (lines 26-95):
```python
scores = build_amd_score_reports_for_problem(
    definition_payload=definition,
    workload_path=workload_path,
    traces_payload=traces,
    trace_ref="L1/matmul_demo/traces.json",
)
report = build_amd_native_suite_report(scores).to_dict()

assert report["derived"] is True
assert report["canonical_output"] == "trace_jsonl"
assert report["scores"][0]["evidence_refs"]["trace"] == "L1/matmul_demo/traces.json"
assert report["scores"][0]["evidence_refs"]["baseline"] == (
    "trace.evaluation.performance.reference_latency_ms"
)
assert report["evidence_summary"]["sol_bound"] == 1
```

**Sidecar materialization assertion pattern** (lines 175-236):
```python
sidecar_dir = tmp_path / "sol-bounds"

scores = build_amd_score_reports_for_problem(
    definition_payload=definition,
    workload_path=workload_path,
    traces_payload=traces,
    trace_ref="L1/matmul_demo/traces.json",
    sol_bound_artifact_dir=sidecar_dir,
)

sidecar_path = sidecar_dir / "matmul_demo.matmul-workload.amd-sol-v2.json"
sidecar = json.loads(sidecar_path.read_text())
assert sidecar["schema_version"] == "sol_execbench.amd_sol_bound.v2"
assert scores[0].evidence_refs["sol_bound"] == str(sidecar_path)
```

**Apply to Phase 52:** Extend this file for dataset-runner SOLAR sidecar generation/consumption. Add tests for: sidecar file path, `coverage_summary`, `aggregate_status`, `source_boundary.candidate_solution_execution is False`, score guard propagation by workload UUID, missing sidecar compatibility, and explicit unscored sidecar behavior.

### `tests/sol_execbench/test_public_contract_guardrails.py` (test, public contract guardrails)

**Analog:** `tests/sol_execbench/test_public_contract_guardrails.py`

**Internal field denylist pattern** (lines 69-95):
```python
PHASE51_SCORE_INTERNAL_EVIDENCE_REFS = (
    "solar_derivation",
    "coverage_summary",
    "aggregate_status",
    "family_counts",
    "status_counts",
    ...
    "bound_evidence",
)
PHASE51_INTERNAL_PUBLIC_BOUNDARY_FIELDS = (
    *PHASE51_SCORE_INTERNAL_EVIDENCE_REFS,
    "missing_patterns",
    "unsupported_patterns",
    "provenance",
    "source_boundary",
    "candidate_solution_execution",
)
```

**Noncanonical schema guard pattern** (lines 234-291, 671+):
```python
for payload in (
    definition.model_dump(mode="json"),
    workload.model_dump(mode="json"),
    trace.model_dump(mode="json"),
):
    for field in forbidden:
        assert field not in payload
        assert field not in repr(payload)

canonical_trace_jsonl = json.dumps(trace.model_dump(mode="json"), sort_keys=True)
for field in (*PHASE50_INTERNAL_EVIDENCE_NAMES, *PHASE51_INTERNAL_PUBLIC_BOUNDARY_FIELDS):
    assert field not in canonical_trace_jsonl
```

**Primary CLI guard pattern** (lines 335-350):
```python
result = CliRunner().invoke(cli, ["--help"])
assert result.exit_code == 0
help_text = result.output

for option in (
    "--solar-derivation",
    "--derive-solar",
    "--solar-sidecar",
    ...
):
    assert option not in help_text
```

**Score evidence boundary pattern** (lines 528-591):
```python
unscored = score_amd_native_workload(..., solar_derivation=SolarAggregateStatus(...)).to_dict()
degraded = score_amd_native_workload(..., solar_derivation=SolarAggregateStatus(...)).to_dict()

assert unscored["claim_level"] == "amd-native-derived"
assert degraded["claim_level"] == "amd-native-derived"
for payload in (unscored, degraded):
    for field in PHASE51_SCORE_INTERNAL_EVIDENCE_REFS:
        assert field not in payload["evidence_refs"]
```

**Apply to Phase 52:** Extend exact-key public contract tests after runner/report closure. Guard `Definition`, `Workload`, `Trace`, primary CLI help, canonical trace JSON, and public `evidence_refs`. Prefer exact JSON key checks over broad substring bans so established sidecar-only fields such as `coverage_summary` remain valid in derived artifacts.

### `tests/sol_execbench/test_v1_9_validation_closure.py` (test, docs + claim guardrails)

**Analog:** `tests/sol_execbench/test_v1_9_validation_closure.py`

**Docs inventory pattern** (lines 13-27):
```python
def test_analysis_docs_explain_v2_sidecars_and_rdna4_scope():
    text = _text("docs/analysis.md")

    for expected in (
        "sol_execbench.amd_sol_bound.v2",
        "--amd-sol-bound-dir",
        "operator_work_estimates",
        "aggregate_bound",
        "coverage_summary",
        "hardware_validation_status",
        "model_validation_status",
        "RDNA 4 (`gfx1200`) is the only validation",
        "CDNA 3 / MI300X real-hardware validation and CDNA 4 validation",
    ):
        assert expected in text
```

**Forbidden overclaim pattern** (lines 30-48):
```python
for phrase in forbidden:
    assert phrase not in combined
assert "not claim NVIDIA B200 equivalence" in combined
assert "CDNA 3 / MI300X real-hardware validation" in combined
assert "CDNA 4 validation" in combined
```

**Apply to Phase 52:** Add v1.10 doc assertions here or in a new closure test: require local workflow text for SOLAR sidecars and AMD-native score warnings; forbid claims of paper parity, 124-model / 235-problem extraction, NVIDIA Blackwell/B200 equivalence, hosted leaderboard readiness, CDNA3/MI300X/CDNA4 validation, NVFP4/MXFP4 validation, and new real-hardware validation.

### `tests/sol_execbench/test_solar_derivation_contract.py` (test, contract fixture validation)

**Analog:** `tests/sol_execbench/test_solar_derivation_contract.py`

**No candidate execution pattern** (lines 203-210):
```python
def test_fixture_loader_does_not_execute_reference_text(tmp_path):
    fixture = _valid_fixture()
    marker = tmp_path / "executed"
    fixture["reference"] = f"raise SystemExit; open({str(marker)!r}, 'w').write('bad')"
    (tmp_path / "fixture.json").write_text(json.dumps(fixture))

    load_solar_derivation_fixtures(tmp_path)

    assert not marker.exists()
```

**Scope boundary pattern** (lines 262-272):
```python
for fixture in fixtures:
    boundary = fixture["scope_boundary"]
    case_id = fixture["case_id"]
    assert set(boundary) == REQUIRED_SCOPE_BOUNDARY, case_id
    assert boundary == {
        "paper_scale_dataset": False,
        "hosted_leaderboard_ready": False,
        "nvidia_blackwell_b200_equivalence": False,
        "real_hardware_validation": False,
    }, case_id
```

**Apply to Phase 52:** Mirror these when proving runner-side SOLAR derivation uses canonical definition/workload/static evidence only. Do not execute candidate solution code to derive evidence; assert sidecar `source_boundary` remains false for canonical/public/candidate mutation/execution boundaries.

### `docs/analysis.md` and `docs/internal/solar_derivation_contract.md` (documentation, public/internal contract)

**Analog:** `docs/analysis.md`, `docs/internal/solar_derivation_contract.md`

**Public derived report language pattern** (`docs/analysis.md` lines 179-199):
```markdown
AMD-native score reports are derived artifacts. They can reference trace timing,
ROCm timing evidence, baseline summaries, and AMD SOL bound artifacts, but they
do not add fields to canonical trace JSONL. These reports are AMD ROCm
interpretation artifacts: not NVIDIA B200, SOLAR, or leaderboard equivalence claims.

Dataset runs can optionally write a derived AMD-native suite score report:
...
The report is opt-in. It reads canonical trace output and derived AMD SOL bound
inputs, records trace, timing, SOL-bound, baseline, and hardware-model evidence
references for each workload score, and keeps the output separate from canonical
trace JSONL. Missing timing, baseline, or bound evidence is reported as an
unscored guarded state rather than an invented score.
```

**Sidecar semantics pattern** (`docs/analysis.md` lines 201-217, 251-264):
```markdown
To also materialize the derived AMD SOL bound artifact v2 sidecars used by the
score report, provide a sidecar directory:
...
The sidecars use schema version `sol_execbench.amd_sol_bound.v2`.
...
This is the ROCm port's AMD-local analog of the paper's graph/evidence/SOL-analyzer
artifact boundary; it is not an upstream NVIDIA B200 or full SOLAR reproduction claim.
```

**Internal claim boundary pattern** (`docs/internal/solar_derivation_contract.md` lines 14-25, 27-35, 263-266):
```markdown
This contract is not paper-scale dataset extraction, not hosted leaderboard
readiness, not NVIDIA Blackwell/B200 equivalence, and not new real-hardware
validation.

SOLAR derivation evidence belongs only in derived sidecars or explicitly opted
in reports. The sidecar contract is internal and must not mutate:

- canonical `definition.json`;
- canonical `workload.jsonl`;
- canonical trace JSONL;
- public solution schemas;
- primary `sol-execbench` CLI behavior or default output.
```

**Apply to Phase 52:** Update docs by extending these sections, not by creating a broad tutorial unless needed. Keep public docs concrete: command, output paths, `coverage_summary`, `aggregate_status`, AMD-native score warnings, and explicit claim limits.

## Shared Patterns

### Derived Artifacts Stay Opt-In

**Source:** `scripts/run_dataset.py` lines 620-642, 865-879; `docs/analysis.md` lines 179-199.

Apply to runner/report/doc work. New SOLAR sidecars should require an explicit runner option and write separate JSON artifacts. Canonical trace JSON stays unchanged.

### Public Evidence Refs Stay Narrow

**Source:** `src/sol_execbench/core/scoring/amd_score.py` lines 125-152 and 436-448; `tests/sol_execbench/test_public_contract_guardrails.py` lines 528-591.

Allowed public score evidence ref keys today are `trace`, `timing`, `sol_bound`, `baseline`, and `hardware_model`. Internal SOLAR keys belong in sidecars or explicitly scoped derived metadata, not in public `evidence_refs`.

### SOLAR Sidecar Shape Is Strict

**Source:** `src/sol_execbench/core/scoring/solar_derivation.py` lines 361-386, 566-630, 943-1058.

Use `schema_version`, `derived`, `definition`, `workload_uuid`, `groups`, `tensors`, `warnings`, `source_boundary`, `coverage_summary`, and `aggregate_status`. Parser validation recomputes coverage and aggregate status; tests should assert mismatches fail instead of accepting loose JSON.

### Claim Guardrails Are Positive And Negative

**Source:** `tests/sol_execbench/test_v1_9_validation_closure.py` lines 30-48; `tests/sol_execbench/test_public_contract_guardrails.py` lines 212-231; `docs/internal/solar_derivation_contract.md` lines 14-25.

Require positive AMD-local language such as paper-aligned automatic SOLAR derivation evidence for the ROCm port. Forbid phrasing that implies paper benchmark parity, paper-scale extraction, NVIDIA Blackwell/B200 equivalence, hosted leaderboard readiness, CDNA3/MI300X/CDNA4 validation, NVFP4/MXFP4 validation, or new real-hardware validation.

## Likely Files To Change

| File | Expected Change | Pattern To Copy |
|------|-----------------|-----------------|
| `scripts/run_dataset.py` | Add opt-in SOLAR derivation sidecar read/write options and workload UUID propagation into AMD score guards | `build_amd_score_reports_for_problem()` sidecar/ref handling |
| `src/sol_execbench/core/scoring/amd_score.py` | Possibly add derived-report-only metadata or evidence summary support without changing public claim level | `AmdNativeSuiteReport.to_dict()` and `_evidence_refs()` |
| `src/sol_execbench/core/scoring/solar_derivation.py` | Likely no core shape change; use builder/parser from runner | `build_solar_derivation_evidence()` and `solar_derivation_from_dict()` |
| `tests/sol_execbench/test_run_dataset_amd_score.py` | Add runner sidecar generation/consumption tests | Existing tmp_path sidecar assertions |
| `tests/sol_execbench/test_public_contract_guardrails.py` | Extend public contract tests for runner/report closure | Existing exact schema/key/CLI/evidence ref guards |
| `tests/sol_execbench/test_v1_9_validation_closure.py` | Add v1.10 docs and overclaim guardrails | Existing docs inventory and forbidden phrase tests |
| `tests/sol_execbench/test_solar_derivation_contract.py` | Extend no-execution/scope boundary tests if runner touches derivation inputs | Existing fixture no-execution and boundary tests |
| `docs/analysis.md` | Document local v1.10 sidecar/report workflow and warnings | AMD-native score interpretation section |
| `docs/internal/solar_derivation_contract.md` | Clarify Phase 52 dataset-level sidecar/report consumption boundary | Sidecar-only artifact rule and downstream phase consumption |

## Pitfalls

- Do not add SOLAR options to primary `sol-execbench` CLI; runner-only options are the established surface.
- Do not mutate canonical `definition.json`, `workload.jsonl`, public schemas, or canonical trace JSONL.
- Do not execute submitted candidate solution code for SOLAR derivation. Use canonical definition/workload inputs and static evidence.
- Do not put `solar_derivation`, `coverage_summary`, `aggregate_status`, `formula_evidence`, `byte_evidence`, or `bound_evidence` into public score `evidence_refs`.
- Distinguish missing sidecar data from explicit `aggregate_status.status == "unscored"`.
- Keep `claim_level` exactly `amd-native-derived`.
- Avoid broad substring guardrails that fail legitimate "not claimed" historical context. Prefer exact key checks and targeted forbidden positive claims.
- Keep tests deterministic and ROCm-free by constructing definitions, workloads, traces, and sidecar JSON under `tmp_path`.

## No Analog Found

No target file lacks a close codebase analog. New helper functions should be patterned from the existing runner/report/sidecar helpers above rather than introducing a new architecture.

## Metadata

**Analog search scope:** `scripts/`, `src/sol_execbench/core/scoring/`, `tests/sol_execbench/`, `docs/`, `docs/internal/`
**Files scanned:** 11 focused files plus phase context
**Pattern extraction date:** 2026-05-23
