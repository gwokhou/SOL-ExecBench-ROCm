# Phase 52: Dataset Runner And Public Contract Closure - Research

**Researched:** 2026-05-23  
**Domain:** Dataset runner derived reports, SOLAR sidecars, public contract and claim guardrails  
**Confidence:** HIGH

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

### Runner Surface
- Use the existing dataset-runner derived artifact pattern instead of adding
  primary `sol-execbench` CLI behavior.
- If runner integration is needed, keep it opt-in and sidecar/report oriented:
  derived SOLAR sidecars and AMD-native score reports may be written as
  separate artifacts, while canonical trace JSONL remains unchanged.
- Do not execute submitted candidate solution code to derive SOLAR evidence.
  Derivation must continue to use canonical problem/reference/workload inputs
  and existing static evidence paths.
- Dataset-runner behavior should remain compatible when no SOLAR derivation
  sidecar is requested or available.

### Derived Report Evidence References
- Derived reports must keep `claim_level` at `amd-native-derived`.
- Evidence references should make formula evidence, hardware model evidence,
  coverage evidence, and score eligibility auditable in derived artifacts.
- Public score `evidence_refs` must preserve existing public keys unless a
  derived report-specific internal field is explicitly scoped and guarded.
- The runner and report should distinguish missing sidecar data from explicit
  `unscored` SOLAR evidence.

### Documentation Surface
- Documentation should explain how to consume v1.10 SOLAR sidecars and
  AMD-native score guard outputs from local derived reports.
- Documentation must state that v1.10 sidecars are AMD ROCm derived evidence,
  not paper-scale dataset extraction, not upstream SOLAR parity, not NVIDIA
  B200/Blackwell equivalence, not hosted leaderboard readiness, and not new
  real-hardware validation.
- Prefer updating existing analysis/internal docs over adding a broad new
  public tutorial unless the planner finds a small dedicated doc clearer.

### Public Contract Guardrails
- Preserve canonical `Definition`, `Workload`, `Trace`, primary CLI help,
  canonical trace JSONL, existing public benchmark semantics, and dependency
  set.
- TEST-04 should prove public contracts remain unchanged even after runner and
  derived report closure.
- Guardrails should inspect exact JSON keys where possible to avoid false
  positives from established v2 fields such as `coverage_summary`.

### Claim Guardrails
- TEST-05 should prevent v1.10 artifacts and docs from implying:
  paper benchmark parity, paper-scale 124-model / 235-problem extraction,
  NVIDIA Blackwell or B200 equivalence, hosted leaderboard readiness, CDNA 3 /
  MI300X / CDNA 4 validation, NVFP4/MXFP4 validation, or new real-hardware
  validation.
- Claim guardrails should preserve positive AMD-local language: v1.10 provides
  paper-aligned automatic SOLAR derivation evidence for the ROCm port.
- Avoid banning historical mentions of B200 or original-paper context when
  those mentions are explicitly framed as not claimed or out of scope.

### the agent's Discretion
- Exact option names, helper names, report field names, and doc placement are
  at the agent's discretion if they remain deterministic, local, and guarded.
- The planner may split work into runner/report integration, documentation,
  and public/claim guardrail closure.

### Deferred Ideas (OUT OF SCOPE)
- Original-paper 124-model / 235-problem extraction and curation remains
  future work.
- MI300X, CDNA 3, CDNA 4, NVFP4, and MXFP4 real-hardware validation remain
  future work.
- Hosted leaderboard or submission service remains future work.
- NVIDIA Blackwell/B200 comparison methodology, if ever desired, must be a
  separate non-ROCm claim analysis effort.
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| REPORT-04 | Derived reports preserve AMD-native-derived claim boundaries and include evidence references for formulas, hardware models, coverage, and score eligibility. | Existing AMD-native reports already expose `claim_level`, `evidence_refs`, and `evidence_summary`; Phase 52 should add derived-report-only SOLAR refs without changing canonical trace JSONL or primary CLI. [VERIFIED: repo `src/sol_execbench/core/scoring/amd_score.py:28-153`, `scripts/run_dataset.py:461-525`] |
| TEST-04 | Public contract guardrails prove canonical schemas, trace JSONL, primary CLI behavior, and existing public benchmark semantics remain unchanged. | Existing guardrail tests inspect Pydantic dumps, CLI help, exact JSON keys, noncanonical derived fields, and derived artifact boundaries. [VERIFIED: repo `tests/sol_execbench/test_public_contract_guardrails.py`] |
| TEST-05 | Claim guardrails prevent v1.10 artifacts from implying paper benchmark parity, NVIDIA Blackwell or B200 equivalence, hosted leaderboard readiness, or new real-hardware validation. | Existing docs and tests already encode no-claim phrases and fixture scope-boundary booleans; Phase 52 should extend the same pattern to runner outputs and updated docs. [VERIFIED: repo `docs/internal/solar_derivation_contract.md:7-41`, `tests/sol_execbench/test_solar_derivation_contract.py`] |
</phase_requirements>

## Project Constraints (from AGENTS.md)

- Python package code lives under `src/sol_execbench/`; tests live under `tests/`, with package tests under `tests/sol_execbench/`. [VERIFIED: repo `AGENTS.md`]
- Use Python 3.12+ and Ruff style; keep focused changes consistent with nearby modules and avoid broad refactors. [VERIFIED: repo `AGENTS.md`, `pyproject.toml`]
- Pytest is the test framework; use existing markers for ROCm/GPU-sensitive tests and prefer small unit tests for schema and driver logic. [VERIFIED: repo `AGENTS.md`, `pyproject.toml`]
- Do not commit credentials, proprietary kernels, downloaded datasets, local cache, build output, or benchmark output. [VERIFIED: repo `AGENTS.md`]
- Preserve SOL ExecBench benchmark semantics and public schemas unless a ROCm-specific change is unavoidable. [VERIFIED: repo `AGENTS.md`, `.planning/PROJECT.md`]
- Start source-code changes through a GSD workflow; this research phase edits only `.planning/` artifacts per user instruction. [VERIFIED: repo `AGENTS.md`]
- Commit messages use the `#<Issue Number> - <Commit Title>` format and DCO sign-off. [VERIFIED: repo `AGENTS.md`]

## Summary

Phase 52 should close v1.10 by extending the existing dataset-runner derived artifact path, not the primary `sol-execbench` CLI. [VERIFIED: repo `scripts/run_dataset.py:619-643`, `docs/internal/solar_derivation_contract.md:27-41`] The current runner already writes canonical per-problem `traces.json`, optional derived AMD-native suite score reports, and optional AMD SOL bound v2 sidecars. [VERIFIED: repo `scripts/run_dataset.py:809-823`, `scripts/run_dataset.py:865-879`] The missing integration is a runner/report surface for SOLAR derivation sidecars and their aggregate guard data. [VERIFIED: repo `scripts/run_dataset.py:461-525`, `src/sol_execbench/core/scoring/amd_score.py:233-305`]

The AMD-native score layer already supports SOLAR guard inputs through `solar_derivation` / `SolarAggregateStatus`, keeps `claim_level` fixed to `amd-native-derived`, suppresses numeric scores for explicit SOLAR `unscored` status, and preserves degraded scores with warnings. [VERIFIED: repo `src/sol_execbench/core/scoring/amd_score.py:28-57`, `src/sol_execbench/core/scoring/amd_score.py:156-216`, `tests/sol_execbench/test_amd_native_score.py`] The safest implementation path is to have the dataset runner optionally build or read SOLAR derivation sidecars by workload UUID, pass parsed aggregate evidence into `score_amd_native_trace_workload()`, and record auditable derived-report-only references without adding canonical trace fields. [VERIFIED: repo `src/sol_execbench/core/scoring/amd_score.py:233-305`, `src/sol_execbench/core/scoring/solar_derivation.py:390-427`]

Public contract risk is high enough to require exact-key tests rather than broad substring bans. [VERIFIED: repo `tests/sol_execbench/test_public_contract_guardrails.py`] Claim guardrails should permit historical/contextual mentions of B200, Blackwell, SOLAR, paper scale, leaderboard, MI300X, CDNA, NVFP4, and MXFP4 only when framed as deferred, not claimed, or out of scope. [VERIFIED: repo `.planning/phases/52-dataset-runner-and-public-contract-closure/52-CONTEXT.md`, `docs/internal/solar_derivation_contract.md:14-25`]

**Primary recommendation:** Add opt-in dataset-runner SOLAR sidecar/report integration using existing `build_solar_derivation_evidence()`, `solar_derivation_from_dict()`, and `score_amd_native_trace_workload(..., solar_derivation=...)`; keep all new refs in derived artifacts and guard them with exact JSON-key contract tests. [VERIFIED: repo `src/sol_execbench/core/scoring/solar_derivation.py:390-397`, `src/sol_execbench/core/scoring/amd_score.py:233-305`]

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|--------------|----------------|-----------|
| Dataset problem discovery and per-problem execution | Script / Runner | CLI subprocess | `scripts/run_dataset.py` discovers problems, constructs reference/custom solutions, calls `sol-execbench`, writes traces, and summarizes results. [VERIFIED: repo `scripts/run_dataset.py:60-83`, `scripts/run_dataset.py:781-863`] |
| Canonical benchmark execution and trace emission | CLI / Driver | Data schemas | The runner invokes the installed `sol-execbench` command with `--json`; Phase 52 must not change primary CLI behavior or trace JSONL semantics. [VERIFIED: repo `scripts/run_dataset.py:181-204`, `tests/sol_execbench/test_public_contract_guardrails.py`] |
| AMD-native score calculation | Core scoring library | Dataset runner | `score_amd_native_trace_workload()` owns score computation, claim level, warnings, and evidence refs; the runner should only assemble inputs and write reports. [VERIFIED: repo `src/sol_execbench/core/scoring/amd_score.py:233-305`] |
| SOLAR derivation evidence generation/parsing | Core scoring library | Dataset runner | `build_solar_derivation_evidence()` derives sidecar evidence from `Definition` and `Workload`, while `solar_derivation_from_dict()` validates sidecar payloads. [VERIFIED: repo `src/sol_execbench/core/scoring/solar_derivation.py:390-397`, `tests/sol_execbench/test_solar_derivation_evidence.py`] |
| Public contract protection | Tests | Docs | Existing static tests guard schema dumps, CLI help, trace JSONL, and derived artifact boundaries. [VERIFIED: repo `tests/sol_execbench/test_public_contract_guardrails.py`] |
| Claim boundary protection | Tests + docs | Runner/report output | Existing docs define no-claim phrases and out-of-scope boundaries; Phase 52 should extend those tests to runner output and new docs. [VERIFIED: repo `docs/internal/solar_derivation_contract.md:7-41`, `docs/analysis.md:162-281`] |

## Standard Stack

### Core

| Library / Module | Version | Purpose | Why Standard |
|------------------|---------|---------|--------------|
| Python | 3.12.13 available; project requires `>=3.12,<3.14` | Runtime for runner, scoring, docs tests, and guardrails | Repository baseline and active environment match. [VERIFIED: command `python --version`, `pyproject.toml`] |
| pytest | 9.0.3 available; `pytest>=9.0.2` configured | Deterministic contract and scoring tests | Existing test suite and markers are pytest-based. [VERIFIED: command `pytest --version`, `pyproject.toml`] |
| Click | `click>=8.0` configured | Primary CLI help and options | Public CLI guardrails inspect Click help through `CliRunner`. [VERIFIED: repo `pyproject.toml`, `tests/sol_execbench/test_public_contract_guardrails.py`] |
| Pydantic v2 | `pydantic>=2.12.5` configured | Public schema models for Definition, Workload, Solution, Trace | Guardrails use `model_dump(mode="json")` on canonical models. [VERIFIED: repo `pyproject.toml`, `tests/sol_execbench/test_public_contract_guardrails.py`] |
| Local scoring modules | In-repo | AMD-native scores, AMD SOL v2 sidecars, SOLAR derivation sidecars | Existing modules already implement the target contracts and should be extended instead of duplicated. [VERIFIED: repo `src/sol_execbench/core/scoring/amd_score.py`, `src/sol_execbench/core/scoring/amd_sol_v2.py`, `src/sol_execbench/core/scoring/solar_derivation.py`] |

### Supporting

| Library / Tool | Version | Purpose | When to Use |
|----------------|---------|---------|-------------|
| uv | 0.11.15 available | Run tests and scripts in project environment | Use for validation commands and dataset runner examples. [VERIFIED: command `uv --version`] |
| rocprofv3 | `/usr/bin/rocprofv3` available | Optional timing evidence collection | Only needed when validating profiler-backed timing evidence, not required for static Phase 52 contract tests. [VERIFIED: command `which rocprofv3`, `scripts/run_dataset.py:644-663`] |
| rocm-smi | `/usr/bin/rocm-smi` available | Optional clock-locking and GPU environment support | Only needed for benchmark-grade or GPU validation runs. [VERIFIED: command `which rocm-smi`, `docs/analysis.md:31-45`] |

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Extend `scripts/run_dataset.py` derived artifact path | Add primary `sol-execbench` options | Rejected by locked context and existing contract docs because primary CLI must stay stable. [VERIFIED: repo `.planning/phases/52-dataset-runner-and-public-contract-closure/52-CONTEXT.md`, `docs/internal/solar_derivation_contract.md:27-41`] |
| Reuse scoring + derivation modules | Build custom report parser/aggregator | Rejected because existing score functions already accept SOLAR guards and preserve claim boundaries. [VERIFIED: repo `src/sol_execbench/core/scoring/amd_score.py:156-216`] |
| Add broad forbidden-word tests | Use exact-key and phrase-context tests | Broad bans would catch valid historical/no-claim context; existing pattern uses exact fields and explicit phrases. [VERIFIED: repo `tests/sol_execbench/test_public_contract_guardrails.py`, `.planning/phases/52-dataset-runner-and-public-contract-closure/52-CONTEXT.md`] |

**Installation:** No new package installation is recommended for Phase 52. [VERIFIED: repo `pyproject.toml`, phase scope]

## Package Legitimacy Audit

No external packages are recommended or newly installed for this phase. [VERIFIED: repo `pyproject.toml`, phase scope]

| Package | Registry | Age | Downloads | Source Repo | slopcheck | Disposition |
|---------|----------|-----|-----------|-------------|-----------|-------------|
| None | N/A | N/A | N/A | N/A | N/A | No package changes |

**Packages removed due to slopcheck [SLOP] verdict:** none  
**Packages flagged as suspicious [SUS]:** none

## Current Behavior Inventory

### Dataset Runner

- `scripts/run_dataset.py` discovers dataset categories or a single problem directory, then writes per-problem outputs under `<output>/<category>/<problem>/`. [VERIFIED: repo `scripts/run_dataset.py:60-83`, `scripts/run_dataset.py:675-735`]
- The runner writes canonical trace payloads to `traces.json` after collecting JSON lines from primary `sol-execbench --json`. [VERIFIED: repo `scripts/run_dataset.py:181-239`, `scripts/run_dataset.py:809-811`]
- `--amd-score-report` is opt-in and writes a suite report after all problems finish. [VERIFIED: repo `scripts/run_dataset.py:619-624`, `scripts/run_dataset.py:865-879`]
- `--amd-sol-bound-dir` is opt-in and writes per-workload AMD SOL v2 sidecars named `<definition>.<workload_uuid>.amd-sol-v2.json`. [VERIFIED: repo `scripts/run_dataset.py:498-506`, `tests/sol_execbench/test_run_dataset_amd_score.py`]
- Current runner score construction does not pass `solar_derivation` into `score_amd_native_trace_workload()`. [VERIFIED: repo `scripts/run_dataset.py:507-523`]

### AMD-Native Score Reports

- Report schema is `sol_execbench.amd_native_score.v1`; score `claim_level` is constant `amd-native-derived`. [VERIFIED: repo `src/sol_execbench/core/scoring/amd_score.py:28-30`, `src/sol_execbench/core/scoring/amd_score.py:80-93`]
- Suite reports are derived artifacts with `canonical_output: trace_jsonl`, `mean_score`, `scored_count`, `unscored_count`, `warnings`, `baseline_summary`, `evidence_summary`, and `scores`. [VERIFIED: repo `src/sol_execbench/core/scoring/amd_score.py:96-153`]
- Existing public `evidence_refs` keys are `trace`, `timing`, `sol_bound`, `baseline`, and `hardware_model`. [VERIFIED: repo `src/sol_execbench/core/scoring/amd_score.py:124-138`, `src/sol_execbench/core/scoring/amd_score.py:380-398`]
- Explicit SOLAR `unscored` aggregate status suppresses the workload score, while degraded status preserves numeric scoring and emits degradation warnings. [VERIFIED: repo `src/sol_execbench/core/scoring/amd_score.py:170-203`, `tests/sol_execbench/test_amd_native_score.py`]

### SOLAR Sidecars

- `SolarDerivationEvidence.to_dict()` emits `schema_version`, `derived`, `definition`, `workload_uuid`, `groups`, `tensors`, `warnings`, `source_boundary`, `coverage_summary`, and `aggregate_status`. [VERIFIED: repo `src/sol_execbench/core/scoring/solar_derivation.py:360-387`]
- `coverage_summary` contains `family_counts`, `status_counts`, `families`, `missing_patterns`, `unsupported_patterns`, degraded/unsupported/estimated node IDs, and provenance. [VERIFIED: repo `src/sol_execbench/core/scoring/solar_derivation.py:306-335`]
- `aggregate_status` contains `status`, `score_eligible`, `reason`, `group_ids`, `node_ids`, and `warnings`. [VERIFIED: repo `src/sol_execbench/core/scoring/solar_derivation.py:338-357`]
- SOLAR derivation is built from canonical `Definition` and `Workload` inputs through `build_bound_graph()` and `estimate_bound_work()`, not by executing submitted candidate solution code. [VERIFIED: repo `src/sol_execbench/core/scoring/solar_derivation.py:390-397`]

## Recommended Decisions

1. Add one opt-in dataset-runner option for SOLAR derivation sidecar output, with deterministic sidecar names keyed by `definition` and `workload_uuid`. [VERIFIED: repo pattern `scripts/run_dataset.py:498-506`]
2. Add one opt-in dataset-runner option for consuming an existing SOLAR derivation sidecar directory only if needed; if both build and read paths are too much scope, prioritize build-and-pass-through because `build_solar_derivation_evidence()` is already deterministic from canonical inputs. [VERIFIED: repo `src/sol_execbench/core/scoring/solar_derivation.py:390-397`]
3. Pass parsed `SolarDerivationEvidence` or `SolarAggregateStatus` into `score_amd_native_trace_workload()` by workload UUID when available. [VERIFIED: repo `src/sol_execbench/core/scoring/amd_score.py:233-305`]
4. Preserve existing score `evidence_refs` keys; add SOLAR sidecar references only as a derived-report-specific field if needed, such as a suite-level `derived_artifacts` or per-score `internal_evidence_refs`, and guard that field as noncanonical. [VERIFIED: repo `src/sol_execbench/core/scoring/amd_score.py:124-153`, `.planning/phases/52-dataset-runner-and-public-contract-closure/52-CONTEXT.md`]
5. Document local consumption in `docs/analysis.md` and tighten `docs/internal/solar_derivation_contract.md` if the new option names or sidecar paths need to be named. [VERIFIED: repo `docs/analysis.md:184-281`, `docs/internal/solar_derivation_contract.md:258-266`]

## Architecture Patterns

### System Architecture Diagram

```text
Dataset root or problem dir
  -> scripts/run_dataset.py discovery
  -> per-problem Definition + Workload load
  -> reference/custom solution JSON construction
  -> sol-execbench --json subprocess
  -> canonical traces.json + summary.json
  -> optional derived artifact branch:
       -> AMD SOL v2 artifact build/write
       -> optional SOLAR derivation sidecar build/read
       -> score_amd_native_trace_workload(trace, sol_bound, solar_derivation=...)
       -> AMD-native suite report JSON
  -> docs/tests verify derived outputs remain noncanonical
```

### Recommended Project Structure

```text
scripts/
└── run_dataset.py                         # runner option parsing, sidecar IO, score report assembly
src/sol_execbench/core/scoring/
├── amd_score.py                           # claim level, warnings, evidence refs, score guard behavior
└── solar_derivation.py                    # sidecar builder/parser and aggregate status contract
docs/
├── analysis.md                            # user-facing local workflow and interpretation boundaries
└── internal/solar_derivation_contract.md  # internal sidecar contract and no-claim language
tests/sol_execbench/
├── test_run_dataset_amd_score.py          # runner/report sidecar behavior
├── test_public_contract_guardrails.py     # canonical schema/CLI/trace guardrails
└── test_solar_derivation_evidence.py      # sidecar schema/aggregate behavior
```

### Pattern 1: Derived Artifact Sidecar Branch

**What:** Keep canonical traces and summary untouched, then branch into optional derived sidecar/report writing. [VERIFIED: repo `scripts/run_dataset.py:809-823`, `scripts/run_dataset.py:865-879`]  
**When to use:** Use for SOLAR sidecar output and report refs. [VERIFIED: repo `docs/internal/solar_derivation_contract.md:27-41`]  
**Example:**

```python
# Existing pattern in scripts/run_dataset.py
if args.amd_score_report is not None:
    amd_scores.extend(
        build_amd_score_reports_for_problem(
            definition_payload=definition,
            workload_path=workload_path,
            traces_payload=traces,
            trace_ref=str(traces_path.relative_to(output_dir)),
            baseline_artifact=scoring_baseline,
            sol_bound_artifact_dir=args.amd_sol_bound_dir,
        )
    )
```

### Pattern 2: Score Guard by Workload UUID

**What:** Build maps by `workload.uuid`, then pass aggregate guard data into existing scoring functions. [VERIFIED: repo `src/sol_execbench/core/scoring/amd_score.py:308-338`, `tests/sol_execbench/test_amd_native_score.py`]  
**When to use:** Use when the runner needs to distinguish absent SOLAR sidecar data from explicit `unscored` SOLAR evidence. [VERIFIED: repo `src/sol_execbench/core/scoring/amd_score.py:186-203`]

### Pattern 3: Exact-Key Contract Tests

**What:** Inspect `model_dump(mode="json")`, recursive JSON keys, and CLI help text; do not rely on generic substring bans. [VERIFIED: repo `tests/sol_execbench/test_public_contract_guardrails.py`]  
**When to use:** Use for TEST-04 and for any new derived-report-specific field. [VERIFIED: repo `.planning/phases/52-dataset-runner-and-public-contract-closure/52-CONTEXT.md`]

### Anti-Patterns to Avoid

- **Adding primary CLI options for SOLAR derivation:** The sidecar contract says primary `sol-execbench` behavior/default output must not mutate. [VERIFIED: repo `docs/internal/solar_derivation_contract.md:27-41`]
- **Putting `coverage_summary`, `aggregate_status`, or formula/byte/bound evidence into `Trace`:** Existing guardrails explicitly keep these fields noncanonical. [VERIFIED: repo `tests/sol_execbench/test_public_contract_guardrails.py`]
- **Treating missing sidecar data as `unscored`:** Missing sidecars should be neutral compatibility behavior, while explicit `aggregate_status.status == "unscored"` should suppress scores. [VERIFIED: repo `tests/sol_execbench/test_amd_native_score.py`]
- **Banning all B200/Blackwell/paper mentions:** Existing docs legitimately mention these as out-of-scope/no-claim contexts. [VERIFIED: repo `docs/analysis.md:162-217`, `docs/internal/solar_derivation_contract.md:14-25`]

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| SOLAR sidecar generation | Ad hoc JSON dicts in the runner | `build_solar_derivation_evidence()` | It derives from canonical `Definition`/`Workload`, emits source boundary data, coverage summary, and aggregate status. [VERIFIED: repo `src/sol_execbench/core/scoring/solar_derivation.py:360-397`] |
| SOLAR sidecar validation | Manual key checks in runner | `solar_derivation_from_dict()` | Existing parser rejects unknown fields, invalid versions, malformed phase51 fields, and mismatched aggregate/coverage data. [VERIFIED: repo `tests/sol_execbench/test_solar_derivation_evidence.py`] |
| AMD-native score warnings and claim level | Runner-side score dict construction | `score_amd_native_trace_workload()` and `build_amd_native_suite_report()` | Existing functions centralize claim level, warnings, baseline source, and evidence refs. [VERIFIED: repo `src/sol_execbench/core/scoring/amd_score.py:156-305`] |
| Public contract detection | Broad string grep only | Existing exact-key model/trace/CLI tests | Exact tests avoid false positives from legitimate derived artifacts and no-claim language. [VERIFIED: repo `tests/sol_execbench/test_public_contract_guardrails.py`] |

**Key insight:** Phase 52 is a wiring and guardrail phase; custom schema/report logic increases the chance of public contract drift and overclaiming. [VERIFIED: repo `.planning/phases/52-dataset-runner-and-public-contract-closure/52-CONTEXT.md`, existing scoring modules]

## Common Pitfalls

### Pitfall 1: SOLAR Refs Leak Into Public `evidence_refs`

**What goes wrong:** Adding `solar_derivation`, `coverage_summary`, or `aggregate_status` to public score `evidence_refs` breaks existing public-boundary assumptions. [VERIFIED: repo `tests/sol_execbench/test_public_contract_guardrails.py`]  
**Why it happens:** REPORT-04 asks for auditable formula/coverage/score-eligibility refs, but the context says existing public keys must be preserved unless a derived report-specific field is explicitly scoped. [VERIFIED: repo `.planning/phases/52-dataset-runner-and-public-contract-closure/52-CONTEXT.md`]  
**How to avoid:** Keep `evidence_refs` keys stable and put internal refs in a clearly derived/internal field with tests proving it stays out of canonical schemas. [VERIFIED: repo `src/sol_execbench/core/scoring/amd_score.py:124-153`]  
**Warning signs:** `test_public_contract_guardrails.py` starts failing on recursive key checks or primary CLI help. [VERIFIED: repo `tests/sol_execbench/test_public_contract_guardrails.py`]

### Pitfall 2: Missing Sidecar Treated Like Explicit Unscored Evidence

**What goes wrong:** Existing reports change score behavior when no SOLAR sidecar option is used. [VERIFIED: repo `tests/sol_execbench/test_amd_native_score.py`]  
**Why it happens:** The scorer distinguishes `solar_derivation=None` from `SolarAggregateStatus(status="unscored")`. [VERIFIED: repo `src/sol_execbench/core/scoring/amd_score.py:170-203`]  
**How to avoid:** Only pass `solar_derivation` when a sidecar is built/read and parsed for that workload UUID. [VERIFIED: repo `src/sol_execbench/core/scoring/amd_score.py:233-305`]  
**Warning signs:** Existing `test_absent_solar_derivation_preserves_existing_workload_score_behavior` or suite missing-guard tests regress. [VERIFIED: repo `tests/sol_execbench/test_amd_native_score.py`]

### Pitfall 3: Claim Guardrails Become Too Broad

**What goes wrong:** Tests reject valid docs that say "not NVIDIA Blackwell/B200 equivalence" or list future validation scope. [VERIFIED: repo `docs/internal/solar_derivation_contract.md:14-25`]  
**Why it happens:** Raw substring bans cannot distinguish claims from no-claim language. [VERIFIED: repo `.planning/phases/52-dataset-runner-and-public-contract-closure/52-CONTEXT.md`]  
**How to avoid:** Test forbidden positive claim phrases and require no-claim phrases; do not ban all contextual mentions. [VERIFIED: repo `tests/sol_execbench/test_v1_9_validation_closure.py`, `tests/sol_execbench/test_public_contract_guardrails.py`]

### Pitfall 4: Candidate Solution Execution in Derivation

**What goes wrong:** Derivation accidentally depends on submitted solution source or runtime output. [VERIFIED: repo `.planning/REQUIREMENTS.md`, `docs/internal/solar_derivation_contract.md:27-41`]  
**Why it happens:** The runner already constructs and executes reference/custom solutions for benchmarking, so derivation code placed nearby could accidentally use solution payloads. [VERIFIED: repo `scripts/run_dataset.py:760-793`]  
**How to avoid:** Build SOLAR sidecars only from `Definition(**definition)` and `Workload` records read from `workload.jsonl`, mirroring `build_solar_derivation_evidence(definition, workload)`. [VERIFIED: repo `src/sol_execbench/core/scoring/solar_derivation.py:390-397`]

## Code Examples

### Existing SOLAR Guard Path

```python
# Existing scorer API supports this integration.
score_amd_native_trace_workload(
    trace,
    artifact,
    trace_ref=refs.get("trace"),
    timing_evidence_ref=refs.get("timing"),
    sol_bound_ref=refs.get("sol_bound"),
    baseline_ref=refs.get("baseline"),
    baseline_artifact=baseline_artifact,
    hardware_model_ref=refs.get("hardware_model"),
    solar_derivation=solar_derivations_by_workload_uuid.get(trace.workload.uuid),
)
```

Source: `src/sol_execbench/core/scoring/amd_score.py:323-337`. [VERIFIED: repo]

### Existing SOLAR Sidecar Builder

```python
def build_solar_derivation_evidence(
    definition: Definition,
    workload: Workload,
) -> SolarDerivationEvidence:
    graph = build_bound_graph(definition, workload)
    estimates = estimate_bound_work(graph)
    return derive_solar_derivation_evidence(definition, workload, graph, estimates)
```

Source: `src/sol_execbench/core/scoring/solar_derivation.py:390-397`. [VERIFIED: repo]

### Existing Derived Report Shape

```python
return {
    "schema_version": self.schema_version,
    "derived": self.derived,
    "canonical_output": self.canonical_output,
    "mean_score": self.mean_score,
    "scored_count": scored_count,
    "unscored_count": len(self.scores) - scored_count,
    "warnings": list(self.warnings),
    "baseline_summary": self.baseline_summary,
    "evidence_summary": self.evidence_summary,
    "scores": [score.to_dict() for score in self.scores],
}
```

Source: `src/sol_execbench/core/scoring/amd_score.py:140-153`. [VERIFIED: repo]

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Baseline-relative score interpretation only | AMD-native derived reports with explicit SOL bound, baseline, timing, trace, and hardware refs | Present before Phase 52 | Phase 52 should extend report evidence surfaces, not invent a new scoring report. [VERIFIED: repo `docs/analysis.md:162-249`] |
| AMD SOL v2 sidecars only | SOLAR derivation sidecars now include coverage summary and aggregate status | Phase 51 context and tests present in repo | Runner can now propagate SOLAR `degraded` / `unscored` status into AMD-native score guards. [VERIFIED: repo `src/sol_execbench/core/scoring/solar_derivation.py:306-387`, `src/sol_execbench/core/scoring/amd_score.py:170-203`] |
| v1.9 no-equivalence docs | v1.10 no-claim contract names Blackwell/B200, paper scale, leaderboard, real hardware validation, MI300X/CDNA/NVFP4/MXFP4 scope | Phase 52 context and internal contract | Claim guardrails must be more specific than previous v1.9 tests. [VERIFIED: repo `.planning/phases/52-dataset-runner-and-public-contract-closure/52-CONTEXT.md`, `docs/internal/solar_derivation_contract.md:14-25`] |

**Deprecated/outdated:** Adding derived SOLAR evidence directly to canonical `Definition`, `Workload`, `Trace`, public solution schemas, or primary CLI is out of scope for v1.10. [VERIFIED: repo `docs/internal/solar_derivation_contract.md:27-41`]

## Exact Integration Risks for Phase 52

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| New sidecar refs widen public `evidence_refs` unexpectedly | Medium | TEST-04 failure and public contract drift | Keep old keys stable; add internal derived refs only with exact-key tests. [VERIFIED: repo `src/sol_execbench/core/scoring/amd_score.py:124-153`] |
| Runner skip behavior omits sidecar/report updates for already-passing traces | High | `--amd-score-report` with existing traces may not include skipped problems because the current loop `continue`s before report generation. | Plan should explicitly decide whether derived report closure must score skipped traces; if yes, build derived artifacts from existing `traces.json` before `continue`. [VERIFIED: repo `scripts/run_dataset.py:737-745`] |
| Sidecar path identity is absolute while trace refs are output-relative | Medium | Reports become less portable or tests become brittle | Prefer output-relative refs where possible; existing sol-bound sidecar test currently accepts absolute `str(sidecar_path)`, so any change needs compatibility tests. [VERIFIED: repo `scripts/run_dataset.py:498-506`, `tests/sol_execbench/test_run_dataset_amd_score.py`] |
| Generated SOLAR sidecars from truncated workloads use output-local workload path | Medium | Correct but easy to miss in docs/tests | Use the active `workload_path` after `--max-workloads` truncation when deriving sidecars, matching AMD SOL v2 behavior. [VERIFIED: repo `scripts/run_dataset.py:752-758`, `scripts/run_dataset.py:813-823`] |
| Hardware model evidence and SOLAR aggregate evidence get conflated | Medium | Overclaiming hardware validation | Keep hardware refs under existing AMD SOL/hardware fields; keep SOLAR refs as derivation/coverage/eligibility evidence, not validation claims. [VERIFIED: repo `docs/analysis.md:267-281`] |
| Claim guardrails block valid no-claim text | Medium | Brittle docs tests | Test forbidden positive phrases and required no-claim phrases instead of banning raw tokens. [VERIFIED: repo `docs/internal/solar_derivation_contract.md:14-25`] |
| Candidate solution execution leaks into derivation | Low | Violates v1.10 scope and claim boundary | Build sidecars before/after CLI using only definition/workload objects, never solution source or trace performance. [VERIFIED: repo `src/sol_execbench/core/scoring/solar_derivation.py:390-397`] |

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | No assumptions were needed; findings are based on repository files and executed local commands. | All | N/A |

## Open Questions

1. **Should Phase 52 support reading pre-existing SOLAR sidecars or only writing them?**
   - What we know: The scorer accepts parsed SOLAR guard objects and the runner already writes derived sidecars for AMD SOL v2. [VERIFIED: repo `src/sol_execbench/core/scoring/amd_score.py:233-305`, `scripts/run_dataset.py:498-506`]
   - What's unclear: The context allows exact option names at planner discretion, but does not lock whether sidecars are generated, consumed, or both. [VERIFIED: repo `.planning/phases/52-dataset-runner-and-public-contract-closure/52-CONTEXT.md`]
   - Recommendation: Prioritize write-and-pass-through from canonical inputs; add read-existing only if needed for local report workflows. [VERIFIED: repo `src/sol_execbench/core/scoring/solar_derivation.py:390-397`]
2. **Where should report-only internal SOLAR refs live?**
   - What we know: Existing public `evidence_refs` keys are stable and counted by `evidence_summary`. [VERIFIED: repo `src/sol_execbench/core/scoring/amd_score.py:124-153`]
   - What's unclear: The context allows a derived report-specific internal field if explicitly scoped and guarded. [VERIFIED: repo `.planning/phases/52-dataset-runner-and-public-contract-closure/52-CONTEXT.md`]
   - Recommendation: Add a suite-level or per-score `internal_evidence_refs` / `derived_evidence_refs` field only in AMD-native derived reports, then assert it never appears in canonical models/traces. [VERIFIED: repo `tests/sol_execbench/test_public_contract_guardrails.py`]

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|-------------|-----------|---------|----------|
| Python | Tests and scripts | yes | 3.12.13 | None needed. [VERIFIED: command `python --version`] |
| uv | Validation commands | yes | 0.11.15 | Use installed `pytest` directly for static tests if uv is unavailable. [VERIFIED: command `uv --version`] |
| pytest | Contract validation | yes | 9.0.3 | None needed. [VERIFIED: command `pytest --version`] |
| sol-execbench console script | Real dataset runner invocation | not on PATH directly | available through `uv run` project environment | Use `uv run sol-execbench ...`; runner builds command from active Python environment. [VERIFIED: command `which sol-execbench`, `scripts/run_dataset.py:181-204`] |
| rocprofv3 | Optional timing evidence | yes | path `/usr/bin/rocprofv3` | Not required for static Phase 52 tests. [VERIFIED: command `which rocprofv3`] |
| rocm-smi | Optional clock locking / GPU checks | yes | path `/usr/bin/rocm-smi` | Not required for static Phase 52 tests. [VERIFIED: command `which rocm-smi`] |

**Missing dependencies with no fallback:** none for static planning and unit guardrails. [VERIFIED: local commands]  
**Missing dependencies with fallback:** direct `sol-execbench` PATH entry is missing; use `uv run` project environment. [VERIFIED: command `which sol-execbench`]

## Validation Architecture

### Test Framework

| Property | Value |
|----------|-------|
| Framework | pytest 9.0.3 [VERIFIED: command `pytest --version`] |
| Config file | `pyproject.toml` [VERIFIED: repo `pyproject.toml`] |
| Quick run command | `uv run pytest tests/sol_execbench/test_run_dataset_amd_score.py tests/sol_execbench/test_amd_native_score.py tests/sol_execbench/test_public_contract_guardrails.py -q` |
| Full relevant suite command | `uv run pytest tests/sol_execbench/test_run_dataset_amd_score.py tests/sol_execbench/test_amd_native_score.py tests/sol_execbench/test_public_contract_guardrails.py tests/sol_execbench/test_solar_derivation_evidence.py -q` |

### Phase Requirements -> Test Map

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|--------------|
| REPORT-04 | Derived reports keep `claim_level: amd-native-derived` and include formula, hardware, coverage, and score-eligibility refs without widening canonical artifacts. | unit/contract | `uv run pytest tests/sol_execbench/test_run_dataset_amd_score.py tests/sol_execbench/test_amd_native_score.py -q` | yes |
| TEST-04 | Canonical schemas, trace JSONL, primary CLI help, and public semantics remain unchanged. | static contract | `uv run pytest tests/sol_execbench/test_public_contract_guardrails.py -q` | yes |
| TEST-05 | Docs and artifacts avoid paper parity, B200/Blackwell equivalence, leaderboard readiness, CDNA/MI300X/CDNA4 validation, NVFP4/MXFP4 validation, and new hardware validation claims. | static claim guardrail | `uv run pytest tests/sol_execbench/test_public_contract_guardrails.py tests/sol_execbench/test_solar_derivation_contract.py tests/sol_execbench/test_v1_9_validation_closure.py -q` | yes |

### Sampling Rate

- **Per task commit:** `uv run pytest tests/sol_execbench/test_run_dataset_amd_score.py tests/sol_execbench/test_public_contract_guardrails.py -q`
- **Per wave merge:** `uv run pytest tests/sol_execbench/test_run_dataset_amd_score.py tests/sol_execbench/test_amd_native_score.py tests/sol_execbench/test_public_contract_guardrails.py tests/sol_execbench/test_solar_derivation_evidence.py -q`
- **Phase gate:** Full relevant suite plus any updated docs tests green before `$gsd-verify-work`.

### Wave 0 Gaps

- [ ] Extend `tests/sol_execbench/test_run_dataset_amd_score.py` for SOLAR derivation sidecar write/pass-through behavior and missing-vs-unscored behavior. [VERIFIED: repo current file lacks runner SOLAR sidecar tests]
- [ ] Extend `tests/sol_execbench/test_public_contract_guardrails.py` for Phase 52 derived-report-specific internal refs and unchanged primary CLI help. [VERIFIED: repo current guardrails target through Phase 51]
- [ ] Add or extend claim guardrail tests covering `docs/analysis.md`, `docs/internal/solar_derivation_contract.md`, runner option help text, and sample report keys for Phase 52 no-claim language. [VERIFIED: repo current docs/tests]

### Verification Already Run During Research

```bash
uv run pytest tests/sol_execbench/test_run_dataset_amd_score.py \
  tests/sol_execbench/test_amd_native_score.py \
  tests/sol_execbench/test_public_contract_guardrails.py \
  tests/sol_execbench/test_solar_derivation_evidence.py -q
```

Result: 125 passed in 5.17s. [VERIFIED: command output]

## Security Domain

### Applicable ASVS Categories

| ASVS Category | Applies | Standard Control |
|---------------|---------|------------------|
| V2 Authentication | no | No auth surface in local runner/docs phase. [VERIFIED: phase scope] |
| V3 Session Management | no | No session surface in local runner/docs phase. [VERIFIED: phase scope] |
| V4 Access Control | no | No multi-user authorization surface in local runner/docs phase. [VERIFIED: phase scope] |
| V5 Input Validation | yes | Parse sidecars with existing strict Pydantic/dataclass parsers and exact-key tests. [VERIFIED: repo `src/sol_execbench/core/scoring/solar_derivation.py`, tests] |
| V6 Cryptography | no | No cryptographic feature in phase. [VERIFIED: phase scope] |

### Known Threat Patterns for This Stack

| Pattern | STRIDE | Standard Mitigation |
|---------|--------|---------------------|
| Candidate code influences derivation evidence | Tampering | Derive SOLAR sidecars only from canonical definition/workload inputs; do not execute candidate solution code for derivation. [VERIFIED: repo `src/sol_execbench/core/scoring/solar_derivation.py:390-397`, `.planning/REQUIREMENTS.md`] |
| Malformed sidecar widens claims | Tampering / Repudiation | Use `solar_derivation_from_dict()` strict parsing and reject unknown fields / mismatched coverage and aggregate status. [VERIFIED: repo `tests/sol_execbench/test_solar_derivation_evidence.py`] |
| Report wording overclaims validation | Spoofing / Repudiation | Static claim guardrails with required no-claim language and forbidden positive claim phrases. [VERIFIED: repo `docs/internal/solar_derivation_contract.md:14-25`, tests] |
| Local output accidentally committed | Information disclosure | Keep outputs in `out/` or temp dirs and do not commit benchmark outputs. [VERIFIED: repo `AGENTS.md`] |

## Sources

### Primary (HIGH confidence)

- `scripts/run_dataset.py` - dataset discovery, CLI invocation, trace output, AMD-native report, AMD SOL v2 sidecar behavior. [VERIFIED: repo]
- `src/sol_execbench/core/scoring/amd_score.py` - report schema, claim level, evidence refs, SOLAR aggregate score guard behavior. [VERIFIED: repo]
- `src/sol_execbench/core/scoring/solar_derivation.py` - SOLAR sidecar schema, coverage summary, aggregate status, builder/parser boundaries. [VERIFIED: repo]
- `docs/analysis.md` - current AMD-native score report and AMD SOL v2 documentation plus claim boundaries. [VERIFIED: repo]
- `docs/internal/solar_derivation_contract.md` - v1.10 sidecar-only contract and no-claim phrases. [VERIFIED: repo]
- `tests/sol_execbench/test_run_dataset_amd_score.py` - existing runner/report tests. [VERIFIED: repo]
- `tests/sol_execbench/test_amd_native_score.py` - score guard and suite report tests. [VERIFIED: repo]
- `tests/sol_execbench/test_public_contract_guardrails.py` - public schema/CLI/trace and claim-boundary guardrails. [VERIFIED: repo]
- `tests/sol_execbench/test_solar_derivation_evidence.py` and `tests/sol_execbench/test_solar_derivation_contract.py` - sidecar parsing and fixture contract tests. [VERIFIED: repo]
- `.planning/phases/52-dataset-runner-and-public-contract-closure/52-CONTEXT.md`, `.planning/REQUIREMENTS.md`, `.planning/ROADMAP.md` - phase scope and locked decisions. [VERIFIED: repo]

### Secondary (MEDIUM confidence)

- None. [VERIFIED: research scope used repository sources only]

### Tertiary (LOW confidence)

- None. [VERIFIED: no unverified web or training-only claims used]

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - verified from `pyproject.toml` and local commands.
- Architecture: HIGH - verified from runner/scoring source and tests.
- Pitfalls: HIGH - derived from existing failing-risk boundaries encoded in tests and context.
- Claim guardrails: HIGH - verified from context, docs, and existing claim tests.

**Research date:** 2026-05-23  
**Valid until:** 2026-06-22 for repository-local architecture; re-check if scoring/report schemas change before planning.
