# Architecture Research

**Domain:** ROCm benchmark research-credibility evidence infrastructure  
**Project:** SOL ExecBench ROCm Port v1.19 Research Credibility Without New Hardware  
**Researched:** 2026-05-31  
**Confidence:** HIGH for repository integration points; MEDIUM for exact phase count because roadmap slicing may choose smaller or larger implementation phases.

## Executive Recommendation

v1.19 should integrate as an additive evidence/reporting layer around the existing dataset, Matrix, and scoring sidecar architecture. Do not change canonical `Trace` semantics, evaluator subprocess behavior, correctness rules, timing rules, or score authority. The safest architecture is to add focused core modules for the new contracts, then call them from the existing script/CLI surfaces that already own the relevant IO.

The main new package components should live near existing bounded domains:

- Denominator accounting: `src/sol_execbench/core/dataset/denominator.py`
- Matrix diff and schema export: `src/sol_execbench/core/compatibility_diff.py` and small additions to `src/sol_execbench/core/compatibility.py`
- Run-dataset hardening: focused helpers under `src/sol_execbench/core/dataset/execution_closure.py`, with thin adapter calls from `scripts/run_dataset.py`
- AMD SOL/SOLAR sanity evidence: `src/sol_execbench/core/scoring/amd_sanity.py`

Keep user-facing command exposure small. The primary CLI in `src/sol_execbench/cli/main.py` can expose GPU-free metadata/report commands if needed, but batch-run behavior should stay in `scripts/run_dataset.py` because it already owns dataset fan-out, resume behavior, derived evidence sidecars, and execution closure.

## Existing Architecture Baseline

```text
----------------------------------------------------------------
| CLI / Script Entry Points                                     |
| src/sol_execbench/cli/main.py                                 |
| scripts/run_dataset.py                                       |
----------------------------------------------------------------
                              |
                              v
----------------------------------------------------------------
| Core Evidence Domains                                         |
| dataset/: manifest, inventory, readiness, ready_subset        |
| compatibility.py: strict ROCm Matrix Entry/Report contracts   |
| scoring/: AMD SOL/SOLAR and AMD-native derived score reports  |
----------------------------------------------------------------
                              |
                              v
----------------------------------------------------------------
| Canonical Evaluator Boundary                                  |
| core/data/Trace, driver/problem_packager.py, eval_driver.py   |
| correctness/timing/reward-hack behavior                       |
----------------------------------------------------------------
                              |
                              v
----------------------------------------------------------------
| Artifact Outputs                                              |
| trace JSONL, sidecar JSON, docs claim tables, guardrail tests |
----------------------------------------------------------------
```

The important invariant is already present: `Trace` remains the canonical benchmark output, while optional compatibility, profiler, static evidence, score, dataset readiness, and closure artifacts are sidecars. v1.19 should follow that same invariant.

## Recommended v1.19 System Shape

```text
                                 -------------------------------
                                 | Existing Trace JSONL         |
                                 | core/data/trace.py           |
                                 | NO v1.19 schema changes      |
                                 -------------------------------
                                                |
                                                | read-only input
                                                v
--------------------------       -------------------------------       --------------------------
| Dataset inventory      |       | Dataset runner / closure     |       | AMD SOL/SOLAR inputs   |
| core/dataset/*.py      |------>| scripts/run_dataset.py       |------>| core/scoring/*.py      |
| manifest/readiness     |       | thin orchestration only      |       | bound artifacts        |
--------------------------       -------------------------------       --------------------------
            |                                   |                                  |
            v                                   v                                  v
--------------------------       -------------------------------       --------------------------
| denominator report     |       | hardened execution closure   |       | sanity evidence report |
| core/dataset/           |       | core/dataset/                 |       | core/scoring/          |
| denominator.py          |       | execution_closure.py          |       | amd_sanity.py          |
--------------------------       -------------------------------       --------------------------

--------------------------       -------------------------------       --------------------------
| Matrix entries/reports  |------>| Matrix diff report          |------>| schema export JSON     |
| core/compatibility.py   |       | core/compatibility_diff.py   |       | model_json_schema()    |
--------------------------       -------------------------------       --------------------------
```

## Component Responsibilities

| Component | New or Modified | Responsibility | Exact Integration Point |
|-----------|-----------------|----------------|-------------------------|
| Denominator report model | New | Summarize original paper dataset denominator status as ready, blocked, unsupported, deferred, missing evidence, attempted, passed, failed, and scored without implying paper parity. | Add `src/sol_execbench/core/dataset/denominator.py`; consume `DatasetInventory`, `DatasetReadiness`, `ReadySubset`, execution closure, AMD score, SOL/SOLAR sidecars. |
| Dataset inventory/readiness | Existing, narrow extension only if needed | Continue owning static dataset parsing and ROCm readiness classification. | `src/sol_execbench/core/dataset/inventory.py`, `readiness.py`, `ready_subset.py`; do not duplicate inventory parsing in new code. |
| Execution closure model/helper | New or extracted | Move closure status vocabulary, record construction, totals, manifest consistency checks, and deterministic output building out of the large runner. | Extract from `scripts/run_dataset.py` into `src/sol_execbench/core/dataset/execution_closure.py`; keep script CLI args and file writes in the script. |
| Dataset runner adapter | Modified, minimal | Keep batch orchestration, solution construction, CLI invocation, resume behavior, and optional derived report flags. Use core helpers for status classification and report payloads. | `scripts/run_dataset.py`; avoid broad refactor. |
| Matrix diff report | New | Compare two `RocmCompatibilityMatrixReport` or `MatrixEntry` payloads and classify status, target, dependency, image, runtime, clock/evidence, and artifact changes. | Add `src/sol_execbench/core/compatibility_diff.py`; validate inputs with `RocmCompatibilityMatrixReport` from `compatibility.py`. |
| Matrix schema export | Modified, small | Export JSON Schema for `MatrixEntry`, `RocmCompatibilityMatrixReport`, and diff report so downstream CI/evidence producers can validate sidecars. | Add helper functions in `src/sol_execbench/core/compatibility.py` or a small `compatibility_schema.py`; expose through CLI/subcommand or `python -m` style module. |
| Main CLI metadata dispatch | Modified only if public command is required | Add GPU-free commands such as `matrix schema` or `matrix diff` if roadmap wants `sol-execbench` exposure. | `src/sol_execbench/cli/main.py`; mirror existing `contract`, `doctor`, `toolchain` dispatch style. |
| AMD SOL/SOLAR sanity evidence | New | Evaluate existing RDNA 4/Docker evidence against provisional AMD SOL/SOLAR model expectations and emit a diagnostic-only sanity report. | Add `src/sol_execbench/core/scoring/amd_sanity.py`; read existing AMD score, `amd_sol_v2`, `solar_derivation`, timing evidence, and compatibility sidecars. |
| Claim guardrail docs/tests | Modified | Keep denominator, Matrix, closure, and sanity claims separate from paper parity, leaderboard authority, and new hardware validation. | `docs/CLAIMS.md`, `docs/TESTING.md`, focused tests under `tests/sol_execbench/`. |

## Data Flow Changes

### 1. Denominator Accounting Flow

```text
dataset root
  -> build_dataset_inventory()
  -> classify_rocm_readiness()
  -> build_ready_subset()
  -> optional execution_closure.json from scripts/run_dataset.py
  -> optional AMD score / SOL / SOLAR evidence sidecars
  -> denominator report sidecar
```

The denominator report should be a new sidecar contract, not a replacement for `parity_gap.py`. `src/sol_execbench/core/dataset/parity_gap.py` already aggregates several denominator-style counts, but v1.19 needs a clearer paper-denominator status vocabulary. Build on its source-loading and checksum patterns, but use a new schema version such as `sol_execbench.paper_denominator_report.v1`.

Recommended status mapping:

| Status | Meaning | Inputs |
|--------|---------|--------|
| `ready` | Static ROCm blockers absent; ready to attempt bounded execution. | `DatasetReadiness.workloads[].status == "ready"` |
| `blocked` | Schema/input/runtime blockers prevent clean attempt. | readiness statuses such as `schema_input_blocked`, `custom_input_blocked`, `runtime_blocked` |
| `unsupported` | NVIDIA-only or ROCm-unsupported path. | readiness reason `nvidia_cuda_runtime_hint` or explicit unsupported classifications |
| `deferred` | In scope conceptually but outside current milestone/hardware evidence. | low precision, Quant, CDNA-only, or configured defer reason |
| `missing_evidence` | Could be reported only if required sidecars are absent. | closure `derived_evidence_missing`, missing SOL/SOLAR/timing refs |
| `attempted_passed` | Attempted and canonical trace passed. | execution closure |
| `attempted_failed` | Attempted and canonical trace failed or no trace emitted. | execution closure |

Keep category/problem/workload denominators explicit. The report should include source checksums for manifest, inventory, readiness, ready subset, execution closure, and derived evidence reports so readers can reconstruct the count path.

### 2. Matrix Diff And Schema Export Flow

```text
old matrix JSON + new matrix JSON
  -> validate as RocmCompatibilityMatrixReport
  -> index entries by target_id + validation_scope + requested ROCm/user-space identity
  -> compare stable fields
  -> emit diagnostic diff sidecar
```

The diff should compare validated Pydantic models, not raw dicts. It should classify changes by field family:

- Target changes: requested ROCm version, target id, validation scope, intended architecture.
- Image changes: Docker repository, tag, digest.
- Dependency changes: PyTorch/TorchVision/Triton versions, ROCm wheel target, policy IDs.
- Runtime changes: host/container ROCm, HIP version, GPU arch/name/count, device availability.
- Status changes: `MatrixCompatibilityStatus`, reason code, benchmark/probe/smoke decision.
- Evidence changes: artifact refs added/removed, missing evidence, clock/evidence notes where represented in artifacts or linked sidecars.

Schema export should use Pydantic v2 `model_json_schema()` on strict models. The exported schema files should remain sidecar/report contracts, not public `Trace` contracts. A practical first target is:

- `MatrixEntry.model_json_schema()`
- `RocmCompatibilityMatrixReport.model_json_schema()`
- `RocmCompatibilityMatrixDiff.model_json_schema()`

The schema exporter can live in a small helper rather than bloating `compatibility.py`. If exposed through `sol-execbench`, follow the existing `contract`, `doctor`, and `toolchain` metadata-command dispatch in `src/sol_execbench/cli/main.py`.

### 3. Run-Dataset Hardening Flow

`scripts/run_dataset.py` is already the correct orchestration boundary, but it is a large file. v1.19 should reduce risk by extracting deterministic policy and report-building pieces instead of moving the runner wholesale.

Recommended extraction:

```text
scripts/run_dataset.py
  owns argparse, discovery, solution construction, subprocess invocation, artifact paths
  calls:
    core/dataset/execution_closure.py
      - ExecutionClosureRecord Pydantic model
      - ExecutionClosureReport Pydantic model
      - failure classification vocabulary
      - resume/manifest consistency validators
      - closure totals
      - deterministic serialization
```

Hardening should add explicit classifications for cases currently collapsed into "CLI returned no traces" or broad attempted failures:

- `cli_timeout`
- `compile_failed`
- `evaluation_nonzero_no_trace`
- `trace_parse_failed`
- `trace_missing_workload`
- `manifest_checksum_mismatch`
- `ready_subset_checksum_mismatch`
- `resume_output_incomplete`
- `derived_evidence_missing`

The runner should still produce `summary.json`, per-problem `traces.json`, optional AMD score reports, and optional closure reports. The hardening artifact should make resume behavior reproducible by recording input checksums, selected workload refs, existing trace status, evidence refs, and why an existing pass was reused or rejected.

### 4. AMD SOL/SOLAR Sanity Evidence Flow

```text
trace/performance evidence
  + amd_sol_v2 sidecars
  + solar_derivation sidecars
  + AMD-native score report
  + Matrix/runtime evidence
  -> diagnostic sanity report
```

The sanity report should validate relationships and evidence completeness, not create new score authority. It should answer questions such as:

- Is every scored workload linked to a SOL bound artifact and SOLAR derivation sidecar?
- Are hardware model refs consistent with observed RDNA 4 evidence?
- Are degraded SOLAR derivations counted separately from complete derivations?
- Are performance/timing artifacts unlocked, missing, or container-only?
- Are provisional model warnings present when evidence is RDNA 4/Docker-only?

Recommended schema fields:

- `schema_version`
- `created_at`
- `sources`
- `workload_checks`
- `status_counts`
- `model_risk_summary`
- `claim_boundary`

Claim boundary flags should be explicit:

- `diagnostic_sanity_evidence: true`
- `score_authority: false`
- `paper_parity_authority: false`
- `leaderboard_authority: false`
- `new_hardware_validation: false`
- `cdna3_validated: false`
- `cdna4_validated: false`

## Build Order Recommendation

1. **Extract/define closure contracts first**
   - Create `src/sol_execbench/core/dataset/execution_closure.py`.
   - Move the status vocabulary and deterministic report model out of `scripts/run_dataset.py`.
   - Add unit tests for totals, status validation, checksum mismatch, and deterministic ordering.
   - Rationale: denominator accounting depends on reliable closure inputs, and this reduces risk in the largest touched module.

2. **Add denominator accounting**
   - Create `src/sol_execbench/core/dataset/denominator.py`.
   - Consume manifest/inventory/readiness/ready-subset/closure/score sidecars.
   - Add tests with small fixture payloads covering ready, blocked, unsupported, deferred, missing evidence, attempted pass/fail.
   - Rationale: this is the core milestone claim and should land before docs wording.

3. **Add Matrix schema export**
   - Add schema helper/exporter for `MatrixEntry` and `RocmCompatibilityMatrixReport`.
   - Add tests that schemas include strict authority flags and status enums.
   - Rationale: low runtime risk, builds directly on existing strict Pydantic models.

4. **Add Matrix diff**
   - Create `src/sol_execbench/core/compatibility_diff.py`.
   - Compare validated reports by stable entry identity.
   - Add tests for status changes, dependency changes, image changes, evidence additions/removals, and missing entries.
   - Rationale: schema export establishes the contract; diff then gets its own contract.

5. **Harden `scripts/run_dataset.py` with core helpers**
   - Replace local closure helper logic with calls into `core/dataset/execution_closure.py`.
   - Add failure classification around subprocess timeout, no traces, partial traces, resume consistency, and sidecar evidence checks.
   - Rationale: do this after core contracts exist so script edits are mechanical and contained.

6. **Add AMD SOL/SOLAR sanity evidence**
   - Create `src/sol_execbench/core/scoring/amd_sanity.py`.
   - Read existing derived sidecars and Matrix/runtime evidence.
   - Add CPU-safe tests using fixture JSON; no new hardware validation.
   - Rationale: this relies on denominator/closure evidence and should be positioned as a diagnostic capstone.

7. **Docs and guardrails**
   - Update `docs/CLAIMS.md`, `docs/TESTING.md`, researcher docs, and schema docs.
   - Add wording tests for "paper parity", "leaderboard", "native host", "new hardware", "CDNA 3", and "CDNA 4" claims.
   - Rationale: docs should reflect stable artifact contracts, not lead implementation.

## Patterns To Follow

### Pattern: Versioned Pydantic Sidecar Contracts

**What:** Define strict Pydantic models with schema version constants, deterministic `to_json()` or `to_dict()` helpers, checksum fields where useful, and explicit claim boundaries.

**Use for:** denominator reports, closure reports, Matrix diffs, schema export manifests, AMD sanity reports.

**Why:** Existing code already uses this for dataset manifests, readiness, ready subsets, Matrix entries, and AMD scoring artifacts. It keeps reports auditable and testable without touching canonical trace.

### Pattern: Script As Adapter, Core As Policy

**What:** Keep `scripts/run_dataset.py` responsible for CLI arguments, paths, subprocess calls, and file writes. Move classification and payload construction to `src/sol_execbench/core/dataset/`.

**Use for:** failure classification, resume consistency, closure totals, denominator source normalization.

**Why:** The runner is a known large orchestration hotspot. Small extra branches in that file are acceptable; new policy clusters should be in tested core modules.

### Pattern: Read Canonical Trace, Do Not Extend It

**What:** v1.19 reports may read `Trace` payloads, but they must not add fields to `src/sol_execbench/core/data/trace.py` or reinterpret evaluator statuses.

**Use for:** denominator pass/fail counts, AMD sanity checks, closure status.

**Why:** `Trace` is the public benchmark contract. Research credibility artifacts are sidecars and reports.

### Pattern: Source Checksums For Research Claims

**What:** Every aggregate report should include source paths, schema versions, and checksums when available.

**Use for:** denominator reports, Matrix diffs, AMD sanity reports.

**Why:** v1.19 is about auditability. A count without source identity is hard to reproduce.

## Anti-Patterns To Avoid

### Mutating `Trace` For Denominator Or Sanity Fields

**Why bad:** It changes the public benchmark contract and risks confusing diagnostic evidence with evaluator authority.

**Do instead:** Add sidecar models under `core/dataset/`, `core/compatibility*`, or `core/scoring/`.

### Adding A Second Dataset Discovery Implementation

**Why bad:** It will drift from `build_dataset_inventory()`, readiness, and ready-subset semantics.

**Do instead:** Consume `DatasetInventory`, `DatasetReadiness`, and `ReadySubset` payloads.

### Broad Refactor Of `scripts/run_dataset.py`

**Why bad:** The file combines CLI fan-out, trace interpretation, scoring, closure, timing evidence, and resume logic. Large reshaping will create regression risk.

**Do instead:** Extract stable helper modules, then make small adapter edits.

### Treating Matrix Diff As Validation Authority

**Why bad:** Matrix entries are diagnostic compatibility evidence. A diff can explain changes, not upgrade benchmark, host, paper, or leaderboard claims.

**Do instead:** Add explicit diff claim boundaries and preserve existing Matrix authority flags.

### Letting AMD Sanity Evidence Become A New Score

**Why bad:** The sanity report is a consistency and risk report over existing evidence. It must not create new SOL/SOLAR or leaderboard authority.

**Do instead:** Keep sanity status separate from `AmdNativeScore` and make provisional model warnings visible.

## Risk Containment

| Risk | Containment |
|------|-------------|
| Large runner regression | Add core helpers first; keep `scripts/run_dataset.py` edits narrow and covered by dry-run/fixture tests. |
| Claim overreach | Every new report has explicit authority flags set false for paper parity, leaderboard, score authority where applicable, native host validation, and new hardware validation. |
| Denominator ambiguity | Store counts by category, problem, workload, and status; include source checksums and reason-code rollups. |
| Matrix diff instability | Compare validated models by stable target identity and classify field families instead of emitting raw recursive diffs only. |
| Sidecar proliferation | Use existing sidecar naming/schema patterns and document how denominator, closure, Matrix, and sanity reports relate. |
| Hardware dependency creep | Keep all new tests CPU-safe with fixture JSON. Live ROCm/Docker evidence may be consumed but not required to validate core logic. |

## Testing Implications

Recommended focused tests:

- `tests/sol_execbench/test_dataset_execution_closure.py`
- `tests/sol_execbench/test_dataset_denominator_report.py`
- `tests/sol_execbench/test_rocm_compatibility_matrix_schema.py`
- `tests/sol_execbench/test_rocm_compatibility_matrix_diff.py`
- `tests/sol_execbench/test_run_dataset_hardening.py`
- `tests/sol_execbench/test_amd_sanity_evidence.py`
- docs guardrails near existing Matrix and claims tests

Core assertions:

- New reports validate with Pydantic and serialize deterministically.
- `Trace` schema and evaluator output remain unchanged.
- Matrix authority flags remain false where existing contract requires them.
- Denominator reports distinguish ready, blocked, unsupported, deferred, missing evidence, attempted pass, and attempted fail.
- Dataset runner resume rejects or flags mismatched source checksums instead of silently reusing stale closure.
- AMD sanity report can be built from fixture sidecars without ROCm hardware.

## Roadmap Implications

Suggested phase ordering:

1. **Closure Contract Foundation** - Extract and test execution closure models/helpers from the runner.
2. **Paper Denominator Accounting** - Build denominator reports from manifest/inventory/readiness/closure evidence.
3. **Matrix Contract Utilities** - Add schema export, then Matrix diff.
4. **Dataset Runner Hardening** - Wire new failure/resume classifications into `scripts/run_dataset.py`.
5. **AMD SOL/SOLAR Sanity Evidence** - Add diagnostic consistency reports over existing RDNA 4/Docker evidence.
6. **Docs And Guardrails** - Lock claim wording after artifact contracts exist.

This order minimizes risk because each later phase consumes earlier sidecar contracts. The only phase that should touch large orchestration code materially is runner hardening, and even that should be mostly adapter wiring after the core helper exists.

## Sources

- `.planning/PROJECT.md`
- `.planning/codebase/ARCHITECTURE.md`
- `.planning/codebase/STRUCTURE.md`
- `.planning/codebase/CONCERNS.md`
- `src/sol_execbench/core/compatibility.py`
- `src/sol_execbench/core/dataset/inventory.py`
- `src/sol_execbench/core/dataset/readiness.py`
- `src/sol_execbench/core/dataset/ready_subset.py`
- `src/sol_execbench/core/dataset/parity_gap.py`
- `scripts/run_dataset.py`
- `src/sol_execbench/cli/main.py`
