# Stack Research

**Domain:** ROCm benchmark research-credibility infrastructure
**Researched:** 2026-05-31
**Confidence:** HIGH for repository stack and integration points; MEDIUM for exact phase split because v1.19 requirements are milestone-context only, not yet active in `.planning/REQUIREMENTS.md`.

## Executive Recommendation

v1.19 should add **no new external runtime dependencies, services, databases, hardware targets, or broad framework changes**. The requested features are contract/reporting/tooling additions over sidecar JSON that the repository already models with Python 3.12+, Pydantic v2, Click/argparse CLIs, Rich where user-facing tables are useful, pytest, and deterministic JSON/Markdown artifacts.

Use the current stack because the new work is about auditable accounting, diffs, schema export, runner hardening, and sanity checks over existing RDNA 4/Docker evidence. Adding a service, data store, validation farm, dataframe dependency, or new hardware abstraction would increase claim risk without improving credibility for this milestone.

## Recommended Stack Additions

### Core Technologies

| Technology | Version | Purpose | Why Recommended |
|------------|---------|---------|-----------------|
| Python | `>=3.12,<3.14` from `pyproject.toml` | Implement pure helpers, CLIs, report builders, and tests. | Existing package and scripts are Python; all v1.19 features can be CPU-safe transformations over JSON sidecars. |
| Pydantic | `>=2.12.5` from `pyproject.toml` | Define strict public sidecar contracts, JSON schema export, diff result models, denominator reports, and bound sanity reports. | Existing Matrix, Docker target, runtime evidence, and dataset sidecars already use strict Pydantic models; `model_json_schema()` is enough for schema export without adding a JSON Schema library. |
| JSON sidecars | Existing `schema_version` pattern | Store paper denominator accounting, compatibility diff reports, exported schemas, runner closure details, and AMD SOL/SOLAR sanity output. | The repo intentionally keeps evidence outside canonical trace JSONL; v1.19 should preserve that boundary. |
| Markdown reports | Existing docs/report pattern | Human-readable summaries for denominator status, Matrix diffs, dataset-runner closure, and bound sanity. | Research credibility depends on auditable explanation, not only machine JSON. Existing docs guardrails treat report wording as part of the product contract. |
| Click / argparse | Click `>=8.0`, stdlib `argparse` | Integrate with `sol-execbench` for package-facing commands and with existing script/module CLIs for dataset/Docker tooling. | Main CLI already uses Click; `scripts/run_dataset.py`, `core/docker_matrix.py`, and `core/runtime_evidence.py` already use argparse. Extend in place instead of migrating CLIs. |
| Rich | `>=13.0` | Optional tables for user-facing CLI summaries. | Already used by the CLI; useful for readable Matrix diff and denominator summaries, but JSON output should remain primary. |

### Supporting Libraries

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| `pathlib`, `json`, `datetime`, `hashlib`, `difflib` | Python stdlib | Deterministic file IO, stable timestamps/checksums, and compact text diffs. | Use for all v1.19 reports and Matrix diff tooling before considering dependencies. |
| `enum`, `typing.Literal`, `dataclasses` | Python stdlib | Bounded statuses, reason codes, and internal immutable records. | Follow existing Matrix and scoring patterns for status vocabularies and sidecar fields. |
| `pytest` | `>=9.0.2` | CPU-safe contract, schema, diff, runner, and guardrail coverage. | Use for new sidecar contracts and docs wording. Marker-gate only checks that genuinely require ROCm. |
| `pytest-xdist` | `>=3.5` | Existing parallel test execution. | No new configuration needed; keep new tests deterministic and independent. |
| Ruff | `>=0.4` | Lint and format. | No style stack change needed. |
| Ty | `>=0.0.39` | Type checking for `src` and tests. | Keep helper APIs typed and small enough for the existing checker. |

## Feature-Specific Stack Choices

### Paper Dataset Denominator Accounting

Use existing dataset modules under `src/sol_execbench/core/dataset/`:

| Existing Surface | Extend With | Why |
|------------------|-------------|-----|
| `inventory.py`, `readiness.py`, `ready_subset.py`, `parity_gap.py` | A focused denominator-accounting model/report, likely `paper_denominator.py` or an extension to `parity_gap.py`. | v1.11 already tracks discovered, parsed, ready, blocked, skipped, attempted, passed, failed, scored, degraded, and unscored counts. v1.19 needs paper-denominator vocabulary: ready, blocked, unsupported, deferred, missing evidence. |
| `DatasetReadiness`, `ParityGapReport` | Explicit reason-code buckets and status totals by problem and workload. | The new feature is accounting clarity, not a new dataset loader. |
| `checksums.py` and manifest checksum types | Stable source checksums in denominator reports. | Denominator claims need reproducible provenance. |
| `docs/original_parity.md`, `docs/v1_11_release_closure.md`, `docs/CLAIMS.md` | Wording updates and guardrail tests. | Prevent denominator accounting from becoming a full paper-parity claim. |

Recommended shape:

- Add a Pydantic sidecar such as `sol_execbench.paper_denominator.v1`.
- Inputs should be existing `dataset_manifest`, `inventory`, `readiness`, `ready_subset`, `execution_closure`, and optional score/evidence sidecars.
- Status categories should be explicit and non-overlapping: `ready`, `blocked`, `unsupported`, `deferred`, `evidence_missing`.
- Preserve workload-level and problem-level counts separately; paper claims usually care about problem denominators, while runner closure often operates at workload granularity.

Do not add `pandas`, SQLite, DuckDB, or a metrics database. The denominators are small JSON reports and deterministic Python aggregation is enough.

### Compatibility Matrix Diff Tooling

Use existing Matrix modules:

| Existing Surface | Extend With | Why |
|------------------|-------------|-----|
| `src/sol_execbench/core/compatibility.py` | Add pure diff result models, for example `MatrixDiffReport`, `MatrixEntryDiff`, `MatrixFieldChange`. | The Matrix contract already owns status, reason, target, observed evidence, artifacts, and claim boundaries. Diff semantics should live beside the contract. |
| `src/sol_execbench/core/runtime_evidence.py` | Add an `aggregate` sibling command such as `diff --base old.json --head new.json --output diff.json`. | This module already loads Matrix entries and writes aggregate reports. |
| `scripts/run_docker.sh` | Optional pass-through flag for writing or comparing Matrix reports only if roadmap requires wrapper integration. | Keep shell changes minimal; Python should do diff logic. |
| `tests/sol_execbench/test_rocm_compatibility_matrix.py`, `test_runtime_evidence_reports.py`, `test_rocm_matrix_docs.py` | Add CPU-safe diff tests. | Diffing is pure JSON comparison and should not require Docker or ROCm. |

Recommended diff behavior:

- Key Matrix entries by `target.target_id` and `target.validation_scope`.
- Report added/removed targets.
- Report changed fields for status, reason code, dependency versions, image repository/tag/digest, ROCm user-space version, GPU architecture/device name, runtime availability evidence, artifacts, and claim-boundary booleans.
- Keep raw before/after values in JSON; render a concise Markdown or Rich table for humans.
- Treat claim-boundary changes as high-signal changes; they affect what users may say.

Do not add a generic deep-diff dependency. Matrix payloads are bounded, typed, and small; a domain-specific diff is clearer and safer.

### Compatibility JSON Schema Export

Use Pydantic v2 schema generation:

| Existing Surface | Extend With | Why |
|------------------|-------------|-----|
| `MatrixEntry`, `RocmCompatibilityMatrixReport`, Docker target manifest models | Export `model_json_schema()` outputs. | Pydantic already owns the public contracts; exported schemas should be generated from those models, not duplicated by hand. |
| `src/sol_execbench/core/data/contract.py` | Optionally list new schema capabilities in the evaluator contract metadata. | Existing contract metadata already communicates public schema capabilities. |
| `sol-execbench` Click CLI or `python -m sol_execbench.core.runtime_evidence` | Add a schema export command. | Use Click if this is a user-facing `sol-execbench contract/schema` feature; use module argparse if scoped to Matrix tooling. |
| `tests/sol_execbench/test_contract.py`, `test_rocm_compatibility_matrix.py` | Assert schema export includes stable `$defs`, `schema_version`, Target/Matrix Entry fields, and forbidden authority claims. | This is the public downstream CI contract. |

Recommended exported schemas:

- `sol_execbench.rocm_compatibility_matrix.v1.entry.schema.json`
- `sol_execbench.rocm_compatibility_matrix.v1.report.schema.json`
- `sol_execbench.rocm_docker_targets.v1.schema.json` if downstream producers need target manifests.
- Include versioned filenames and schema IDs if implemented, but keep model field names unchanged.

Do not add a separate JSON Schema authoring layer. It would risk drift from Pydantic validation.

### Dataset-Runner Hardening

Use existing `scripts/run_dataset.py` first, with small helper extraction only where needed:

| Existing Surface | Extend With | Why |
|------------------|-------------|-----|
| `scripts/run_dataset.py` | Failure classification, resume/manifest consistency checks, closure stability fields. | The runner already owns discovery, execution, score sidecars, SOL/SOLAR sidecars, and `execution_closure.json`. |
| `EXECUTION_CLOSURE_SCHEMA_VERSION` and closure helpers | New closure statuses or reason-code subfields, not trace fields. | Keeps runner hardening outside canonical trace semantics. |
| `src/sol_execbench/core/dataset/checksums.py` | Manifest/ready-subset/closure checksum comparisons. | Resume consistency should be deterministic and auditable. |
| `tests/sol_execbench/test_run_dataset_execution_closure.py`, `test_run_dataset_amd_score.py`, `test_parity_gap_report.py` | Focused CPU-safe cases for classification, resume mismatch, stale manifest, missing evidence, stable output ordering. | This is the highest-risk module for v1.19 because it is large and orchestrates many sidecars. |

Recommended hardening fields:

- `failure_classification`: bounded category, reason code, command exit status, and truncated stderr/log ref.
- `resume_consistency`: expected and observed checksums for dataset manifest, ready subset, readiness, solution mode, and selected categories.
- `closure_inputs`: stable paths and checksums for all sidecars used to build closure.
- `closure_status_reason`: separate from existing `closure_status` so existing status vocabulary does not become overloaded.

Only extract a new module if it reduces risk, for example `src/sol_execbench/core/dataset/execution_closure.py` for pure closure models/helpers. Do not rewrite the runner or move CLI execution in this milestone.

### AMD SOL/SOLAR Bound Sanity Checks

Use existing scoring modules and existing RDNA 4/Docker evidence only:

| Existing Surface | Extend With | Why |
|------------------|-------------|-----|
| `src/sol_execbench/core/scoring/amd_sol_v2.py` | Sanity-check helpers over bound artifacts: finite values, non-negative values, aggregate consistency, limiting-resource consistency, coverage/warning consistency. | This file already owns v2 AMD SOL bound sidecars and aggregate semantics. |
| `src/sol_execbench/core/scoring/solar_derivation.py` | Cross-check SOLAR derivation evidence against AMD SOL v2 where both are present. | v1.19 is about clarifying provisional model risk, not changing bound derivation. |
| `src/sol_execbench/data/amd_hardware_models/gfx1200.json` and `default_amd_hardware_models()` | Keep checks scoped to existing `gfx1200` RDNA 4 model evidence. | The milestone explicitly excludes new CDNA 3, MI300X, or CDNA 4 validation scope. |
| `scripts/run_dataset.py` | Optional `--bound-sanity-report` or generated report when SOL/SOLAR dirs are supplied. | Runner already writes AMD SOL and SOLAR sidecars. |
| `tests/sol_execbench/test_amd_sol_v2.py`, `test_solar_derivation_evidence.py`, `test_run_dataset_amd_score.py`, docs guardrail tests | Add sanity report and claim-boundary cases. | CPU-safe sanity checks can be fixture-driven. |

Recommended sidecar:

- `sol_execbench.amd_bound_sanity.v1`
- Inputs: AMD SOL v2 sidecars, SOLAR derivation sidecars, optional trace status, hardware model ref, Matrix/Docker evidence refs.
- Outputs: `passed`, `warning`, `failed`, or `not_applicable`; bounded reason codes; source artifact refs; claim boundary flags all false for score authority, paper parity, leaderboard authority, and new hardware validation.

Do not add new performance modeling libraries or hardware probes. The purpose is sanity evidence over existing artifacts, not a new bound model.

## Development Tools

| Tool | Purpose | Notes |
|------|---------|-------|
| `uv` | Existing dependency and command runner. | Keep `uv.lock` unchanged unless implementation discovers an unavoidable dependency, which research does not recommend. |
| `pytest` | Contract and guardrail verification. | New v1.19 tests should be CPU-safe by default. Use `requires_rocm`/`requires_rdna4` only for optional evidence replay. |
| `bash -n scripts/run_docker.sh` | Shell syntax check if wrapper flags change. | Any `run_docker.sh` edit should keep existing non-privileged Docker behavior. |
| `uv run ty check` | Type checking. | Recommended after adding typed helper modules. |
| `uv run ruff check .` | Lint. | No formatter/config changes needed. |

## Installation

No dependency installation changes are recommended.

```bash
# Existing project setup remains sufficient
uv sync --all-groups

# Existing focused verification style
uv run pytest tests/sol_execbench/test_rocm_compatibility_matrix.py tests/sol_execbench/test_runtime_evidence_reports.py -q
uv run pytest tests/sol_execbench/test_dataset_inventory_readiness.py tests/sol_execbench/test_run_dataset_execution_closure.py tests/sol_execbench/test_parity_gap_report.py -q
uv run pytest tests/sol_execbench/test_amd_sol_v2.py tests/sol_execbench/test_solar_derivation_evidence.py tests/sol_execbench/test_run_dataset_amd_score.py -q
```

## Alternatives Considered

| Recommended | Alternative | Why Not for v1.19 |
|-------------|-------------|-------------------|
| Pydantic `model_json_schema()` | Hand-authored JSON Schema files | Hand-authored schemas can drift from runtime validation. |
| Domain-specific Matrix diff models | Generic deep-diff library | Matrix fields have domain meaning; generic diffs cannot classify claim-boundary, status, dependency, and image changes cleanly. |
| Deterministic JSON aggregation | Pandas/DataFrame reporting | Adds dependency weight for small bounded sidecars and weakens public schema discipline. |
| Existing script/module CLIs | New service or dashboard | v1.19 needs reproducible artifacts, not hosted infrastructure. |
| Existing RDNA 4/Docker evidence refs | New hardware validation harness | New hardware is explicitly out of scope. |
| Small helper extraction from `run_dataset.py` | Broad runner rewrite | The runner is a known large orchestration hotspot; broad rewrites increase regression risk. |

## What NOT to Use

| Avoid | Why | Use Instead |
|-------|-----|-------------|
| New hardware validation scope, MI300X/CDNA 3/CDNA 4 claims | The milestone says no new hardware validation. | Existing RDNA 4 and Docker evidence with explicit claim boundaries. |
| New database, warehouse, or service | Reports are bounded sidecars and should remain reproducible in the repo/artifact directory. | Versioned JSON sidecars plus Markdown summaries. |
| New package dependencies for diffing/reporting | Existing Python stdlib and Pydantic are enough. | Domain-specific helpers using `json`, `pathlib`, `difflib`, and Pydantic. |
| Mutating `Trace`, `Definition`, `Workload`, score authority, or benchmark timing schemas | Existing concerns flag sidecar proliferation and public contract boundaries. | Keep all v1.19 evidence sidecar-only. |
| Docker privilege expansion | Matrix evidence currently avoids `--privileged` and records container user-space scope. | Keep existing `/dev/kfd`, `/dev/dri`, group, seccomp, and IPC behavior. |
| Treating bound sanity as validation or SOLAR equivalence | Sanity checks can only identify provisional model risk over existing artifacts. | Use `diagnostic_only`/authority-false claim fields. |
| Re-locking PyTorch ROCm versions | v1.19 does not need runtime stack changes. | Keep `torch==2.10.0+rocm7.1`, `torchvision==0.25.0+rocm7.1`, `triton-rocm==3.6.0` project defaults and Docker target policies as-is. |

## Stack Patterns by Variant

**If implementing a new public sidecar contract:**
- Use Pydantic v2 models with `extra="forbid"`, `strict=True`, explicit `schema_version`, bounded enums/reason codes, deterministic JSON serialization, and contract tests.
- Because downstream consumers need stable validation and schema export.

**If adding a user-facing command:**
- Use the existing Click `sol-execbench` CLI for broad public commands, or module-level argparse for narrow tooling beside `runtime_evidence.py` and `docker_matrix.py`.
- Because the repo already uses both patterns and migration would not add value.

**If integrating with dataset runs:**
- Extend `scripts/run_dataset.py` with pure helpers or extract `core/dataset/execution_closure.py` only for testable closure logic.
- Because dataset-runner behavior is high-risk and should change incrementally.

**If evidence is diagnostic:**
- Write a sidecar and docs guardrail; do not add canonical trace fields or scoring semantics.
- Because the project has explicit claim boundaries for Matrix, runtime evidence, AMD score, AMD SOL, SOLAR, and static evidence.

## Version Compatibility

| Component | Compatible With | Notes |
|-----------|-----------------|-------|
| Python `>=3.12,<3.14` | Existing package and CI matrix | Keep helpers compatible with Python 3.12 and 3.13. |
| Pydantic `>=2.12.5` | Existing strict model contracts | Use v2 APIs such as `model_validate`, `model_dump(mode="json")`, and `model_json_schema()`. |
| PyTorch `2.10.0+rocm7.1` project default | ROCm `7.1` default target | Do not change for v1.19. Matrix diff/reporting should record dependency changes, not require them. |
| Docker target `rocm-7.0.2`, `rocm-7.1.1`, `rocm-7.2.0` | `docker/rocm-targets.json` | Keep target policy as source of truth. Diff tooling should compare reports across these targets without adding target entries. |
| Triton ROCm `3.6.0` | Existing PyTorch ROCm wheel policy | No stack change needed for denominator, Matrix, closure, or sanity reports. |
| pytest `>=9.0.2` + xdist `>=3.5` | Existing test suite | New tests must be deterministic under parallel execution. |

## Recommended Integration Points

| Feature | Primary Files | CLI / Report Surface | Tests |
|---------|---------------|----------------------|-------|
| Paper denominator accounting | `src/sol_execbench/core/dataset/parity_gap.py`, possible `paper_denominator.py`, `readiness.py`, `inventory.py` | Dataset report command or `scripts/run_dataset.py` post-processing path | `test_dataset_inventory_readiness.py`, `test_parity_gap_report.py`, new denominator tests |
| Matrix diff | `src/sol_execbench/core/compatibility.py`, `src/sol_execbench/core/runtime_evidence.py` | `python -m sol_execbench.core.runtime_evidence diff ...` or `sol-execbench matrix diff ...` | `test_rocm_compatibility_matrix.py`, `test_runtime_evidence_reports.py`, docs guardrails |
| Matrix schema export | `compatibility.py`, `docker_matrix.py`, `core/data/contract.py` | `sol-execbench contract --json` extension or `schema export` command | `test_contract.py`, `test_rocm_compatibility_matrix.py` |
| Dataset-runner hardening | `scripts/run_dataset.py`, possible `core/dataset/execution_closure.py`, `checksums.py` | Existing `run_dataset.py` flags plus optional stricter resume/manifest checks | `test_run_dataset_execution_closure.py`, `test_run_dataset_amd_score.py` |
| AMD SOL/SOLAR sanity | `amd_sol_v2.py`, `solar_derivation.py`, `amd_hardware_models.py`, `scripts/run_dataset.py` | Optional bound sanity sidecar/report emitted alongside AMD SOL/SOLAR dirs | `test_amd_sol_v2.py`, `test_solar_derivation_evidence.py`, `test_run_dataset_amd_score.py` |

## Sources

- `.planning/PROJECT.md` - v1.19 target features and no-new-hardware scope.
- `.planning/REQUIREMENTS.md` - current deferred Matrix tooling requirements and absence of active requirements.
- `.planning/ROADMAP.md` - current milestone position.
- `.planning/codebase/STACK.md` - existing Python, Pydantic, Click/Rich, PyTorch ROCm, Triton ROCm, Docker, pytest, Ruff, Ty stack.
- `.planning/codebase/CONCERNS.md` - large runner risk, sidecar boundary, Matrix claim boundaries, Docker diagnostic scope, and docs guardrail risks.
- `pyproject.toml` - dependency versions and dev tooling.
- `docker/rocm-targets.json` - declared Docker ROCm targets and PyTorch ROCm policy.
- `src/sol_execbench/core/compatibility.py` - Matrix Entry/report contracts.
- `src/sol_execbench/core/runtime_evidence.py` - runtime Matrix entry and aggregate report writer.
- `src/sol_execbench/core/docker_matrix.py` - Docker target manifest and preflight integration.
- `src/sol_execbench/core/dataset/readiness.py` and `parity_gap.py` - existing dataset readiness and denominator/gap reporting.
- `scripts/run_dataset.py` - execution closure, AMD score, AMD SOL v2, and SOLAR sidecar integration.
- `src/sol_execbench/core/scoring/amd_sol_v2.py` and `solar_derivation.py` - existing bound and SOLAR evidence contracts.

---
*Stack research for: v1.19 Research Credibility Without New Hardware*
*Researched: 2026-05-31*
