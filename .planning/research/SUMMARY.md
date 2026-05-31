# Project Research Summary

**Project:** SOL ExecBench ROCm Port
**Domain:** ROCm benchmark research-credibility evidence infrastructure
**Researched:** 2026-05-31
**Confidence:** HIGH for repo-local stack, features, architecture, and pitfalls; MEDIUM for downstream external-CI schema needs and exact roadmap slicing

## Executive Summary

v1.19 is a credibility milestone for an existing ROCm-only GPU benchmark port, not a runtime expansion. Expert implementations in this space preserve canonical benchmark contracts and add auditable, versioned evidence sidecars around them: strict schemas, deterministic JSON/Markdown reports, source checksums, bounded status vocabularies, and claim guardrails. The research consensus is clear: do not add CDNA 3, MI300X, CDNA 4, native-host validation, new services, databases, or new package dependencies for this milestone.

The recommended approach is to build additive reporting and contract layers over existing v1.11 dataset inventory/readiness/closure artifacts, v1.18 ROCm Compatibility Matrix Entries, and v1.9-v1.10 AMD SOL/SOLAR sidecars. Use Python 3.12+, Pydantic v2, stdlib JSON/path/checksum helpers, existing Click/argparse entry points, pytest, Ruff, and Ty. Keep `Trace`, correctness, timing, evaluator subprocess behavior, and score authority unchanged.

The main risk is overclaiming. A denominator report can look like paper parity, Docker Matrix rows can be mistaken for native-host validation, and AMD SOL/SOLAR sanity checks can be read as model validation. Mitigate this with explicit authority-false fields, separate ready/blocked/unsupported/deferred/evidence-missing buckets, semantic Matrix diffs, provenance checks for resumed traces, and docs/tests that enforce no paper parity, no leaderboard authority, no native-host upgrade from Docker evidence, and no new hardware validation.

## Key Findings

### Recommended Stack

No new external runtime dependencies are recommended. v1.19 should use the existing Python/Pydantic/reporting stack to produce deterministic sidecars and human summaries. Pydantic models should remain the source of truth for public contracts, and JSON Schema export should be generated from those models rather than maintained by hand.

**Core technologies:**
- Python `>=3.12,<3.14`: pure report builders, CLIs, checksums, and tests.
- Pydantic `>=2.12.5`: strict sidecar contracts, diff models, denominator reports, sanity reports, and `model_json_schema()` export.
- JSON sidecars: machine-readable evidence that stays outside canonical trace JSONL.
- Markdown reports: reviewer-facing summaries with explicit claim boundaries.
- Click / argparse: expose public package commands through existing Click CLI where appropriate; keep narrow dataset/Docker tooling beside existing argparse scripts.
- Rich: optional readable CLI tables; JSON remains the primary automation output.
- pytest, Ruff, Ty: CPU-safe contract coverage, formatting/linting, and typed helper APIs.

**Do not add:** pandas, DuckDB, SQLite, generic deep-diff libraries, hosted services, dashboards, new hardware probes, Docker privilege expansion, or PyTorch/ROCm dependency relocking.

### Expected Features

v1.19 should let researchers audit what exists, what changed, what was attempted, and what remains unsupported or missing without inflating authority.

**Must have (table stakes):**
- Paper dataset denominator report with JSON and Markdown outputs, problem/workload rollups, source refs/checksums, and ready/blocked/unsupported/deferred/evidence-missing classifications.
- Stable status and reason-code vocabulary that distinguishes missing evidence from benchmark failure.
- Dataset-runner resume/manifest consistency checks, failure classification, and deterministic closure outputs.
- Compatibility Matrix diff tooling with machine JSON and human summaries for status, dependency, image, runtime, clock/evidence, artifact, and claim-boundary changes.
- Compatibility Matrix JSON Schema export for `MatrixEntry` and `RocmCompatibilityMatrixReport`.
- AMD SOL/SOLAR bound sanity checks over existing RDNA 4/Docker evidence only, with provisional model-risk language.
- Claim-boundary guardrails and CPU-safe tests for every new report.

**Should have (competitive):**
- Denominator ledger with problem-level and workload-level rollups by category, op family, readiness status, closure status, and evidence gap.
- Claim-safe upgrade-readiness hints that say what evidence is missing before stronger claims are possible.
- Severity-ranked Matrix diffs that highlight status downgrades, claim-boundary escalation, dependency/image drift, runtime unavailability, and GPU arch drift.
- Schema bundle manifest for downstream evidence producers if external CI needs it.
- Dataset-runner preflight mode for closure consistency.
- Cross-report consistency checks spanning denominator, Matrix, closure, AMD score, SOL/SOLAR, and docs.

**Defer (v2+ / separate milestone):**
- Full public 235-problem real-hardware validation.
- Native host ROCm matrix validation.
- CDNA 3, MI300X, or CDNA 4 live validation.
- Upstream SOLAR equivalence comparison.
- Hosted leaderboard or remote submission workflow.
- Original 124-model / 7,400-subgraph extraction reproduction.

### Architecture Approach

The architecture should remain sidecar-first and additive. New contracts belong near their owning evidence domains, with scripts acting as adapters and core package modules owning policy, validation, and deterministic serialization. Canonical traces remain read-only inputs for reports; they must not gain denominator, Matrix, or sanity fields.

**Major components:**
1. `core/dataset/execution_closure.py` — extracted closure models, status vocabulary, failure classification, provenance checks, totals, and deterministic output helpers.
2. `core/dataset/denominator.py` — paper denominator accounting over inventory, readiness, ready subset, closure, score, SOL, and SOLAR sidecars.
3. `core/compatibility_diff.py` — semantic ROCm Compatibility Matrix diff models and comparison logic.
4. Compatibility schema helper/exporter — `model_json_schema()` output for public Matrix contracts and diff reports.
5. `core/scoring/amd_sanity.py` — diagnostic AMD SOL/SOLAR sanity evidence over existing RDNA 4/Docker artifacts.
6. `scripts/run_dataset.py` adapter changes — minimal wiring for hardened closure, classification, resume checks, and optional report emission.
7. Docs and guardrail tests — public wording and authority boundaries for denominator, Matrix diff/schema, closure, and bound sanity artifacts.

### Critical Pitfalls

1. **Turning denominator accounting into paper parity** — keep denominator accounted-for separate from validation; include explicit false authority flags and separate status buckets.
2. **Collapsing statuses into one skip bucket** — preserve ready, blocked, unsupported, deferred, evidence-missing, attempted-passed, and attempted-failed with reason codes and separate problem/workload rollups.
3. **Creating conflicting sidecar sources of truth** — define artifact ownership and use refs/checksums in aggregate reports instead of duplicating full payloads.
4. **Destabilizing `scripts/run_dataset.py`** — extract policy into tested core helpers and keep the runner’s default behavior unchanged unless closure/hardening options require new diagnostics.
5. **Reusing stale traces during resume** — accept `skipped_existing_pass` only when manifest, ready subset, config, workload identity, solution mode, and evidence requirements match.
6. **Producing noisy or misleading Matrix diffs** — compare validated models semantically, normalize volatile metadata, and severity-rank status, dependency, image, runtime, and authority changes.
7. **Letting AMD SOL/SOLAR sanity become model validation** — report degraded, unsupported, unscored, and missing-evidence counts; keep upstream SOLAR parity and new hardware validation false.
8. **Leaking sensitive evidence details** — use relative refs, checksums, bounded logs, and redaction; do not embed tokens, raw datasets, proprietary kernels, or unnecessary absolute paths.

## Implications for Roadmap

Based on research, suggested phase structure:

### Phase 1: Closure Contracts And Provenance Foundation

**Rationale:** Denominator accounting depends on reliable execution-closure inputs, and `scripts/run_dataset.py` is the highest-regression-risk module.

**Delivers:** Extracted `execution_closure` core module, strict closure record/report models, status vocabulary, failure classification enums, checksum/provenance validators, deterministic totals, and unit tests.

**Addresses:** Reproducible closure outputs, resume/manifest consistency checks, failure/evidence classification foundations.

**Avoids:** Runner destabilization, stale trace reuse, skip-bucket collapse, and sidecar drift.

### Phase 2: Paper Denominator Accounting And Claim Boundaries

**Rationale:** This is the core v1.19 research-credibility deliverable and should build on stable closure semantics.

**Delivers:** `paper_denominator_report.v1` JSON/Markdown reports with source refs/checksums, problem/workload/category rollups, explicit status/reason-code taxonomy, evidence gaps, and authority-false fields.

**Addresses:** Paper denominator status report, stable denominator vocabulary, evidence-missing separation, claim guardrails.

**Avoids:** Paper-parity overclaims, hidden unsupported/deferred buckets, and evidence gaps being counted as failures or successes.

### Phase 3: Compatibility Matrix Schema Export And Semantic Diff

**Rationale:** Schema export and diff share the existing Matrix Pydantic contract, and both are CPU-safe infrastructure work. Export first to publish the contract; diff second to compare validated reports.

**Delivers:** Versioned JSON Schema files for public Matrix contracts, optional schema manifest, semantic Matrix diff JSON/Markdown, severity-ranked transitions, and tests for external payload boundaries.

**Uses:** Pydantic `model_json_schema()`, stdlib JSON/path helpers, existing `MatrixEntry` and `RocmCompatibilityMatrixReport`.

**Avoids:** Raw noisy diffs, accidental native-host authority from Docker rows, internal model leakage, and dependency drift being misclassified as benchmark failure.

### Phase 4: Dataset Runner Hardening Integration

**Rationale:** After closure policy is tested in core modules, runner changes can be contained to adapter wiring and mode-specific behavior.

**Delivers:** Hardened `scripts/run_dataset.py` behavior for resume consistency, manifest/ready-subset mismatch detection, `skipped_existing_pass` provenance, log refs, no-trace classification, missing derived evidence, and stable closure writes.

**Addresses:** Dataset-runner consistency, failure classification, reproducible closure outputs, evidence hygiene.

**Avoids:** Default workflow regressions, stale output reuse, broad main-loop rewrites, and sensitive log/path leakage.

### Phase 5: AMD SOL/SOLAR Bound Sanity Evidence

**Rationale:** Bound sanity is a diagnostic capstone over existing AMD sidecars and should consume stable closure/denominator evidence rather than inventing new authority.

**Delivers:** `amd_bound_sanity.v1` report with workload checks, status counts, degraded/unscored/unsupported/missing-evidence buckets, model-risk summary, source artifact refs, and explicit no-new-hardware claim boundary.

**Addresses:** AMD SOL/SOLAR sanity on existing RDNA 4/Docker evidence, provisional model-risk reporting, score eligibility guardrails.

**Avoids:** Upstream SOLAR equivalence claims, model-validation overclaims, hidden unscored workloads, and CDNA 3/MI300X/CDNA4 scope creep.

### Phase 6: Documentation, Examples, And Guardrail Tests

**Rationale:** Docs should reflect stable artifact contracts and are part of the product behavior for this milestone.

**Delivers:** Updates to `docs/CLAIMS.md`, `docs/TESTING.md`, researcher docs, schema docs, report examples, and CPU-safe wording tests for denominator, Matrix diff/schema, runner closure, and bound sanity artifacts.

**Addresses:** Claim-boundary guardrails, external producer expectations, no-new-hardware scope, evidence hygiene.

**Avoids:** Documentation overclaims, artifact examples without boundaries, and downstream misuse of new reports.

### Phase Ordering Rationale

- Closure contracts come first because denominator and runner-hardening credibility depend on stable provenance and status semantics.
- Denominator accounting should land before docs wording because it is the milestone’s primary claim surface and determines the vocabulary guardrails.
- Matrix schema export and diff are isolated from dataset execution and can proceed as CPU-safe contract work.
- Runner integration follows core helper extraction to keep changes in the large script mechanical and testable.
- AMD SOL/SOLAR sanity comes after closure/denominator foundations so it can report exclusions and evidence gaps honestly.
- Documentation and guardrails close the milestone after artifact semantics are fixed.

### Research Flags

Phases likely needing deeper research during planning:
- **Phase 4:** `scripts/run_dataset.py` has broad orchestration behavior; planning should inspect current resume, trace loading, score/SOL/SOLAR, and closure paths in detail.
- **Phase 5:** AMD SOL/SOLAR sanity needs careful review of existing sidecar semantics, degraded statuses, and score eligibility wording.
- **Phase 6:** Docs guardrails need current wording audits across claims/testing/researcher docs and tests.

Phases with standard patterns (skip research-phase):
- **Phase 1:** Pydantic sidecar contracts, deterministic serialization, and checksum/provenance helpers are established repo patterns; implementation needs code inspection, not external research.
- **Phase 2:** Dataset inventory/readiness/ready-subset/parity patterns are already local and well documented.
- **Phase 3:** Pydantic schema export and semantic domain diffing are bounded, CPU-safe, and rely on existing Matrix models.

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Stack | HIGH | Repository stack, dependency versions, CLI patterns, Pydantic contracts, and test tools are directly supported by local files. |
| Features | HIGH | Target features match `.planning/PROJECT.md` and existing v1.11/v1.18/v1.9-v1.10 artifact surfaces; external-CI schema details remain MEDIUM. |
| Architecture | HIGH | Integration points are clear and align with existing sidecar-first architecture; exact phase count is MEDIUM because roadmap may merge or split work. |
| Pitfalls | HIGH | Risks are grounded in existing docs, tests, and known hot spots such as runner complexity, claim guardrails, Docker/native-host boundaries, and AMD SOL/SOLAR authority. |

**Overall confidence:** HIGH for the recommended roadmap direction; MEDIUM for exact downstream schema bundle requirements and final phase slicing.

### Gaps to Address

- External CI producer expectations are not yet concrete; start with Matrix Entry/report schema export and defer broader schema bundles until a consumer requires them.
- Exact denominator rollup rules for mixed-status problems need implementation-time decisions and tests, especially when workloads within one problem have different readiness or evidence states.
- Runner provenance comparison fields must be chosen by inspecting current output metadata and resume behavior so hardening does not break non-closure runs.
- AMD SOL/SOLAR sanity thresholds and reason-code names need fixture-driven validation against existing RDNA 4/Docker sidecars.
- Documentation examples must be checked after implementation so wording matches actual report fields and commands.

## Sources

### Primary (HIGH confidence)

- `.planning/PROJECT.md` — v1.19 scope, no-new-hardware constraint, target features.
- `.planning/research/STACK.md` — recommended stack, dependency/version boundaries, integration points.
- `.planning/research/FEATURES.md` — table stakes, differentiators, anti-features, acceptance criteria.
- `.planning/research/ARCHITECTURE.md` — system shape, component boundaries, data flow, build order.
- `.planning/research/PITFALLS.md` — critical risks, phase mapping, evidence hygiene, recovery strategies.
- `.planning/codebase/STACK.md` — existing Python, Pydantic, Click/Rich, PyTorch ROCm, Triton ROCm, Docker, pytest, Ruff, and Ty stack.
- `.planning/codebase/CONCERNS.md` — runner risk, sidecar proliferation, docs guardrails, Docker/native-host claim boundaries.
- `pyproject.toml` — dependency versions and development tooling.
- `src/sol_execbench/core/compatibility.py` — Matrix contracts and claim-boundary fields.
- `src/sol_execbench/core/dataset/` — inventory, readiness, ready-subset, parity/checksum patterns.
- `scripts/run_dataset.py` — dataset execution, resume behavior, AMD score, SOL/SOLAR, timing, and closure integration.
- `src/sol_execbench/core/scoring/amd_sol_v2.py` and `solar_derivation.py` — AMD SOL/SOLAR evidence contracts.
- `docs/CLAIMS.md`, `docs/TESTING.md`, `docs/original_parity.md`, `docs/RESEARCHER-GUIDE.md` — public claim boundaries and evidence wording.

### Secondary (MEDIUM confidence)

- Downstream external-CI and evidence-producer needs — inferred from Matrix schema export requirements, not yet from a concrete external consumer.
- Exact phase split — inferred from dependency structure and risk grouping; roadmap may choose smaller implementation phases.

---
*Research completed: 2026-05-31*
*Ready for roadmap: yes*
