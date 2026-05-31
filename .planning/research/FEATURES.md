# Feature Research

**Domain:** ROCm benchmark research-credibility infrastructure
**Milestone:** v1.19 Research Credibility Without New Hardware
**Researched:** 2026-05-31
**Confidence:** HIGH for local feature shape; MEDIUM for downstream external-CI schema needs

## Feature Landscape

v1.19 should improve how researchers audit existing evidence. It should not
expand hardware scope, score authority, paper parity, leaderboard authority, or
native host validation. The useful product is a stricter accounting and
reporting layer over evidence that already exists: v1.11 dataset inventory,
readiness, ready-subset, execution-closure, parity-gap reports; v1.18 Docker
compatibility Matrix Entries; and v1.9-v1.10 AMD SOL/SOLAR sidecars.

The most important behavior is explicit denominator accounting. A researcher
should be able to answer: "Of the paper/public benchmark denominator available
to this repo, which problems are ready, blocked, unsupported, deferred, missing
evidence, attempted, passed, failed, or unattempted, and why?" The answer must
be machine-readable and must also carry claim-boundary fields showing that this
is not full paper parity unless the full denominator and required evidence are
actually complete.

### Table Stakes (Users Expect These)

Features users assume exist. Missing these = product feels incomplete.

| Feature | Why Expected | Complexity | Notes |
|---------|--------------|------------|-------|
| Paper dataset denominator status report | Benchmark/reproducibility users need a complete count of the paper/public dataset surface before trusting any subset claim. | MEDIUM | Build from existing manifest, inventory, readiness, ready-subset, execution-closure, and parity-gap data. Must classify problem and workload denominators into ready, blocked, unsupported, deferred, missing evidence, attempted, passed, failed, skipped, filtered, and not attempted without mutating canonical dataset files. |
| Stable denominator status vocabulary | Counts are only credible if statuses are deterministic and testable. | MEDIUM | Reuse current statuses where possible: readiness statuses such as `ready`, `runtime_blocked`, `unsupported_nvidia_only_path`, `needs_hardware_evidence`, and closure statuses such as `attempted_passed`, `attempted_failed`, `missing_trace`, `derived_evidence_missing`, `not_attempted`, `filtered`, `skipped_existing_pass`. Add new labels only when they cannot be represented by existing fields. |
| Evidence-missing classification separated from failure | Missing AMD SOL/SOLAR/timing/static/runtime evidence is not the same as kernel failure. | MEDIUM | Treat missing sidecars, missing traces, missing score inputs, and missing compatibility evidence as explicit evidence gaps. Do not convert them into correctness failures or zero-cost score inputs. Existing `derived_evidence_missing` behavior is the pattern to extend. |
| Compatibility Matrix diff tooling | Matrix reports are useful only if researchers can see what changed between runs. | MEDIUM | Diff two `sol_execbench.rocm_compatibility_matrix.v1` reports by Target id and validation scope. Highlight status, reason code, requested Target values, observed host/container/Python/toolchain/GPU fields, dependency policy, clock policy evidence when present, image repository/tag/digest, and artifact references. |
| Compatibility diff machine output and human summary | CI and reviewers need both JSON for automation and readable Markdown/text for PR review. | MEDIUM | JSON should include added/removed/changed/unchanged entries, per-field changes, status transitions, claim-boundary changes, and severity. Human output should summarize high-risk transitions such as `container_validated -> mixed_version`, image digest drift, dependency drift, and any attempted `native_host_validated` upgrade. |
| Compatibility JSON Schema export | External evidence producers need a contract they can validate without importing Python models. | LOW-MEDIUM | Export JSON Schema for `MatrixEntry` and `RocmCompatibilityMatrixReport` from the Pydantic models. Include schema id/version metadata and forbid extra fields in the schema. This should not change the Matrix payload schema version unless the payload fields change. |
| Dataset runner resume/manifest consistency checks | Long dataset batches need repeatable closure outputs, and resume mistakes are easy to miss. | HIGH | Harden `scripts/run_dataset.py` around ready-subset checksum, readiness checksum, manifest checksum, problem ids, workload uuid/row-index matching, output directory reuse, `--rerun`, `--limit`, `--max-workloads`, and skipped-existing-pass semantics. Mismatches should be explicit blocked/diagnostic states, not silent reuse. |
| Dataset runner failure classification | Researchers need to distinguish schema/input blockers, build failures, runtime failures, correctness failures, timeouts, missing traces, and missing derived evidence. | HIGH | Extend closure records or supporting report fields with stable reason codes and log refs. Preserve canonical trace JSONL semantics; classification belongs in closure/report sidecars. |
| Reproducible closure outputs | The same inputs and sidecars should produce stable counts and deterministic ordering. | MEDIUM | Keep sorted records, stable checksums, source path refs, git commit, command line, filters, selected categories, and claim-boundary fields. Tests should pin output shape on small fixtures. |
| AMD SOL/SOLAR bound sanity checks for existing RDNA 4/Docker evidence | The milestone goal includes clarifying provisional model risk without adding hardware. | MEDIUM-HIGH | Check current RDNA 4 `gfx1200` artifacts and Docker rows for internally consistent hardware model refs, aggregate statuses, coverage summaries, warnings, score eligibility, and validation-status wording. This is a sanity audit, not new model validation. |
| Claim-boundary guardrails for every new report | Existing docs make claim boundaries part of the product contract. | MEDIUM | New reports and docs must explicitly set/retain false authority for paper parity, score authority where not applicable, leaderboard authority, upstream SOLAR parity, native host validation, CDNA 3/MI300X/CDNA4 validation, and new hardware validation. |
| CPU-safe tests for report generation | Most milestone value is schema/report logic and should be covered without live ROCm. | MEDIUM | Add fixture-driven unit tests for denominator accounting, Matrix diffing, schema export, closure consistency checks, failure classification, and docs guardrails. Live RDNA 4 evidence can be referenced or smoke-checked only where already available. |

### Differentiators (Competitive Advantage)

Features that set the project apart. Not required for a minimal audit layer, but
valuable for research credibility.

| Feature | Value Proposition | Complexity | Notes |
|---------|-------------------|------------|-------|
| Denominator ledger with problem-level and workload-level rollups | Makes the ROCm port unusually honest about what has and has not been evaluated. | HIGH | A ledger-style JSON/Markdown report should join paper/public denominator, readiness, closure, and derived evidence. Rollups by category, op family, readiness status, closure status, and evidence gap make claims review easier. |
| Claim-safe "upgrade readiness" hints | Shows exactly what evidence is still needed before a stronger claim can be made. | MEDIUM | For each blocked/deferred/evidence-missing bucket, emit next actions such as acquire dataset asset, port NVIDIA-only reference path, collect missing trace, generate AMD SOL sidecar, or run full denominator. Hints must not imply the stronger claim is already met. |
| Severity-ranked Matrix diffs | Helps reviewers focus on compatibility regressions instead of raw JSON churn. | MEDIUM | Treat status downgrade, claim-boundary escalation, target mismatch, dependency drift, runtime unavailability, image digest drift, and GPU arch drift as higher severity than timestamp or artifact-list-only changes. |
| Schema bundle export for evidence producers | Makes downstream CI integration easier than asking consumers to import project internals. | MEDIUM | Provide a command that writes a directory containing Matrix Entry/report schemas and a small manifest of schema names, versions, model class, and source commit. Useful for independent Docker runners or hosted artifact collectors later. |
| Dataset runner preflight mode for closure consistency | Prevents wasted long runs by checking manifests, ready subsets, output reuse, and selected workload refs before execution. | MEDIUM | A `--preflight-closure` or similar mode can validate sidecar coherence and emit planned closure counts without running kernels. This is especially useful under the no-new-hardware constraint. |
| Bound sanity report with model-risk language | Converts AMD SOL/SOLAR uncertainty into a concrete audit artifact. | MEDIUM-HIGH | Report counts for `scored`, `degraded`, `unscored`, missing hardware model, provisional hardware/model validation, unsupported operation families, and warnings across the existing RDNA 4 evidence. Wording must stay at "sanity" or "risk" level, not validation parity. |
| Cross-report consistency checks | Catches contradictions between denominator, Matrix, closure, score, and docs claims. | HIGH | Examples: a report cannot mark paper denominator complete while closure has `not_attempted`; a Docker row cannot imply native host validation; a Matrix diff cannot treat mixed-version override output as clean validation; an AMD score report cannot hide unscored workloads. |
| Researcher-facing compact Markdown summaries | Makes artifacts usable in papers, PRs, and release notes without hand-summarizing JSON. | LOW-MEDIUM | Summaries should include denominator counts, evidence gaps, Matrix transitions, bound sanity outcomes, explicit no-new-hardware scope, and links to source sidecars. |

### Anti-Features (Commonly Requested, Often Problematic)

Features that seem useful but would weaken the milestone.

| Feature | Why Requested | Why Problematic | Alternative |
|---------|---------------|-----------------|-------------|
| Claiming paper parity from denominator accounting | A complete-looking denominator table is tempting to cite as paper validation. | Denominator accounting is audit metadata. It does not prove execution, scoring, upstream SOLAR equivalence, original 124-model extraction, or official leaderboard comparability. | Report "paper/public dataset denominator status" and keep `paper_parity` / `paper_parity_authority` false unless future evidence satisfies the claim-upgrade rule. |
| Adding CDNA 3, MI300X, CDNA 4, or native host validation rows | Broader hardware coverage looks more credible. | The milestone explicitly forbids new hardware validation scope. Schema support or Docker evidence cannot become live hardware validation. | Keep those rows deferred/not validated unless archived real-hardware evidence already exists; focus on RDNA 4 and current Docker evidence. |
| Treating Docker Matrix Entries as native host validation | Docker rows are easier to collect than host reinstall validation. | Existing claims state Docker rows validate container ROCm user-space on recorded host driver/devices only. Native host validation needs direct native-host evidence. | Diff and report Docker rows as `container_user_space` scope; keep `native_host_validated=false`. |
| Making Matrix diff status affect benchmark correctness or score | Users may want regressions to fail benchmark runs. | Compatibility is diagnostic sidecar evidence. Coupling it to trace correctness/timing would change benchmark semantics. | Allow CI to fail on diff severity outside the benchmark run, while keeping trace JSONL, correctness, timing, and scoring unchanged. |
| Bumping canonical trace or problem schemas for credibility fields | New evidence fields may feel easier to consume if placed in traces. | Existing architecture keeps derived evidence sidecar-only. Changing trace schemas creates consumer drift and overstates authority. | Write sidecars/reports and reference canonical traces by path/checksum. |
| Auto-filling missing evidence with defaults | Makes reports look complete and avoids unscored rows. | Missing sidecars, missing timing, or unsupported bounds must not be treated as zero-cost, passed, or scored evidence. | Emit explicit `missing_*` gaps and `unscored` or `derived_evidence_missing` statuses. |
| Upgrading AMD SOL/SOLAR sanity checks into upstream SOLAR equivalence | Side-by-side equivalence would be a stronger research claim. | v1.19 has no upstream SOLAR comparison, no paper-scale extraction, and no new hardware validation. | Produce sanity/risk checks over local AMD-derived sidecars only; document evidence required for upstream SOLAR parity. |
| Full 235-problem execution as part of this milestone | It would make the denominator report more complete. | The milestone is "without new hardware" and focuses on infrastructure. Full execution requires dataset availability, runtime time, artifacts, and validation scope decisions. | Make full-denominator gaps visible and test tooling on fixtures plus existing evidence. |
| Hosted leaderboard/submission service | External CI schema export can look like a step toward hosted leaderboard support. | Leaderboard policy needs stable hardware, anti-cheat, baselines, submission format, and official scoring authority. | Export schemas for evidence validation only; keep leaderboard authority false. |
| Replacing existing v1.11/v1.18 sidecars with a new monolithic format | A single report sounds simpler. | It would duplicate contracts and increase migration risk. | Add joining reports and diff/schema-export tools that consume existing sidecars. |

## Feature Dependencies

```text
Existing dataset manifest/inventory/readiness
    └──requires──> Paper dataset denominator status report
                       └──enhances──> Parity gap report and closure docs

Existing ready-subset + execution_closure.v1
    └──requires──> Dataset-runner resume/manifest consistency checks
                       └──requires──> Reproducible closure outputs
                       └──enhances──> Denominator ledger

Existing rocm_compatibility_matrix.v1 models
    └──requires──> Compatibility Matrix diff tooling
    └──requires──> Compatibility JSON Schema export
                       └──enhances──> External CI/downstream evidence producers

Existing AMD SOL v2 + SOLAR derivation + AMD-native score reports
    └──requires──> AMD SOL/SOLAR bound sanity checks
                       └──enhances──> Denominator evidence-missing classification

Claim-boundary guardrails
    └──constrains──> All new reports, diffs, schemas, docs, and tests

No-new-hardware constraint
    └──conflicts──> CDNA 3/MI300X/CDNA4 validation, native host matrix validation,
                    hosted leaderboard authority, full paper parity claims
```

### Dependency Notes

- **Denominator reporting requires existing v1.11 sidecars:** The report should
  join, not replace, manifest, inventory, readiness, ready subset, execution
  closure, and parity gap artifacts.
- **Dataset-runner hardening is upstream of credible closure:** If resume or
  manifest mismatches are silent, the denominator report can be wrong even when
  counts look stable.
- **Matrix diff and schema export share the compatibility model:** Both should
  use `MatrixEntry` and `RocmCompatibilityMatrixReport` as the source of truth.
  JSON Schema export should be a contract publication layer, not a new schema.
- **Bound sanity checks depend on current AMD sidecars:** They should inspect
  hardware model refs, aggregate statuses, coverage summaries, warnings, and
  evidence refs already generated for RDNA 4/Docker evidence.
- **Claim guardrails constrain every feature:** A report can improve credibility
  only if it prevents overclaiming. Authority flags and docs wording are
  functional requirements, not prose polish.

## MVP Definition

### Launch With (v1.19)

Minimum viable milestone scope.

- [ ] Paper dataset denominator status report with JSON and Markdown outputs,
      problem/workload rollups, stable statuses, evidence gaps, source refs,
      and claim-boundary fields.
- [ ] Dataset runner consistency hardening for ready-subset/readiness/manifest
      checksums, workload ref matching, skipped-existing-pass handling, and
      deterministic closure ordering.
- [ ] Dataset runner failure/evidence classification that keeps missing trace,
      execution failure, correctness failure, runtime/build failure, and missing
      derived evidence distinct.
- [ ] Compatibility Matrix diff command/report with status transitions,
      dependency/image/runtime/evidence changes, severity, JSON output, and a
      human-readable summary.
- [ ] Compatibility JSON Schema export for Matrix Entry and Matrix Report,
      generated from strict Pydantic models.
- [ ] AMD SOL/SOLAR bound sanity report over existing RDNA 4/Docker evidence,
      with provisional model-risk wording and no new hardware claim.
- [ ] Documentation and guardrail tests preserving boundaries around paper
      parity, score authority, leaderboard authority, Docker evidence, native
      host validation, and deferred hardware.

### Add After Validation (v1.19.x or Later)

Useful but not required to make the milestone credible.

- [ ] Schema bundle manifest covering more sidecars than Matrix entries, once
      downstream consumers actually need them.
- [ ] CI policy helper that exits nonzero for selected Matrix diff severities,
      while keeping benchmark execution semantics unchanged.
- [ ] Cross-report contradiction linter spanning denominator, compatibility,
      score, SOL/SOLAR, and docs outputs.
- [ ] Richer Markdown tables grouped by op family, category, and evidence gap
      after initial denominator report shape stabilizes.

### Future Consideration (v2+ / Separate Milestone)

Explicitly defer these from v1.19.

- [ ] Full public 235-problem real-hardware validation.
- [ ] Native host ROCm 7.0.x / 7.1.x / 7.2.x matrix validation.
- [ ] CDNA 3 / MI300X / CDNA 4 live validation.
- [ ] Upstream SOLAR equivalence comparison.
- [ ] Hosted leaderboard or remote submission workflow.
- [ ] Original 124-model / 7,400-subgraph extraction reproduction.

## Feature Prioritization Matrix

| Feature | User Value | Implementation Cost | Priority |
|---------|------------|---------------------|----------|
| Paper dataset denominator status report | HIGH | MEDIUM | P1 |
| Stable denominator status vocabulary | HIGH | MEDIUM | P1 |
| Evidence-missing classification separated from failure | HIGH | MEDIUM | P1 |
| Dataset runner resume/manifest consistency checks | HIGH | HIGH | P1 |
| Dataset runner failure classification | HIGH | HIGH | P1 |
| Reproducible closure outputs | HIGH | MEDIUM | P1 |
| Compatibility Matrix diff tooling | HIGH | MEDIUM | P1 |
| Compatibility diff machine output and human summary | HIGH | MEDIUM | P1 |
| Compatibility JSON Schema export | MEDIUM | LOW-MEDIUM | P1 |
| AMD SOL/SOLAR bound sanity checks | HIGH | MEDIUM-HIGH | P1 |
| Claim-boundary guardrails for every new report | HIGH | MEDIUM | P1 |
| CPU-safe tests for report generation | HIGH | MEDIUM | P1 |
| Denominator ledger rollups by category/op family/evidence gap | MEDIUM | HIGH | P2 |
| Claim-safe upgrade-readiness hints | MEDIUM | MEDIUM | P2 |
| Severity-ranked Matrix diffs | MEDIUM | MEDIUM | P2 |
| Schema bundle export for external producers | MEDIUM | MEDIUM | P2 |
| Dataset runner preflight mode for closure consistency | MEDIUM | MEDIUM | P2 |
| Cross-report consistency checks | MEDIUM | HIGH | P3 |

**Priority key:**
- P1: Must have for v1.19.
- P2: Should have if it fits after the core report contracts are stable.
- P3: Useful later, but not required for the no-new-hardware credibility goal.

## Requirements-Ready Acceptance Criteria

These criteria are phrased so they can be copied into milestone requirements.

| Feature Area | Acceptance Criteria |
|--------------|---------------------|
| Denominator accounting | Given fixture manifest/inventory/readiness/closure/score sidecars, the tool emits JSON and Markdown reports with deterministic category, problem, workload, readiness, closure, and evidence-gap counts. |
| Denominator claim boundary | Report payload includes explicit false values or equivalent wording for full paper parity, leaderboard authority, upstream SOLAR parity, score authority where not applicable, native host validation, and new hardware validation. |
| Evidence gaps | Missing traces, missing AMD SOL sidecars, missing SOLAR derivation, missing timing evidence, and missing Matrix evidence are reported as evidence gaps, not correctness failures or scored results. |
| Dataset runner consistency | A ready-subset/readiness/manifest checksum mismatch is detected before or during closure generation and is represented with a stable diagnostic/error path. |
| Dataset runner resume | Existing passing traces are skipped only when they match the selected problem/workload identity and closure provenance records `skipped_existing_pass`; `--rerun` forces reattempt. |
| Failure classification | Fixture failures are classified separately for build/runtime/timeout/nonzero CLI/missing trace/correctness failure/derived evidence missing, with log refs when available. |
| Matrix diff | Given two Matrix reports, the diff output identifies added, removed, unchanged, and changed Targets and lists changed fields with old/new values. |
| Matrix diff guardrails | A diff cannot turn Docker container evidence into native-host validation; any claim-boundary escalation is high severity and visible in JSON and human output. |
| Schema export | A command writes valid JSON Schema for Matrix Entry and Matrix Report, with schema version/source metadata and `additionalProperties` disabled where the Pydantic model forbids extras. |
| Bound sanity | The sanity report summarizes existing RDNA 4/Docker AMD SOL/SOLAR artifacts by aggregate status, coverage, warnings, missing evidence, and provisional hardware/model validation risk. |
| No-new-hardware | Tests and docs state that v1.19 adds no CDNA 3/MI300X/CDNA4 or native host validation and uses only existing RDNA 4/Docker evidence surfaces. |

## Source-Backed Constraints

| Constraint | Source | Confidence |
|------------|--------|------------|
| v1.19 goal is credibility without expanding hardware validation. | `.planning/PROJECT.md` | HIGH |
| Existing active requirements are closed; deferred items include native host matrix and Matrix tooling. | `.planning/REQUIREMENTS.md` | HIGH |
| Docker Matrix Entries validate container ROCm user-space only, not native host validation. | `docs/CLAIMS.md`, `docs/TESTING.md` | HIGH |
| Dataset execution closure statuses are sidecar-only and bounded. | `scripts/run_dataset.py`, `docs/RESEARCHER-GUIDE.md` | HIGH |
| Matrix status/reason vocabularies and claim boundaries are strict Pydantic models. | `src/sol_execbench/core/compatibility.py` | HIGH |
| AMD SOL/SOLAR artifacts are local AMD-native-derived evidence, not upstream SOLAR or B200 equivalence. | `docs/analysis.md`, `docs/CLAIMS.md` | HIGH |
| Documentation wording around evidence authority is behavior-affecting. | `.planning/codebase/CONCERNS.md`, `tests/sol_execbench/test_public_contract_guardrails.py` search results | HIGH |

## Sources

- `.planning/PROJECT.md`
- `.planning/REQUIREMENTS.md`
- `.planning/ROADMAP.md`
- `.planning/codebase/CONCERNS.md`
- `docs/CLAIMS.md`
- `docs/TESTING.md`
- `docs/RESEARCHER-GUIDE.md`
- `docs/analysis.md`
- `docs/v1_11_release_closure.md`
- `src/sol_execbench/core/compatibility.py`
- `src/sol_execbench/core/dataset/inventory.py`
- `src/sol_execbench/core/dataset/readiness.py`
- `scripts/run_dataset.py`
- `scripts/report_parity_gaps.py`

---
*Feature research for: v1.19 Research Credibility Without New Hardware*
*Researched: 2026-05-31*
