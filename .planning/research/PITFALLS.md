# v1.19 Pitfalls: Research Credibility Without New Hardware

**Project:** SOL ExecBench ROCm Port
**Milestone:** v1.19 Research Credibility Without New Hardware
**Domain:** ROCm benchmark evidence, denominator accounting, compatibility Matrix reporting, dataset-runner closure, and AMD SOL/SOLAR sanity checks
**Researched:** 2026-05-31
**Confidence:** HIGH for repo-local risks; MEDIUM for external dependency behavior because live package and image availability can change.

## Research Basis

Repo evidence reviewed:

- `.planning/PROJECT.md` defines v1.19 as credibility work without CDNA 3, CDNA 4, MI300X, or expanded hardware validation.
- `.planning/codebase/CONCERNS.md` identifies the main hot spots: large orchestration modules, sidecar proliferation, claim wording, live ROCm gaps, Docker/native-host boundaries, dataset/token handling, and external dependency movement.
- `docs/CLAIMS.md` allows ROCm-port, runtime, profiling, toolchain routing, static kernel, AMD-native-derived, and research-preview evidence, but blocks paper parity, leaderboard authority, upstream SOLAR equivalence, Docker-as-native-host validation, and CDNA 3/CDNA 4 validation without archived direct evidence.
- `docs/TESTING.md` documents marker-gated live ROCm validation, CPU-safe Matrix guardrails, Docker container validation scope, and recorded RDNA 4 Docker rows.
- `docs/static_kernel_evidence.md` models the correct sidecar pattern: optional, diagnostic-only, explicit statuses, no mutation of trace/correctness/timing/scoring semantics.
- `docs/original_parity.md` classifies upstream SOL ExecBench public surfaces and states that full NVIDIA leaderboard parity and CDNA 3 hardware validation remain out of scope.
- `scripts/run_dataset.py` is a 1700-line runner that now handles discovery, reference/custom solution wrapping, trace parsing, AMD score reports, SOL/SOLAR sidecars, timing evidence, ready-subset execution closure, manifest provenance, and closure totals.
- Existing tests already guard adjacent surfaces: `test_dataset_inventory_readiness.py`, `test_run_dataset_execution_closure.py`, `test_parity_gap_report.py`, `test_matrix_claim_guardrails.py`, `test_rocm_matrix_docs.py`, `test_original_parity_docs.py`, `test_static_kernel_evidence.py`, `test_trace_reporting_and_score_guardrails.py`, `test_amd_sol_bounds.py`, `test_amd_native_score.py`, and SOLAR derivation fixture validators.

## Suggested v1.19 Phase Placement

| Phase | Name | Pitfalls Owned |
| --- | --- | --- |
| Phase 1 | Paper Denominator And Claim Boundary Accounting | Dataset denominator status, paper-parity wording, ready/blocked/unsupported/deferred/evidence-missing taxonomy. |
| Phase 2 | Matrix Diff And Schema Export | Matrix diff semantics, schema compatibility, target-vs-observed separation, external producer contracts. |
| Phase 3 | Dataset Runner Closure Hardening | Resume/manifest consistency, failure classification, large-module containment, closure determinism. |
| Phase 4 | AMD SOL/SOLAR Sanity Evidence | Provisional model boundaries, bound sanity checks, degraded/unscored status handling, evidence refs. |
| Phase 5 | Documentation And Guardrail Tests | Docs wording, CPU-safe regression tests, artifact examples, no-new-hardware boundary enforcement. |

## Critical Pitfalls

### Pitfall 1: Turning Denominator Accounting Into A Paper-Parity Claim

**What goes wrong:**
The new denominator report says or implies that the ROCm port has "paper parity", "full benchmark validation", or "235-problem validation" because every original problem has a status row. A complete accounting table is mistaken for completed execution evidence.

**Why it happens:**
v1.19 wants to explain the paper denominator. That naturally produces full-looking tables. Existing docs already warn that curated-slice results and Docker Matrix Entries are not paper-level benchmark results, but a new denominator report can accidentally flatten `ready`, `blocked`, `unsupported`, `deferred`, and `evidence_missing` into "covered".

**How to avoid:**
Make the denominator artifact explicitly count categories separately: original denominator total, discovered, parsed, ready-to-attempt, attempted, passed, failed, blocked, unsupported, deferred, and evidence-missing. Every JSON and Markdown report should include claim flags equivalent to `full_235_problem_validation=false`, `paper_parity=false`, and `leaderboard_result=false` unless direct complete evidence exists. Use wording like "denominator accounted for" rather than "validated".

**Warning signs:**
- Markdown contains "paper parity", "full validation", "complete benchmark", or "leaderboard" without a nearby negative boundary.
- Status totals add blocked or unsupported problems into a "covered" or "validated" numerator.
- A denominator report has no separate `evidence_missing` bucket.
- A report headline uses only pass counts and hides denominator status counts.

**Phase to address:**
Phase 1 and Phase 5.

---

### Pitfall 2: Collapsing Ready, Blocked, Unsupported, Deferred, And Evidence-Missing Into One Skip Bucket

**What goes wrong:**
Dataset status becomes unauditable because "skipped" covers missing safetensors assets, custom input support gaps, NVIDIA-only runtime hints, low-precision hardware evidence needs, intentionally deferred hardware validation, and missing derived evidence. Downstream users cannot tell whether a problem can be run, should not be run on ROCm, or requires evidence the project explicitly deferred.

**Why it happens:**
`scripts/run_dataset.py` already has closure statuses such as `filtered`, `not_attempted`, `missing_trace`, and `derived_evidence_missing`; dataset readiness has statuses such as `runtime_blocked`, `needs_hardware_evidence`, and `unsupported_nvidia_only_path`. A new denominator layer can simplify these too aggressively.

**How to avoid:**
Preserve both high-level status and reason code. At minimum, keep separate top-level classes for `ready`, `blocked`, `unsupported`, `deferred`, and `evidence_missing`, with stable reason-code vocabularies. Keep workload-level and problem-level counts distinct because one problem can contain both runnable and blocked workloads.

**Warning signs:**
- New code reports only `passed`, `failed`, and `skipped`.
- Reason codes from readiness records are dropped from closure or parity output.
- Problem-level totals are inferred from workload-level totals without documenting the rollup rule.
- Tests assert only aggregate counts, not reason-code preservation.

**Phase to address:**
Phase 1 and Phase 3.

---

### Pitfall 3: Letting Sidecar Proliferation Create Conflicting Sources Of Truth

**What goes wrong:**
Matrix diffs, schema exports, denominator reports, execution closure, AMD score reports, SOL bound sidecars, SOLAR derivation sidecars, timing evidence, static evidence, and runtime evidence all claim overlapping facts with slightly different status names or refs. A reviewer cannot reconstruct which artifact is authoritative for which question.

**Why it happens:**
This repo deliberately keeps evidence out of canonical trace JSONL. That is the right pattern, but v1.19 adds more meta-evidence around existing sidecars. Without a source-of-truth map, every sidecar can become a partial summary of every other sidecar.

**How to avoid:**
Define artifact ownership: canonical traces own workload execution result; readiness owns runnable/blocker classification; execution closure owns attempted/not-attempted closure; Matrix Entry owns ROCm target compatibility; Matrix diff owns changes between Matrix artifacts; AMD score owns derived local score; SOL/SOLAR sidecars own bound derivation evidence. New reports should reference source artifacts by path/checksum rather than duplicating detailed payloads.

**Warning signs:**
- The same status appears with different names across reports, for example `missing`, `unavailable`, and `evidence_gap` for the same condition.
- Matrix diff output embeds full Matrix Entries instead of path/checksum refs plus changed fields.
- Denominator reports copy SOL/SOLAR warnings instead of linking evidence refs.
- Canonical trace JSONL is changed to carry Matrix or denominator fields.

**Phase to address:**
Phase 1, Phase 2, Phase 3, and Phase 5.

---

### Pitfall 4: Destabilizing `scripts/run_dataset.py` While Hardening It

**What goes wrong:**
A runner hardening change breaks existing single-problem execution, dataset discovery, reference solution wrapping, `--solution-name`, `--max-workloads`, `--rerun`, AMD score generation, timing evidence, or ready-subset closure. The milestone improves auditability but regresses ordinary benchmark workflows.

**Why it happens:**
`scripts/run_dataset.py` is a large orchestration module combining IO, CLI fan-out, trace parsing, derived reports, skip handling, and artifact closure. Small changes to shared paths such as workload staging, `problem_output_dir`, trace loading, or summary writing can affect many modes.

**How to avoid:**
Treat the runner as a compatibility surface. Add focused helper functions for new classification and manifest checks; do not rewrite the main loop. Preserve default behavior when `--ready-subset`, `--execution-closure`, Matrix diff, or denominator options are not supplied. Extend `test_run_dataset_execution_closure.py` with resume and manifest-mismatch tests before changing control flow.

**Warning signs:**
- A patch changes the main loop and derived score paths in the same diff.
- Existing no-`--ready-subset` behavior now writes new required sidecars or exits differently.
- `summary.json` shape changes to support closure reporting.
- A failed CLI invocation with no traces is counted as a pass, filtered item, or non-attempt.

**Phase to address:**
Phase 3.

---

### Pitfall 5: Resume Logic Reuses Stale Traces Against A Different Manifest

**What goes wrong:**
The runner skips an existing passing `traces.json` even though the ready subset, workload file, dataset manifest, benchmark config, solution mode, iteration count, clock policy, or git commit changed. Closure then appears reproducible but actually mixes old execution with new denominator metadata.

**Why it happens:**
Current runner behavior skips existing passing results unless `--rerun`. v1.19 adds stronger manifest and closure semantics, so "existing pass" must be scoped to the same workload selection and evidence requirements.

**How to avoid:**
For closure-producing runs, compare stored provenance before accepting existing traces: ready subset checksum, readiness checksum, dataset manifest checksum, selected categories, limit, max workloads, solution mode/name, benchmark config, and derived evidence options. If they mismatch, either force rerun or mark `stale_existing_trace`/`manifest_mismatch` explicitly and keep the closure non-authoritative.

**Warning signs:**
- `skipped_existing_pass` records lack provenance comparison fields.
- Changing `--max-workloads`, `--iterations`, or `--solution-name` still reuses traces silently.
- Closure records cite an old `solution.json` or workload staging file that no longer matches the ready subset.
- A closure report has checksums in `provenance` but no decision logic uses them.

**Phase to address:**
Phase 3.

---

### Pitfall 6: Matrix Diff Treats Textual Changes As Semantic Compatibility Changes

**What goes wrong:**
Matrix diffs become noisy or misleading: timestamps, ordering, absolute paths, stdout tails, or unrelated artifact locations are reported as compatibility regressions, while meaningful changes such as status, reason code, PyTorch ROCm target, image tag, clock policy, runtime availability, or evidence authority flags are buried.

**Why it happens:**
Compatibility Matrix Entries include both target/requested values and observed evidence. Some observed fields are intentionally environment-specific and volatile. A raw JSON diff is easy to implement but poor evidence.

**How to avoid:**
Build a semantic diff. Normalize paths and ordering. Group changes into target identity, requested stack, observed host/container/Python/toolchain/GPU evidence, status/reason, clock/runtime evidence, claim boundary, and artifact refs. Assign severity: `breaking`, `claim_boundary`, `compatibility_status`, `dependency`, `environment`, `metadata`.

**Warning signs:**
- Diff output is line-oriented JSON with no status classification.
- A timestamp-only change marks the Matrix as changed.
- A change from `not_tested` to `container_validated` is not highlighted as a status upgrade.
- Claim-boundary flag changes are treated as ordinary metadata.

**Phase to address:**
Phase 2.

---

### Pitfall 7: Schema Export Freezes Internal Models As An External Contract

**What goes wrong:**
Downstream CI or external evidence producers depend on every internal Matrix model field, including unstable implementation details. Future internal refactors become breaking changes, or external sidecars start submitting fields the project cannot validate consistently.

**Why it happens:**
Pydantic models make JSON Schema export tempting. But internal strict models can include fields whose meaning depends on local collection code, while external producers need a narrower ingestion contract with explicit schema versioning and allowed authority flags.

**How to avoid:**
Export versioned schemas for public artifacts only, not arbitrary internals. Include `$id`, `schema_version`, `additionalProperties` policy, required fields, enums, and claim-boundary flags. Document producer responsibilities: target identity, observed evidence provenance, checksum/ref fields, and prohibition on setting validation authority without matching evidence.

**Warning signs:**
- Schema export includes private helper names or local-only fields.
- There is no schema compatibility test that validates a representative external payload.
- Schema files omit `schema_version` or enum value tests.
- External producers can set `native_host_validated=true` in a container-scope payload.

**Phase to address:**
Phase 2 and Phase 5.

---

### Pitfall 8: AMD SOL/SOLAR Sanity Checks Become Model Validation Claims

**What goes wrong:**
Bound sanity checks are presented as proof that the AMD SOL/SOLAR model is correct, upstream SOLAR-equivalent, or valid for CDNA 3/CDNA 4. In reality, v1.19 only uses existing RDNA 4 and Docker evidence to clarify provisional model risk.

**Why it happens:**
Sanity checks often produce reassuring pass/fail results. Existing AMD SOL artifacts can say RDNA 4 hardware validation is recorded while model validation remains provisional. A new "sanity passed" field can obscure that distinction.

**How to avoid:**
Separate hardware evidence, model validation status, and sanity-check outcome. Use terms such as `provisional_model_sanity`, `bound_ordering_check`, or `requires_external_validation`, not `validated_model`. Keep upstream SOLAR equivalence and paper-scale validation flags false unless direct comparison evidence exists.

**Warning signs:**
- Docs say "SOLAR validated" or "model validated" from local derived sidecars.
- Sanity output has only `passed=true` with no confidence/status/warnings.
- CDNA 3/CDNA 4 hardware models appear in positive examples without archived live evidence.
- Unsupported or degraded SOLAR fixture classes are converted into scored results.

**Phase to address:**
Phase 4 and Phase 5.

---

### Pitfall 9: Bound Sanity Checks Ignore Degraded And Unscored Evidence

**What goes wrong:**
Sanity checks average or compare only numeric scores and silently exclude degraded, inexact, unsupported, missing, or unscored workloads. The report looks clean because the risky cases were omitted from the numerator.

**Why it happens:**
Existing SOL/SOLAR code has nuanced statuses: supported, inexact, unsupported, degraded, unscored, missing evidence, and warning prefixes. A simple bound sanity report may focus on numeric bounds only.

**How to avoid:**
Make sanity reports denominator-aware: count checked, degraded, unscored, unsupported, missing sidecar, missing timing, and missing trace separately. Require every excluded workload to appear with a reason code. Treat missing requested bound evidence as a degraded closure state, matching the existing `derived_evidence_missing` pattern.

**Warning signs:**
- Sanity report denominator is smaller than execution closure denominator without explanation.
- Unsupported operators disappear from output.
- Report has no warning count or no list of unscored workloads.
- `UNSCORED_SOL_BOUND_WARNING`-style warnings are not surfaced in summaries.

**Phase to address:**
Phase 4.

---

### Pitfall 10: External Dependency Drift Is Misclassified As Benchmark Failure

**What goes wrong:**
A PyTorch ROCm wheel, Triton ROCm package, ROCm Docker image, `rocprofv3`, `llvm-objdump`, `readelf`, or ROCm library package changes or disappears. Reports classify the resulting failure as problem correctness failure, paper denominator blocker, or score regression instead of dependency/environment drift.

**Why it happens:**
The project already depends on external moving parts, and v1.18 added Docker target-specific PyTorch policy. v1.19 diff tooling can expose changes, but only if dependency and target evidence are separated from benchmark results.

**How to avoid:**
Use explicit dependency statuses and reason codes in Matrix and runner reports: wheel unavailable, runtime unavailable, mixed version, tool unavailable, profiler unavailable, image unavailable, library header missing. Matrix diff should highlight dependency drift separately from benchmark trace changes. Dataset closure should link dependency failure logs and keep trace status absent or failed with the correct environment reason.

**Warning signs:**
- A missing wheel or image produces `attempted_failed` without dependency reason code.
- Matrix diff reports only final benchmark pass/fail.
- The runner stderr log is saved but not referenced from closure records.
- Docs update live Matrix rows without updating dependency policy examples.

**Phase to address:**
Phase 2, Phase 3, and Phase 5.

---

### Pitfall 11: Documentation Guardrails Lag Behind New Evidence Surfaces

**What goes wrong:**
Code-side artifacts correctly preserve authority flags, but docs or examples overclaim the new features. Users cite Matrix diffs, schema exports, denominator reports, or bound sanity checks as evidence for paper parity, leaderboard readiness, native host validation, or new hardware validation.

**Why it happens:**
Existing docs guardrails cover Docker Matrix, static evidence, original parity, AMD scores, and testing. v1.19 adds new named artifacts that need the same explicit negative wording.

**How to avoid:**
Update `docs/CLAIMS.md`, `docs/TESTING.md`, and any new docs with exact allowed and forbidden wording. Add tests patterned after `test_rocm_matrix_docs.py`, `test_original_parity_docs.py`, and `test_amd_native_score.py` that search for claim-boundary statements and representative commands.

**Warning signs:**
- New docs introduce "credible", "validated", "complete", or "parity" without defining evidence scope.
- Examples show denominator or sanity reports without warning blocks.
- Tests cover schemas but not public wording.
- README or researcher guide summaries omit the no-new-hardware boundary.

**Phase to address:**
Phase 5.

---

### Pitfall 12: New Artifacts Leak Dataset Payloads, Tokens, Or Host-Specific Paths

**What goes wrong:**
Denominator or closure artifacts include Hugging Face tokens, proprietary dataset paths, safetensors payload metadata beyond refs, local usernames, absolute build paths, or full stderr/stdout containing sensitive environment details.

**Why it happens:**
Dataset acquisition and benchmark execution touch downloaded assets, user-supplied code, and external paths. Evidence artifacts are meant to be shareable, so they need stricter content boundaries than local logs.

**How to avoid:**
Keep artifacts to relative refs, checksums, bounded stdout/stderr tails, reason codes, and redacted paths where possible. Preserve the existing pattern of not committing downloaded datasets. For external evidence schemas, document that raw data payloads, credentials, and proprietary kernels are forbidden.

**Warning signs:**
- JSON reports include absolute `/home/...` paths where relative refs would work.
- Sidecars embed full safetensors paths outside the dataset root or raw input metadata.
- Logs are copied wholesale into Markdown reports.
- New schema fields invite environment dumps rather than bounded evidence refs.

**Phase to address:**
Phase 1, Phase 3, and Phase 5.

## Technical Debt Patterns

| Shortcut | Immediate Benefit | Long-term Cost | When Acceptable |
| --- | --- | --- | --- |
| Add v1.19 logic directly inside the `run_dataset.py` main loop | Fast implementation | More fragile runner, hidden coupling between closure, scoring, timing, and denominator paths | Only for minimal argument wiring; classification and validation should be helper functions with tests. |
| Use raw JSON diffs for Matrix comparison | Quick diff output | Noisy, unactionable diffs that hide claim-boundary and dependency changes | Acceptable only as a debug appendix behind a semantic summary. |
| Export schemas from every Pydantic model | Easy downstream schema generation | Internal fields become accidental public API | Never for undocumented internals; export only versioned public artifact schemas. |
| Treat `skipped` as a denominator status | Simpler reporting | Erases blocked vs unsupported vs deferred vs evidence-missing distinctions | Never for v1.19 roadmap outputs. |
| Copy sidecar details into aggregate reports | Self-contained reports | Conflicting sources of truth and hard-to-audit drift | Only for small summaries; include refs/checksums to source artifacts. |
| Add positive "sanity passed" badges | Clear demo output | Encourages model-validation overclaims | Only if paired with provisional model and no-new-hardware boundaries. |

## Integration Gotchas

| Integration | Common Mistake | Correct Approach |
| --- | --- | --- |
| Paper dataset inventory/readiness | Counting only discovered or parsed problems as the denominator | Preserve original denominator accounting and separate discovered, parsed, ready, blocked, unsupported, deferred, attempted, passed, and evidence-missing counts. |
| Execution closure | Reusing existing traces without checking ready-subset or manifest provenance | Compare checksums and run configuration before `skipped_existing_pass`; otherwise rerun or mark stale/mismatched. |
| ROCm Compatibility Matrix | Diffing volatile raw JSON fields | Produce semantic diffs grouped by target, observed evidence, status/reason, dependencies, clock/runtime state, and claim-boundary flags. |
| JSON Schema export | Letting external producers set authority flags freely | Validate scope/status/authority consistency and ship representative external-payload tests. |
| AMD SOL/SOLAR sidecars | Treating derived bound evidence as upstream SOLAR equivalence | Keep derived AMD-native, provisional, degraded, and unscored states visible with source refs. |
| PyTorch ROCm / Docker images / Triton ROCm | Reporting package/image unavailability as benchmark correctness failure | Classify as dependency or runtime evidence state and include Matrix diff dependency changes. |
| Docs and examples | Showing artifact commands without claim boundaries | Pair every new evidence command with "not paper parity / not leaderboard / not new hardware validation" wording. |

## Performance And Scale Traps

| Trap | Symptoms | Prevention | When It Breaks |
| --- | --- | --- | --- |
| Loading full sidecars into aggregate reports | Large JSON/Markdown outputs, duplicate evidence, slow review | Store refs/checksums and compact summaries | Full original dataset plus derived SOL/SOLAR/timing artifacts. |
| Per-workload schema export or validation with repeated model generation | Slow CLI and large schema directories | Generate one schema per public artifact version | External CI schema bundles. |
| Matrix raw diff across many volatile fields | Massive diffs after every run | Normalize and classify semantic fields | Multiple ROCm Targets or repeated Docker runs. |
| Recomputing SOL/SOLAR sidecars unnecessarily on resumed runs | Slow dataset closure and changed timestamps/checksums | Reuse only when provenance matches; otherwise mark stale or rerun deterministically | Ready subsets with many workloads. |
| Writing closure for every discovered problem without rollups | Reports become hard to inspect | Provide workload records plus problem/category/total summaries | Paper-scale denominator reports. |

## Security And Evidence Hygiene Mistakes

| Mistake | Risk | Prevention |
| --- | --- | --- |
| Embedding raw downloaded dataset content in evidence artifacts | Licensing/privacy leakage and oversized reports | Store relative refs, checksums, status, and reason codes only. |
| Recording credentials or tokens from environment/logs | Secret leakage in shareable research artifacts | Redact environment values; bound stdout/stderr tails; never include Hugging Face tokens. |
| Trusting external schema payload authority flags | Downstream producer can overclaim validation | Re-validate scope/status/evidence consistency on ingest. |
| Treating subprocess staging as untrusted-code containment | Users overestimate isolation of arbitrary solutions | Preserve SECURITY/docs language that benchmark execution is not a hard sandbox. |
| Using absolute host paths as stable evidence refs | Non-reproducible and privacy-sensitive reports | Prefer paths relative to output root plus checksums. |

## "Looks Done But Isn't" Checklist

- [ ] **Denominator report:** Has original denominator, status buckets, workload/problem rollups, and explicit `paper_parity=false`.
- [ ] **Readiness taxonomy:** Preserves reason codes for blocked, unsupported, deferred, and evidence-missing cases.
- [ ] **Matrix diff:** Highlights status/reason, target, dependency, image, clock/runtime, evidence, and authority changes separately from timestamps.
- [ ] **Schema export:** Ships versioned public schemas with representative valid and invalid external payload tests.
- [ ] **Runner hardening:** Default no-closure dataset run behavior is unchanged and covered by focused tests.
- [ ] **Resume handling:** Existing traces are accepted only when provenance and evidence requirements match.
- [ ] **AMD SOL/SOLAR sanity:** Reports degraded, unsupported, unscored, and missing-evidence counts, not just numeric pass/fail.
- [ ] **Docs:** New artifact docs repeat no paper parity, no leaderboard authority, no native-host upgrade from Docker, and no new hardware validation.
- [ ] **Evidence hygiene:** Artifacts avoid tokens, raw dataset payloads, proprietary kernels, and unnecessary absolute paths.

## Recovery Strategies

| Pitfall | Recovery Cost | Recovery Steps |
| --- | --- | --- |
| Denominator report overclaims paper parity | MEDIUM | Rename statuses, add claim-boundary flags, update docs/tests, regenerate affected artifacts. |
| Skip bucket hides blockers | MEDIUM | Reprocess readiness/closure records with reason-code taxonomy; add regression tests for each status class. |
| Sidecars conflict | HIGH | Define source-of-truth ownership, replace duplicated fields with refs/checksums, add schema/doc tests. |
| Runner regression | HIGH | Bisect around `scripts/run_dataset.py`, restore default behavior, add mode-specific tests before reintroducing hardening. |
| Stale trace reuse | MEDIUM | Add provenance comparison, mark old closure unauditable, rerun affected ready subset. |
| Noisy Matrix diff | LOW | Add semantic normalization and severity grouping; keep raw diff only as debug output. |
| Overbroad schema export | MEDIUM | Version public schemas, remove internal fields, add invalid external authority-flag fixtures. |
| SOL/SOLAR sanity overclaim | MEDIUM | Reword docs/artifacts, force provisional status, surface degraded/unscored counts. |
| External dependency drift misclassified | MEDIUM | Add dependency reason codes and Matrix diff classification; relabel affected benchmark results as environment-blocked. |
| Evidence leaks sensitive data | HIGH | Remove leaked artifacts from working tree/history as needed, redact generation paths, add hygiene tests. |

## Pitfall-to-Phase Mapping

| Pitfall | Prevention Phase | Verification |
| --- | --- | --- |
| Denominator accounting becomes paper-parity claim | Phase 1, Phase 5 | Denominator JSON/Markdown has false paper/leaderboard flags and tests reject overclaim wording. |
| Status taxonomy collapses into `skipped` | Phase 1, Phase 3 | Tests cover ready, blocked, unsupported, deferred, evidence-missing, and reason-code preservation. |
| Sidecar proliferation creates conflicting truth | Phase 1, Phase 2, Phase 3, Phase 5 | Artifact docs define ownership; aggregate reports use refs/checksums; trace schema remains unchanged. |
| Runner hardening destabilizes existing behavior | Phase 3 | Existing runner tests plus new resume/manifest tests pass; default CLI behavior remains unchanged. |
| Resume uses stale traces | Phase 3 | Mismatched ready subset, manifest, config, solution, or evidence options force rerun or stale status. |
| Matrix diff is raw/noisy | Phase 2 | Diff fixtures classify meaningful vs volatile changes and flag authority changes prominently. |
| Schema export freezes internals | Phase 2, Phase 5 | Only versioned public schemas are exported; invalid external authority payloads are rejected. |
| SOL/SOLAR sanity becomes model validation | Phase 4, Phase 5 | Sanity artifact distinguishes hardware evidence, provisional model status, and sanity result. |
| Degraded/unscored bounds disappear | Phase 4 | Sanity report denominator includes degraded, unsupported, unscored, and missing evidence counts. |
| External dependency drift becomes benchmark failure | Phase 2, Phase 3 | Dependency/image/tool unavailability reason codes appear in Matrix diff and closure records. |
| Docs lag behind new artifacts | Phase 5 | Docs guardrail tests cover denominator, Matrix diff/schema, runner closure, and bound sanity wording. |
| Evidence leaks sensitive data | Phase 1, Phase 3, Phase 5 | Artifact tests use relative refs/checksums and bounded logs; docs forbid tokens/raw payloads. |

## Sources

- `.planning/PROJECT.md`
- `.planning/codebase/CONCERNS.md`
- `docs/CLAIMS.md`
- `docs/TESTING.md`
- `docs/static_kernel_evidence.md`
- `docs/original_parity.md`
- `scripts/run_dataset.py`
- `tests/sol_execbench/test_dataset_inventory_readiness.py`
- `tests/sol_execbench/test_run_dataset_execution_closure.py`
- `tests/sol_execbench/test_parity_gap_report.py`
- `tests/sol_execbench/test_matrix_claim_guardrails.py`
- `tests/sol_execbench/test_rocm_matrix_docs.py`
- `tests/sol_execbench/test_original_parity_docs.py`
- `tests/sol_execbench/test_static_kernel_evidence.py`
- `tests/sol_execbench/test_trace_reporting_and_score_guardrails.py`
- `tests/sol_execbench/test_amd_sol_bounds.py`
- `tests/sol_execbench/test_amd_native_score.py`
- `tests/sol_execbench/solar_derivation_fixtures.py`

---
*Pitfalls research for: v1.19 Research Credibility Without New Hardware*
*Researched: 2026-05-31*
