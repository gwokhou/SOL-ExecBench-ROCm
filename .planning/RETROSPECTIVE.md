# Project Retrospective

*A living document updated after each milestone. Lessons feed forward into future planning.*

## Milestone: v1.0 - ROCm Port

**Shipped:** 2026-05-21
**Phases:** 6 | **Plans:** 21 | **Tasks:** 18

### What Was Built

- ROCm Docker/runtime baseline with HIP compiler tooling, ROCm profiling tools, PyTorch ROCm, Triton ROCm, and dependency smoke tests.
- ROCm-native solution schema and HIP/C++ native build flow with AMD gfx offload architecture handling.
- ROCm-compatible evaluation runtime, timing path, AMD environment snapshots, and clock tooling.
- Migrated examples, library replacement documentation, ROCm pytest markers, and RDNA 4 full-suite validation.
- User-facing README, setup, schema, trace, analysis, and compliance documentation for the ROCm-only port.

### What Worked

- Horizontal phase ordering kept hard dependencies clear: environment, schema/build, runtime/timing, examples, validation, then docs.
- Focused audit tests were useful for preventing CUDA/NVIDIA tooling regressions while still allowing PyTorch ROCm's `torch.cuda` compatibility API.
- Recording the hardware matrix separately made the RDNA 4 pass and CDNA 3 gap explicit instead of mixing hardware claims into general test status.

### What Was Inefficient

- Some summary files lacked extractable one-line metadata, so milestone accomplishment extraction depended on manual synthesis.
- CDNA 3 validation was discovered as a closure gap late in the milestone and had to be deferred instead of planned as an available hardware run.

### Patterns Established

- Keep legacy CUDA/NVIDIA strings only when they are compatibility API names, rejection tests, migration guidance, or attribution.
- Add source audits for high-risk migration surfaces instead of relying on broad text search alone.
- Treat hardware support claims as evidence-backed: schema values and docs should follow recorded full-suite runs.

### Key Lessons

1. ROCm ports should distinguish API namespace compatibility from runtime dependency residue; PyTorch ROCm still legitimately uses `torch.cuda`.
2. Hardware matrix evidence should be planned as first-class milestone work whenever multiple AMD architecture families are part of the goal.
3. Documentation should name unsupported NVIDIA runtime features directly so users do not infer dual-backend support from retained upstream attribution.

### Cost Observations

- Model mix: not recorded.
- Sessions: not recorded.
- Notable: phase archives and focused verification files now give enough structure for future milestone closure with lower context load.

---

## Milestone: v1.2 - Engineering Practice Harvest and Compatibility Guardrails

**Shipped:** 2026-05-22
**Phases:** 4 | **Plans:** 4 | **Tasks:** 0

### What Was Built

- Internal adaptation map for selected `hip-execbench` engineering practices.
- ROCm diagnostic helpers for tool readiness, gfx classification, local gfx
  detection, and profiler backend fallback reasoning.
- Pure trace summary helpers that preserve the existing trace JSONL contract.
- SOL-Score interpretation guardrails for unsupported AMD-native performance
  claims.
- Public contract tests for schemas, CLI help, trace behavior, examples, and
  CDNA 3 validation deferral language.

### What Worked

- Treating `hip-execbench` as a practice source instead of a port target kept
  changes narrow and avoided public interface churn.
- Pure helper modules made diagnostics/reporting easy to test without requiring
  GPU hardware in CI.
- Contract tests gave concrete protection for the user's "do not change public
  formats" constraint.

### What Was Inefficient

- The SDK autonomous runner failed immediately, so the workflow had to be
  executed inline.
- Summary extraction produced generic "Completed:" one-liners, requiring manual
  milestone accomplishment synthesis.

### Patterns Established

- Borrow practices through an explicit accept/reject/defer map before touching
  runtime code.
- Add public-contract guardrail tests whenever internal quality work risks
  drifting schemas, CLI help, examples, or trace output.
- Keep score interpretation warnings separate from score calculation so legacy
  benchmark semantics remain stable.

### Key Lessons

1. Practice-harvest milestones need compatibility tests as first-class outputs,
   not just implementation improvements.
2. ROCm diagnostic routes can be modeled as descriptive readiness metadata
   before becoming user-facing commands.
3. AMD performance claims should be guarded in code and docs until a validated
   AMD-native interpretation model exists.

### Cost Observations

- Model mix: not recorded.
- Sessions: 1 autonomous session with inline fallback after SDK runner failure.
- Notable: phase artifacts and commits were kept small enough to audit.

---

## Milestone: v1.5 - AMD-native SOL Scoring and ROCm Profiler Timing

**Shipped:** 2026-05-22
**Phases:** 4 | **Plans:** 4 | **Tasks:** 0

### What Was Built

- Source-specific timing policy for HIP native, Triton, PyTorch, mixed, and
  unknown workloads.
- `rocprofv3` timing evidence helpers with command construction, CSV parsing,
  policy-aware default selection, and fallback metadata.
- AMD SOL bound artifacts with graph nodes, FLOP/byte work estimates, hardware
  model metadata, per-op bounds, and aggregate bounds.
- Derived AMD-native score reports with per-workload scores, suite aggregation,
  evidence references, and CDNA3 no-validation guardrails.

### What Worked

- Keeping timing policy, profiler evidence, SOL bounds, and scoring reports as
  separate layers made each phase testable without GPU hardware.
- Derived artifacts preserved the canonical trace JSONL contract while still
  giving future scoring workflows auditable evidence.
- Explicit CDNA3 no-claim wording kept readiness scaffolding separate from
  hardware validation claims.

### What Was Inefficient

- Summary extraction still produced generic `Status:` entries, so milestone
  accomplishments required manual synthesis.
- The `rocprofv3` work remains an evidence-helper foundation; live benchmark
  integration is still future work.

### Patterns Established

- Treat timing source type as part of measurement semantics, not just metadata.
- Carry confidence and rationale through graph extraction, FLOP/byte estimates,
  hardware model data, and final score reports.
- Keep AMD-native performance interpretation as a derived report until the
  hardware model and validation evidence are strong enough for public claims.

### Key Lessons

1. A chimney-style timing model is preferable when one timing backend would
   conflate HIP runtime, kernel activity, Triton, and PyTorch operator costs.
2. SOL-like score reports need evidence references and claim guardrails as much
   as they need formulas.
3. CDNA3 support scaffolding should remain visibly unvalidated until a real
   `gfx94*` full-suite pass exists.

### Cost Observations

- Model mix: not recorded.
- Sessions: 1 autonomous session plus milestone completion.
- Notable: phase separation kept the final audit small; 42 milestone tests and
  focused ruff checks passed at closure.

---

## Milestone: v1.7 - Baseline, Timing, Reward-Hack Hardening, and ROCm Library Migration

**Shipped:** 2026-05-22
**Phases:** 5 | **Plans:** 5 | **Tasks:** 0

### What Was Built

- Release-scoped optimized scoring baseline artifacts with explicit
  `baseline_source` reporting and dataset-runner integration.
- Source-specific `rocprofv3` timing evidence collection with backend,
  aggregation, warmup, clock-lock, architecture, and fallback metadata.
- Static reward-hack source review before submitted Python import for stream
  hiding, semantic caches, unauthorized loaders, opaque payloads, and precision
  downgrade patterns.
- A runnable hipBLAS SGEMM public example with native staging tests and support
  docs distinguishing supported and candidate ROCm library categories.
- MI300X validation handoff docs and evidence gates that prevent premature
  MI300X-on-CDNA3 hardware-validation claims.

### What Worked

- Keeping scoring baselines and timing evidence as derived artifacts preserved
  canonical trace JSONL while improving auditability.
- The reward-hack source review was integrated before execution, which reduced
  subprocess failure ambiguity and kept findings in existing `REWARD_HACK`
  semantics.
- Promoting only hipBLAS to supported status avoided overclaiming MIOpen, CK,
  and rocWMMA without blocking future migration.

### What Was Inefficient

- Automatic milestone accomplishment extraction still produced generic
  `Completed:` entries, requiring manual rewrite during closure.
- Formal Nyquist `*-VALIDATION.md` artifacts were absent for Phases 31-35, so
  the milestone audit recorded this as discovery-only.

### Patterns Established

- Label every AMD-native score with its baseline source.
- Record timer backend and fallback reason as first-class timing evidence.
- Separate validation readiness from validation claims through pure evidence
  gates and public-contract tests.

### Key Lessons

1. ROCm-only parity reviews should remove NVIDIA hardware expectations from the
   missing-feature list while preserving original benchmark semantics.
2. Library category support should advance one runnable, tested path at a time;
   candidate categories need tests that prevent public overclaiming.
3. MI300X/FP8 validation can be prepared before hardware access, but status
   upgrades must remain gated by full-suite evidence.

### Cost Observations

- Model mix: not recorded.
- Sessions: 1 autonomous session plus audit and milestone completion.
- Notable: the final audit aggregate covered 67 focused tests across scoring,
  profiler timing, reward-hack review, hipBLAS examples, and MI300X guardrails.

---

## Milestone: v1.8 - ROCm Library Ecosystem Completion

**Shipped:** 2026-05-22
**Phases:** 5 | **Plans:** 5 | **Tasks:** 0

### What Was Built

- Reusable ROCm library dependency diagnostics and Docker dependency checks for
  MIOpen, Composable Kernel, rocWMMA, and hipBLAS.
- Native public examples for MIOpen softmax, CK small GEMM, and rocWMMA
  matrix-core GEMM, each with metadata, source consistency, staging, and RDNA 4
  E2E registration coverage.
- Public docs mapping former NVIDIA library/DSL categories to supported ROCm
  examples, compatibility examples, or deferred validation targets.
- Nyquist validation artifacts for all v1.8 phases and a passed milestone audit
  with 23/23 requirements satisfied.

### What Worked

- Keeping RDNA 4 as the only validation scope made support claims crisp and
  prevented CDNA 3/CDNA 4 overclaiming.
- Promoting one scoped runnable example per library category gave each support
  claim concrete source, metadata, test, and documentation evidence.
- Public-contract tests tied support wording to runnable examples, which closed
  the previous candidate-category ambiguity.

### What Was Inefficient

- Summary frontmatter and Nyquist validation artifacts were missing at first,
  so milestone audit needed a follow-up automatic validation pass.
- Local native E2E coverage remains environment-dependent because complete
  ROCm development/library headers are not always installed.

### Patterns Established

- Library support status requires four artifacts: public example, native source
  or metadata evidence, focused tests, and docs that state operation scope.
- CDNA validation wording should remain explicit in every support matrix until
  real hardware evidence exists.
- Milestone audits should run after Nyquist validation files are created, not
  before.

### Key Lessons

1. Candidate ROCm library categories should become supported only through
   small, runnable, operation-specific examples.
2. Validation artifacts are part of the shipped state; missing planning
   metadata can block closure even when implementation evidence is strong.
3. Environment-gated native E2E tests are acceptable only when the guard reason
   is documented and support claims remain scoped.

### Cost Observations

- Model mix: not recorded.
- Sessions: 1 autonomous validation/closure session after phase execution.
- Notable: final validation ran 41 focused tests, Ruff, Docker entrypoint syntax,
  summary requirement extraction, and Nyquist artifact checks.

---

## Milestone: v1.9 - AMD SOL/SOLAR Bound Modeling Completion

**Shipped:** 2026-05-23
**Phases:** 6 | **Plans:** 17 | **Tasks:** 21

### What Was Built

- Strict packaged and external AMD hardware model artifacts with RDNA 4
  `gfx1200` defaults and explicit validation status metadata.
- A structured bound graph IR with workload-aware tensor metadata, dataflow
  edges, dynamic-trace-first extraction, AST fallback, and unsupported/inexact
  evidence.
- Rich operator FLOP, byte, and movement modeling for GEMM/BMM, elementwise,
  activation, reduction, normalization, softmax, data movement, and dtype
  conversion families.
- AMD SOL bound artifact v2 sidecars with graph, estimates, per-op bounds,
  aggregate bound, hardware model reference, coverage summaries, and warnings.
- AMD-native score and dataset integration that consumes v2 sidecars while
  preserving canonical trace JSONL and public schema contracts.
- Documentation, guardrails, and RDNA 4 validation evidence for the v2 bound
  pipeline and no-claim boundaries.

### What Worked

- Keeping v2 bound artifacts as derived sidecars let the milestone add
  paper-aligned evidence without changing canonical traces.
- Separating graph extraction, operator estimates, artifact serialization, and
  score integration made each layer independently testable.
- The user's repeated "keep it paper-consistent" constraint helped prioritize
  explicit formulas, rationale, unsupported evidence, and claim guardrails over
  optimistic scoring.

### What Was Inefficient

- Some phase summaries and validation files still lack machine-friendly
  metadata, so the milestone audit recorded planning-artifact hygiene debt even
  though implementation and verification passed.
- The ROADMAP analyzer still expects a narrower completion marker than the
  narrative completion rows used in this milestone.
- Automatic milestone accomplishment extraction again produced noisy summary
  lines and required manual cleanup in `MILESTONES.md`.

### Patterns Established

- Bound-model claims should move through a layered evidence chain: hardware
  model -> graph IR -> operator estimate -> v2 sidecar -> score report.
- Unsupported or inexact evidence must degrade scoring or warn deterministically
  instead of silently improving aggregate bounds.
- Hardware validation status belongs in artifacts, docs, and guardrail tests,
  not just in prose.

### Key Lessons

1. Paper consistency for AMD SOL/SOLAR modeling is mostly an evidence-contract
   problem: formulas, byte buckets, confidence, rationale, and unsupported
   evidence need to be inspectable at every layer.
2. RDNA 4-only validation scope works when deferred CDNA3-family validation, including MI300X, and CDNA 4
   claims are guarded in code, docs, and tests.
3. Planning metadata quality matters at milestone close; future phases should
   keep summary frontmatter and validation task tables synchronized as work
   lands.

### Cost Observations

- Model mix: not recorded.
- Sessions: one autonomous milestone run plus audit, validation, and completion.
- Notable: final validation included 40 focused AMD bound/model/score/contract
  tests and `uv build`; the audit found 28/28 requirements satisfied with only
  planning metadata hygiene debt.

---

## Milestone: v1.10 - Paper-Aligned SOLAR Automatic Derivation

**Shipped:** 2026-05-23
**Phases:** 6 | **Plans:** 23 | **Tasks:** 41

### What Was Built

- Sidecar-only SOLAR derivation contract and fixture matrix covering positive,
  degraded, unsupported, and negative family cases.
- Internal SOLAR evidence builder with semantic groups, tensor/source
  provenance, strict parser behavior, and conservative confidence/status rules.
- Formula, byte, and AMD SOL-style bound evidence for linear projection,
  attention, convolution, embedding/positional/gather, and rotary-like memory
  structures.
- Conservative MoE and SSM/Mamba evidence paths that preserve degraded and
  unscored semantics without fabricating dynamic routing or recurrence data.
- Coverage summaries, aggregate scored/degraded/unscored sidecar status,
  AMD-native score guards, and dataset-runner generated SOLAR sidecars with
  report-only `derived_evidence_refs`.
- Public contract, claim-boundary, and sidecar filename traversal guardrails for
  the v1.10 reporting surface.

### What Worked

- The phase sequence matched the evidence stack: contract, extraction,
  high-confidence families, degraded complex families, sidecar/status reporting,
  then runner/docs/guardrails.
- Review/fix loops caught semantic and security issues that tests initially
  missed, especially degraded score eligibility and sidecar filename traversal.
- Keeping all SOLAR fields sidecar-only allowed the implementation to become
  richer without changing canonical trace JSONL or primary CLI behavior.

### What Was Inefficient

- One documentation executor stalled after writing docs, requiring manual
  takeover and verification.
- `gsd-sdk` milestone completion archived files but left ROADMAP/PROJECT
  evolution and audit movement for manual cleanup.
- Auto-generated milestone accomplishments contained noisy test-fix lines and
  still need manual cleanup for polished release notes.

### Patterns Established

- SOLAR derivation should derive from canonical `Definition` and `Workload`
  inputs, not candidate execution or timing results.
- Degraded evidence may remain score-eligible when numeric AMD-native inputs
  are complete, but must carry deterministic warnings and parseable status.
- New report evidence belongs in derived-report-only metadata, not public score
  `evidence_refs` or canonical trace fields.
- File names derived from benchmark identifiers must use safe deterministic
  path components before writing sidecars.

### Key Lessons

1. Paper-aligned automatic derivation needs strict evidence contracts as much as
   family recognition: provenance, formulas, byte buckets, confidence, and
   status all need parser coverage.
2. "No public contract drift" should be protected both positively and
   negatively: exact-key tests for canonical surfaces, and explicit no-claim
   phrase tests for documentation.
3. Dataset-runner skip paths are part of derived-report correctness; already
   passing traces still need sidecar/report generation when report flags are
   requested.

### Cost Observations

- Model mix: not recorded.
- Sessions: one autonomous milestone continuation with executor/reviewer/fixer
  agents plus main-thread verification.
- Notable: final gates included 196 Phase 51 tests, 169 Phase 52 tests, Ruff
  checks, clean re-reviews, and a PASS milestone audit.

---

## Milestone: v1.19 - Research Credibility Without New Hardware

**Shipped:** 2026-05-31
**Phases:** 6 | **Plans:** 13 | **Tasks:** 29

### What Was Built

- Strict execution closure contracts and provenance checks with deterministic
  sidecar serialization.
- Paper denominator reports that account for coverage and evidence gaps without
  upgrading claims to paper parity.
- Matrix schema export and semantic diff tooling with diagnostic-only
  JSON/Markdown outputs.
- Dataset-runner closure hardening for provenance-gated reuse, rerun behavior,
  failure classification, and bounded logs.
- AMD SOL/SOLAR bound sanity reports over existing evidence only.
- v1.19 evidence guide, demo fixtures, strict model validation, and public
  contract guardrails.

### What Worked

- Treating each credibility surface as a sidecar/report kept canonical
  benchmark contracts stable while still improving auditability.
- Repeated integration checks were effective at catching docs-to-real-CLI drift
  before milestone close.
- User scope boundaries around no new hardware validation kept the milestone
  focused and prevented Docker, lockfile, and hardware-probe churn.

### What Was Inefficient

- Several docs/examples issues surfaced only during milestone-level integration
  audit, after Phase 88 had already passed local verification.
- Some early SUMMARY frontmatter omitted `requirements-completed`, which created
  extra planning metadata cleanup during audit.
- The Matrix diff and AMD bound sanity docs needed multiple command-shape and
  wording passes to match the real parser/model surfaces exactly.

### Patterns Established

- Every researcher-facing command example should be guarded by a CPU-safe test
  that checks the actual script option names.
- Demo JSON fixtures should validate against the real Pydantic report models,
  not only parse as JSON.
- Sidecar docs should name non-authority explicitly and avoid implying inputs
  that scripts do not consume.

### Key Lessons

1. Claim-boundary milestones need integration tests that connect docs, examples,
   strict models, and script parsers, not just isolated unit tests.
2. Planning metadata should be updated as part of each phase summary so
   milestone audits can rely on three independent sources without repair work.
3. "No new hardware validation" is a productive constraint when preserved
   throughout tests, docs, examples, and verification notes.

### Cost Observations

- Model mix: not recorded.
- Sessions: one autonomous milestone run with targeted subagent review,
  verification, and integration passes.
- Notable: late integration checks paid off by catching four command/example
  mismatches before archival.

---

## Milestone: v1.20 - Cross-Report Consistency and Evaluation Stability

**Shipped:** 2026-05-31
**Phases:** 5 | **Plans:** 10 | **Tasks:** 23 requirements

### What Was Built

- Strict `consistency_report.v1` sidecars and a standalone consistency script
  for contradiction checks across closure, denominator, Matrix, runtime/static
  evidence, AMD score, AMD SOL/SOLAR, and bound sanity reports.
- Strict `evaluation_stability.v1` sidecars and a standalone stability script
  for timing-quality classification, dispersion metrics, clock policy, backend
  state, and profiling-overhead risk.
- Claim-upgrade rule reports that evaluate stronger claim prerequisites without
  mutating source report authority fields.
- Trust summaries that combine consistency, stability, claim-upgrade, evidence
  completeness, source refs, checksums, and next steps into bounded review
  artifacts.
- v1.20 evidence-quality docs, demo fixtures, public contract guardrails, and a
  full consistency -> stability -> claim-upgrade -> trust-summary E2E chain.

### What Worked

- Building local sidecars kept canonical benchmark schemas and evaluator
  semantics stable while giving researchers stronger audit tools.
- Milestone audit caught real integration gaps around AMD SOL/SOLAR propagation,
  fixture coverage, verification artifacts, and public docs command wiring.
- Adding a full cross-script E2E chain turned separate report builders into a
  verified workflow rather than isolated utilities.

### What Was Inefficient

- Automatic milestone accomplishment extraction still produced generic
  `Status:` entries, requiring manual rewrite during archival.
- Some verification and validation artifacts were created after audit instead
  of as part of each phase close.
- Docs command wiring lagged implementation once AMD SOL/SOLAR became required
  downstream evidence.

### Patterns Established

- Any evidence source accepted by an upstream sidecar should be propagated or
  explicitly rejected by downstream claim and trust reports.
- Public guide command examples need tests that inspect exact option names for
  each script in the workflow.
- Milestone audits should require verification, validation, fixtures, and
  end-to-end command chains before archive.

### Key Lessons

1. Evidence-quality systems need chain tests, not only report-level unit tests.
2. Claim gates should require both source refs and checksums for every evidence
   type that can influence stronger authority language.
3. Diagnostic-only wording is easier to preserve when examples, docs, models,
   and public contract tests all share the same negative claim vocabulary.

### Cost Observations

- Model mix: not recorded.
- Sessions: one autonomous milestone run plus audit/fix/close passes.
- Notable: final closure included a passed milestone audit, 74 focused
  evidence-quality tests, docs command regression tests, and open-artifact
  cleanup before archival.

---

## Milestone: v1.29 — Dataset Migration and Compliance

**Shipped:** 2026-06-04
**Phases:** 5 | **Plans:** 5

### What Was Built

- Dataset source, license, provenance, and redistribution policy for
  NVIDIA/SOL-ExecBench, FlashInfer Trace, generated local migration artifacts,
  and project-owned ROCm code.
- Deterministic local SOL-ExecBench and FlashInfer Trace migration commands
  with manifests, checksums, source revisions, license boundaries, and blocker
  states.
- ROCm readiness classification and ready-subset generation preserving
  denominators, blocker reports, closure inputs, and no-claim boundaries.
- CPU-safe low-precision compatibility helpers for NVIDIA/Blackwell-style
  NVFP4/MXFP4 semantics with explicit unvalidated-CDNA4 evidence markers.
- Dataset runner closure integration that records migration, readiness,
  license, blocker, and requested-evidence provenance without allowing public
  redistribution of restricted dataset payloads.

### What Worked

- Running the five phases sequentially kept legal policy, migration, readiness,
  compatibility, and runner integration layered cleanly.
- CPU-safe synthetic fixtures were enough to verify migration contracts,
  readiness blockers, low-precision semantics, and public guardrails without
  needing external datasets or GPU hardware.
- The milestone audit caught planning metadata drift in requirement checkboxes
  and CDNA3/CDNA4 wording before archival.

### What Was Inefficient

- The milestone archive helper extracted generic date-only accomplishments, so
  `MILESTONES.md` needed a manual rewrite.
- The complete-milestone helper was accidentally invoked through two equivalent
  command forms, which created a duplicate v1.29 milestone entry that needed
  cleanup.
- Open-artifact audit still reports historical quick-task artifacts that are
  not current milestone blockers; they had to be explicitly acknowledged as
  deferred at close.

### Patterns Established

- Dataset migration must stay local-only when upstream redistribution rights
  are restricted; manifests and docs should prove source boundaries without
  checking in payloads.
- Readiness classification should preserve denominator accounting for blocked,
  skipped, missing, and unvalidated workloads instead of silently dropping them.
- Compatibility implementation is acceptable without hardware validation only
  when every downstream report carries explicit unvalidated evidence markers.

### Key Lessons

1. Legal provenance and runner closure need to be connected by machine-readable
   metadata, not just documentation.
2. Hardware-specific semantic compatibility should have CPU-safe round-trip
   tests plus separate hardware-evidence blockers.
3. Public docs for dataset migration should phrase NVIDIA data handling as
   user-managed local migration, not project redistribution.

### Cost Observations

- Model mix: parent plus worker agents; exact split not recorded.
- Sessions: one autonomous milestone run plus local audit/close cleanup.
- Notable: Phase 135 verified 74 focused runner, migration, readiness,
  redistribution, prerelease, and public-doc tests.

---

## Cross-Milestone Trends

### Process Evolution

| Milestone | Sessions | Phases | Key Change |
|-----------|----------|--------|------------|
| v1.0 | not recorded | 6 | Established ROCm-only milestone workflow with phase verification, validation artifacts, and hardware matrix evidence |
| v1.2 | 1 | 4 | Added practice-harvest workflow with explicit public-contract guardrails |
| v1.5 | 1 | 4 | Added AMD-native timing/SOL/scoring evidence layers while preserving public contracts |
| v1.7 | 1 | 5 | Added baseline/timing/reward-hack/library/MI300X readiness hardening before real commercial GPU validation |
| v1.8 | 1 | 5 | Completed scoped ROCm library replacement support for RDNA 4 and added Nyquist validation artifacts |
| v1.9 | 1+ | 6 | Completed the RDNA 4-scoped AMD SOL/SOLAR bound modeling evidence chain from hardware model through score report |
| v1.10 | 1+ | 6 | Completed paper-aligned automatic SOLAR derivation sidecars, score guards, dataset-runner report integration, and claim guardrails |
| v1.19 | 1+ | 6 | Added sidecar evidence credibility surfaces without expanding hardware validation |
| v1.20 | 1+ | 5 | Added consistency, stability, claim-upgrade, and trust-summary gates over existing evidence |
| v1.29 | 1+ | 5 | Added local dataset migration, provenance, readiness subsets, low-precision compatibility, and runner guardrails without redistributing restricted data |

### Cumulative Quality

| Milestone | Tests | Coverage | Deferred Items |
|-----------|-------|----------|----------------|
| v1.0 | 462 passed, 58 skipped on RDNA 4 | v1 requirements 38/39 complete, 1 deferred | TEST-05 CDNA 3 full-suite validation |
| v1.2 | 16 focused tests passed; ruff clean | v1.2 requirements 14/14 complete | CDNA 3 real hardware validation; AMD-native scoring model |
| v1.5 | 42 focused milestone tests passed; ruff clean | v1.5 requirements 20/20 complete | Real CDNA 3 hardware validation; broader AMD SOL operator coverage |
| v1.7 | 67 focused audit tests passed | v1.7 requirements 21/21 complete | MI300X full-suite validation on CDNA3; FP8 real-hardware validation; paper extraction; full SOLAR parity |
| v1.8 | 41 focused tests passed; ruff clean; Docker entrypoint syntax passed | v1.8 requirements 23/23 complete | CDNA 3/CDNA 4 library validation; complete local ROCm development headers for native E2E |
| v1.9 | 40 focused AMD bound/model/score/contract tests passed; `uv build` passed | v1.9 requirements 28/28 complete | CDNA3-family real-hardware validation, including MI300X; CDNA 4 validation; paper extraction; broader upstream SOLAR parity |
| v1.10 | 196 Phase 51 tests and 169 Phase 52 tests passed; Ruff clean; milestone audit PASS | v1.10 requirements 21/21 complete | Paper-scale 124-model/235-problem extraction; MI300X-on-CDNA3 validation; CDNA4 validation; NVFP4/MXFP4 validation; hosted leaderboard |
| v1.19 | 74 focused audit tests passed; Ruff clean | v1.19 requirements 28/28 complete | New hardware validation; paper-scale validation; hosted leaderboard |
| v1.20 | 74 evidence-quality tests passed; docs regression passed; audit PASS | v1.20 requirements 23/23 complete | MI300X-on-CDNA3 validation; CDNA4 validation; full paper validation; hosted leaderboard; profiling diagnostics |
| v1.29 | 74 Phase 135 focused tests passed; phase-specific suites passed; Ruff clean; milestone audit PASS | v1.29 requirements 24/24 complete | Real CDNA3 or CDNA4 full-suite execution; CDNA4 low-precision validation/performance authority; FlashInfer kernel performance tuning; NVIDIA dataset redistribution |

### Top Lessons

1. Hardware-specific requirements need hardware-specific evidence before support claims are made.
2. Compatibility wrappers can preserve caller stability while changing the underlying ROCm implementation path.
3. Engineering-practice borrowing should be protected by explicit public-contract tests.
4. Support matrices should only use "supported" for categories with runnable
   examples, tests, and scope-specific docs.
5. Derived scoring artifacts need explicit degradation semantics so unsupported
   evidence cannot inflate reported AMD-native scores.
6. Rich derived sidecars should remain isolated from canonical benchmark
   contracts and backed by exact parser, public-key, and claim-boundary tests.
7. Evidence-quality milestones need verified end-to-end report chains so
   downstream claim and trust reports cannot drift from upstream evidence inputs.
