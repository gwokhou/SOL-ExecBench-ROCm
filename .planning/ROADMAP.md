# Roadmap: SOL ExecBench ROCm Port

## Milestones

- 🚧 **v1.11 Paper Dataset Parity Inventory and ROCm Execution Closure** —
  Phases 53-57 (active). This milestone makes the public paper dataset surface
  concrete and auditable for the ROCm port through acquisition/layout metadata,
  machine-readable inventory, ROCm readiness classification, bounded ready-set
  execution closure, parity gap reports, and claim guardrails.

- ✅ **v1.10 Paper-Aligned SOLAR Automatic Derivation** — Phases 47-52
  (shipped 2026-05-23). See `.planning/milestones/v1.10-ROADMAP.md`.

- ✅ **v1.9 AMD SOL/SOLAR Bound Modeling Completion** — Phases 41-46
  (shipped 2026-05-23). See `.planning/milestones/v1.9-ROADMAP.md`.

- ✅ **v1.8 ROCm Library Ecosystem Completion** — Phases 36-40 (shipped
  2026-05-22). See `.planning/milestones/v1.8-ROADMAP.md`.

- ✅ **v1.7 Baseline, Timing, Reward-Hack Hardening, and ROCm Library
  Migration** — Phases 31-35 (shipped 2026-05-22). See
  `.planning/milestones/v1.7-ROADMAP.md`.

- ✅ **v1.6 AMD SOLAR Coverage, Live Profiler Timing, and Scoring Workflow** —
  Phases 27-30 (shipped 2026-05-22). See
  `.planning/milestones/v1.6-ROADMAP.md`.

- ✅ **v1.5 AMD-native SOL Scoring and ROCm Profiler Timing** — Phases 23-26
  (shipped 2026-05-22). See `.planning/milestones/v1.5-ROADMAP.md`.

- ✅ **v1.4 hip-execbench Engineering Experience Adaptation + Validation
  Workflow Readiness** — shipped 2026-05-22. See
  `.planning/milestones/v1.4-ROADMAP.md`.

- ✅ **v1.3 Non-CDNA Issue Closure** — shipped 2026-05-22. See
  `.planning/milestones/v1.3-ROADMAP.md`.

- ✅ **v1.2 Engineering Practice Harvest and Compatibility Guardrails** —
  shipped 2026-05-22. See `.planning/milestones/v1.2-ROADMAP.md`.

- ✅ **v1.1 CDNA 3 Support and Migration Closure** — shipped 2026-05-21. See
  `.planning/milestones/v1.1-ROADMAP.md`.

- ✅ **v1.0 ROCm Port** — shipped 2026-05-21. See
  `.planning/milestones/v1.0-ROADMAP.md`.

## Current Position

**Active milestone:** v1.11 Paper Dataset Parity Inventory and ROCm Execution
Closure.

**Status:** planning. Requirements are defined; roadmap and phase plans are
next.

## Active Milestone: v1.11 Paper Dataset Parity Inventory and ROCm Execution Closure

### Phase 53: Dataset Contract And Acquisition Metadata (completed 2026-05-23)

**Goal:** Establish a reproducible dataset acquisition and local-layout contract
for the public SOL-ExecBench benchmark categories without implying ROCm
execution parity.

**Requirements:** DATA-01, DATA-02, DATA-03, DATA-04

**Success criteria:**
- Public dataset roots can be checked for `L1`, `L2`, `Quant`, and
  `FlashInfer-Bench` category layout.
- Acquisition or local-layout manifests capture source, category set,
  provenance, discovered counts, and checksum metadata.
- Downloader/local-layout behavior is idempotent, category-selective, and keeps
  downloaded benchmark assets out of committed source files.
- Metadata and docs distinguish acquisition/layout completion from readiness,
  execution, and paper-level validation.

### Phase 54: Paper Inventory And ROCm Readiness Classification (completed 2026-05-23)

**Goal:** Generate a deterministic machine-readable inventory for every
discovered problem/workload and classify ROCm readiness with explicit blockers
and evidence.

**Requirements:** INV-01, INV-02, INV-03, INV-04, INV-05, READY-01, READY-02,
READY-03, READY-04, READY-05

**Success criteria:**
- Inventory generation uses the current Pydantic `Definition` and `Workload`
  contracts and records problem/workload metadata needed for parity audit.
- Category and suite denominators distinguish discovered, parsed, schema-failed,
  and missing-file states.
- Readiness classification is deterministic and uses stable statuses, reason
  codes, evidence paths, and next actions.
- Custom inputs, safetensors, low-precision, Quant, and hardware-evidence
  dependencies are represented explicitly without fabricating runnable data.
- Ready-subset manifests are sidecar artifacts and never mutate canonical
  dataset files.

### Phase 55: Ready Subset Selection And Bounded Execution Closure (completed 2026-05-23)

**Goal:** Run bounded ready subsets through the existing benchmark execution
path and join results back to readiness and evidence artifacts.

**Requirements:** EXEC-01, EXEC-02, EXEC-03, EXEC-04, EXEC-05

**Success criteria:**
- Ready-subset execution flows through `scripts/run_dataset.py` and the primary
  `sol-execbench` subprocess path rather than a second benchmark runner.
- Execution can be constrained by category, limit, workload cap, timeout,
  warmup, iteration count, rerun policy, and derived-evidence flags.
- Closure reports join readiness, canonical traces, summaries, logs,
  skipped-existing-pass states, missing traces, failures, and not-attempted
  items.
- Existing AMD-native score, AMD SOL v2, SOLAR derivation, and timing evidence
  artifacts can be referenced without mutating canonical trace JSONL.
- Closure artifacts record command, dataset checksum, git commit, solution mode,
  and benchmark config provenance.

### Phase 56: Parity Gap Reporting And Evidence Review (completed 2026-05-23)

**Goal:** Produce deterministic JSON and Markdown reports that explain dataset,
readiness, execution, scoring, and evidence gaps by category and suite.

**Requirements:** GAP-01, GAP-02, GAP-03, GAP-04, GAP-05

**Success criteria:**
- Gap reports combine acquisition, inventory, readiness, and execution-closure
  artifacts.
- Reports expose complete denominators for discovered, parsed, ready, blocked,
  not attempted, skipped, attempted, passed, failed, scored, degraded, and
  unscored items.
- Blockers are grouped by stable reason code with concrete next actions.
- Evidence completeness distinguishes plain execution, trace, timing,
  AMD-native score, AMD SOL, and SOLAR derivation evidence.
- Artifact references use safe deterministic paths.

### Phase 57: Claim Guardrails, Docs, And Release Closure (completed 2026-05-23)

**Goal:** Prevent v1.11 artifacts from being presented as full paper validation,
leaderboard parity, upstream SOLAR parity, or unsupported hardware validation.

**Requirements:** CLAIM-01, CLAIM-02, CLAIM-03, CLAIM-04, CLAIM-05

**Success criteria:**
- Public docs and generated reports clearly separate inventory/readiness/closure
  from full 235-problem ROCm validation and leaderboard equivalence.
- Deferred scope remains explicit for original extraction, upstream SOLAR
  equivalence, CDNA 3 / MI300X, CDNA 4, NVFP4, and MXFP4 validation.
- Public contract tests protect canonical schemas, trace JSONL, primary CLI
  behavior, AMD SOL v2 sidecars, and SOLAR derivation sidecars.
- Report wording tests prevent derived AMD-native scores or bounded execution
  results from being described as paper-level results.
- Milestone closure summarizes remaining gaps and evidence boundaries.

## Progress

| Milestone | Phases | Plans | Status | Shipped |
|-----------|--------|-------|--------|---------|
| v1.11 Paper Dataset Parity Inventory and ROCm Execution Closure | 53-57 | 14/14 | Complete | 2026-05-23 |
| v1.10 Paper-Aligned SOLAR Automatic Derivation | 47-52 | 23/23 | Complete | 2026-05-23 |
| v1.9 AMD SOL/SOLAR Bound Modeling Completion | 41-46 | 17/17 | Complete | 2026-05-23 |
| v1.8 ROCm Library Ecosystem Completion | 36-40 | 5/5 | Complete | 2026-05-22 |
| v1.7 Baseline, Timing, Reward-Hack Hardening, and ROCm Library Migration | 31-35 | 5/5 | Complete | 2026-05-22 |
| v1.6 AMD SOLAR Coverage, Live Profiler Timing, and Scoring Workflow | 27-30 | 4/4 | Complete | 2026-05-22 |
| v1.5 AMD-native SOL Scoring and ROCm Profiler Timing | 23-26 | 4/4 | Complete | 2026-05-22 |
| v1.4 hip-execbench Engineering Experience Adaptation + Validation Workflow Readiness | - | - | Complete | 2026-05-22 |
| v1.3 Non-CDNA Issue Closure | - | - | Complete | 2026-05-22 |
| v1.2 Engineering Practice Harvest and Compatibility Guardrails | - | - | Complete | 2026-05-22 |
| v1.1 CDNA 3 Support and Migration Closure | - | - | Complete | 2026-05-21 |
| v1.0 ROCm Port | - | - | Complete | 2026-05-21 |

## Future Candidate Work

- Original paper-scale 124-model / 235-problem extraction and curation.
- MI300X, CDNA 3, and CDNA 4 real-hardware validation.
- NVFP4 and MXFP4 validation if a suitable AMD hardware path exists.
- Hosted leaderboard or submission service.
- NVIDIA Blackwell/B200 comparison methodology, if ever scoped as a separate
  non-ROCm claim analysis effort.
