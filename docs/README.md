# Documentation Map

This page classifies the repository documentation by audience, purpose, and
authority. It is the entry point for documentation under `docs/`; the root
[`README.md`](../README.md) remains the shortest path to installation and a
first evaluation.

## Read by task

| Need | Read | Status |
| --- | --- | --- |
| Install, configure, and run a first evaluation | [Getting Started](GETTING-STARTED.md), [Cookbook](COOKBOOK.md), [Configuration](CONFIGURATION.md) | Current user guide |
| Understand the stable CLI and integration boundary | [Evaluator Contract](EVALUATOR-CONTRACT.md), [Trace Schema](trace.md), [Schema Boundaries](schema-boundaries.md) | Current reference |
| Create or adapt kernel solutions | [Researcher Guide](RESEARCHER-GUIDE.md), [Solution Schema](solution.md), [ROCm Libraries](rocm_libraries.md) | Current workflow/reference |
| Develop or validate the project | [Architecture](ARCHITECTURE.md), [Development](DEVELOPMENT.md), [Testing](TESTING.md) | Current maintainer guide |
| Work with ROCm runtime, timing, or tool diagnostics | [ROCm Setup](rocm.md), [ROCm Timing](rocm_timing.md), [ROCm Toolchain Routing](rocm_toolchain_routing.md) | Current operational reference |
| Interpret evidence or make public claims | [Claims](CLAIMS.md), [Confirmed Evidence](confirmed_evidence.md), [Evidence Lifecycle](EVIDENCE-LIFECYCLE.md), [Evidence Publication](EVIDENCE-PUBLICATION.md) | Current authority/process reference |
| Review a release or research result | [Research Preview](research_preview.md), [Prerelease Readiness](prerelease_readiness.md), and the relevant item in [`releases/`](releases/) | Release-scoped or historical evidence |

## Current product and maintainer documentation

### User workflows and configuration

- [Getting Started](GETTING-STARTED.md), [Cookbook](COOKBOOK.md), and
  [Configuration](CONFIGURATION.md)
- [Researcher Guide](RESEARCHER-GUIDE.md) and [Analysis and Profiling](analysis.md)
- [ROCm Setup](rocm.md), [ROCm Timing](rocm_timing.md),
  [ROCm Toolchain Routing](rocm_toolchain_routing.md), and
  [ROCm Library Category Readiness](rocm_libraries.md)

### Contracts, schemas, and diagnostic sidecars

- [Evaluator Contract](EVALUATOR-CONTRACT.md), [Schema Boundaries](schema-boundaries.md),
  [Definition Schema](definition.md), [Workload Schema](workload.md),
  [Solution Schema](solution.md), and [Trace Schema](trace.md)
- [Static Kernel Evidence](static_kernel_evidence.md),
  [Agent Feedback Sidecar](agent_feedback_sidecar.md),
  [Profile Summary Sidecar](profile_summary_sidecar.md),
  [Decision Sidecar](decision_sidecar.md), and
  [Decision Sidecar Contract](decision_sidecar_contract.md)

### Project governance and development

- [Architecture](ARCHITECTURE.md), [Development](DEVELOPMENT.md), and [Testing](TESTING.md)
- [Claims and Evidence Boundaries](CLAIMS.md), [Provenance](provenance.md), and
  [Compliance](compliance.md)
- [Evidence Lifecycle](EVIDENCE-LIFECYCLE.md), [Evidence Publication](EVIDENCE-PUBLICATION.md),
  [Confirmed Evidence Consumer Guide](confirmed_evidence.md), and
  [AMD SOL Bound Evidence](amd_sol.md)

## Evidence, research, and release material

These documents describe bounded evidence. They do not expand the current
product contract or turn derived, diagnostic, or historical artifacts into
benchmark authority. Use [Claims](CLAIMS.md) to interpret their limits.

- Current review workflows: [Research Preview](research_preview.md),
  [Prerelease Artifact Bundle](prerelease_artifact_bundle.md),
  [Prerelease Readiness](prerelease_readiness.md),
  [Release Candidate Validation](release_candidate_validation.md), and
  [Public Prerelease Publishing Guide](public_prerelease.md)
- Versioned guides and closure records: [v1.19 Evidence Guide](v1_19_evidence_guide.md),
  [v1.20 Evidence Quality Guide](v1_20_evidence_quality_guide.md),
  [v1.25 Engineering Prerelease Notes](v1_25_release_notes.md),
  [v1.25 Prerelease Checklist](v1_25_prerelease_checklist.md),
  [v1.15 Release Closure](v1_15_release_closure.md), and
  [v1.11 Release Closure](v1_11_release_closure.md)
- Research and compatibility analyses: [AMD Authority Slice](authority_slice.md),
  [Curated ROCm Benchmark Slice](curated_rocm_slice.md),
  [Original SOL ExecBench Parity](original_parity.md),
  [SOL Score Gap and AMD Reuse Report](sol_score_gap_and_amd_reuse_report.md), and
  [Decision Sidecar Modeling Research](decision-modeling-research.md)

## Directory classification

| Location | Contents | How to use it |
| --- | --- | --- |
| `docs/` | Current guides, contracts, schemas, policies, plus explicitly versioned evidence records. | Start from the task map above; check each document's stated evidence boundary. |
| [`examples/`](examples/) | Small, non-authoritative example evidence fixtures. | Use only to understand artifact shape or test fixtures. |
| [`releases/`](releases/) | Release drafts and Git-tracked release manifests/lifecycle records. | Treat JSON lifecycle/manifest files as release-scoped evidence metadata; Markdown drafts are not a current product contract. |
| [`internal/`](internal/) | Maintainer decision records, validation attempts, inventories, and closure notes. | Internal context only; do not cite as public capability or authority without the corresponding public evidence. |
| [`superpowers/specs/`](superpowers/specs/) | Historical design specifications. | Background for an implemented change, not an implementation-status source. |
| [`superpowers/plans/`](superpowers/plans/) | Historical implementation plans. | Background only; inspect code, tests, and current guides for actual behavior. |

## Maintenance rule

When behavior, a contract, or an evidence boundary changes, update the matching
current guide above and its focused tests. Keep release, internal, and planning
records in their classified locations; do not silently rewrite them as current
product documentation.
