# Documentation Map

This page is the entry point for documentation under `docs/`. The root
[`README.md`](../README.md) remains the quickest installation guide.

## User documentation

[`user/`](user/) contains the documents intended for benchmark users,
integrators, and researchers. Start with:

- [Getting Started](user/GETTING-STARTED.md), [Cookbook](user/COOKBOOK.md), and
  [Configuration](user/CONFIGURATION.md)
- [Architecture](user/ARCHITECTURE.md), [Development](user/DEVELOPMENT.md), and
  [Testing](user/TESTING.md)
- [Evaluator Contract](user/EVALUATOR-CONTRACT.md), [Definition Schema](user/definition.md),
  [Workload Schema](user/workload.md), [Solution Schema](user/solution.md), and
  [Trace Schema](user/trace.md)
- [Researcher Guide](user/RESEARCHER-GUIDE.md), [ROCm Setup](user/rocm.md),
  [ROCm Timing](user/rocm_timing.md),
  [ROCm Toolchain Routing](user/rocm_toolchain_routing.md), and
  [ROCm Libraries](user/rocm_libraries.md)
- [Release and Official Score Workflow](user/RELEASE-SCORING.md),
  [AMD SOL Bound Evidence](user/amd_sol.md), and
  [Confirmed Evidence](user/confirmed_evidence.md)
- [Static Kernel Evidence](user/static_kernel_evidence.md),
  [Decision Sidecar](user/decision_sidecar.md),
  [Decision Sidecar Contract](user/decision_sidecar_contract.md),
  [Profile Summary Sidecar](user/profile_summary_sidecar.md), and
  [Agent Feedback Sidecar](user/agent_feedback_sidecar.md)
- [Schema Boundaries](user/schema-boundaries.md),
  [Research Preview](user/research_preview.md), and
  [Public Prerelease](user/public_prerelease.md)
- [Claims](user/CLAIMS.md), [Provenance](user/provenance.md), and
  [Compliance](user/compliance.md)

User documentation defines supported interfaces and public claim boundaries.
It may link to evidence records, but those records do not silently expand the
product contract.

## Internal documentation

[`internal/`](internal/) contains maintainer-oriented architecture notes,
validation records, release processes, research analyses, decision records, and
historical plans. It is not user-facing product documentation.

Key entry point: [Analysis](internal/analysis.md). The AMD SOL accuracy audit
and its closure strategy are also internal records:
[accuracy gap](internal/amd_sol_bound_accuracy_gap.md) and
[closure strategy](internal/amd_sol_bound_closure_strategy.md).
Historical implementation plans and design specifications are kept under
[`internal/superpowers/`](internal/superpowers/).

Internal documents may explain implementation tradeoffs or record incomplete
validation. They must not be cited as public capability or authority without
corresponding user documentation and evidence.

## Examples and release records

| Location | Contents | How to use it |
| --- | --- | --- |
| [`examples/`](examples/) | Small, non-authoritative example evidence fixtures. | Use only to understand artifact shape or test fixtures. |
| [`releases/`](releases/) | Versioned release drafts and machine-readable evidence manifests/lifecycle records. | Treat each record as release-scoped evidence, not a current product contract. |

## Maintenance rule

Put user-visible behavior, supported schemas, configuration, and public policy
in `docs/user/`. Put implementation plans, audits, validation attempts,
maintainer procedures, and historical analyses in `docs/internal/`. Update the
matching focused tests whenever a path or contract changes.
