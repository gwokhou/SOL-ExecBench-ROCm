# Documentation

Start with [Getting started](user/GETTING-STARTED.md), then use the
[Cookbook](user/COOKBOOK.md) for common commands or the
[Researcher guide](user/RESEARCHER-GUIDE.md) for experiment-recording
requirements.

## Public contracts and workflows

- [Definition](user/definition.md), [Workload](user/workload.md),
  [Solution](user/solution.md), and [Trace](user/trace.md) define the accepted
  benchmark inputs and canonical result.
- [Schema boundaries](user/schema-boundaries.md) and the
  [Evaluator contract](user/EVALUATOR-CONTRACT.md) describe strict parsing and
  the machine-readable capability surface.
- [Architecture](user/ARCHITECTURE.md), [Configuration](user/CONFIGURATION.md),
  [Cross-path comparison](user/CROSS-PATH-COMPARISON.md),
  and the [SOLAR boundary](SOLAR-BOUNDARY.md) describe process ownership and
  formal analysis.
- [Scoring contract](SCORING-V3.md) and
  [Release scoring](user/RELEASE-SCORING.md) distinguish diagnostic formula
  results from the publisher-authored, content-addressed official-score workflow.
- [Current claim boundaries](user/CLAIMS.md) state exactly what checked-in and
  generated evidence can establish.

## ROCm operation and validation

- [ROCm environment](user/rocm.md)
- [ROCm library categories](user/rocm_libraries.md)
- [Timing and profiling](user/rocm_timing.md)
- [Toolchain routing](user/rocm_toolchain_routing.md)
- [RDNA4 validation scope](user/RDNA4-VALIDATION.md)
- [Testing](user/TESTING.md)

## Diagnostic sidecars

- [Static-kernel evidence](user/static_kernel_evidence.md)
- [Profile summary](user/profile_summary_sidecar.md)
- [Decision sidecar](user/decision_sidecar.md) and its
  [derivation contract](user/decision_sidecar_contract.md)
- [Agent feedback](user/agent_feedback_sidecar.md)

These artifacts are diagnostic. Their guides state the current schema
identifiers and authority limits; none can replace canonical Trace JSONL or
grant official-score authority.

## Policy and maintenance

- [Provenance](user/provenance.md) and [Compliance](user/compliance.md)
- [Development](user/DEVELOPMENT.md)
- [Architecture navigation](internal/architecture_navigation.md) and
  [coupling governance](internal/coupling_governance.md)
- [AKA task-source research](internal/aka-sol-task-source-research.md),
  [corpus friendliness policy](internal/aka-expansion-friendliness.md), and
  [decision-modeling research](internal/decision-modeling-research.md)

Internal research documents record the design basis for policies still
referenced by the implementation. They are not current readiness reports or
release-status claims.

This tree documents only the current implementation. Superseded contracts,
completed readiness snapshots, and engineering history are available from Git
history rather than checked-in archive files.
