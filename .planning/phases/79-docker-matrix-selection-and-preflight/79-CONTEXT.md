# Phase 79: Docker Matrix Selection And Preflight - Context

**Gathered:** 2026-05-28
**Status:** Ready for planning

<domain>
## Phase Boundary

This phase adds declared Docker Matrix Target selection and conservative Docker
preflight classification before benchmark execution. It builds on the Phase 78
Matrix Entry contract by connecting checked-in ROCm Docker Targets to
`scripts/run_docker.sh`, `docker/Dockerfile`, and CPU-safe tests. It does not
solve PyTorch ROCm wheel/index policy, full runtime evidence collection,
aggregate report emission, or documentation/CI guardrail closure beyond what is
needed to prove Docker Target selection and preflight behavior.

</domain>

<decisions>
## Implementation Decisions

### Docker Target Manifest Scope
- Use a checked-in manifest of declared Docker Targets rather than ad hoc
  discovery of arbitrary ROCm image tags.
- Include logical ROCm Target entries for configured 7.0.x, 7.1.x, and 7.2.x
  where usable Docker image tags are known.
- Preserve the current ROCm 7.1 Docker behavior as the default path so existing
  `./scripts/run_docker.sh` and `docker/Dockerfile` usage continues to work.
- Reject unknown Targets unless the user supplies an explicit unsafe or
  untested override.

### Docker Preflight Classification
- Classify Docker Desktop or otherwise unsupported Docker contexts as
  `runtime_unavailable` before benchmark execution.
- Classify missing `/dev/kfd`, missing `/dev/dri`, and inaccessible GPU device
  access as `runtime_unavailable` before benchmark execution.
- Preflight failures should produce diagnostic Matrix-compatible evidence and
  should not continue into benchmark execution.
- These classifications are compatibility evidence only and must not imply
  correctness, timing, scoring, paper-parity, or leaderboard authority.

### Docker Evidence Fields
- Record exact requested image repository and image tag for the selected Target.
- Record resolved image digest when available, but treat digest resolution as
  best-effort and non-blocking.
- Record Docker build arguments used for the selected Target, including ROCm
  base-image selection inputs.
- Record selected Target id, validation scope, and runtime preflight details so
  downstream phases can emit full Matrix Entries and aggregate reports.

### Unsafe Or Untested Override Boundaries
- Unknown Target overrides may allow diagnostic probes or smoke-style setup
  checks only.
- Unknown or untested override paths must not emit `container_validated`,
  `host_validated`, benchmark eligibility, score authority, paper-parity
  authority, or leaderboard authority.
- Use Phase 78 `mixed_version`, `runtime_unavailable`, and `not_tested`
  semantics where applicable instead of inventing new status words.
- Keep override naming explicit enough that logs, tests, and docs cannot present
  the path as normal validation.

### the agent's Discretion
The agent may choose the exact manifest format, helper module names, and script
flag spelling as long as the result is checked in, auditable, CPU-testable, and
keeps the existing ROCm 7.1 default behavior stable.

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- Phase 78 added `src/sol_execbench/core/compatibility.py` with strict Matrix
  Target, observed evidence, claim-boundary, execution-decision, and report
  models.
- `scripts/run_docker.sh` already checks Docker context, `/dev/kfd`, and
  `/dev/dri`, then builds/runs `docker/Dockerfile` with ROCm devices mounted.
- `docker/Dockerfile` currently uses `rocm/dev-ubuntu-24.04:7.1.1-complete` as
  its fixed base image.
- Existing Docker dependency smoke tests live under `tests/docker/dependencies/`.

### Established Patterns
- Keep reusable logic in Python modules under `src/sol_execbench/core/`; keep
  shell scripts thin where possible.
- Prefer CPU-safe unit tests under `tests/sol_execbench/` for parsing,
  selection, command construction, and preflight classification.
- Runtime failures should be classified before expensive benchmark execution and
  should remain diagnostic evidence.
- Project claim boundaries forbid Docker container user-space validation from
  being described as native host validation.

### Integration Points
- `scripts/run_docker.sh` is the primary user entry point for Docker execution.
- `docker/Dockerfile` needs parameterized ROCm base-image selection while
  retaining the 7.1 default.
- Phase 79 should feed the Phase 78 `MatrixTarget`, `MatrixObservedEvidence`,
  `MatrixCompatibilityStatus.RUNTIME_UNAVAILABLE`, and claim-boundary
  vocabulary so later phases can emit complete compatibility reports.
- Phase 80 will own uv/PyTorch ROCm wheel coordination; Phase 79 should record
  selected Target policy fields without resolving wheel availability.

</code_context>

<specifics>
## Specific Ideas

- Consider a small checked-in JSON manifest for Docker Targets, with stable
  Target ids and requested image repository/tag fields.
- Keep default script invocation behavior unchanged when no Target is selected.
- Add shell-testable helpers or generated command previews so unknown Target
  rejection and build argument construction can be tested without live Docker or
  ROCm hardware.
- Treat missing image digest as explicit `None`/unavailable evidence, not as a
  failing validation result.

</specifics>

<deferred>
## Deferred Ideas

- PyTorch ROCm wheel local-version tags, uv index selection, and mixed-version
  dependency policy belong to Phase 80.
- Full host/container/Python/toolchain/GPU runtime evidence collection and
  aggregate compatibility report emission belong to Phase 81.
- User-facing validation workflow docs, CI guardrails, and live marker-gated
  ROCm validation guidance belong to Phase 82.

</deferred>
