# Phase 80: uv And PyTorch ROCm Wheel Coordination - Context

**Gathered:** 2026-05-28
**Status:** Ready for planning

<domain>
## Phase Boundary

This phase adds per-Target PyTorch ROCm dependency policy and dependency
preflight classification before any clean validation claim is made. It builds on
the Phase 79 Docker Target manifest and wrapper integration by recording the
expected PyTorch ROCm wheel/index policy for each declared Target, preserving the
default ROCm 7.1 dependency path, detecting wrong or unavailable dependency
stacks, and blocking illegal mixed-version validation by default. It does not
emit full runtime evidence reports, aggregate compatibility reports, live ROCm
validation docs, or CI documentation guardrails beyond the focused tests needed
to prove dependency policy behavior.

</domain>

<decisions>
## Implementation Decisions

### Dependency Policy Source
- Extend `docker/rocm-targets.json` so each declared Docker Target records its
  PyTorch ROCm wheel/index policy near the Target that consumes it.
- Record expected `torch` and `torchvision` local-version tags, the uv index or
  lock strategy, and any `triton-rocm` policy needed for mismatch detection.
- Preserve the current `pyproject.toml` ROCm 7.1 dependency path as the default
  project install path unless a per-Target dependency workflow is explicitly
  selected and recorded.
- For v1, record recommended uv index and command/strategy metadata rather than
  forcing automatic per-Target lockfile generation.

### Dependency Detection And Classification
- Detect installed PyTorch stack state from installed package metadata plus
  runtime probes such as `torch.__version__`, `torch.version.hip`,
  `torch.version.cuda`, and PyTorch device availability.
- Classify missing or policy-unsupported PyTorch ROCm wheels as
  `pytorch_wheel_unavailable`, not as benchmark failures.
- Classify installed-but-wrong CPU, CUDA, wrong-index, wrong-ROCm, Triton ROCm,
  or toolchain mismatches as `mixed_version` with specific reason codes where
  the Phase 78 vocabulary permits it.
- Treat dependency probe failures conservatively as diagnostic blockers for
  clean validation, without changing canonical benchmark correctness, timing,
  scoring, or exit semantics.

### Blocking And Debug Override
- Block illegal `mixed_version` dependency states during preflight before clean
  validation or benchmark-authority claims by default.
- Provide an explicit debug override that may continue dependency probes or
  smoke execution, while keeping `container_validated`, `host_validated`,
  `benchmark_allowed`, score authority, paper-parity authority, and leaderboard
  authority false for the resulting Matrix decision.
- Wire dependency policy/preflight JSON into `scripts/run_docker.sh` before
  Docker build/run so illegal combinations do not proceed silently.
- Do not reuse the unknown Docker Target unsafe override for dependency
  mismatches; dependency mismatch override naming should be explicit in logs,
  tests, and diagnostics.

### Test Strategy
- Add CPU-safe tests for dependency policy schema, default 7.1 preservation,
  missing-wheel classification, mixed-version classification, CLI JSON output,
  script blocking, and debug override behavior.
- Avoid live ROCm tests in this phase; live host/container/Python/GPU evidence
  and marker-gated validation guidance belong to later phases.
- Keep tests focused on dependency policy and classification while reusing the
  Phase 78 Matrix contract and Phase 79 Docker Target selection paths.

### the agent's Discretion
The agent may choose exact helper names, JSON field names, and internal model
shape as long as the policy is checked in, auditable, CPU-testable, tied to
declared Docker Targets, and keeps the existing ROCm 7.1 default install path
stable.

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- Phase 78 added `src/sol_execbench/core/compatibility.py` with bounded Matrix
  statuses including `mixed_version` and `pytorch_wheel_unavailable`, reason
  codes, claim boundaries, and benchmark execution decisions.
- Phase 79 added `docker/rocm-targets.json` and
  `src/sol_execbench/core/docker_matrix.py`, which already load Target policy,
  emit JSON previews/preflight output, and gate `scripts/run_docker.sh`.
- `pyproject.toml` currently defaults Linux/Windows torch and torchvision to
  `2.10.0+rocm7.1` and `0.25.0+rocm7.1` through the explicit
  `pytorch-rocm71` uv index.
- `pyproject.toml` currently sources `triton-rocm==3.6.0` from the explicit
  `pytorch-rocm-root` uv index on Linux.

### Established Patterns
- Keep reusable compatibility and selection logic in Python modules under
  `src/sol_execbench/core/`; keep shell script parsing and command assembly thin.
- Preserve default behavior when no explicit Target or override is selected.
- Use CPU-safe tests under `tests/sol_execbench/` for schema validation,
  classifier behavior, CLI JSON, shell dry-runs, and preflight gating.
- Diagnostic compatibility evidence must not grant score, paper-parity,
  leaderboard, or native-host authority.

### Integration Points
- `docker/rocm-targets.json` is the natural source for per-Target dependency
  policy because it already stores declared ROCm Docker Target policy.
- `src/sol_execbench/core/docker_matrix.py` can expose dependency policy in
  preview/preflight payloads or delegate to a focused dependency helper.
- `scripts/run_docker.sh` already reads helper JSON, supports `--target`,
  `--allow-unknown-target`, and `--preflight-only`, and gates Docker build/run
  on Matrix decisions.
- Phase 81 will consume dependency evidence alongside host/container/Python/GPU
  evidence to emit per-Target and aggregate compatibility reports.

</code_context>

<specifics>
## Specific Ideas

- Prefer manifest fields that make the expected wheel local-version visible,
  for example `expected_torch_local_version` and
  `expected_torchvision_local_version`.
- Preserve the current ROCm 7.1 lock/install path and make per-Target dependency
  workflows explicit rather than automatic.
- Treat unsupported ROCm wheel availability as a declared policy outcome when
  upstream wheels are unavailable, not as a runtime benchmark failure.
- Add invalid/mismatched installed-stack fixtures rather than importing live
  PyTorch in most unit tests.

</specifics>

<deferred>
## Deferred Ideas

- Full host/container/Python/toolchain/GPU runtime evidence collection and
  aggregate compatibility report emission belong to Phase 81.
- User-facing validation workflow docs, CI guardrails, and marker-gated live
  ROCm validation guidance belong to Phase 82.
- Automatic host ROCm driver management, arbitrary undeclared ROCm image tags,
  and paper-scale validation remain out of scope for v1.18.

</deferred>
