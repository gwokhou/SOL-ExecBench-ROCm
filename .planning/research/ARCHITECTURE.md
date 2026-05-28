# Architecture Research: v1.18 ROCm Version Matrix via Docker

**Project:** SOL ExecBench ROCm Port
**Scope:** Integration architecture for Docker-selectable ROCm user-space versions, uv/PyTorch ROCm wheel handling, compatibility evidence, reports, docs, and tests.
**Researched:** 2026-05-28
**Confidence:** HIGH for repository integration points, MEDIUM for exact future PyTorch ROCm wheel availability.

## Current Integration Points

### CLI and Sidecars

- `src/sol_execbench/cli/main.py` is the right integration point for run-adjacent evidence because it already owns trace output paths and writes optional sidecars after evaluation.
- Current sidecar pattern:
  - trace JSONL remains canonical benchmark output.
  - environment snapshots are opt-in and written as `<trace>.environment.json` through `SOL_EXECBENCH_ENVIRONMENT_SNAPSHOT*`.
  - profiling and static kernel evidence are diagnostic sidecars and explicitly avoid score/correctness/timing authority.
- `sol-execbench doctor --json` is the existing GPU/environment diagnostics command and should expose matrix-readiness diagnostics without requiring a benchmark run.
- `sol-execbench contract --json` is the existing compatibility metadata surface and should carry claim-boundary text if v1.18 changes public evidence expectations.

### Data Models

- Canonical `Trace` models live in `src/sol_execbench/core/data/trace.py`. Do not add Docker matrix state to `Trace` or `Evaluation.environment` unless public benchmark semantics intentionally change.
- Optional environment evidence lives in `src/sol_execbench/core/environment.py` with strict Pydantic payloads and injectable probe runners. This is the best home for host/container/PyTorch/toolchain version evidence.
- Generic derived-report helpers live in `src/sol_execbench/core/reporting.py`. Compatibility matrix aggregation should follow this pattern: derived, trace-adjacent, and non-authoritative.
- Tool availability/routing already lives in `src/sol_execbench/core/toolchain.py` and `src/sol_execbench/core/diagnostics.py`; use those only for toolchain checks, not Docker orchestration.

### Docker Scripts

- `docker/Dockerfile` currently pins `rocm/dev-ubuntu-24.04:7.1.1-complete` and runs `uv sync --frozen --all-groups`.
- `scripts/run_docker.sh` owns image tag/name, build args, native Linux Docker checks, device passthrough, repository mount, and runtime env passthrough.
- `docker/entrypoint.sh` owns container startup checks and clock locking before delegating to the user command.
- Docker version selection should be implemented at this operational boundary, then reflected into package diagnostics through environment variables and probes. The core evaluator should not call Docker.

### uv and PyTorch ROCm Wheels

- `pyproject.toml` currently pins Linux/Windows PyTorch to `2.10.0+rocm7.1` and `torchvision` to `0.25.0+rocm7.1`, with `pytorch-rocm71` and root PyTorch wheel indexes.
- v1.18 needs a project-owned selection/detection layer because Docker base ROCm user-space and PyTorch wheel tags can diverge.
- Wheel availability is an external moving target. The architecture should support `available`, `unavailable`, and `unknown/not_tested` outcomes rather than assuming every ROCm Docker tag has a matching current PyTorch wheel.

### Docs and Guardrails

- Existing docs already distinguish ROCm claims, Docker usage, environment snapshots, profiling/static evidence, and original parity boundaries.
- v1.18 must add an explicit boundary: container user-space validation is not the same as native host ROCm validation. Host kernel/driver compatibility still matters because `/dev/kfd` and `/dev/dri` are passed through from the host.

## Proposed Components

### 1. Docker Matrix Configuration

**Modify:** `docker/Dockerfile`, `scripts/run_docker.sh`
**Optional new file:** `docker/rocm-matrix.json`

- Add `ARG ROCM_VERSION=7.1.1` and construct the base image as `rocm/dev-ubuntu-24.04:${ROCM_VERSION}-complete`.
- Add `ROCM_VERSION`, `PYTORCH_ROCM_INDEX`, and `PYTORCH_ROCM_TAG` build args/env vars, but keep defaults matching the current repository state.
- Make `scripts/run_docker.sh` accept a small, explicit flag such as `--rocm 7.1.1` and derive:
  - image tag, for example `sol-execbench:rocm7.1.1`.
  - Docker build arg `ROCM_VERSION=7.1.1`.
  - PyTorch ROCm wheel selector env/build arg.
- Prefer a checked-in mapping table over ad hoc shell string parsing:
  - Docker ROCm user-space tag.
  - expected PyTorch wheel index suffix/tag.
  - support status: `supported`, `wheel_unavailable`, `experimental`, `not_tested`.

**Why:** Docker matrix selection is operational infrastructure. Keeping it in Docker scripts avoids contaminating benchmark models with container mechanics.

### 2. PyTorch ROCm Wheel Strategy

**Modify:** `pyproject.toml`, `uv.lock`, `docker/Dockerfile`
**Optional new script:** `scripts/check_pytorch_rocm_wheel.py`

- Keep the default lockfile on one validated baseline for normal local development.
- For Docker matrix builds, pass a selected PyTorch index through `uv` configuration rather than editing canonical project metadata at runtime.
- If multiple ROCm wheels are officially available, represent them as named uv indexes in `pyproject.toml` and choose through build-time env/constraints.
- If a requested matrix row has no matching wheel, fail early during Docker build or emit a `pytorch_wheel_unavailable` report row without pretending the environment was validated.

**Why:** The repository needs reproducible default dependencies, but matrix Docker builds need controlled variant selection. Variant handling belongs to Docker build orchestration and diagnostics, not trace serialization.

### 3. Compatibility Evidence Models

**Modify:** `src/sol_execbench/core/environment.py`
**New module likely:** `src/sol_execbench/core/compatibility.py`

Add a small compatibility model family:

- `RocmVersionEvidence`
  - host driver/runtime evidence from host-side probes when available.
  - container ROCm user-space evidence from `/opt/rocm/.info/version*`, `hipcc --version`, `rocminfo`, and environment variables.
  - PyTorch evidence from `torch.__version__`, `torch.version.hip`, device availability, device name, and gfx target.
  - Triton/toolchain evidence from `triton`, `triton-rocm`, `hipcc`, `rocprofv3`, `rocm_agent_enumerator`.
- `CompatibilityMatrixRow`
  - requested ROCm version.
  - container image tag.
  - selected/observed PyTorch ROCm tag.
  - host ROCm evidence.
  - container ROCm evidence.
  - GPU architecture.
  - status enum: `host_validated`, `container_validated`, `mixed_version`, `pytorch_wheel_unavailable`, `runtime_unavailable`, `not_tested`.
  - warnings and claim boundaries.
- `CompatibilityMatrixReport`
  - schema version such as `sol_execbench.rocm_compatibility_matrix.v1`.
  - generated timestamp.
  - rows.
  - `diagnostic_only=true`.
  - authority booleans all false for correctness, timing, scoring, paper parity, and leaderboard claims.

Keep this separate from `Trace`. `EnvironmentSnapshot` can embed detailed version evidence, while `CompatibilityMatrixReport` aggregates one or more snapshots into a researcher-facing report.

### 4. CLI Surfaces

**Modify:** `src/sol_execbench/cli/main.py`

Recommended additions:

- `sol-execbench doctor --json` includes compatibility checks derived from the current environment.
- A focused command such as `sol-execbench compatibility --json` can emit the current row/report without running a benchmark.
- Evaluation CLI may gain `--compatibility-report auto|none` only if reports must be trace-adjacent for benchmark runs. Default should be `none` or environment-gated to preserve current behavior.

Do not make Docker matrix execution a `sol-execbench` subcommand. Let `scripts/run_docker.sh` run Docker and call `sol-execbench doctor --json` or `sol-execbench compatibility --json` inside the container.

### 5. Docker-to-Evidence Handshake

**Modify:** `scripts/run_docker.sh`, `docker/entrypoint.sh`, `src/sol_execbench/core/environment.py`

Pass these environment variables into the container:

- `SOL_EXECBENCH_REQUESTED_ROCM_VERSION`
- `SOL_EXECBENCH_CONTAINER_IMAGE`
- `SOL_EXECBENCH_PYTORCH_ROCM_INDEX`
- `SOL_EXECBENCH_COMPATIBILITY_REPORT`

The package records them as requested configuration, then separately records observed values from probes. Status is computed from requested-vs-observed evidence. This prevents shell scripts from being the source of truth for validation.

### 6. Docs and Claim Guardrails

**Modify:** `README.md`, `docs/rocm.md`, `docs/CONFIGURATION.md`, `docs/CLAIMS.md`
**New doc likely:** `docs/rocm_version_matrix.md`

Documentation should state:

- how to build/run each supported Docker ROCm user-space image.
- how PyTorch ROCm wheels are selected and what unavailable means.
- what report files are produced.
- why container validation is `container_validated` unless host-native evidence is separately collected.
- that trace JSONL remains the canonical benchmark result, while compatibility reports are environment/reproducibility evidence.

## Data Flow

```text
User selects ROCm row
  |
  v
scripts/run_docker.sh --rocm X.Y.Z
  |
  |-- validates native Linux Docker + /dev/kfd + /dev/dri
  |-- derives image tag and wheel selector
  |-- docker build --build-arg ROCM_VERSION --build-arg PYTORCH_ROCM_*
  v
Docker image
  |
  |-- base image supplies ROCm user-space
  |-- uv installs selected/default PyTorch ROCm wheel
  v
docker/entrypoint.sh
  |
  |-- records requested env
  |-- performs existing clock-lock startup behavior
  v
sol-execbench doctor/compatibility/evaluate
  |
  |-- core.environment probes observed ROCm/PyTorch/Triton/toolchain/GPU values
  |-- core.compatibility classifies status
  |-- CLI writes report sidecar or standalone JSON
  v
Artifacts
  |
  |-- trace.jsonl                         canonical benchmark output
  |-- trace.jsonl.environment.json        optional environment evidence
  |-- trace.jsonl.compatibility.json      optional diagnostic compatibility report
  |-- matrix report JSON/Markdown         derived compatibility overview
```

Rules:

- The generated evaluation driver continues to emit only strict trace JSONL.
- Compatibility collection happens in the parent CLI or standalone diagnostic command after/beside evaluation.
- `scripts/run_dataset.py` may pass through compatibility-report options and copy report paths into run closure metadata, but should not compute compatibility status itself.
- Docker scripts provide requested matrix coordinates; Python probes provide observed evidence; the compatibility model compares them.

## Build Order

1. **Schema and status vocabulary**
   - Add compatibility evidence/report models and tests first.
   - Lock status values: `host_validated`, `container_validated`, `mixed_version`, `pytorch_wheel_unavailable`, `runtime_unavailable`, `not_tested`.
   - Add authority/claim-boundary fields from the start.

2. **Environment probe expansion**
   - Extend `EnvironmentSnapshot` or add adjacent compatibility collection to record ROCm user-space version, PyTorch ROCm tag, Triton version, hipcc version, and requested Docker matrix env vars.
   - Keep probe runners injectable and timeout-bounded.

3. **CLI diagnostic/report surface**
   - Add `compatibility --json` or extend `doctor --json`.
   - Add optional trace-adjacent compatibility sidecar only after standalone reporting works.

4. **Docker matrix plumbing**
   - Parameterize Docker base image.
   - Add `scripts/run_docker.sh --rocm`.
   - Add matrix mapping and early failure/report behavior for unavailable PyTorch wheels.

5. **uv/PyTorch selection**
   - Add uv index/source support for validated wheel rows.
   - Keep default lock stable.
   - Document and test mixed-version detection.

6. **Reports and dataset integration**
   - Add matrix report generation and Markdown/JSON rendering.
   - Let `scripts/run_dataset.py` attach/copy reports without changing trace parsing or scoring.

7. **Docs and claim guardrails**
   - Document commands and claim boundaries.
   - Add doc tests ensuring Docker/container validation is not described as host-native validation.

## Test Strategy

### Unit Tests

- `tests/sol_execbench/test_rocm_compatibility.py`
  - report models round-trip strict JSON.
  - status vocabulary is locked.
  - authority booleans are false.
  - requested-vs-observed matching yields `container_validated`.
  - host/native evidence can yield `host_validated`.
  - mismatched requested/container/PyTorch versions yields `mixed_version`.
  - unavailable PyTorch wheel yields `pytorch_wheel_unavailable`.
  - missing runtime/GPU/PyTorch availability yields `runtime_unavailable`.
  - unexecuted rows stay `not_tested`.

- `tests/sol_execbench/test_environment_snapshot.py`
  - expanded probes are injectable and GPU-free.
  - ROCm version files and `hipcc --version` parsing are bounded and conservative.
  - PyTorch ROCm build tag parsing handles `+rocm7.1`, missing tag, and import errors.

- `tests/sol_execbench/test_cli_environment_snapshot.py`
  - compatibility sidecar path tracks trace output.
  - report writing is nonfatal and does not mutate trace JSONL.
  - `doctor --json` or `compatibility --json` includes expected schema/status.

### Script Tests

- Add shell/static tests for `scripts/run_docker.sh` parsing:
  - `--rocm 7.1.1` produces expected image tag/build arg/env.
  - unknown ROCm version fails with an actionable message.
  - explicit Docker args and command splitting still work.
- Add Dockerfile text tests:
  - base image uses `ARG ROCM_VERSION`.
  - current default remains the validated baseline.
  - build args/env vars for PyTorch ROCm wheel selection are present.

### Integration Tests

- Keep GPU integration tests marker-gated with `requires_rocm`.
- Add one smoke path inside Docker for the default matrix row:
  - `./scripts/run_docker.sh --build --rocm <default> -- sol-execbench doctor --json`
  - `./scripts/run_docker.sh --rocm <default> -- uv run pytest tests/sol_execbench/test_rocm_compatibility.py`
- Full matrix validation should produce report artifacts but remain opt-in because it depends on external images, wheel availability, host driver compatibility, and hardware access.

### Guardrail Tests

- Assert docs use `container_validated` and explain host dependency for `/dev/kfd` and `/dev/dri`.
- Assert docs do not claim Docker user-space validation is native host ROCm validation.
- Assert compatibility reports state they are diagnostic/reproducibility evidence, not correctness, timing, scoring, paper-parity, or leaderboard authority.
- Assert canonical trace JSONL schema remains unchanged for v1.18 unless a deliberate contract version bump is separately approved.

## Phase Split Suggestions

1. **Contracts and probe architecture**: compatibility models, evidence probes, status classifier, unit tests.
2. **CLI/report integration**: `doctor`/`compatibility` JSON, optional trace-adjacent report sidecar, nonfatal write behavior.
3. **Docker and uv matrix plumbing**: Docker ARGs, `run_docker.sh --rocm`, wheel selection/detection, early unavailable states.
4. **Matrix reports and dataset handoff**: report renderer, dataset-runner passthrough/copying, no scoring integration.
5. **Docs and guardrails**: command docs, claim boundaries, doc tests, compatibility matrix examples.

## Key Architectural Decision

Treat ROCm version matrix support as **environment compatibility evidence around the benchmark**, not as benchmark result data. Docker scripts select and label the environment, Python probes observe and classify it, reports preserve the evidence, and trace JSONL remains unchanged as the canonical benchmark execution contract.
