# Phase 80: uv And PyTorch ROCm Wheel Coordination - Research

**Researched:** 2026-05-28
**Domain:** Python dependency policy, uv indexes, PyTorch ROCm wheels, ROCm compatibility preflight
**Confidence:** HIGH

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
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

### Deferred Ideas (OUT OF SCOPE)
## Deferred Ideas

- Full host/container/Python/toolchain/GPU runtime evidence collection and
  aggregate compatibility report emission belong to Phase 81.
- User-facing validation workflow docs, CI guardrails, and marker-gated live
  ROCm validation guidance belong to Phase 82.
- Automatic host ROCm driver management, arbitrary undeclared ROCm image tags,
  and paper-scale validation remain out of scope for v1.18.
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| DEPS-01 | Each Matrix Entry records its PyTorch ROCm wheel/index policy, including expected wheel local-version tag and uv index or lock strategy. | Extend `docker/rocm-targets.json` and strict manifest models with dependency policy; uv supports named explicit indexes and package-to-index sources. [VERIFIED: codebase grep][CITED: https://docs.astral.sh/uv/concepts/indexes/] |
| DEPS-02 | The default project dependency path remains ROCm 7.1 unless a per-Target dependency workflow is explicitly selected and recorded. | Keep current `pyproject.toml`/`uv.lock` default pins and add policy metadata without changing default sources. [VERIFIED: codebase grep] |
| DEPS-03 | Missing or unsupported PyTorch ROCm wheels are classified as `pytorch_wheel_unavailable`, not as benchmark failures. | Phase 78 already includes `pytorch_wheel_unavailable` status and execution decision semantics. [VERIFIED: codebase grep] |
| DEPS-04 | CPU, CUDA, wrong-index, or wrong-ROCm PyTorch wheels are detected from installed package metadata and runtime probes. | Use `importlib.metadata` for installed distributions and injectable observations for `torch.__version__`, `torch.version.hip`, `torch.version.cuda`, and availability. [CITED: https://docs.python.org/3/library/importlib.metadata.html][VERIFIED: codebase grep] |
| DEPS-05 | A requested Target whose observed PyTorch ROCm wheel, container ROCm user-space, Triton ROCm package, or toolchain version does not match policy is classified as `mixed_version`. | Phase 78 defines `mixed_version` as Target/observed mismatch, and existing observed evidence has Python dependency and toolchain slots. [VERIFIED: codebase grep] |
| DEPS-06 | Illegal `mixed_version` Targets are blocked during preflight before benchmark execution by default. | `classify_matrix_entry_for_execution` blocks `mixed_version` unless explicit debug override is set. [VERIFIED: codebase grep] |
| DEPS-07 | An explicit mixed-version debug override may allow probes or smoke execution to continue, but the resulting entry must remain ineligible for validation and authority claims. | Existing Phase 78 execution decision supports `allow_mixed_version_debug=True` with `benchmark_allowed=false` and authority false. [VERIFIED: codebase grep] |
</phase_requirements>

## Summary

Phase 80 should add a dependency-policy layer beside the existing Docker Target policy, not replace the project dependency system. The checked-in default is already ROCm 7.1: `torch==2.10.0+rocm7.1`, `torchvision==0.25.0+rocm7.1`, and `triton-rocm==3.6.0`, with explicit uv sources for PyTorch indexes. [VERIFIED: codebase grep] The planner should preserve that path and make any per-Target dependency workflow explicit metadata rather than automatic lockfile generation. [VERIFIED: Phase 80 CONTEXT.md]

Official PyTorch indexes currently show different available wheel families by ROCm target: ROCm 7.0 has `torch 2.10.0+rocm7.0` and `torchvision 0.25.0+rocm7.0`; ROCm 7.1 includes `torch 2.10/2.11/2.12 +rocm7.1` and `torchvision 0.25/0.26/0.27 +rocm7.1`; ROCm 7.2 includes `torch 2.11/2.12 +rocm7.2` and `torchvision 0.26/0.27 +rocm7.2`. [VERIFIED: pip index versions + PyTorch simple index][CITED: https://download.pytorch.org/whl/rocm7.0/][CITED: https://download.pytorch.org/whl/rocm7.1/][CITED: https://download.pytorch.org/whl/rocm7.2/] This means a ROCm 7.2 Docker Target cannot be cleanly validated with the current default `2.10.0+rocm7.1` project stack; it must either remain dependency-ineligible or use an explicit per-Target dependency workflow. [VERIFIED: pip index versions + codebase grep]

**Primary recommendation:** implement a new pure helper such as `src/sol_execbench/core/dependency_matrix.py` that reads dependency policy from `docker/rocm-targets.json`, classifies injectable installed-stack observations into Phase 78 statuses, emits shell-consumable JSON, and is called by `scripts/run_docker.sh` after Docker Target selection/preflight but before build/run. [VERIFIED: codebase grep]

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| Per-Target dependency policy | Config / Manifest | Core Python model | `docker/rocm-targets.json` is the declared Target source; strict Pydantic parsing belongs in `src/sol_execbench/core/`. [VERIFIED: codebase grep] |
| Installed dependency observation | Python runtime | Shell wrapper | Python can inspect installed metadata and runtime probes; shell should only pass explicit test overrides and consume JSON. [CITED: https://docs.python.org/3/library/importlib.metadata.html][VERIFIED: codebase grep] |
| Compatibility classification | Core Python model | Shell wrapper | Phase 78 statuses and execution decisions already live in `core/compatibility.py`; Bash should not duplicate policy. [VERIFIED: codebase grep] |
| Docker build/run gating | Shell wrapper | Core Python JSON helper | `scripts/run_docker.sh` is the user entry point and already gates on helper JSON from Phase 79. [VERIFIED: codebase grep] |
| Default dependency preservation | pyproject/uv lock | Manifest policy | The default install path is encoded in `pyproject.toml` and `uv.lock`; per-Target policies should describe alternatives without mutating defaults. [VERIFIED: codebase grep] |

## Project Constraints (from AGENTS.md)

- Python package code lives under `src/sol_execbench/`; CLI entry point is `sol_execbench.cli:cli`. [VERIFIED: AGENTS.md]
- Tests belong under `tests/`, with package tests under `tests/sol_execbench/`. [VERIFIED: AGENTS.md]
- Use Python 3.12+ and Ruff style; use `snake_case` for functions/modules and `PascalCase` for classes/Pydantic models. [VERIFIED: AGENTS.md]
- Pytest is the test framework; use existing markers for environment-sensitive tests and prefer CPU-safe unit tests for schema and driver logic. [VERIFIED: AGENTS.md]
- Do not commit proprietary kernels, credentials, Hugging Face tokens, or downloaded datasets. [VERIFIED: AGENTS.md]
- ROCm >= 7.0 is the supported software baseline; RDNA 4 and CDNA 3 are project targets, with full CDNA 3 validation deferred. [VERIFIED: AGENTS.md]
- Preserve SOL ExecBench semantics and public schemas unless a ROCm-specific change is unavoidable; compatibility evidence must not change scoring/timing/correctness semantics. [VERIFIED: AGENTS.md][VERIFIED: .planning/REQUIREMENTS.md]

## Standard Stack

### Core

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| Python stdlib `importlib.metadata` | Python 3.12 stdlib | Inspect installed distribution versions and metadata for `torch`, `torchvision`, and `triton-rocm`. | Official stdlib API for installed package metadata. [CITED: https://docs.python.org/3/library/importlib.metadata.html] |
| Pydantic v2 | Existing project dependency `>=2.12.5` | Strict dependency policy, observation, and preflight result models. | Existing Phase 78/79 contracts use frozen strict Pydantic models. [VERIFIED: codebase grep] |
| Click/Rich | Existing project dependencies | Add or extend CLI JSON entry points if needed. | Existing CLI layer uses Click/Rich; Phase 79 module CLI uses argparse for helper JSON. [VERIFIED: AGENTS.md][VERIFIED: codebase grep] |
| uv | Installed `0.11.15` | Maintain default lock/sync path and document per-Target index strategy. | uv supports explicit named indexes and `tool.uv.sources` package-to-index pins. [VERIFIED: local command][CITED: https://docs.astral.sh/uv/concepts/indexes/] |
| `torch` | Default `2.10.0+rocm7.1` | PyTorch ROCm runtime dependency. | Current project default and PyTorch official previous-version command for ROCm 7.1. [VERIFIED: codebase grep][CITED: https://pytorch.org/get-started/previous-versions/] |
| `torchvision` | Default `0.25.0+rocm7.1` | Companion PyTorch package pinned to matching ROCm wheel tag. | Current project default and PyTorch official previous-version command for ROCm 7.1. [VERIFIED: codebase grep][CITED: https://pytorch.org/get-started/previous-versions/] |
| `triton-rocm` | Default `3.6.0` | Triton ROCm dependency status for mismatch detection. | Current project default; PyTorch index lists `3.6.0` and newer `3.7.0`. [VERIFIED: codebase grep][CITED: https://download.pytorch.org/whl/triton-rocm/] |

### Supporting

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| `packaging.version` | Transitive via project dependencies [ASSUMED] | Parse PEP 440 local-version tags if string matching becomes too brittle. | Use only if needed; PEP 440 defines local version identifiers after `+`. [CITED: https://peps.python.org/pep-0440/] |
| `json` / `argparse` stdlib | Python 3.12 stdlib | Emit shell-consumable JSON helper output. | Phase 79 already uses this pattern. [VERIFIED: codebase grep] |

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Manifest policy in `docker/rocm-targets.json` | Separate dependency manifest | Separate files reduce coupling but make Target + dependency policy harder to audit together; user locked policy near Target. [VERIFIED: Phase 80 CONTEXT.md] |
| Explicit policy metadata | Automatic per-Target lockfiles | Automatic lock generation increases blast radius and conflicts with v1 decision to record command/strategy metadata only. [VERIFIED: Phase 80 CONTEXT.md] |
| `importlib.metadata` + injectable runtime observation | Always import live PyTorch in tests | Live imports are slower and hardware/environment-sensitive; CPU-safe tests require fixtures. [VERIFIED: Phase 80 CONTEXT.md][CITED: https://docs.python.org/3/library/importlib.metadata.html] |

**Installation:**

No new dependency installation is required for Phase 80. [VERIFIED: codebase grep] Keep the existing default install path:

```bash
uv sync --all-groups
```

If a future per-Target debug workflow is recorded, model it as metadata such as `uv_index_name`, `uv_index_url`, and `suggested_uv_command`; do not mutate `pyproject.toml` automatically in Phase 80. [VERIFIED: Phase 80 CONTEXT.md]

**Version verification:** Verified with `pip index versions` on 2026-05-28:
- `torch --index-url https://download.pytorch.org/whl/rocm7.0`: latest/listed `2.10.0+rocm7.0`. [VERIFIED: pip index versions]
- `torch --index-url https://download.pytorch.org/whl/rocm7.1`: listed `2.10.0+rocm7.1`, `2.11.0+rocm7.1`, `2.12.0+rocm7.1`. [VERIFIED: pip index versions]
- `torch --index-url https://download.pytorch.org/whl/rocm7.2`: listed `2.11.0+rocm7.2`, `2.12.0+rocm7.2`. [VERIFIED: pip index versions]
- `torchvision` ROCm indexes match the 0.25/0.26/0.27 family pattern described in the summary. [VERIFIED: pip index versions]
- `triton-rocm --index-url https://download.pytorch.org/whl/`: listed `3.6.0` and `3.7.0`. [VERIFIED: pip index versions][CITED: https://download.pytorch.org/whl/triton-rocm/]

## Package Legitimacy Audit

> Phase 80 does not need new packages, but it coordinates existing external dependency names, so this audit covers the coordinated packages. [VERIFIED: codebase grep]

| Package | Registry | Age | Downloads | Source Repo | slopcheck | Disposition |
|---------|----------|-----|-----------|-------------|-----------|-------------|
| `torch` | PyPI + PyTorch wheel index | Established [VERIFIED: pip index versions] | Not checked | PyTorch official package [CITED: https://pytorch.org/get-started/previous-versions/] | OK [VERIFIED: slopcheck] | Approved; keep existing pin unless explicit per-Target workflow is selected. |
| `torchvision` | PyPI + PyTorch wheel index | Established [VERIFIED: pip index versions] | Not checked | PyTorch official package [CITED: https://pytorch.org/get-started/previous-versions/] | OK [VERIFIED: slopcheck] | Approved; keep existing pin unless explicit per-Target workflow is selected. |
| `triton-rocm` | PyTorch wheel index | Established in PyTorch index [CITED: https://download.pytorch.org/whl/triton-rocm/] | Not checked | PyTorch wheel index [CITED: https://download.pytorch.org/whl/triton-rocm/] | OK [VERIFIED: slopcheck] | Approved as existing dependency; record policy and mismatch behavior. |

**Packages removed due to slopcheck [SLOP] verdict:** none. [VERIFIED: slopcheck]
**Packages flagged as suspicious [SUS]:** none. [VERIFIED: slopcheck]

Note: `slopcheck install torch torchvision triton-rocm --json` was unsupported by the installed slopcheck CLI; `slopcheck install torch torchvision triton-rocm` returned three `[OK]` verdicts, then attempted `pip install` and failed due the externally managed system environment. No project files changed. [VERIFIED: local command]

## Architecture Patterns

### System Architecture Diagram

```text
User / script flags
  -> scripts/run_docker.sh parses --target and dependency debug override
  -> docker_matrix.py resolves declared Docker Target from docker/rocm-targets.json
  -> dependency_matrix.py loads the same Target dependency policy
  -> dependency observation source
       -> normal: importlib.metadata + lightweight torch runtime probe
       -> tests: injected JSON/env observations
  -> dependency classifier
       -> missing or policy-unsupported wheel => pytorch_wheel_unavailable
       -> installed CPU/CUDA/wrong ROCm/wrong index/Triton/toolchain mismatch => mixed_version
       -> matching default or explicitly selected policy => not_tested until runtime validation
  -> MatrixExecutionDecision
       -> block build/run by default for mixed_version or unavailable wheel
       -> allow explicit dependency debug probes/smoke only, with authority false
  -> Docker build/run only if preflight JSON allows it
```

### Recommended Project Structure

```text
src/sol_execbench/core/
├── compatibility.py        # existing Phase 78 statuses and execution decisions
├── docker_matrix.py        # existing Phase 79 Docker Target selection and preflight
└── dependency_matrix.py    # new Phase 80 dependency policy and classifier

tests/sol_execbench/
├── test_dependency_matrix_policy.py
├── test_dependency_matrix_classification.py
├── test_dependency_matrix_cli.py
└── test_run_docker_dependency_preflight.py
```

### Pattern 1: Target-Adjacent Dependency Policy

**What:** Add a nested dependency policy object to each declared Docker Target, for example `pytorch_dependency_policy`. [VERIFIED: Phase 80 CONTEXT.md]

**When to use:** Every declared Target should expose expected `torch`, `torchvision`, uv index/lock strategy, and Triton policy before runtime validation can make a clean claim. [VERIFIED: .planning/REQUIREMENTS.md]

**Example:**

```json
{
  "target_id": "rocm-7.1.1-ubuntu-24.04-container",
  "requested_rocm_user_space_version": "7.1.1",
  "pytorch_rocm_target": "rocm7.1",
  "pytorch_dependency_policy": {
    "policy_id": "pytorch-2.10.0-rocm7.1-default",
    "wheel_availability": "available",
    "torch_version": "2.10.0+rocm7.1",
    "torchvision_version": "0.25.0+rocm7.1",
    "expected_local_version": "rocm7.1",
    "uv_index_name": "pytorch-rocm71",
    "uv_index_url": "https://download.pytorch.org/whl/rocm7.1",
    "lock_strategy": "project_default",
    "triton_rocm_version": "3.6.0",
    "triton_rocm_index_name": "pytorch-rocm-root"
  }
}
```

Source: existing `pyproject.toml` and uv docs for explicit indexes. [VERIFIED: codebase grep][CITED: https://docs.astral.sh/uv/concepts/indexes/]

### Pattern 2: Injectable Observation Classifier

**What:** Classifier accepts a structured observation object instead of importing PyTorch internally in all paths. [VERIFIED: Phase 80 CONTEXT.md]

**When to use:** Use injected observations for unit tests and shell dry-runs; provide a small collection function for normal execution. [VERIFIED: Phase 79 SUMMARY]

**Example:**

```python
def classify_dependency_preflight(
    *,
    target: DockerTargetManifestEntry,
    policy: PytorchDependencyPolicy,
    observation: PytorchDependencyObservation,
    allow_mixed_version_debug: bool = False,
) -> DependencyPreflightResult:
    entry = build_dependency_matrix_entry(target, policy, observation)
    decision = classify_matrix_entry_for_execution(
        entry,
        allow_mixed_version_debug=allow_mixed_version_debug,
    )
    return DependencyPreflightResult(entry=entry, decision=decision)
```

Source: Phase 78/79 local helper patterns. [VERIFIED: codebase grep]

### Pattern 3: Shell-Consumable JSON Contract

**What:** Add a module CLI such as `python -m sol_execbench.core.dependency_matrix preflight --manifest ...` and return JSON fields already used by `scripts/run_docker.sh`: `status`, `reason_code`, `reason`, `benchmark_allowed`, `probes_allowed`, `smoke_allowed`, and authority flags. [VERIFIED: codebase grep]

**When to use:** `scripts/run_docker.sh` should call this helper before Docker build/run, just as it calls `docker_matrix.py`. [VERIFIED: Phase 80 CONTEXT.md][VERIFIED: codebase grep]

### Anti-Patterns to Avoid

- **Bash-only dependency policy:** Bash cannot safely duplicate uv/PyTorch policy, enum semantics, or strict model validation; keep shell thin. [VERIFIED: codebase grep]
- **Changing default pins to support every Target:** The default ROCm 7.1 path is a locked decision; per-Target workflows must be explicit. [VERIFIED: Phase 80 CONTEXT.md]
- **Treating ROCm 7.2 as clean with the default lock:** Current default wheels are `+rocm7.1`; ROCm 7.2 wheels start at newer torch/torchvision families. [VERIFIED: pip index versions]
- **Classifying dependency setup failures as benchmark failures:** Phase 78 requires diagnostic statuses and benchmark authority false. [VERIFIED: codebase grep]
- **Reusing `--allow-unknown-target`:** Dependency mismatches need explicit debug override wording, not unknown-target unsafe override wording. [VERIFIED: Phase 80 CONTEXT.md]

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Installed package metadata inspection | Custom `site-packages` scanners | `importlib.metadata.version()` / `distribution()` | Official stdlib metadata API. [CITED: https://docs.python.org/3/library/importlib.metadata.html] |
| uv package index resolution | Custom URL probing as resolver | uv `[[tool.uv.index]]` + `[tool.uv.sources]` metadata | uv already models explicit package-to-index sources. [CITED: https://docs.astral.sh/uv/concepts/indexes/] |
| Version local-tag semantics | Ad hoc split for every case | Exact expected local tag checks, optionally `packaging.version` if needed | PEP 440 defines local version identifiers after `+`; exact string comparison is sufficient for known policy tags. [CITED: https://peps.python.org/pep-0440/] |
| Matrix execution policy | New status words or shell booleans | Phase 78 `MatrixCompatibilityStatus` and `MatrixExecutionDecision` | Bounded statuses already exist and are requirement-backed. [VERIFIED: codebase grep] |
| Docker gating policy | Independent shell exits | Python helper JSON consumed by `scripts/run_docker.sh` | Phase 79 established this integration pattern. [VERIFIED: codebase grep] |

**Key insight:** The hard part is not installing PyTorch; it is preventing a user from interpreting a selected ROCm Docker Target as cleanly validated when the installed Python dependency stack belongs to a different ROCm target. [VERIFIED: Phase 80 CONTEXT.md]

## Common Pitfalls

### Pitfall 1: Default ROCm 7.1 Drift

**What goes wrong:** Adding per-Target policy accidentally changes `pyproject.toml` or `uv.lock`, breaking the current default install. [VERIFIED: Phase 80 CONTEXT.md]
**Why it happens:** ROCm 7.0/7.2 need different wheel policies, which tempts a global dependency change. [VERIFIED: pip index versions]
**How to avoid:** Store policy in `docker/rocm-targets.json` and keep default `pyproject.toml` pins unchanged in Phase 80. [VERIFIED: Phase 80 CONTEXT.md]
**Warning signs:** Diffs in `pyproject.toml` or `uv.lock` that are not purely deliberate default-policy changes. [VERIFIED: codebase grep]

### Pitfall 2: CPU/CUDA Wheels Look Importable

**What goes wrong:** `import torch` succeeds, but `torch.version.hip` is `None` or `torch.version.cuda` is set; clean ROCm validation would be invalid. [VERIFIED: codebase grep]
**Why it happens:** PyTorch publishes CPU/CUDA wheels under the same distribution names but different indexes/build tags. [CITED: https://pytorch.org/get-started/previous-versions/]
**How to avoid:** Classify by installed metadata plus runtime fields: `torch.__version__`, local tag, `torch.version.hip`, `torch.version.cuda`, and device availability. [VERIFIED: Phase 80 CONTEXT.md]
**Warning signs:** `torch_version` has no `+rocm*` local tag, has `+cu*`, `torch_hip_version` is null, or `torch_cuda_version` is non-null. [VERIFIED: codebase grep]

### Pitfall 3: Wheel Unavailable vs Wrong Wheel

**What goes wrong:** Unsupported policy is reported as `mixed_version` or benchmark failure instead of `pytorch_wheel_unavailable`. [VERIFIED: .planning/REQUIREMENTS.md]
**Why it happens:** Missing upstream wheels and installed mismatches both block validation, but they mean different things. [VERIFIED: Phase 80 CONTEXT.md]
**How to avoid:** If the Target policy says no supported wheel exists for the requested stack, classify `pytorch_wheel_unavailable`; if a stack is installed but does not match policy, classify `mixed_version`. [VERIFIED: Phase 80 CONTEXT.md]
**Warning signs:** ROCm 7.2 Target with default `torch 2.10.0+rocm7.1` is treated as a benchmark failure. [VERIFIED: pip index versions + codebase grep]

### Pitfall 4: Triton Package Name/Index Confusion

**What goes wrong:** Planner checks PyPI only and misses `triton-rocm` on the PyTorch wheel index. [VERIFIED: local command]
**Why it happens:** `pip index versions triton-rocm` on default PyPI returned no match, while `--index-url https://download.pytorch.org/whl/` listed `3.6.0` and `3.7.0`. [VERIFIED: local command][CITED: https://download.pytorch.org/whl/triton-rocm/]
**How to avoid:** Record `triton-rocm` index policy separately from PyPI and classify absence/wrong version as dependency policy evidence. [VERIFIED: codebase grep]
**Warning signs:** A classifier says `triton-rocm` is unavailable solely because it queried default PyPI. [VERIFIED: local command]

### Pitfall 5: Debug Override Grants Authority

**What goes wrong:** `--allow-mixed-version-dependencies` allows smoke execution and accidentally sets clean validation or benchmark authority true. [VERIFIED: Phase 80 CONTEXT.md]
**Why it happens:** Override naming is confused with normal validation. [VERIFIED: Phase 80 CONTEXT.md]
**How to avoid:** Reuse `classify_matrix_entry_for_execution(..., allow_mixed_version_debug=True)` and assert authority flags remain false in tests. [VERIFIED: codebase grep]
**Warning signs:** JSON under debug override has `benchmark_allowed=true`, `container_user_space_validated=true`, or authority fields true. [VERIFIED: codebase grep]

## Code Examples

Verified patterns from official/local sources:

### Installed Metadata Probe

```python
from importlib import metadata

def installed_version(distribution_name: str) -> str | None:
    try:
        return metadata.version(distribution_name)
    except metadata.PackageNotFoundError:
        return None
```

Source: Python `importlib.metadata.version()` docs. [CITED: https://docs.python.org/3/library/importlib.metadata.html]

### Runtime PyTorch Probe Fields

```python
torch_version = str(getattr(torch, "__version__", ""))
version = getattr(torch, "version", None)
hip_version = getattr(version, "hip", None)
cuda_version = getattr(version, "cuda", None)
available = bool(torch.cuda.is_available()) and hip_version is not None
```

Source: existing `collect_pytorch_rocm_summary()`. [VERIFIED: codebase grep]

### Existing Mixed-Version Debug Decision

```python
decision = classify_matrix_entry_for_execution(
    entry,
    allow_mixed_version_debug=True,
)
assert decision.benchmark_allowed is False
assert decision.probes_allowed is True
assert decision.smoke_allowed is True
assert decision.score_authority is False
```

Source: existing Phase 78 tests. [VERIFIED: codebase grep]

### uv Explicit Index Pattern

```toml
[tool.uv.sources]
torch = [
  { index = "pytorch-rocm71", marker = "sys_platform == 'linux' or sys_platform == 'win32'" },
]

[[tool.uv.index]]
name = "pytorch-rocm71"
url = "https://download.pytorch.org/whl/rocm7.1"
explicit = true
```

Source: current `pyproject.toml` and uv explicit-index docs. [VERIFIED: codebase grep][CITED: https://docs.astral.sh/uv/concepts/indexes/]

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Single global ROCm dependency assumption | Target-specific dependency policy with default ROCm 7.1 preserved | Phase 80 scope, 2026-05-28 [VERIFIED: Phase 80 CONTEXT.md] | Prevents false clean validation across ROCm 7.0/7.1/7.2 Docker Targets. |
| Benchmark execution discovers dependency mismatch late | Dependency preflight blocks mismatches before build/run | Phase 80 scope, 2026-05-28 [VERIFIED: .planning/REQUIREMENTS.md] | Wrong wheels become compatibility diagnostics, not benchmark failures. |
| Ad hoc package source selection | uv explicit indexes and `tool.uv.sources` | uv docs current as crawled 2026-05 [CITED: https://docs.astral.sh/uv/concepts/indexes/] | Keeps PyTorch packages pinned to intended indexes and reduces dependency-confusion risk. |

**Deprecated/outdated:**
- Treating Docker ROCm user-space match as sufficient for clean validation is invalid because Python dependency evidence is a separate observed dimension. [VERIFIED: Phase 78 CONTEXT.md]
- Treating `torch` distribution presence as ROCm readiness is invalid because CPU/CUDA/ROCm wheels share distribution names. [CITED: https://pytorch.org/get-started/previous-versions/]

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | `packaging.version` is available transitively if exact string checks need richer parsing. | Standard Stack | Low; planner can avoid by using exact local-tag checks only. |

## Open Questions (RESOLVED)

1. **Should ROCm 7.2 policy be recorded as available with explicit `torch 2.11/2.12 +rocm7.2`, or unavailable for the current default `torch 2.10` family?**
   - What we know: PyTorch ROCm 7.2 index lists `torch 2.11/2.12 +rocm7.2`, not `2.10.0+rocm7.2`. [VERIFIED: pip index versions]
   - Resolution: Record ROCm 7.2 as `available_with_explicit_workflow` using a newer supported torch/torchvision ROCm 7.2 family, but block clean validation under the default ROCm 7.1 project dependency path unless that explicit per-Target workflow is selected and recorded. [VERIFIED: Phase 80 CONTEXT.md]

2. **Should ROCm 7.0 policy use `torch 2.10.0+rocm7.0` as an explicit workflow?**
   - What we know: PyTorch ROCm 7.0 index lists `torch 2.10.0+rocm7.0` and `torchvision 0.25.0+rocm7.0`. [VERIFIED: pip index versions]
   - Resolution: Record the explicit ROCm 7.0 uv index/command metadata and keep the default lock unchanged in Phase 80. Do not generate a separate per-Target lockfile yet. [VERIFIED: Phase 80 CONTEXT.md]

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|-------------|-----------|---------|----------|
| Python | Models/tests/helper CLI | ✓ | 3.12.13 | None needed. [VERIFIED: local command] |
| uv | Default lock/sync path | ✓ | 0.11.15 | None needed for tests; do not regenerate locks unless planned. [VERIFIED: local command] |
| Docker | Wrapper integration path | ✓ | 29.4.3 | CPU-safe dry-run tests use env overrides. [VERIFIED: local command][VERIFIED: Phase 79 SUMMARY] |
| hipcc | Toolchain mismatch policy context | ✓ | ROCm 7.1.1 / HIP 7.1.52802 | CPU-safe tests should inject observations. [VERIFIED: local command] |
| slopcheck | Package legitimacy audit | ✓ | CLI present; `--json` unsupported | Use plain output and record results. [VERIFIED: local command] |
| PyTorch wheel indexes | Policy verification | ✓ | ROCm 7.0/7.1/7.2 indexes reachable by `pip index versions` | If unavailable in CI, use checked-in policy fixtures. [VERIFIED: local command] |

**Missing dependencies with no fallback:** none for CPU-safe planning/execution. [VERIFIED: local command]

**Missing dependencies with fallback:** live ROCm hardware is not required for Phase 80 tests; use injectable observations. [VERIFIED: Phase 80 CONTEXT.md]

## Validation Architecture

### Test Framework

| Property | Value |
|----------|-------|
| Framework | pytest `>=9.0.2` from dev dependencies. [VERIFIED: codebase grep] |
| Config file | `pyproject.toml` `[tool.pytest.ini_options]`. [VERIFIED: codebase grep] |
| Quick run command | `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/sol_execbench/test_dependency_matrix_policy.py tests/sol_execbench/test_dependency_matrix_classification.py -q` |
| Full suite command | `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/sol_execbench/test_rocm_compatibility_matrix.py tests/sol_execbench/test_matrix_claim_guardrails.py tests/sol_execbench/test_docker_matrix_targets.py tests/sol_execbench/test_docker_matrix_preflight.py tests/sol_execbench/test_run_docker_matrix_script.py tests/sol_execbench/test_dependency_matrix_policy.py tests/sol_execbench/test_dependency_matrix_classification.py tests/sol_execbench/test_dependency_matrix_cli.py tests/sol_execbench/test_run_docker_dependency_preflight.py -q` |

### Phase Requirements -> Test Map

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|--------------|
| DEPS-01 | Manifest policy exposes expected local tag and uv strategy per Target. | unit | `uv run pytest tests/sol_execbench/test_dependency_matrix_policy.py -q` | ❌ Wave 0 |
| DEPS-02 | Default ROCm 7.1 `pyproject.toml`/lock path remains unchanged. | unit/static | `uv run pytest tests/sol_execbench/test_dependency_matrix_policy.py::test_default_project_dependency_path_remains_rocm_7_1 -q` | ❌ Wave 0 |
| DEPS-03 | Unsupported/missing wheel policy classifies `pytorch_wheel_unavailable`. | unit | `uv run pytest tests/sol_execbench/test_dependency_matrix_classification.py::test_unsupported_policy_is_pytorch_wheel_unavailable -q` | ❌ Wave 0 |
| DEPS-04 | CPU/CUDA/wrong-index/wrong-ROCm observations are detected. | unit | `uv run pytest tests/sol_execbench/test_dependency_matrix_classification.py -q` | ❌ Wave 0 |
| DEPS-05 | PyTorch/Triton/toolchain mismatch becomes `mixed_version`. | unit | `uv run pytest tests/sol_execbench/test_dependency_matrix_classification.py::test_triton_or_toolchain_mismatch_is_mixed_version -q` | ❌ Wave 0 |
| DEPS-06 | `scripts/run_docker.sh` blocks illegal mixed-version preflight by default. | script subprocess | `uv run pytest tests/sol_execbench/test_run_docker_dependency_preflight.py -q` | ❌ Wave 0 |
| DEPS-07 | Explicit debug override permits probes/smoke but no validation or authority. | unit + script subprocess | `uv run pytest tests/sol_execbench/test_dependency_matrix_classification.py tests/sol_execbench/test_run_docker_dependency_preflight.py -q` | ❌ Wave 0 |

### Sampling Rate

- **Per task commit:** Run the narrow test file for the touched helper/script plus `bash -n scripts/run_docker.sh` for shell changes. [VERIFIED: Phase 79 SUMMARY]
- **Per wave merge:** Run all Phase 78/79 compatibility and Docker tests plus new Phase 80 tests. [VERIFIED: Phase 79 SUMMARY]
- **Phase gate:** Full suite above green before `$gsd-verify-work`. [VERIFIED: .planning/config.json]

### Wave 0 Gaps

- [ ] `tests/sol_execbench/test_dependency_matrix_policy.py` — covers DEPS-01 and DEPS-02.
- [ ] `tests/sol_execbench/test_dependency_matrix_classification.py` — covers DEPS-03, DEPS-04, DEPS-05, and DEPS-07.
- [ ] `tests/sol_execbench/test_dependency_matrix_cli.py` — covers shell-consumable JSON output.
- [ ] `tests/sol_execbench/test_run_docker_dependency_preflight.py` — covers DEPS-06 and script-side DEPS-07.

## Security Domain

### Applicable ASVS Categories

| ASVS Category | Applies | Standard Control |
|---------------|---------|------------------|
| V2 Authentication | no | No auth surface in this phase. [VERIFIED: Phase 80 CONTEXT.md] |
| V3 Session Management | no | No session surface in this phase. [VERIFIED: Phase 80 CONTEXT.md] |
| V4 Access Control | yes | Validation authority is denied by Phase 78 claim flags unless evidence supports it. [VERIFIED: codebase grep] |
| V5 Input Validation | yes | Strict Pydantic models with `extra="forbid"` for manifest/policy JSON. [VERIFIED: codebase grep] |
| V6 Cryptography | no | No cryptographic operation in this phase. [VERIFIED: Phase 80 CONTEXT.md] |
| V14 Configuration | yes | uv explicit indexes and manifest policy prevent accidental wrong-index dependency resolution. [CITED: https://docs.astral.sh/uv/concepts/indexes/] |

### Known Threat Patterns for uv/PyTorch ROCm Coordination

| Pattern | STRIDE | Standard Mitigation |
|---------|--------|---------------------|
| Dependency confusion/wrong index | Tampering | Use uv explicit indexes and `tool.uv.sources`; record expected index URL/name in policy. [CITED: https://docs.astral.sh/uv/concepts/indexes/] |
| False validation claim under mixed dependencies | Spoofing | Classify as `mixed_version`, block benchmark by default, authority false. [VERIFIED: codebase grep] |
| Diagnostic override misuse | Elevation of privilege | Use explicit dependency debug override and assert no validation/authority flags become true. [VERIFIED: Phase 80 CONTEXT.md] |
| Environment-specific test flakiness | Repudiation | Use injectable observations and CPU-safe tests; reserve live ROCm evidence for Phase 81/82. [VERIFIED: Phase 80 CONTEXT.md] |

## Sources

### Primary (HIGH confidence)

- `AGENTS.md` — project structure, test, style, ROCm, and security constraints. [VERIFIED: AGENTS.md]
- `.planning/phases/80-uv-and-pytorch-rocm-wheel-coordination/80-CONTEXT.md` — locked Phase 80 decisions and deferrals. [VERIFIED: Phase 80 CONTEXT.md]
- `.planning/REQUIREMENTS.md` — DEPS-01 through DEPS-07 requirement wording. [VERIFIED: .planning/REQUIREMENTS.md]
- `src/sol_execbench/core/compatibility.py` — Matrix statuses, reason codes, claim boundaries, and execution decisions. [VERIFIED: codebase grep]
- `src/sol_execbench/core/docker_matrix.py`, `docker/rocm-targets.json`, `scripts/run_docker.sh` — Phase 79 manifest/helper/script integration. [VERIFIED: codebase grep]
- `pyproject.toml`, `uv.lock` — current default ROCm 7.1 pins and uv index/source setup. [VERIFIED: codebase grep]
- uv package indexes docs — explicit indexes and `tool.uv.sources`. [CITED: https://docs.astral.sh/uv/concepts/indexes/]
- PyTorch previous versions docs — official ROCm 7.1 install command for torch 2.10.0 / torchvision 0.25.0. [CITED: https://pytorch.org/get-started/previous-versions/]
- PyTorch wheel simple indexes for ROCm 7.0/7.1/7.2 and Triton ROCm. [CITED: https://download.pytorch.org/whl/rocm7.0/][CITED: https://download.pytorch.org/whl/rocm7.1/][CITED: https://download.pytorch.org/whl/rocm7.2/][CITED: https://download.pytorch.org/whl/triton-rocm/]
- Python `importlib.metadata` docs. [CITED: https://docs.python.org/3/library/importlib.metadata.html]
- PEP 440 local version identifier docs. [CITED: https://peps.python.org/pep-0440/]

### Secondary (MEDIUM confidence)

- `pip index versions` against PyTorch indexes on 2026-05-28 for exact available versions. [VERIFIED: local command]
- `slopcheck install torch torchvision triton-rocm` plain output on 2026-05-28. [VERIFIED: local command]

### Tertiary (LOW confidence)

- None used for recommendations. [VERIFIED: source audit]

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — current project pins, official uv docs, official PyTorch docs/indexes, and local registry probes agree. [VERIFIED: codebase grep][CITED: https://docs.astral.sh/uv/concepts/indexes/][CITED: https://pytorch.org/get-started/previous-versions/]
- Architecture: HIGH — follows existing Phase 78/79 Python-helper + shell-JSON integration pattern. [VERIFIED: codebase grep]
- Pitfalls: HIGH — derived from locked phase decisions, existing tests, and verified wheel/index differences. [VERIFIED: Phase 80 CONTEXT.md][VERIFIED: pip index versions]

**Research date:** 2026-05-28
**Valid until:** 2026-06-04 for PyTorch wheel availability; uv/index semantics likely stable for 30 days. [ASSUMED]
