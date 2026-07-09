---
generated_by: gsd-map-codebase
generated_on: 2026-07-09
last_mapped_commit: cc007cd3af3e5100f7d86f155a40d5e51ffb57e5
focus: concerns
---

# Concerns

## Execution Isolation Is Not A Sandbox

The evaluator runs submitted code in a local subprocess with static and runtime
guardrails, but it is not a hardened multi-tenant sandbox. This is documented in
`README.md` and `docs/ARCHITECTURE.md`. Any workflow that evaluates untrusted
solutions must use external isolation such as Docker, a VM, or a dedicated ROCm
host.

Relevant paths:

- `src/sol_execbench/driver/templates/eval_driver.py`
- `src/sol_execbench/core/bench/reward_hack/`
- `src/sol_execbench/core/data/solution_models.py`
- `docs/SECURITY.md`

## Large Surface Area

The project has grown beyond a simple evaluator. `src/sol_execbench/core/`
contains benchmark runtime, dataset migration, execution closure, readiness,
reporting, scoring, ROCm platform diagnostics, profiler integration, and
release evidence. This breadth increases regression risk when changing shared
models such as `Trace`, `Solution`, `Workload`, or evidence refs.

Changes to `src/sol_execbench/core/data/` and `src/sol_execbench/core/bench/`
should run targeted unit tests plus at least one CLI/evaluator integration path.

## Generated Cache Files Present

The working tree contains many `__pycache__` files under `src/`, `tests/`, and
`scripts/`. They are ignored by tooling, but their presence can make file scans
noisy and increases the chance of accidental artifact handling in ad-hoc
scripts.

## Legacy CUDA/NVIDIA Residue Is Intentional But Fragile

The codebase intentionally retains some CUDA/NVIDIA strings for attribution,
negative tests, compatibility namespace explanations, migration guidance, and
dataset provenance. The distinction between allowed residue and accidental
runtime residue is enforced by tests such as
`tests/sol_execbench/core/platform/test_rocm_migration_residue_audit.py`.

When editing docs, examples, or schema validators, avoid introducing unclassified
CUDA/NVIDIA terms. If a new term is legitimate, update the classifier test with
a precise reason.

## PyTorch ROCm Uses CUDA Namespace Compatibility

ROCm PyTorch still exposes many device APIs through `torch.cuda` and related
names. The project documents this, but it remains easy to misread as NVIDIA CUDA
runtime support. Examples include `torch.cuda.is_available()` in
`tests/conftest.py` and device selection in
`src/sol_execbench/driver/templates/eval_driver.py`.

Public docs and error messages should continue to distinguish PyTorch namespace
compatibility from actual CUDA backend support.

## Native Build Boundary Is Sensitive

Native ROCm builds accept user-supplied source and limited compile options.
`src/sol_execbench/core/data/solution_models.py` rejects response files,
external path injection, and dynamic loader control, while allowing documented
ROCm system include/library paths. Any relaxation of compile flag validation
should be treated as security-sensitive and tested directly.

`ProblemPackager` also auto-injects HIP offload architecture flags through
`src/sol_execbench/driver/build_config.py`. Changes there can affect both local
developer workflows and reproducibility on RDNA4/CDNA3 hosts.

## Profiler Evidence Is Diagnostic

`rocprofv3` integration is optional and can fail or partially produce artifacts.
The CLI intentionally falls back to normal evaluation when profiler collection
fails. Reports and docs must not treat profiler sidecars as correctness or
official score authority.

Relevant paths:

- `src/sol_execbench/cli/evaluation/phases.py`
- `src/sol_execbench/cli/evaluation/runtime.py`
- `src/sol_execbench/core/bench/rocm_profiler/`
- `src/sol_execbench/core/dataset/profiler_timing_coverage/`
- `docs/rocm_timing.md`

## Hardware Coverage Is Bounded

The project target includes RDNA4 and CDNA3, but current documented validation
is bounded. RDNA4 has recorded validation evidence; CDNA3/MI300X validation has
handoff/deferred records under `.planning/milestones/` and `docs/internal/`.
Docs should keep claims specific to the exact hardware, commands, and artifacts
that were actually validated.

## CI Is CPU-Safe, Not Full ROCm Validation

GitHub Actions run Ruff, Ty, and CPU-safe pytest subsets. Hardware tests are
skipped unless a ROCm-capable environment with `/dev/kfd`, `/dev/dri`, headers,
and supported GPU architecture is available. A green CI run does not prove GPU
runtime behavior.

## Dataset Licensing Boundary

The repository does not redistribute upstream restricted datasets. Migration
and execution workflows assume the operator supplies local dataset assets under
the applicable license. Scripts and docs around `scripts/download_solexecbench.py`,
`src/sol_execbench/core/dataset/migration/`, and `docs/provenance.md` should
continue preserving this boundary.

## Subprocess And Parallelism Resource Risk

PyTorch+ROCm imports can be memory-heavy, and profiler/GPU timing runs are
resource-sensitive. `pyproject.toml` caps pytest-xdist at eight workers, and
dataset/profiler scripts include serial GPU timing paths. Avoid introducing
unbounded parallel GPU subprocess execution.
