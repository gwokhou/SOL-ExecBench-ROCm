---
last_mapped_commit: dd5731c42d9f3d417acf552b3a02004cd5039df2
last_mapped_date: 2026-06-08
focus: concerns
---

# Concerns

## Summary

The repository has strong ROCm migration guardrails and broad tests, but the
durable risk areas are still concentrated around untrusted code execution,
hardware-dependent validation, timing authority, dataset-scale orchestration,
and derived score/evidence interpretation.

Largest maintenance surfaces:

- `scripts/run_dataset.py` is the main dataset orchestrator and is large enough
  that changes can affect sharding, reuse, closure, timing, scoring, and claim
  evidence at once.
- `src/sol_execbench/cli/main.py` owns the public evaluator CLI, subprocess
  execution, compile flow, optional sidecars, and trace output.
- `src/sol_execbench/core/scoring/solar_derivation.py` and
  `src/sol_execbench/core/scoring/amd_bound_graph.py` encode complex derived
  scoring and semantic extraction policy.
- `src/sol_execbench/driver/templates/eval_driver.py` is generated into staging
  directories and owns benchmark-critical correctness, timing, reward-hack, and
  trace emission behavior.

## Security And Isolation

### Evaluation Is Not A Hard Sandbox

`SECURITY.md` correctly frames evaluation as subprocess isolation rather than a
security sandbox. User and reference code are imported and executed by
`src/sol_execbench/driver/templates/eval_driver.py` using helpers in
`src/sol_execbench/core/bench/eval_runtime.py`.

Controls:

- `src/sol_execbench/core/data/solution.py` rejects absolute source paths,
  parent traversal, legacy CUDA/NVIDIA language values, CUDA compile option
  keys, response files, linker-loader flags, and most host-path compile flags.
- `src/sol_execbench/core/bench/eval_runtime.py` blocks
  `torch.utils.cpp_extension.load()` and `load_inline()` for Python solutions.
- `src/sol_execbench/core/bench/reward_hack.py` and
  `src/sol_execbench/driver/templates/eval_driver.py` check static source
  policy, monkey-patching, lazy outputs, semantic caches, side threads, and
  integrity of critical functions.

Residual risk:

- User Python can still perform arbitrary process-local actions available to the
  evaluator subprocess before or outside heuristic detection.
- Native HIP/C++ extensions loaded from `benchmark_kernel.so` run in-process and
  can crash, hang, corrupt process state, or exhaust GPU/host resources.
- The subprocess timeout in `src/sol_execbench/cli/main.py` limits hangs but is
  not resource isolation for memory, disk, device state, or host process effects.
- Generated staging directories are removed with best-effort cleanup in
  `src/sol_execbench/driver/problem_packager.py`; cleanup failures are ignored
  unless `--keep-staging` is used.

### Compile And Staging Boundary Is Sensitive

Native builds go through `src/sol_execbench/driver/templates/build_ext.py` and
`torch.utils.cpp_extension.load()`. The compile flag policy is protective, but
the build still trusts the ROCm toolchain and the temporary staging directory.

Watch points:

- `_ALLOWED_ROCM_SYSTEM_PATH_FLAGS` in `src/sol_execbench/core/data/solution.py`
  permits `/opt/rocm/include` and `/opt/rocm/lib`, so host ROCm installation
  contents remain part of the trust boundary.
- `src/sol_execbench/driver/problem_packager.py` auto-injects HIP offload
  architecture flags from requested hardware or local `rocm_agent_enumerator` /
  `rocminfo`; mixed hosts or stale tool output can create misleading binaries.
- `FLASHINFER_TRACE_DIR` is used by both staging and runtime safetensors lookup
  in `src/sol_execbench/driver/problem_packager.py` and
  `src/sol_execbench/driver/templates/eval_driver.py`; treat it as trusted input.

### Derived Graph Extraction Can Execute Reference Code

`src/sol_execbench/core/scoring/amd_bound_graph.py` attempts Torch FX tracing by
executing `Definition.reference` with `exec()` before falling back to AST
extraction. This is useful for derived evidence but extends executable trust
beyond the main evaluator path. Do not use derived scoring on untrusted
definition payloads without the same operational caution as evaluation.

## Correctness And Benchmark Semantics

### Evaluation Driver Is Fragile By Design

`src/sol_execbench/driver/templates/eval_driver.py` coordinates source review,
reference import, user import, ten correctness rounds, output validation,
latency measurement, optional reference timing, and canonical Trace JSONL. Small
changes can alter benchmark semantics.

Fragile details:

- The driver redirects stdout to stderr before importing PyTorch so canonical
  JSONL can be written to the saved original stdout. Any future direct file
  descriptor writes can interfere with trace parsing.
- Reference code and user code share the same subprocess; integrity snapshots
  reduce mutation risk but are not process separation.
- Correctness depends on the alignment of `gen_inputs()`,
  `call_and_collect_outputs()`, destination-passing style, output allocation,
  shape/dtype checks, and repeated numerical rounds.
- The driver ends with `os._exit(0)` to avoid lingering TorchInductor/ROCm
  worker threads. That intentionally skips normal Python teardown.

### Trace Parsing Is Tolerant

`src/sol_execbench/driver/problem_packager.py` parses only stdout lines that
start with `{`; `src/sol_execbench/core/dataset/runner.py` skips JSON decode
failures while parsing trace JSONL from captured stdout. This keeps library noise
from breaking runs, but malformed or partial stdout can collapse into
`no_parseable_traces` or saved CLI logs instead of a precise parse error.

### CPU Fallbacks Are Test Helpers, Not Runtime Authority

`src/sol_execbench/core/bench/eval_runtime.py` permits CPU timing only when
`SOL_EXECBENCH_ALLOW_CPU_TIMING=1`. That is useful for subprocess tests, but CPU
paths do not validate real ROCm timing or GPU memory behavior.

## Timing And Performance Risks

### Device-Event Timing Is A Compatibility Path

`src/sol_execbench/core/bench/timing.py` uses PyTorch ROCm's historical
`torch.cuda` namespace and event API. This is valid for PyTorch ROCm, but it
remains easy to misread as NVIDIA CUDA timing.

Risks:

- Event timing is not equivalent to profiler-backed kernel activity timing.
- L2 cache clearing allocates a buffer at least twice the detected L2 cache size;
  this can perturb memory pressure on smaller or fragmented devices.
- `ShiftingMemoryPoolAllocator` increases allocation pressure to vary tensor
  pointers across iterations.
- Timing-sensitive tests marked `timing_serial` in `tests/conftest.py` are
  skipped by default unless explicitly selected.

### `rocprofv3` Evidence Is Diagnostic And Best Effort

`src/sol_execbench/core/bench/rocm_profiler.py` records policy, fallback, and
sidecar metadata, but collection failures intentionally fall back or become
diagnostic sidecars. `src/sol_execbench/cli/main.py` reruns normal evaluation if
optional `--profile rocprofv3` fails.

Implication: profiler sidecars help interpretation, but canonical correctness
and score claims must not silently rely on absent or fallback profiler output.

## Hardware And Validation Gaps

### Green Tests Can Hide Hardware Skips

`tests/conftest.py` skips hardware tests when `/dev/kfd` or `/dev/dri` are
missing, PyTorch is not a ROCm build, PyTorch cannot see a GPU, target
architectures do not match, or ROCm development headers are absent.

Important skipped surfaces include:

- `requires_rocm` GPU behavior.
- `requires_rocm_dev` HIP extension builds.
- `requires_ck` and `requires_rocwmma` native library categories.
- `requires_rdna4` and `requires_cdna3` architecture-specific behavior.
- `requires_cutile`, which is permanently skipped as NVIDIA-only legacy
  coverage in this ROCm-only port.

### CDNA3 And MI300X Claims Remain Bounded

`docs/internal/cdna3_gfx942_validation_attempt.md` records meaningful MI308X
(`gfx942`) progress, including full adapted pytest validation and operational
dataset infrastructure after follow-up fixes. It also records remaining dataset
timeout blockers and states that MI308X evidence does not complete exact MI300X
benchmark-grade validation.

`docs/research_preview.md` keeps MI300X/CDNA3 evidence bounded:

- Current CDNA3 evidence was recorded on MI308X, not exact MI300X.
- Known timeout shards remain in dataset validation.
- Benchmark-grade timing, score, FP8, and deferred NVFP4/MXFP4 evidence still
  need exact-hardware records.
- CDNA4 validation is unavailable without suitable hardware.

### RDNA4 Evidence Is Bounded And Timing Is Non-Authoritative

`docs/research_preview.md` records bounded RDNA4 ready-subset evidence, not a
full paper-scale validation. It also states RDNA4 timing remains
non-authoritative because recorded timing used PyTorch/device-event fallback
rather than profiler-backed `rocprofv3` kernel activity timing.

## Dataset And Evidence Pipeline Debt

### Dataset Runner Has Many Coupled Responsibilities

`scripts/run_dataset.py` handles discovery, ready-subset filtering, safetensors
prechecks, workload caps, sharding, timeout policy, long-tail exclusions,
serial/GPU/derived phases, trace reuse, closure records, timing evidence, AMD
score reports, and provenance. The file is highly covered but still a high-risk
change surface because many claim-boundary artifacts are assembled there.

Fragile areas:

- Reuse decisions depend on provenance, trace status, closure status, requested
  evidence requirements, and stale checksum detection.
- Timeout classification spans direct `TimeoutExpired`, nested CLI logs, and
  synthetic timeout traces.
- Derived phase behavior can reuse existing traces, so it must not be mistaken
  for new execution evidence.
- `--jobs` is only parallel for derived work; GPU and profiler phases remain
  serial despite accepting shared orchestration flags.

### Safetensors Blobs Are Operationally Fragile

FlashInfer-style safetensors are located through repo-relative paths and
`FLASHINFER_TRACE_DIR` in `src/sol_execbench/driver/problem_packager.py`,
`src/sol_execbench/driver/templates/eval_driver.py`, and
`scripts/run_dataset.py`. Missing blobs can produce no-trace or runtime errors,
while large blobs can make staging expensive if symlink creation falls back to
copying.

### Claim-Boundary Artifacts Need Discipline

The project has many diagnostic and derived reports:

- Environment snapshots in `src/sol_execbench/core/environment.py`.
- Static kernel evidence in `src/sol_execbench/core/bench/static_kernel_evidence.py`.
- Runtime evidence in `src/sol_execbench/core/runtime_evidence.py`.
- Matrix, consistency, claim upgrade, trust summary, and release scripts under
  `src/sol_execbench/core/` and `scripts/`.

Risk: these reports are helpful, but they are not canonical Trace JSONL and are
often diagnostic-only. Keep authority boundaries explicit when adding new
release or README claims.

## Scoring And Derived Analysis Risks

### AMD SOL / SOLAR Derivation Is Complex And Provisional

`src/sol_execbench/core/scoring/amd_bound_graph.py`,
`src/sol_execbench/core/scoring/amd_bound_estimates.py`,
`src/sol_execbench/core/scoring/amd_sol_v2.py`, and
`src/sol_execbench/core/scoring/solar_derivation.py` encode derived operator
graphs, hardware models, confidence levels, byte/flop estimates, and score
eligibility.

Risks:

- Unsupported and inexact operator families can dominate real dataset coverage.
- `src/sol_execbench/data/amd_hardware_models/gfx1200.json` is packaged, while
  CDNA3/MI300X model authority remains more limited in the current evidence.
- Derived AMD-native scores are not upstream SOLAR parity, NVIDIA B200
  equivalence, or leaderboard authority.
- Derived sidecar exclusions and long-tail exclusions must stay visible and
  reversible.

## Dependency And Environment Risks

### ROCm Stack Is Narrowly Pinned

`pyproject.toml` pins Linux x86_64 to PyTorch `2.10.0+rocm7.1`,
torchvision `0.25.0+rocm7.1`, and `triton-rocm==3.6.0` from explicit PyTorch
ROCm indexes. `docker/Dockerfile` defaults to ROCm `7.1.1-complete` and then
installs matching wheels.

Risks:

- Non-Linux or non-x86_64 installs receive non-ROCm PyTorch wheels, so GPU
  behavior can be unavailable while CPU-safe checks pass.
- Docker targets can override wheel versions after frozen sync, so runtime
  image contents may diverge from the checked-in `uv.lock`.
- ROCm minor-version drift can affect compiler flags, profiler output CSV
  formats, device names, and `torch.cuda` compatibility behavior.

### Docker Clock-Lock Privilege Is Delicate

`docker/Dockerfile` grants passwordless SMI commands to the runtime user, and
`docker/entrypoint.sh` attempts to lock and unlock clocks through
`src/sol_execbench/core/bench/clock_lock.py`.

Risks:

- Clock lock availability depends on host/container SMI permissions and exact
  command support.
- Failure to lock clocks is allowed by the entrypoint unless benchmark config
  explicitly requires locked clocks.
- Clock state reset depends on process exit traps and may not run after all host
  or container failure modes.

## Documentation And Provenance Risks

### Upstream Attribution Must Stay Exact

Many source files retain NVIDIA copyright notices while independent ROCm changes
add contributor notices. `docs/provenance.md`, `provenance.toml`,
`THIRD_PARTY_NOTICES.txt`, and `docs/research_preview.md` are important claim
and attribution controls.

Risk: new docs or generated bundles must not imply NVIDIA or AMD endorsement,
NVIDIA B200 equivalence, paper-scale validation, upstream SOLAR parity, or
hosted leaderboard readiness.

### Local Outputs Must Stay Out Of Release Scope

The repository contains ignored/generated surfaces such as `out/`, `.artifacts/`,
`.uv-cache/`, `.venv/`, `.ruff_cache/`, `.pytest_cache/`, `dist/`, and
`data/`. These are useful locally but can contain large artifacts, downloaded
datasets, or host-specific evidence. Keep release bundles curated and governed
by redistribution checks such as `scripts/check_dataset_redistribution.py`.
