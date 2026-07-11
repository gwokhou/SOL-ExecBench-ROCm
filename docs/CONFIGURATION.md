<!-- generated-by: gsd-doc-writer -->
# Configuration

SOL ExecBench ROCm Port is configured through CLI flags, optional benchmark
configuration JSON, `pyproject.toml`, Docker target metadata, Docker wrapper
inputs, and runtime environment variables. The repository does not contain a
required application-level `.env` file.

## How To Use This Reference

Most users only need:

- `--config`, `--compile-timeout`, `--timeout`, `--output`, `--profile`, and
  `--static-evidence` for single benchmark runs.
- `scripts/run_dataset.py` flags for dataset batches and derived reports.
- `./scripts/run_docker.sh --target <id>` when selecting a ROCm container.

The environment-variable table below is intentionally complete. Many variables
are diagnostic or CI-style overrides used by tests, Docker preflight checks, or
evidence sidecars. Treat them as reference material unless a command or failure
message points you to one of them.

## Environment Variables

No environment variable is required for normal host CLI startup. The variables
below are optional runtime, Docker, diagnostic, or build inputs discovered in
`src/`, `scripts/`, and `docker/`.

| Variable | Required | Default | Description |
| --- | --- | --- | --- |
| `PYTORCH_ALLOC_CONF` | Optional | `expandable_segments:True` in compile/eval subprocesses | Set by the CLI subprocess launcher for staged PyTorch ROCm compilation and evaluation. |
| `PYTORCH_ROCM_ARCH` | Optional | Derived from solution `target_hardware` when unset | Overrides the ROCm architecture list used by PyTorch extension builds. |
| `SOL_EXECBENCH_ALLOW_CPU_TIMING` | Optional | Unset | Test/debug escape hatch that lets evaluator timing proceed on CPU tensors when set to `1`; not a GPU validation setting. |
| `SOL_EXECBENCH_GRACEFUL_EXIT` | Optional | Unset | When set to `1`, the eval driver uses `sys.exit(0)` instead of `os._exit(0)` at process end. Automatically injected by the profiler adapter during profiled evaluations. |
| `SOLEXECBENCH_ENV_SNAPSHOT` | Optional | Unset | Set to `1` to write an environment snapshot sidecar next to `--output`. |
| `SOLEXECBENCH_ENV_SNAPSHOT_PATH` | Optional | Unset | Explicit environment snapshot sidecar output path. |
| `HIP_VISIBLE_DEVICES` | Optional | Unset | Device visibility filter recorded in environment and runtime evidence. |
| `ROCR_VISIBLE_DEVICES` | Optional | Unset | ROCr device visibility filter recorded in environment and runtime evidence. |
| `HSA_OVERRIDE_GFX_VERSION` | Optional | Unset | Forced HSA architecture override recorded in environment snapshots. |
| `CUDA_VISIBLE_DEVICES` | Optional | Unset | Compatibility visibility variable recorded by Docker runtime evidence when present. |
| `GPU_DEVICE_ORDINAL` | Optional | Unset | Compatibility visibility variable recorded by Docker runtime evidence when present. |
| `SOL_EXECBENCH_CLOCKS_LOCKED` | Optional | `0` when unset | Set by `docker/entrypoint.sh` after clock-lock attempts; read by clock-lock checks. |
| `FLASHINFER_TRACE_DIR` | Optional | `/sol-execbench/data/flashinfer-trace` under `scripts/run_docker.sh` | Adds the FlashInfer trace safetensors lookup root for evaluation. |
| `IMAGE_NAME` | Optional | `sol-execbench` | Docker wrapper local image name. |
| `IMAGE_TAG` | Optional | `rocm-<selected Docker tag>` | Docker wrapper local image tag. The default target resolves to `rocm-7.1.1-complete`. |
| `ROCM_DOCKER_IMAGE` | Optional | `rocm/dev-ubuntu-24.04` for unknown-target override | Docker image repository override when `--allow-unknown-target` is used. |
| `ROCM_DOCKER_TAG` | Optional | Selected target ID for unknown-target override | Docker image tag override when `--allow-unknown-target` is used. |
| `SOL_EXECBENCH_ALLOW_MIXED_VERSION_DEPENDENCIES` | Optional | `0` | Allows dependency probe diagnostics for mixed-version stacks. |
| `SOL_EXECBENCH_ALLOW_UNTESTED_TARGET_SMOKE` | Optional | `0` | Allows `not_tested` targets to run smoke/E2E commands without validation claims. |
| `SOL_EXECBENCH_RECORD_CONTAINER_VALIDATION` | Optional | `0` | Records successful target-container wrapper benchmark evidence as `container_validated`. |
| `SOL_EXECBENCH_HOST_PYTHON` | Optional | `uv run python` | Host Python executable override for Docker wrapper helper commands. |
| `SOL_EXECBENCH_COMPATIBILITY_ENTRY` | Optional | Unset | Per-target compatibility JSON sidecar path. |
| `SOL_EXECBENCH_COMPATIBILITY_MATRIX` | Optional | Unset | Aggregate compatibility matrix JSON path. |
| `SOL_EXECBENCH_RUN_DOCKER_DRY_RUN` | Optional | `0` | Enables dry-run behavior in `scripts/run_docker.sh`. |
| `SOL_EXECBENCH_DOCKER_CONTEXT` | Optional | `docker context show` output | Test/debug override for Docker context preflight evidence. |
| `SOL_EXECBENCH_DOCKER_HOST` | Optional | `docker context inspect` output | Test/debug override for Docker host preflight evidence. |
| `SOL_EXECBENCH_DEV_KFD_PRESENT` | Optional | Filesystem probe of `/dev/kfd` | Test/debug override for Docker runtime preflight evidence. |
| `SOL_EXECBENCH_DEV_KFD_ACCESSIBLE` | Optional | Read/write probe of `/dev/kfd` | Test/debug override for Docker runtime preflight evidence. |
| `SOL_EXECBENCH_DEV_DRI_PRESENT` | Optional | Filesystem probe of `/dev/dri` | Test/debug override for Docker runtime preflight evidence. |
| `SOL_EXECBENCH_DEV_DRI_ACCESSIBLE` | Optional | Render/card device access probe under `/dev/dri` | Test/debug override for Docker runtime preflight evidence. |
| `SOL_EXECBENCH_GPU_ACCESSIBLE` | Optional | Unset | Test/debug override for Docker runtime preflight GPU accessibility. |
| `SOL_EXECBENCH_HOST_ROCM_VERSION` | Optional | Unset | Runtime evidence override for host ROCm version. |
| `SOL_EXECBENCH_HOST_DRIVER_VERSION` | Optional | Unset | Runtime evidence override for host driver version. |
| `SOL_EXECBENCH_IMAGE_DIGEST` | Optional | Unset | Runtime evidence override for container image digest. |
| `SOL_EXECBENCH_RUNTIME_DEVICE_COUNT` | Optional | Unset | Runtime evidence override for device count. |
| `SOL_EXECBENCH_RUNTIME_DEVICE_NAME` | Optional | Unset | Runtime evidence override for device name. |
| `SOL_EXECBENCH_RUNTIME_GFX_ARCHITECTURE` | Optional | Unset | Runtime evidence override for gfx architecture. |
| `SOL_EXECBENCH_GPU_CLK_MHZ` | Optional | Empty string in Docker wrapper environment | Forwarded into Docker runs for GPU clock diagnostics. |
| `SOL_EXECBENCH_DRAM_CLK_MHZ` | Optional | Empty string in Docker wrapper environment | Forwarded into Docker runs for DRAM clock diagnostics. |
| `SOL_EXECBENCH_DEPENDENCY_TORCH_DISTRIBUTION_VERSION` | Optional | Unset | Dependency preflight/runtime evidence override for installed Torch distribution version. |
| `SOL_EXECBENCH_DEPENDENCY_TORCH_VERSION` | Optional | Unset | Dependency preflight/runtime evidence override for imported Torch version. |
| `SOL_EXECBENCH_DEPENDENCY_TORCH_LOCAL_VERSION` | Optional | Unset | Dependency preflight/runtime evidence override for Torch local version suffix. |
| `SOL_EXECBENCH_DEPENDENCY_TORCH_ROCM_TARGET` | Optional | Unset | Dependency preflight/runtime evidence override for expected ROCm wheel target. |
| `SOL_EXECBENCH_DEPENDENCY_TORCH_HIP_VERSION` | Optional | Unset | Dependency preflight/runtime evidence override for `torch.version.hip`. |
| `SOL_EXECBENCH_DEPENDENCY_TORCH_CUDA_VERSION` | Optional | Unset | Dependency preflight/runtime evidence override for `torch.version.cuda`. |
| `SOL_EXECBENCH_DEPENDENCY_TORCH_DEVICE_AVAILABLE` | Optional | Unset | Dependency preflight/runtime evidence override for Torch device availability. |
| `SOL_EXECBENCH_DEPENDENCY_TORCH_IMPORT_ERROR` | Optional | Unset | Dependency preflight/runtime evidence override for Torch import failure text. |
| `SOL_EXECBENCH_DEPENDENCY_TORCHVISION_DISTRIBUTION_VERSION` | Optional | Unset | Dependency preflight/runtime evidence override for installed torchvision distribution version. |
| `SOL_EXECBENCH_DEPENDENCY_TRITON_ROCM_DISTRIBUTION_VERSION` | Optional | Unset | Dependency preflight/runtime evidence override for installed `triton-rocm` distribution version. |
| `SOL_EXECBENCH_DEPENDENCY_TRITON_ROCM_STATUS` | Optional | Unset | Dependency preflight/runtime evidence override for Triton ROCm status. |
| `SOL_EXECBENCH_DEPENDENCY_CONTAINER_ROCM_USER_SPACE_VERSION` | Optional | Unset | Dependency preflight/runtime evidence override for container ROCm user-space version. |
| `SOL_EXECBENCH_DEPENDENCY_HIPCC_VERSION` | Optional | Unset | Dependency preflight/runtime evidence override for `hipcc` version. |
| `SOL_EXECBENCH_DEPENDENCY_TOOLCHAIN_ROCM_VERSION` | Optional | Unset | Dependency preflight/runtime evidence override for toolchain ROCm version. |
| `ROCM_PATH` | Optional | `/opt/rocm` in Docker image | ROCm installation root in `docker/Dockerfile`. |
| `HIP_PATH` | Optional | `/opt/rocm` in Docker image | HIP installation root in `docker/Dockerfile`. |
| `HIP_PLATFORM` | Optional | `amd` in Docker image | HIP platform selector in `docker/Dockerfile`. |
| `UV_CACHE_DIR` | Optional | `/home/${HOST_USER}/.cache/uv` in Docker image | UV cache directory in `docker/Dockerfile`. |
| `UV_LINK_MODE` | Optional | `copy` in Docker image | UV link behavior in `docker/Dockerfile`. |
| `UV_COMPILE_BYTECODE` | Optional | `1` in Docker image | Enables bytecode compilation in Docker installs. |
| `UV_PYTHON_DOWNLOADS` | Optional | `never` in Docker image | Disables Python downloads during Docker installs. |
| `UV_PROJECT_ENVIRONMENT` | Optional | `/venv` in Docker image | Docker image virtual environment path. |
| `HOST_UID` | Optional | `1000` Docker build argument | Host user ID used when creating the Docker image user. |
| `HOST_GID` | Optional | `1000` Docker build argument | Host group ID used when creating the Docker image group. |
| `HOST_USER` | Optional | `sol-execbench` Docker build argument | Host user name used when creating the Docker image user. |
| `PYTORCH_TORCH_VERSION` | Optional | `2.10.0+rocm7.1` Docker build argument | Torch wheel version installed in the Docker image. |
| `PYTORCH_TORCHVISION_VERSION` | Optional | `0.25.0+rocm7.1` Docker build argument | Torchvision wheel version installed in the Docker image. |
| `PYTORCH_ROCM_INDEX_URL` | Optional | `https://download.pytorch.org/whl/rocm7.1` Docker build argument | PyTorch ROCm wheel index URL. |
| `TRITON_ROCM_VERSION` | Optional | `3.6.0` Docker build argument | `triton-rocm` wheel version installed in the Docker image. |
| `TRITON_ROCM_INDEX_URL` | Optional | `https://download.pytorch.org/whl/` Docker build argument | Extra wheel index URL used for `triton-rocm`. |

## Config File Format

The benchmark evaluator accepts an optional JSON config file through
`--config`. It is loaded into `BenchmarkConfig` from
`src/sol_execbench/core/bench/config/benchmark_config.py`.

```json
{
  "warmup_runs": 10,
  "iterations": 50,
  "lock_clocks": false,
  "benchmark_reference": true,
  "seed": 200
}
```

| Field | Default | Description |
| --- | --- | --- |
| `warmup_runs` | `10` | Number of warmup runs before measurement. Must be greater than or equal to `0`. |
| `iterations` | `50` | Number of timed iterations. Must be greater than `0`. |
| `lock_clocks` | `false` | Whether evaluation requires clocks to be locked. |
| `benchmark_reference` | `true` | Whether to benchmark the reference implementation. |
| `seed` | `200` | Integer seed for benchmark input generation. |

The Docker target manifest at `docker/rocm-targets.json` is another repository
configuration file. It declares `default_target_id`,
`requested_rocm_user_space_version`, Docker image tags, PyTorch ROCm wheel
policies, and Triton ROCm wheel policies for supported container targets.
The current declared targets are:

| Target ID | ROCm User Space | Docker Tag | PyTorch Policy |
| --- | --- | --- | --- |
| `rocm-7.0.2-ubuntu-24.04-container` | `7.0.2` | `7.0.2-complete` | `torch==2.10.0+rocm7.0`, `torchvision==0.25.0+rocm7.0`, `triton-rocm==3.6.0` |
| `rocm-7.1.1-ubuntu-24.04-container` | `7.1.1` | `7.1.1-complete` | `torch==2.10.0+rocm7.1`, `torchvision==0.25.0+rocm7.1`, `triton-rocm==3.6.0` |
| `rocm-7.2.0-ubuntu-24.04-container` | `7.2.0` | `7.2-complete` | `torch==2.11.0+rocm7.2`, `torchvision==0.26.0+rocm7.2`, `triton-rocm==3.6.0` |

These targets describe container user-space and dependency policy. They do not,
by themselves, make native-host, paper-parity, leaderboard, MI300X, CDNA4, or
NVFP4/MXFP4 validation claims.

`provenance.toml` is the machine-readable source attribution policy. It
classifies active files as upstream-retained, derivative-modified, independent
ROCm work, or generated/planning material; defines header policy for each
class; and lists files that may retain NVIDIA SPDX notices. The prerelease
readiness gate consumes this manifest with `docs/provenance.md`.

## Required vs Optional Settings

There are no required environment variables for starting the CLI. Required
inputs are passed as files or positional arguments:

- `sol-execbench <problem_dir>` requires definition and workload JSON/JSONL
  files in the problem directory.
- If no problem directory is used, `--definition` and `--workload` are required.
- A solution must be supplied through `--solution` or by the conventional
  solution JSON file in the problem directory.
- `contract`, `doctor`, and `toolchain` subcommands only support `--json`
  output and raise a Click exception without it.
- `BenchmarkConfig.warmup_runs` must be `>= 0`; `iterations` must be `> 0`.
- `scripts/run_docker.sh --target`, `--compatibility-entry`, and
  `--compatibility-matrix` require argument values when those flags are used.

Clock-lock settings are optional, but `--lock-clocks` changes behavior by
forcing `BenchmarkConfig.lock_clocks = true`. If clocks are not locked,
clock-sensitive evaluation paths can reject the run based on
`SOL_EXECBENCH_CLOCKS_LOCKED`.

## Defaults

| Setting | Default | Source |
| --- | --- | --- |
| Benchmark warmup runs | `10` | `BenchmarkConfig.warmup_runs` |
| Benchmark iterations | `50` | `BenchmarkConfig.iterations` |
| Benchmark clock-lock requirement | `false` | `BenchmarkConfig.lock_clocks` |
| Benchmark reference timing | `true` | `BenchmarkConfig.benchmark_reference` |
| Benchmark seed | `200` | `BenchmarkConfig.seed` |
| CLI compile timeout | `120` seconds | `--compile-timeout` option |
| CLI evaluation timeout | `600` seconds | `--timeout` option |
| CLI profiling mode | `none` | `--profile` option |
| CLI static evidence mode | `none` | `--static-evidence` option |
| Dataset runner output directory | `<repo_root>/out` | `scripts/run_dataset.py --output` |
| Dataset runner timeout | `300` seconds per problem | `scripts/run_dataset.py --timeout` |
| Dataset runner phase | `all` | `scripts/run_dataset.py --phase` |
| Dataset runner execution mode | `serial` | `scripts/run_dataset.py --execution-mode` |
| Baseline comparison win threshold | `2.0` percent | `sol-execbench-baseline --win-pct` |
| Baseline comparison parity threshold | `5.0` percent | `sol-execbench-baseline --parity-pct` |
| Docker target | `rocm-7.1.1-ubuntu-24.04-container` | `docker/rocm-targets.json` |
| Docker base image | `rocm/dev-ubuntu-24.04:7.1.1-complete` | Default Docker target and `docker/Dockerfile` |
| Docker local image name | `sol-execbench` | `scripts/run_docker.sh` |
| Docker local image tag | `rocm-<selected Docker tag>` | `scripts/run_docker.sh` |
| Docker FlashInfer trace root | `/sol-execbench/data/flashinfer-trace` | `scripts/run_docker.sh` |
| Python package version | `1.0.5` | `pyproject.toml`; separate from milestone and prerelease labels |
| Python requirement | `>=3.12,<3.14` | `pyproject.toml` |
| Default ROCm Torch wheel | `torch==2.10.0+rocm7.1` | `pyproject.toml` Linux x86_64 marker and Docker target metadata |
| Default ROCm torchvision wheel | `torchvision==0.25.0+rocm7.1` | `pyproject.toml` Linux x86_64 marker and Docker target metadata |
| Default Triton ROCm wheel | `triton-rocm==3.6.0` | `pyproject.toml` Linux x86_64 marker and Docker target metadata |
| Provenance policy document | `docs/provenance.md` | `provenance.toml` |

## Per-Environment Overrides

The repository does not define `.env.development`, `.env.production`, or
`.env.test` files, and it does not contain deployment-specific configuration
files. Use these source-backed override paths instead:

- Host benchmark runs: pass CLI flags such as `--config`, `--compile-timeout`,
  `--timeout`, `--output`, `--profile`, `--static-evidence`, and
  `--feedback-*` diagnostic metadata flags.
- Per-problem benchmark settings: place the benchmark config JSON next to the
  problem definition and workload files, or pass a config file explicitly with
  `--config`.
- Dataset batch settings: pass `scripts/run_dataset.py` flags such as
  `--phase`, `--limit`, `--max-workloads`, `--workload-shard-size`,
  `--iterations`, `--warmup-runs`, `--timeout`, `--timeout-overrides`,
  `--long-tail-exclusions`, `--blob-precheck`, and `--lock-clocks`.
- RDNA4 profiler timing closure settings: pass
  `scripts/internal/rdna4/run_rdna4_profiler_timing_batch.py` flags such as
  `--only-problem`, `--only-problem-file`, `--skip-problem`,
  `--skip-problem-file`, `--mark-blocked-problem`, `--mark-blocked-only`,
  `--workload-limit`, `--workload-offset`, `--workload-sharded`,
  `--workload-slice-timing-dir`, `--workload-sharded-import-only`,
  `--compact-workload-slices`, `--timeout`, `--subprocess-memory-limit-gib`,
  `--max-estimated-timing-input-gib`, `--auto-estimated-timing-input-cap`,
  `--temp-dir`, `--resume`, `--max-workers`, `--clock-locked`,
  `--hip-runtime-trace`, `--keep-staging`, `--keep-rocprofv3-csv`,
  `--strict-isolation`, `--gpu-device`, and `--calibration-path`.
- Docker ROCm stack selection: use `./scripts/run_docker.sh --target <id>` for
  declared targets in `docker/rocm-targets.json`.
- Docker image overrides for unknown targets: use `--allow-unknown-target` with
  `ROCM_DOCKER_IMAGE` and `ROCM_DOCKER_TAG`.
- Diagnostic and CI-style preflight overrides: set the
  `SOL_EXECBENCH_DEPENDENCY_*`, `SOL_EXECBENCH_RUNTIME_*`,
  `SOL_EXECBENCH_DEV_*`, `SOL_EXECBENCH_DOCKER_*`, and compatibility sidecar
  variables listed above.
- Source attribution policy: update `provenance.toml` and
  `docs/provenance.md`; prior blanket header corrections are handled as
  ordinary commits unless a separate legal review requires history rewriting.

## CLI Flags

Primary evaluator form:

```bash
uv run sol-execbench <problem_dir> --solution solution.json
```

Explicit input form:

```bash
uv run sol-execbench \
  --definition definition.json \
  --workload workload.jsonl \
  --solution solution.json
```

| Flag | Default | Purpose |
| --- | --- | --- |
| `--definition` | None | Path to the definition JSON file when not using a problem directory. |
| `--workload` | None | Path to `workload.jsonl` when not using a problem directory. |
| `--solution` | Conventional solution JSON in problem directory when present | Path to solution JSON. |
| `--config` | None | Optional benchmark config JSON. |
| `--compile-timeout` | `120` | HIP/C++ compilation timeout in seconds. |
| `--timeout` | `600` | Evaluation subprocess timeout in seconds. |
| `-o`, `--output` | None | Trace JSONL output path. |
| `--json` | Disabled | Print trace JSON lines to stdout. |
| `--lock-clocks` | Disabled | Require GPU clocks to be locked. |
| `--keep-staging` | Disabled | Preserve the temporary staging directory. |
| `--profile` | `none` | Use `rocprofv3` for optional diagnostic profiling when set to `rocprofv3`. |
| `--static-evidence` | `none` | Collect optional diagnostic static kernel evidence when set to `auto`. |
| `--feedback-target-id` | None | Consumer target identity persisted in diagnostic agent feedback. |
| `--feedback-run-id` | None | Consumer run identity persisted in diagnostic agent feedback. |
| `--feedback-candidate-id` | None | Consumer candidate identity persisted in diagnostic agent feedback. |
| `--feedback-source-sha256` | None | Consumer source SHA256 identity persisted in diagnostic agent feedback. |
| `--feedback-sol-version` | None | Consumer SOL version or tag identity persisted in diagnostic agent feedback. |
| `-v`, `--verbose` | Disabled | Show subprocess output and staging details. |

No-trace diagnostic sidecars are not controlled by a separate flag. When an
evaluation subprocess produces no parseable trace JSONL, the CLI writes a
bounded diagnostic-only sidecar next to `--output`, in the kept staging
directory, or in the system temp directory depending on the available path.
That sidecar records stdout/stderr tails and line counts and is not canonical
trace JSONL.

Metadata and diagnostic subcommands require `--json`:

```bash
uv run sol-execbench contract --json
uv run sol-execbench doctor --json
uv run sol-execbench toolchain --json
```

`toolchain` also accepts `--evidence-level`, `--artifact-type`, `--gpu-arch`,
`--hardware-generation`, `--rocm-version`, and `--list-registry`.

Local dataset migration subcommands live under the same `sol-execbench` entry
point:

```bash
uv run sol-execbench dataset migrate-sol <source_root> <output_root> \
  --category L1 --manifest out/sol-migration-manifest.json --json
uv run sol-execbench dataset migrate-flashinfer <source_root> <output_root> \
  --manifest out/flashinfer-migration-manifest.json --json
```

| Subcommand | Option | Purpose |
| --- | --- | --- |
| `dataset migrate-sol` | `--category` | Restrict SOL-ExecBench migration to one or more categories. |
| `dataset migrate-sol` | `--source-revision` | Record the source dataset revision or local commit ref in the manifest. |
| `dataset migrate-sol` | `--manifest` | Write the migration manifest to an explicit path. |
| `dataset migrate-sol` | `--json` | Print the migration manifest JSON to stdout. |
| `dataset migrate-flashinfer` | `--source-revision` | Record the FlashInfer Trace source revision or local commit ref in the manifest. |
| `dataset migrate-flashinfer` | `--manifest` | Write the migration manifest to an explicit path. |
| `dataset migrate-flashinfer` | `--json` | Print the migration manifest JSON to stdout. |

## Dataset Runner Options

`scripts/run_dataset.py` runs a single problem directory or a dataset root with
category subdirectories. It shells out to `sol-execbench` for trace collection
and can also build derived AMD/SOLAR/timing reports from existing traces.

```bash
uv run scripts/run_dataset.py data/SOL-ExecBench/benchmark \
  --category L1 L2 --limit 5 -o out/dataset-smoke
```

| Option | Default | Purpose |
| --- | --- | --- |
| `--category` | None | Restrict to one or more of `L1`, `L2`, `FlashInfer-Bench`, or `Quant`. |
| `--limit` | None | Maximum number of problems to evaluate. |
| `--phase` | `all` | Run `all`, `traces`, `derived`, or `timing`. |
| `--jobs` | `1` | Parallel workers for CPU/I/O-only phases; `auto` is accepted. |
| `--execution-mode` | `serial` | Use `serial` ordering or `pipeline` for trace-stage CPU preparation overlapped with serial GPU evaluation. |
| `--prepare-jobs` | `auto` | CPU preparation workers for pipeline mode. |
| `--gpu-jobs` | `1` | GPU evaluation workers for pipeline mode; only `1` is currently accepted. |
| `--timeout-policy` | `record` | Record timeout traces or fail on no-trace timeout behavior. |
| `--timeout-overrides` | None | JSON object with `default`, `problems`, or `workloads` timeout overrides. |
| `--long-tail-exclusions` | None | JSON config for temporarily excluding known long-tail problems, workloads, or workload shards while preserving closure accounting. |
| `--blob-precheck` | `fail` | Precheck safetensors references with `fail`, `warn`, or `off`. |
| `--log-order` | `completion` | Pipeline log ordering: `completion` or `problem`. |
| `-o`, `--output` | `<repo_root>/out` | Output directory for traces and summary files. |
| `--timeout` | `300` | Per-problem GPU evaluation timeout in seconds. |
| `--max-workloads` | None | Limit workloads per problem by truncating the workload input for the run. |
| `--workload-shard-size` | None | Split workload files into temporary shards with this many workloads per CLI invocation. |
| `--iterations` | `BenchmarkConfig.iterations` | Override timed iterations in generated per-run config. |
| `--warmup-runs` | `BenchmarkConfig.warmup_runs` | Override warmup runs in generated per-run config. |
| `--lock-clocks` | Disabled | Require locked GPU clocks for benchmark and timing evidence. |
| `--solution-name` | Definition reference | Use a named solution file in each problem directory. |
| `--rerun` | Disabled | Re-evaluate problems that already have results. |
| `--keep-staging` | Disabled | Preserve `sol-execbench` staging directories. |
| `-v`, `--verbose` | Disabled | Pass verbose mode through to `sol-execbench`. |
| `--amd-score-report` | None | Write a derived AMD-native suite score JSON report. |
| `--scoring-baseline` | None | Use a release-defined scoring baseline artifact for AMD score reports. |
| `--amd-sol-bound-dir` | None | Write derived AMD SOL bound v3 sidecars when score reporting is enabled. |
| `--solar-derivation` | None | Write SOLAR derivation sidecars from definitions and workloads. |
| `--timing-evidence-dir` | None | Write per-problem ROCm timing evidence JSON. |
| `--timing-tool-version` | `rocprofv3` | Tool version string recorded in timing evidence. |
| `--gpu-architecture` | `unknown` | GPU architecture string recorded in timing evidence. |
| `--ready-subset` | None | Bound dataset execution to a ready-subset JSON. |
| `--readiness` | None | Enrich execution-closure blockers from readiness JSON. |
| `--execution-closure` | `<output>/execution_closure.json` when `--ready-subset` is supplied | Write execution closure JSON. |
| `--dataset-manifest` | None | Dataset manifest JSON used for closure provenance. |

`--execution-mode pipeline` currently supports `--phase traces` or `--phase all`,
rejects `--ready-subset` and `--execution-closure`, and still requires
`--gpu-jobs 1`. With `--phase all`, trace collection uses pipeline scheduling
and derived/timing sidecars are produced after the serial GPU trace path.

## Baseline Comparison CLI

The package also exposes `sol-execbench-baseline` for comparing existing trace
JSONL files. It does not run GPU evaluation.

```bash
uv run sol-execbench-baseline \
  --candidate out/current/traces.jsonl \
  --baseline out/baseline/traces.jsonl \
  --format json --output out/baseline-comparison.json
```

| Flag | Default | Purpose |
| --- | --- | --- |
| `--candidate` | Required | Candidate trace JSONL file. |
| `--baseline` | Required, repeatable | One or more baseline trace JSONL files. |
| `--format` | `text` | Output `text` or `json`. |
| `--output` | stdout | Optional output path. |
| `--win-pct` | `2.0` | Candidate must beat baseline by at least this percentage to be a win. |
| `--parity-pct` | `5.0` | Candidate within this percentage of baseline is parity. |
| `--amd-native-claim` | Disabled | Label output as an AMD-native claim and emit guardrail warnings. |

## Docker Wrapper Flags

`./scripts/run_docker.sh` selects a declared ROCm Docker target, performs host
and dependency preflights, optionally builds the image, and then runs a command
inside the container. Arguments after `--` become the container command;
arguments before `--` that are not wrapper flags are forwarded to `docker run`.

```bash
./scripts/run_docker.sh --target rocm-7.1.1-ubuntu-24.04-container -- \
  sol-execbench tests/sol_execbench/samples/rmsnorm \
    --solution tests/sol_execbench/samples/rmsnorm/solution_triton.json
```

| Flag | Purpose |
| --- | --- |
| `--build` | Build the selected local image before running. |
| `--target <id>` | Select a declared target from `docker/rocm-targets.json`. |
| `--allow-unknown-target` | Permit explicit `ROCM_DOCKER_IMAGE` and `ROCM_DOCKER_TAG` override selection. |
| `--allow-mixed-version-dependencies` | Allow dependency probe/smoke diagnostics for mixed-version stacks. |
| `--allow-untested-target-smoke` | Allow `not_tested` targets to run smoke/E2E commands without validation claims. |
| `--record-container-validation` | Record successful wrapper benchmark evidence as `container_validated`. |
| `--preflight-only` | Print preflight JSON and exit before Docker build/run. |
| `--compatibility-entry <path>` | Write a per-target compatibility JSON sidecar. |
| `--compatibility-matrix <path>` | Write or aggregate a compatibility matrix JSON report. |

## RDNA4 Profiler Timing Operator Flags

The RDNA4 profiler closure scripts are evidence-generation tools for existing
dataset layouts and timing sidecars. They do not change canonical Trace JSONL
or create score authority.

`scripts/internal/rdna4/run_rdna4_profiler_timing_coverage.py` accepts:

| Flag | Purpose |
| --- | --- |
| `--dataset-root` | Dataset benchmark root, defaulting to `data/SOL-ExecBench/benchmark`. |
| `--output-dir` | Directory for generated coverage JSON, Markdown, and summary JSON artifacts. |
| `--timing-evidence-dir` | Timing sidecar root; may be repeated. |
| `--category` | Dataset category filter; may be repeated. |
| `--expected-problem-denominator` | Expected denominator, defaulting to `235`. |
| `--no-expected-problem-denominator` | Disable fixed denominator recording. |
| `--require-profiler-complete` | Exit nonzero unless every denominator problem has full profiler-backed timing. |

`scripts/internal/rdna4/run_rdna4_profiler_timing_batch.py` accepts the same dataset/output
roots plus target and workload controls:

| Flag | Purpose |
| --- | --- |
| `--source-timing-dir` | Existing fallback timing sidecar root; may be repeated. |
| `--replacement-timing-dir` | Destination for replacement timing sidecars. |
| `--limit` | Limit selected targets. |
| `--only-problem` | Include only a problem ID; may be repeated. |
| `--only-problem-file` | Read included problem IDs from a file. |
| `--skip-problem` | Skip a problem ID; may be repeated. |
| `--skip-problem-file` | Read skipped problem IDs from a file. |
| `--mark-blocked-problem` | Write a classified profiler-blocked sidecar for a problem ID. |
| `--mark-blocked-only` | Only write requested blocked sidecars; do not profile other targets. |
| `--workload-limit` | Profile a bounded workload slice for a selected problem. |
| `--workload-offset` | Start workload-slice profiling at an offset. |
| `--workload-sharded` | Profile each workload independently and aggregate complete manifests. |
| `--workload-slice-timing-dir` | Import existing workload-slice timing roots before profiling missing slices. |
| `--workload-sharded-import-only` | Aggregate imported slices without profiling missing workloads. |
| `--compact-workload-slices`, `--no-compact-workload-slices` | Compact completed slice timing sidecars and remove raw rocprofv3 run directories after manifest summaries; enabled by default. |
| `--timeout` | Per-profiler subprocess timeout in seconds. |
| `--subprocess-memory-limit-gib` | Optional address-space limit for staged profiler subprocesses. |
| `--max-estimated-timing-input-gib` | Manual cap for estimated timing input footprint before launching profiler subprocesses. |
| `--auto-estimated-timing-input-cap`, `--no-auto-estimated-timing-input-cap` | Dynamically derive the estimated timing input cap from available memory, cgroup remaining memory, and subprocess limit; enabled by default. |

### Calibrated AMD hardware models

`sol-execbench hardware-model calibrate --device 0 --output calibration.json` records
measured or explicitly unknown calibration evidence.  The optional `--offline` and
`--no-auto-install` switches prevent managed ROCm Compute Profiler dependency
installation; unavailable profiler data remains `unknown`.  Convert only a validated
artifact with `hardware-model build --calibration calibration.json --output model.json`,
then select that external model explicitly in bound-generation workflows.

| `--temp-dir` | Parent directory for profiler staging directories. |
| `--resume`, `--no-resume` | Resume from existing sidecars and manifests; enabled by default. |
| `--max-workers` | Maximum target-level workers; default is `4`. |
| `--timing-tool-version` | Timing tool version string recorded in evidence. |
| `--gpu-architecture` | GPU architecture string recorded in evidence. |
| `--clock-locked`, `--no-clock-locked` | Whether evidence records locked clocks; enabled by default. |
| `--hip-runtime-trace`, `--no-hip-runtime-trace` | Include rocprofv3 HIP runtime API tracing; disabled by default. |
| `--keep-staging` | Keep per-target staging directories after profiler subprocesses. |
| `--keep-rocprofv3-csv` | Keep raw rocprofv3 CSV run directories after compact timing sidecars are written. |
| `--strict-isolation` | Abort on timing isolation check failures instead of warning. |
| `--gpu-device` | Set `ROCR_VISIBLE_DEVICES` to a device index for GPU isolation. |
| `--calibration-path` | Include rocprofv3 overhead calibration data from a calibration JSON sidecar. |

### AMD ISA tool layer

`sol_execbench.tools.amd_isa` exposes AMD's machine-readable ISA decoder and
explorer APIs to Python callers. Its first explicit use builds a small local
C++ helper and downloads the project-pinned XML release into the user cache;
imports have no build or network side effect. Set
`SOL_EXECBENCH_AMD_ISA_CACHE` to relocate that cache, or set
`SOL_EXECBENCH_AMD_ISA_OFFLINE=1` to require a pre-populated cache. Missing
tools, a missing cache, or failed integrity checks produce explicit tool-layer
errors rather than fabricated ISA capability claims.

The classification scripts
`scripts/internal/rdna4/run_rdna4_profiler_partial_failures.py` and
`scripts/internal/rdna4/run_rdna4_profiler_sharded_closure.py` accept dataset root, output
directory, timing evidence directories, and denominator controls. The sharded
closure audit also accepts repeated `--target-status` values.

## Package Configuration

`pyproject.toml` defines:

- Package name `sol-execbench`
- Version `1.0.5`, which is the Python package version rather than the v1.26
  research-prerelease milestone tag
- Python range `>=3.12,<3.14`
- Console scripts `sol-execbench` and `sol-execbench-baseline`
- Runtime dependencies, including PyTorch ROCm, torchvision ROCm, `triton-rocm`,
  Pydantic, Click, Rich, datasets, and native build helpers
- Development dependencies for pytest, pytest-xdist, Ruff, Ty, and pre-commit
- Pytest markers
- Ruff exclusions
- Ty source roots
- UV package indexes for PyPI, PyTorch ROCm 7.1, and the PyTorch ROCm package root

## Docker Wrapper Settings

`scripts/run_docker.sh` supports:

- `--build` to build the selected Docker image.
- `--target <id>` to select a declared target from `docker/rocm-targets.json`.
- `--allow-unknown-target` with `ROCM_DOCKER_IMAGE` and `ROCM_DOCKER_TAG`.
- `--allow-mixed-version-dependencies` for mixed-version dependency diagnostics.
- `--allow-untested-target-smoke` for non-authoritative smoke/E2E runs on
  `not_tested` targets.
- `--record-container-validation` to write container validation evidence after
  successful wrapper execution.
- `--preflight-only` to run preflight classification without launching the
  benchmark command.
- `--compatibility-entry <path>` and `--compatibility-matrix <path>` to write
  compatibility sidecars.

The wrapper derives PyTorch and Triton Docker build arguments from
`docker/rocm-targets.json` so ROCm 7.0, 7.1, and 7.2 images can install
target-specific ROCm wheel stacks without changing the project lockfile.

## Dataset Runner Evidence Notes

The full dataset runner flag table appears above in
[Dataset Runner Options](#dataset-runner-options). The most common evidence
flags are:

- `--ready-subset` and `--readiness` to bound execution to ready workloads and
  carry blocker reasons into closure records.
- `--execution-closure` to write attempted, skipped, missing-trace, and
  missing-evidence states.
- `--dataset-manifest` to include migration or acquisition provenance in reuse
  decisions.
- `--amd-score-report`, `--amd-sol-bound-dir`, and `--solar-derivation` to
  generate opt-in derived score and bound sidecars.
- `--timing-evidence-dir` to write source-specific ROCm timing evidence.

Existing traces are reused only when they exist, have no failed workloads,
`--rerun` is not set, and any requested execution-closure provenance matches
the current manifest, ready subset, readiness, solution, and derived-evidence
requirements.

The `auto` job count uses the current CPU count capped by the number of selected
problems and an internal upper bound. Derived sidecar names are scoped by
problem path during dataset runs, so two problems with the same definition name
and workload UUID do not overwrite each other's AMD SOL or SOLAR derivation
files.

## Docker Image Settings

`docker/Dockerfile` sets ROCm and `uv` environment defaults:

| Variable | Default |
| --- | --- |
| `ROCM_PATH` | `/opt/rocm` |
| `HIP_PATH` | `/opt/rocm` |
| `HIP_PLATFORM` | `amd` |
| `PATH` | Includes `/opt/rocm/bin`, `/opt/rocm/llvm/bin`, and `/venv/bin`. |
| `LD_LIBRARY_PATH` | `/opt/rocm/lib` |
| `HOME` | `/home/${HOST_USER}` |
| `UV_CACHE_DIR` | `/home/${HOST_USER}/.cache/uv` |
| `UV_LINK_MODE` | `copy` |
| `UV_COMPILE_BYTECODE` | `1` |
| `UV_PYTHON_DOWNLOADS` | `never` |
| `UV_PROJECT_ENVIRONMENT` | `/venv` |
| `PYTHONPATH` | `/sol-execbench/src` |

Docker build arguments include `ROCM_DOCKER_IMAGE`, `ROCM_DOCKER_TAG`,
`PYTORCH_TORCH_VERSION`, `PYTORCH_TORCHVISION_VERSION`,
`PYTORCH_ROCM_INDEX_URL`, `TRITON_ROCM_VERSION`, `TRITON_ROCM_INDEX_URL`,
`HOST_UID`, `HOST_GID`, and `HOST_USER`.

## Optional Evidence Outputs

These settings create sidecars without changing trace JSONL schema:

- `--profile rocprofv3` writes profiler metadata next to `--output`, or under
  the staging directory when no output file is provided.
- `--static-evidence auto` writes static kernel evidence metadata for native
  solution builds.
- `SOLEXECBENCH_ENV_SNAPSHOT=1` writes an environment sidecar next to
  `--output`.
- `SOLEXECBENCH_ENV_SNAPSHOT_PATH=<path>` writes an environment sidecar to an
  explicit path.

## Configuration Boundaries

No deployment, staging, or production config files are present. For local
benchmark changes, use CLI flags or benchmark config JSON. For ROCm container
runs, use `./scripts/run_docker.sh` so target selection, device mounting,
dependency preflight, and compatibility sidecars stay aligned with repository
logic.
