# Configuration

The v3 CLI is the configuration authority. Generate its exact machine-readable
surface with:

```bash
uv run sol-execbench --format json contract cli
```

Root options such as `--format json` precede the command. Command-specific
options follow it.

## Benchmark config

`evaluate --config FILE` accepts this JSON shape:

```json
{
  "warmup_runs": 10,
  "iterations": 50,
  "trials": 3,
  "min_measurement_time_seconds": null,
  "lock_clocks": true,
  "benchmark_reference": true,
  "seed": 200
}
```

| Field | Default | Constraint |
| --- | --- | --- |
| `warmup_runs` | `10` | integer ≥ 0 |
| `iterations` | `50` | integer > 0 |
| `trials` | `3` | integer > 0 |
| `min_measurement_time_seconds` | `null` | positive number or null |
| `lock_clocks` | `true` | Boolean |
| `benchmark_reference` | `true` | Boolean |
| `seed` | `200` | integer |

The paper timing label requires the exact defaults for warmup, iterations,
trials, minimum duration and clock locking. Changing them produces a diagnostic
custom timing protocol.

## Evaluate

```bash
uv run sol-execbench evaluate PROBLEM_DIR --solution solution.json

uv run sol-execbench evaluate \
  --definition definition.json \
  --workload workload.jsonl \
  --solution solution.json
```

| Option | Default | Meaning |
| --- | --- | --- |
| `--config` | none | benchmark config JSON |
| `--compile-timeout` | `120` | native compile timeout in seconds |
| `--timeout` | `300` | whole evaluation timeout in seconds |
| `--trace-output` | none | durable canonical Trace JSONL path |
| `--lock-clocks` | off | force the clock-lock requirement |
| `--unsafe-local-execution` | off | allow diagnostic host execution |
| `--keep-staging` | off | preserve staged process assets |
| `--profile` | `none` | `none` or `rocprofv3` |
| `--static-evidence` | `none` | `none` or `auto` |
| `--decision` | `none` | `none` or `auto` |
| `--feedback-*` | none | optional derived-sidecar identities |
| `--verbose` | off | print bounded subprocess details |

JSON response mode requires a trace path:

```bash
uv run sol-execbench --format json evaluate PROBLEM_DIR \
  --solution solution.json --trace-output out/run.trace.jsonl
```

## Dataset

Only the pinned public corpus operations are exposed:

```bash
uv run sol-execbench dataset materialize \
  --manifest problems/RX_9060_XT/manifest.yaml \
  --output problems/local/RX_9060_XT

uv run sol-execbench dataset audit problems/local/RX_9060_XT
```

`materialize --source DIR` uses an already downloaded pinned snapshot. Without
`--source`, it downloads the pinned Hugging Face revision. `--cache-dir` selects
the Hugging Face cache. This revision has no dataset batch-runner command;
evaluate materialized problems individually or use external orchestration.

## SOLAR

```bash
uv run sol-execbench solar analyze PROBLEM_DIR \
  --workload WORKLOAD_UUID \
  --output out/solar/WORKLOAD_UUID \
  --orojenesis-home /path/to/orojenesis
```

Options are `--device` (default `cuda:0`), `--timeout` (default 14400 seconds)
and `--orojenesis-home`. Formal analysis is currently constrained by the pinned
gfx1200 architecture audit and fails closed when required evidence is absent.

`solar learn-handler` is an offline candidate-generation workflow. Its output
is forbidden in formal analysis until reviewed and committed under
`src/solar/handlers` with `verification: passed`, `formal_review: approved`, and
matching source SHA-256 metadata.

## Official score

```bash
uv run sol-execbench --format json score status \
  --manifest problems/RX_9060_XT/manifest.yaml
```

The checked-in manifest declares official authority unavailable, and the
official scorer is not implemented. The status command accepts no measurement
or baseline file and cannot promote caller-authored evidence.

## Environment

| Variable | Use |
| --- | --- |
| `FLASHINFER_TRACE_DIR` | additional safetensors root used by the trusted reference worker |
| `SOL_EXECBENCH_CLOCKS_LOCKED` | evaluator-owned clock-lock evidence (`1` or `0`) |
| `SOL_EXECBENCH_GPU_LOCK_DIR` | directory for per-device lock files |
| `SOL_EXECBENCH_ALLOW_CPU_TIMING` | test-only CPU timing escape hatch |
| `SOL_EXECBENCH_SANDBOXED` | marks an externally isolated execution environment |
| `SOL_EXECBENCH_UNSAFE_LOCAL_EXECUTION` | internal marker set by the CLI flag |
| `SOL_EXECBENCH_GRACEFUL_EXIT` | profiler-controlled normal interpreter teardown |
| `SOLEXECBENCH_ENV_SNAPSHOT` | write an environment sidecar when set to `1` |
| `SOLEXECBENCH_ENV_SNAPSHOT_PATH` | explicit environment sidecar path |
| `SOLAR_OROJENESIS_HOME` | default Orojenesis toolchain directory |
| `OPENAI_API_KEY` | optional offline handler-learning client credential |
| `SOL_EXECBENCH_AMD_ISA_CACHE` | static ISA tool/spec cache |
| `SOL_EXECBENCH_AMD_ISA_OFFLINE` | forbid static ISA downloads when `1` |
| `HIP_VISIBLE_DEVICES`, `ROCR_VISIBLE_DEVICES` | ROCm device visibility |

The reference pipe descriptors, token and worker PID use private
`SOL_EXECBENCH_REFERENCE_*` variables. They are created by the staged
orchestrator, removed from the candidate environment when connected and are not
user configuration.

## Docker

Use `./scripts/run_docker.sh --build` or select a target declared in
`docker/rocm-targets.json` with `--target`. The wrapper owns its image, device
and dependency variables; inspect `./scripts/run_docker.sh --help` for the
current surface.
