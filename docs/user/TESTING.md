# Testing

## Default suite

```bash
uv run pytest tests/
```

The repository config caps xdist at eight workers. Do not use `-n auto` on a
ROCm machine. Use `-n 0` when debugging subprocess/native-extension tests.

## Focused CPU-safe suites

```bash
uv run pytest tests/solar -q -n 0
uv run pytest tests/sol_execbench/core -q -n 0
uv run pytest tests/sol_execbench/cli -q -n 0
uv run pytest tests/sol_execbench/driver/test_problem_packager.py -q -n 0
uv run pytest tests/sol_execbench/core/bench/test_reference_protocol.py -q -n 0
```

The generated driver integration suite is intentionally separate because it
launches multiple PyTorch subprocesses:

```bash
uv run pytest tests/sol_execbench/driver/test_eval_driver.py -q -n 0
```

## Architecture guardrails

```bash
uv run --with ruff ruff check .
uv run ty check
uv run python scripts/check_coupling.py
uv run python scripts/check_readability.py
uv run python scripts/check_current_docs.py
```

These gates scan both `sol_execbench` and `solar`, including package-direction
rules, stage entry points and non-increasing SOLAR readability debt.

## Hardware markers

Use the existing markers to select evidence that the host can actually
provide:

- `requires_rocm` / `requires_rocm_gpu` for a visible ROCm device;
- `requires_rocm_dev` and `cpp` for native development headers/toolchain;
- `requires_triton_rocm` for Triton ROCm;
- `requires_rdna4` for gfx1200-class evidence;
- `requires_cdna3` for gfx94x-class evidence;
- `docker_dependency` for the declared container dependency stack.

Examples:

```bash
uv run pytest tests/ -m requires_rocm -n 0
uv run pytest tests/ -m 'requires_rdna4 and cpp' -n 0
```

Do not replace unavailable hardware with a broad `xfail`, mock or skip.
Hardware claims require the exact device/toolchain and should skip only on a
precisely tested missing prerequisite.

## Process-boundary expectations

Driver tests should assert that:

- the orchestrator starts the trusted reference worker before candidate code;
- reference control uses JSON and tensors use safetensors, never pickle;
- the candidate driver neither loads nor invokes reference source;
- input-generation and reference-execution failures retain distinct statuses;
- relative speedup is derived after worker completion;
- shutdown, timeouts and process groups do not leak children.

SOLAR API tests should assert each public stage code and atomic removal of
partial output on failure.

## ROCm Matrix Guardrails

The CPU-safe compatibility checks cover status classification and reason-code classification,
schema serialization, mixed-version blocking, unknown Target rejection,
Docker Target selection and documentation boundaries:

```bash
uv run pytest \
  tests/sol_execbench/core/platform/test_rocm_compatibility_matrix.py \
  tests/sol_execbench/core/reports/test_matrix_claim_guardrails.py \
  tests/sol_execbench/core/platform/test_docker_matrix_targets.py \
  tests/sol_execbench/core/platform/test_docker_matrix_preflight.py \
  tests/sol_execbench/core/platform/test_run_docker_matrix_script.py \
  tests/sol_execbench/core/platform/test_dependency_matrix_policy.py \
  tests/sol_execbench/core/platform/test_dependency_matrix_classification.py \
  tests/sol_execbench/core/platform/test_dependency_matrix_cli.py \
  tests/sol_execbench/core/platform/test_run_docker_dependency_preflight.py \
  tests/sol_execbench/core/reports/test_runtime_evidence_reports.py \
  tests/sol_execbench/core/evidence/test_run_docker_runtime_evidence.py \
  tests/sol_execbench/core/platform/test_rocm_matrix_docs.py -q

bash -n scripts/run_docker.sh
```

## Live ROCm validation

Live ROCm validation is marker-gated. Select `requires_rocm`,
`requires_rdna4`, or `requires_cdna3` only on hosts that provide the matching
device and toolchain. The CDNA3 marker contract lives at
`tests/sol_execbench/core/platform/test_cdna3_hardware_marker.py`; passing it is
not full MI300X hardware-validation evidence, and RDNA4 is not a `gfx94*` validation target.

The current host ROCm 7.2.x environment is the default recorded environment;
default validation does not require host reinstall. ROCm 7.0.x or
ROCm 7.1.x native-host validation requires a matching host. Container rows are
ROCm user-space evidence on the recorded host driver/devices, not native-host
authority.

### Compatibility Matrix Summary

| Target id | Local image tag | Requested ROCm user-space | Evidence summary |
| --- | --- | --- | --- |
| `rocm-7.0.2-ubuntu-24.04-container` | `sol-execbench:rocm-7.0.2-complete` | 7.0.2 | `linear_backward` passed 3/3 workloads with `--record-container-validation`; `container_validated` evidence recorded `CLOCKS_LOCKED=0`. |
| `rocm-7.1.1-ubuntu-24.04-container` | `sol-execbench:rocm-7.1.1-complete` | 7.1.1 | Declared container target with target-specific PyTorch ROCm dependencies and `CLOCKS_LOCKED=1` evidence. |
| `rocm-7.2.0-ubuntu-24.04-container` | `sol-execbench:rocm-7.2-complete` | 7.2.0 | Default target; `linear_backward` passed 3/3 workloads with `CLOCKS_LOCKED=1` evidence. |

Smoke runs selected by `--allow-untested-target-smoke` or
`--allow-mixed-version-dependencies` are diagnostic. A mixed stack reports
`benchmark_allowed=false` and `status=mixed_version`. The target-specific
PyTorch ROCm pins include `torch==2.10.0+rocm7.0` and
`torch==2.11.0+rocm7.2`.

Recorded official wrapper artifacts use
`rocm-7.0.2-linear-wrapper-official.jsonl`,
`rocm-7.0.2-linear-wrapper-official.compatibility.json`,
`rocm-7.2-linear-wrapper-official.jsonl`, and
`rocm-7.2-linear-wrapper-official.compatibility.json`. Diagnostic smoke names
include `rocm-7.2-linear-wrapper-smoke.jsonl` and
`rocm-7.2-linear-wrapper-smoke.compatibility.json`.
