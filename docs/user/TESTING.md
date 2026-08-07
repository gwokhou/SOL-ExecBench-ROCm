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
uv run python scripts/check_production_reachability.py
uv run python scripts/check_current_docs.py
```

These gates scan both `sol_execbench` and `solar`, including package-direction
rules, production reachability, stage entry points and synchronized SOLAR
readability debt.

## Hardware markers

Use the existing markers to select evidence that the host can actually
provide:

- `requires_rocm` / `requires_rocm_gpu` for a visible ROCm device;
- `requires_rocm_dev` and `cpp` for native development headers/toolchain;
- `requires_triton_rocm` for Triton ROCm;
- `requires_rdna4` for the exact validated `gfx1200` target;
- `requires_cdna3` for gfx94x-class evidence;
- `docker_dependency` for the declared container dependency stack.

Examples:

```bash
uv run pytest tests/ -m requires_rocm -n 0
uv run pytest tests/ -m 'requires_rdna4 and cpp' -n 0
```

The remaining multi-GPU isolation debt has an exact entrypoint. On a host
exposing at least two ROCm GPUs (do not constrain it with
`HIP_VISIBLE_DEVICES=0`), run:

```bash
uv run pytest \
  tests/sol_execbench/core/bench/test_timing.py::test_real_multi_gpu_candidate_device_switch_is_rejected \
  -m requires_rocm_gpu -n 0
```

The real HIP translation-unit/dlopen constructor path is a retained regression
entrypoint rather than an active handoff debt. To reproduce that hardware
evidence, run:

```bash
uv run pytest \
  tests/sol_execbench/core/bench/test_reward_hack_native_ctor.py::test_native_constructor_elapsed_time_patch_is_detected \
  -m 'native_extension_serial and requires_rocm and requires_rocm_dev and cpp' \
  -n 0
```

The first test skips precisely when fewer than two ROCm GPUs are visible. The
second requires the ROCm runtime and development toolchain and is excluded from
the default suite by `native_extension_serial`. A skipped node records an unmet
prerequisite; it is not successful hardware evidence.

Do not replace unavailable hardware with a broad `xfail`, mock or skip.
Hardware claims require the exact device/toolchain and should skip only on a
precisely tested missing prerequisite.

GitHub-hosted runners do not provide the recorded RDNA4 GPU. Use the manual
`.github/workflows/rdna4-hardware.yml` only with an administered self-hosted
runner labeled `linux`, `x64`, `rocm`, and `gfx1200`, or run:

```bash
uv run python scripts/internal/rdna4/run_rdna4_validation.py \
  --output-dir out/rdna4-local
uv run python scripts/internal/rdna4/run_rdna4_validation.py \
  --verify out/rdna4-local
```

The same self-hosted workflow also runs the complete SOLAR three-stage corpus
audit and uploads its content-addressed `gfx1200` readiness matrix. For a local
equivalent:

```bash
uv run sol-execbench solar corpus-audit out/solar-corpus-readiness \
  --device cuda:0 \
  --backend torchview_extended_einsum \
  --timeout 14400
```

The exact RX 9060 XT/ROCm/PyTorch/HIP scope and the non-release authority of
these content-addressed bundles are documented in
[RDNA4 Validation Scope](RDNA4-VALIDATION.md).
Run a separate audit with `--backend make_fx_aten` when validating that path;
one matrix never mixes paths or retries failures through the other backend.

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

The configured default container target is ROCm 7.2.x. ROCm 7.0.x, 7.1.x, or
7.2.x native-host validation requires a matching host. Container records are
ROCm user-space evidence on their observed host driver/devices, not native-host
authority.

### Configured container target catalog

| Target id | Local image tag | Requested ROCm user-space | PyTorch ROCm dependency |
| --- | --- | --- | --- |
| `rocm-7.0.2-ubuntu-24.04-container` | `sol-execbench:rocm-7.0.2-complete` | 7.0.2 | `torch==2.10.0+rocm7.0` |
| `rocm-7.1.1-ubuntu-24.04-container` | `sol-execbench:rocm-7.1.1-complete` | 7.1.1 | `torch==2.10.0+rocm7.1` |
| `rocm-7.2.0-ubuntu-24.04-container` | `sol-execbench:rocm-7.2-complete` | 7.2.0 | `torch==2.11.0+rocm7.2` (default) |

Smoke runs selected by `--allow-untested-target-smoke` or
`--allow-mixed-version-dependencies` are diagnostic. A mixed stack reports
`benchmark_allowed=false` and `status=mixed_version`. The target-specific
PyTorch ROCm pins include `torch==2.10.0+rocm7.0` and
`torch==2.11.0+rocm7.2`.

Use `--record-container-validation` to create evidence for a concrete run. The
resulting artifact, rather than this catalog, owns observed versions, clock
state, workload results, and validation status.
