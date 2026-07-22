# Getting started

## Install

Requirements are Python 3.12 and `uv`. Live evaluation additionally requires a
ROCm-capable AMD GPU, a ROCm PyTorch build and access to `/dev/kfd` and
`/dev/dri`.

```bash
git clone https://github.com/gwokhou/SOL-ExecBench-ROCm.git
cd SOL-ExecBench-ROCm
uv sync --all-groups
```

Check the GPU-free contracts and runtime diagnostics:

```bash
uv run sol-execbench --format json contract evaluator
uv run sol-execbench --format json contract cli
uv run sol-execbench --format json environment doctor
```

On a ROCm host, `torch.version.hip` should be set and
`torch.cuda.is_available()` should be true. PyTorch ROCm intentionally exposes
device APIs through PyTorch's public `torch.cuda` namespace on HIP builds.

## First evaluation

Run the small PyTorch sample:

```bash
mkdir -p out
uv run sol-execbench evaluate tests/sol_execbench/samples/custom_inputs_matmul \
  --solution tests/sol_execbench/samples/custom_inputs_matmul/solution_python.json \
  --trace-output out/first-run.trace.jsonl \
  --unsafe-local-execution
```

Use the hardened container path instead of `--unsafe-local-execution` for
untrusted submissions. Host execution is diagnostic-only.

Each line of the output is one canonical Trace. Important fields are:

- `evaluation.status` for correctness/execution outcome;
- `evaluation.correctness` for numeric error evidence;
- `evaluation.performance.latency_ms` for candidate latency;
- `evaluation.performance.reference_latency_ms` for trusted reference timing;
- `evaluation.performance.speedup_factor`, derived by the outer process;
- `evaluation.environment` for recorded runtime context.

Reference code, expected outputs and reference timing execute in a distinct
trusted worker. Candidate code executes in the generated driver and never
imports the reference implementation.

## Triton and native samples

Triton ROCm:

```bash
uv run sol-execbench evaluate tests/sol_execbench/samples/nemotron_rms_norm \
  --solution tests/sol_execbench/samples/nemotron_rms_norm/solution_triton.json \
  --trace-output out/triton.trace.jsonl \
  --unsafe-local-execution
```

HIP/C++ (requires ROCm development headers/toolchain):

```bash
uv run sol-execbench evaluate tests/sol_execbench/samples/rmsnorm \
  --solution tests/sol_execbench/samples/rmsnorm/solution_cuda.json \
  --trace-output out/hip.trace.jsonl \
  --unsafe-local-execution
```

The sample file `solution_cuda.json` contains a ROCm `hip_cpp`
solution; its validated schema is authoritative.

## Optional evidence

Add profiler diagnostics:

```bash
uv run sol-execbench evaluate tests/sol_execbench/samples/nemotron_rms_norm \
  --solution tests/sol_execbench/samples/nemotron_rms_norm/solution_triton.json \
  --profile rocprofv3 --trace-output out/profiled.trace.jsonl \
  --unsafe-local-execution
```

For native solutions, `--static-evidence auto --decision auto` requests static
artifacts and resource hints. These sidecars are diagnostic and never replace
Trace JSONL or change correctness/scoring authority.

## Pinned corpus

Materialize the exact public selection:

```bash
uv run sol-execbench dataset materialize \
  --manifest problems/AMD_AKA/manifest.yaml \
  --device cuda:0

uv run sol-execbench dataset audit problems/local/AMD_AKA/gfx1200
```

The default output suffix is the exact detected architecture (`gfx942`,
`gfx1150`, or `gfx1200`). Use `--target-arch` to assert the expected target and
`--output` to override the destination. Each workload receives a bounded live
probe; excluded workloads and coverage gaps remain in the materialization
manifest. This revision does not ship a batch runner; choose a materialized
problem and invoke `sol-execbench evaluate --device ...` explicitly.

## SOLAR analysis

Formal analysis is separate from candidate evaluation:

```bash
uv run sol-execbench solar analyze PROBLEM_DIR \
  --workload WORKLOAD_UUID \
  --output out/solar/WORKLOAD_UUID \
  --orojenesis-home /path/to/orojenesis
```

The command reports the exact failed stage when architecture evidence,
extraction, conversion, verification or formal analysis cannot complete. It
publishes no partial output directory on failure.

## Docker

```bash
./scripts/run_docker.sh --build
```

The wrapper uses `docker/rocm-targets.json` and configures ROCm device
passthrough. Run `./scripts/run_docker.sh --help` for current target options.

## Common failures

| Symptom | Check |
| --- | --- |
| `torch.version.hip` is empty | ROCm PyTorch wheel was not installed |
| GPU unavailable | `/dev/kfd`, `/dev/dri`, visibility variables and group permissions |
| native compile fails | ROCm headers, `hipcc`, target architecture and dependencies |
| no trace is parsed | inspect the bounded no-trace diagnostics sidecar |
| official score fails | current manifest intentionally has no published release authority |
| SOLAR fails at architecture | required pinned architecture audit is unavailable |
