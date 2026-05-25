<!-- generated-by: gsd-doc-writer -->
# SOL ExecBench ROCm Port

SOL ExecBench ROCm Port is a ROCm-only Python benchmark package for researchers and developers evaluating GPU kernel solutions on AMD hardware.

## Installation

The project requires Python `>=3.12,<3.14` and uses `uv` for environment management. On Linux, the dependency set resolves PyTorch ROCm 7.1 wheels and `triton-rocm`.

```bash
uv sync --all-groups
```

For GPU evaluation, run on a Linux host with ROCm-capable AMD hardware, ROCm drivers, and access to `/dev/kfd` and `/dev/dri`. The Docker helper can build and enter the repository container:

```bash
./scripts/run_docker.sh --build
```

ROCm library example readiness is documented in [ROCm library notes](docs/rocm_libraries.md).
The supported native library categories include hipBLAS, MIOpen, Composable Kernel,
and rocWMMA. CDNA 3 support is schema/build/docs-ready until real hardware
validation is recorded; CDNA 4 validation is also deferred.

Optional benchmark assets can be downloaded into `data/`:

```bash
./scripts/download_data.sh
```

## Quick Start

1. Install dependencies.

   ```bash
   uv sync --all-groups
   ```

2. Run a local sample problem.

   ```bash
   uv run sol-execbench tests/sol_execbench/samples/rmsnorm \
     --solution tests/sol_execbench/samples/rmsnorm/solution_triton.json
   ```

3. Write trace output when you need machine-readable results.

   ```bash
   uv run sol-execbench tests/sol_execbench/samples/linear_backward \
     --solution tests/sol_execbench/samples/linear_backward/solution_python.json \
     --output out/linear_backward.trace.jsonl
   ```

## Usage Examples

Evaluate a problem directory that contains `definition.json` and `workload.jsonl`:

```bash
uv run sol-execbench examples/triton/rmsnorm \
  --solution examples/triton/rmsnorm/solution_triton.json
```

Evaluate by passing the problem files explicitly:

```bash
uv run sol-execbench \
  --definition examples/pytorch/gemma3_swiglu/definition.json \
  --workload examples/pytorch/gemma3_swiglu/workload.jsonl \
  --solution examples/pytorch/gemma3_swiglu/solution_python.json
```

Run a small dataset batch after downloading benchmark data:

```bash
uv run scripts/run_dataset.py data/SOL-ExecBench/benchmark --limit 5
```

Compare candidate trace output against one or more baseline trace JSONL files:

```bash
uv run sol-execbench-baseline \
  --candidate out/candidate.trace.jsonl \
  --baseline out/baseline.trace.jsonl \
  --format text
```

Print the GPU-free evaluator compatibility contract:

```bash
uv run sol-execbench contract --json
```

## CLI Reference

Primary command:

```bash
uv run sol-execbench <problem_dir> --solution solution.json
```

Equivalent explicit-file form:

```bash
uv run sol-execbench \
  --definition definition.json \
  --workload workload.jsonl \
  --solution solution.json
```

Common options:

| Flag | Description |
| --- | --- |
| `--compile-timeout` | HIP/C++ compilation timeout in seconds. |
| `--timeout` | Evaluation subprocess timeout in seconds. |
| `-o`, `--output` | Write trace JSONL to a file. |
| `--json` | Print trace JSON to stdout. |
| `--lock-clocks` | Require GPU clocks to be locked before benchmarking. |
| `--keep-staging` | Keep the staging directory after evaluation. |
| `-v`, `--verbose` | Show subprocess output. |

The baseline comparison command is exposed separately as `sol-execbench-baseline`.

## Documentation

- [Getting Started](docs/GETTING-STARTED.md): prerequisites, installation, first run, and setup issues.
- [Architecture](docs/ARCHITECTURE.md): system overview, components, data flow, and key abstractions.
- [Development](docs/DEVELOPMENT.md): local development commands, code style, branch conventions, and PR process.
- [Testing](docs/TESTING.md): pytest setup, markers, coverage notes, and CI integration.
- [Configuration](docs/CONFIGURATION.md): environment variables and runtime configuration.
- [Definition schema](docs/definition.md): benchmark problem definitions.
- [Workload schema](docs/workload.md): workload JSONL inputs and tolerances.
- [Solution schema](docs/solution.md): ROCm-supported solution metadata.
- [Trace schema](docs/trace.md): evaluation output format.
- [ROCm notes](docs/rocm.md): host, Docker, and validation notes for ROCm evaluation.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

Apache License 2.0. See [LICENSE](LICENSE) and [docs/compliance.md](docs/compliance.md).
