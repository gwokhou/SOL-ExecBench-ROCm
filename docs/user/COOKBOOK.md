# Cookbook

Commands that execute kernels require a ROCm host or the repository container.
The examples below use checked-in test samples that are part of the current
tree.

## Evaluate one PyTorch solution

```bash
uv run sol-execbench --format json evaluate \
  tests/sol_execbench/samples/custom_inputs_matmul \
  --solution tests/sol_execbench/samples/custom_inputs_matmul/solution_python.json \
  --trace-output out/custom-matmul.trace.jsonl \
  --unsafe-local-execution
```

## Evaluate one Triton solution

```bash
uv run sol-execbench --format json evaluate \
  tests/sol_execbench/samples/nemotron_rms_norm \
  --solution tests/sol_execbench/samples/nemotron_rms_norm/solution_triton.json \
  --trace-output out/nemotron-rmsnorm.trace.jsonl \
  --unsafe-local-execution
```

## Compile and evaluate HIP/C++

```bash
uv run sol-execbench --format json evaluate \
  tests/sol_execbench/samples/rmsnorm \
  --solution tests/sol_execbench/samples/rmsnorm/solution_cuda.json \
  --trace-output out/hip-rmsnorm.trace.jsonl \
  --unsafe-local-execution
```

The current sample solution declares `languages:
["hip_cpp"]` and contains HIP source.

## Collect profile diagnostics

```bash
uv run sol-execbench --format json evaluate \
  tests/sol_execbench/samples/nemotron_rms_norm \
  --solution tests/sol_execbench/samples/nemotron_rms_norm/solution_triton.json \
  --profile rocprofv3 \
  --trace-output out/profiled.trace.jsonl \
  --unsafe-local-execution
```

If `rocprofv3` is unavailable or fails, the profiler sidecar records the reason
and the CLI falls back to normal evaluation. Profile evidence remains
diagnostic.

## Collect native static evidence and decisions

```bash
uv run sol-execbench --format json evaluate \
  tests/sol_execbench/samples/rmsnorm \
  --solution tests/sol_execbench/samples/rmsnorm/solution_cuda.json \
  --static-evidence auto --decision auto \
  --trace-output out/native-evidence.trace.jsonl \
  --unsafe-local-execution
```

The sidecars describe build/static resource evidence. They do not modify the
canonical trace or prove a runtime bottleneck.

## Capture environment evidence

```bash
uv run sol-execbench --format json environment doctor > out/doctor.json

SOLEXECBENCH_ENV_SNAPSHOT=1 \
  uv run sol-execbench --format json evaluate \
    tests/sol_execbench/samples/custom_inputs_matmul \
    --solution tests/sol_execbench/samples/custom_inputs_matmul/solution_python.json \
    --trace-output out/environment.trace.jsonl \
    --unsafe-local-execution
```

## Inspect tool routing

```bash
uv run sol-execbench --format json toolchain route \
  --evidence-level profiling \
  --artifact-type executable_run \
  --gpu-arch gfx1200

uv run sol-execbench --format json toolchain list
```

Routing availability is not correctness or performance evidence.

## Materialize and audit the pinned corpus

```bash
uv run sol-execbench dataset materialize \
  --manifest problems/AMD_AKA/manifest.yaml \
  --device cuda:0 --target-arch gfx1200

uv run sol-execbench dataset audit problems/local/AMD_AKA/gfx1200
```

The command performs static target filtering and a bounded trusted-reference /
harness probe for every workload. It writes all include/exclude decisions and
the filtered coverage report, and does not run candidate solutions.

## Analyze one reference with SOLAR

```bash
uv run sol-execbench solar analyze PROBLEM_DIR \
  --workload WORKLOAD_UUID \
  --output out/solar/WORKLOAD_UUID \
  --orojenesis-home /path/to/orojenesis
```

SOLAR publishes an operator graph, the selected path's `einsum_graph.yaml` or
`aten_graph.yaml`, conversion attestation, formal analysis, and manifest only
when all stages pass. Select `--backend make_fx_aten` explicitly when required;
the default is `torchview_extended_einsum`, and there is no fallback. The CLI accepts only the
publication-eligible capacity-constrained Orojenesis bound; the
reviewed-mapper allowlist contains the reproducibly built trusted digest, and
locally substituted binaries fail closed.

## Audit corpus-wide SOLAR readiness

```bash
uv run sol-execbench solar corpus-audit out/solar-corpus-readiness \
  --device cuda:0 \
  --timeout 14400
```

This command derives the scored denominator from the manifest (currently 163
workloads), then audits extraction, strict conversion, and multi-seed replay
without running Orojenesis. It keeps every workload in the content-addressed
matrix, including failed stages and stable reason codes. Use `--resume` only to
continue an audit whose recorded input identities and artifact hashes still
match.

## Inspect official-score availability

```bash
uv run sol-execbench --format json score status \
  --manifest problems/AMD_AKA/manifest.yaml
```

The checked-in manifest reports the content-addressed publisher policy,
canonical baseline, and required evidence classes. Building raw timing JSON
does not create a release bundle. See the
[release and official-score workflow](RELEASE-SCORING.md) for the complete
publisher workflow.

## Inspect the ownership contract

```bash
uv run sol-execbench --format json contract evaluator
uv run sol-execbench --format json contract cli
```

Use these outputs instead of relying on retired command/import paths.
