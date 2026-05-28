# v1.4 Compatibility Inventory

This inventory defines public contracts that v1.4 must preserve while adapting
engineering practices from the separate hip-execbench project.
It is source-grounded and intentionally separate from the user-facing benchmark
schema.

## Public CLI Contract

Source: `src/sol_execbench/cli/main.py`

- Entry point: `sol_execbench.cli:cli`, exposed as `sol-execbench`.
- Supported invocation shapes:
  - `sol-execbench <problem_dir> --solution <solution-file>`
  - `sol-execbench --definition <definition-file> --workload <workload-file> --solution <solution-file>`
- Public options include:
  - `--definition`
  - `--workload`
  - `--solution`
  - `--config`
  - `--compile-timeout`
  - `--timeout`
  - `-o` / `--output`
  - `--json`
  - `--lock-clocks`
  - `--keep-staging`
  - `--verbose` / `-v`
- Normal benchmark output remains trace JSONL when `--json` or `--output` is
  used. Rich table output remains the human-facing default.
- Current public CLI additions include `doctor`, `toolchain`, `--profile`,
  `--static-evidence`, and contract dispatch.

## Definition Schema Contract

Source: `src/sol_execbench/core/data/definition.py`

- `Definition` remains the public model for benchmark definition files.
- Contract fields include `name`, `op_type`, `axes`,
  `custom_inputs_entrypoint`, `inputs`, `outputs`, `reference`,
  `description`, and `hf_id`.
- Axis variants remain `const`, `var`, and `expr`.
- Tensor dtype strings remain owned by `DType`.
- `reference` must remain valid Python code with a top-level `run` function, and
  reference function parameters must match the ordered input keys.

## Workload Schema Contract

Source: `src/sol_execbench/core/data/workload.py`

- `Workload` remains the public model for benchmark workload JSONL files.
- Contract fields include `axes`, `inputs`, `uuid`, and `tolerance`.
- Input variants remain `random`, `scalar`, `safetensors`, and `custom`.
- Workloads must not mix `custom` inputs with non-custom inputs.
- `ToleranceSpec` remains the public numerical tolerance surface.

## Solution Format Contract

Source: `src/sol_execbench/core/data/solution.py`

- `Solution` remains the public model for solution metadata files.
- `BuildSpec.languages` remains the public language selector.
- ROCm-facing native/library language values include `hip_cpp`, `hipblas`,
  `miopen`, `ck`, and `rocwmma`; Python-side values include `pytorch` and
  `triton`.
- `BuildSpec.target_hardware` remains the public hardware metadata field with
  values such as `gfx1200`, `gfx940`, `gfx941`, `gfx942`, and `LOCAL`.
- `BuildSpec.entry_point` retains the `<file_path>::<function_name>` format.
- `CompileOptions.hip_cflags` remains the HIP compiler flag surface.
- `SourceFile.path` must remain relative and must reject absolute paths or
  parent-directory traversal.

## Trace JSONL Contract

Source: `src/sol_execbench/core/data/trace.py`

- `Trace` remains the canonical machine-readable benchmark output.
- Contract fields include `definition`, `workload`, `solution`, and
  `evaluation`.
- Workload-only traces continue to use `solution=None` and `evaluation=None`.
- Evaluation status values remain:
  - `PASSED`
  - `INVALID_REFERENCE`
  - `INCORRECT_SHAPE`
  - `INCORRECT_NUMERICAL`
  - `INCORRECT_DTYPE`
  - `RUNTIME_ERROR`
  - `COMPILE_ERROR`
  - `TIMEOUT`
  - `REWARD_HACK`
- `PASSED` evaluations require correctness and performance data.
- `INCORRECT_NUMERICAL` evaluations require correctness data and must not carry
  performance data.
- Other failure statuses must not carry correctness or performance data.

## Eval-Driver Semantics Contract

Source: `src/sol_execbench/driver/templates/eval_driver.py`

- The eval driver evaluates the staged solution metadata, definition, and
  workload JSONL files.
- It emits normal workload JSONL `Trace` objects to stdout through the existing
  `_emit` path; static source-review reward-hack traces use a direct stdout
  print path in the same generated driver.
- It preserves reward-hack detection, reference execution, user execution,
  shape/dtype checks, numerical correctness checks, timing, and environment
  capture semantics.
- Additional diagnostics and derived evidence must not corrupt stdout trace
  JSONL or add fields to canonical trace objects.

## hip-execbench Adaptation Boundary

Useful hip-execbench ideas may be adapted only when they preserve the contracts
above:

- Internal profiler-readiness diagnostics may adapt hip-execbench-style source
  patterns when they remain private implementation details.
- Internal stage diagnostics may adapt typed error-surface patterns.
- Derived summaries may adapt pure transformation patterns, but they must remain
  separate from canonical trace JSONL.
- Baseline-relative comparison may adapt comparator-style thresholding, but
  repeated-sample statistical claims require a separate repeated-run trace
  contract.

## Phase 19 Non-Goals

- Historical Phase 19 analysis originally avoided public CLI additions; the
  current CLI now includes public diagnostic and metadata surfaces.
- Scope future compatibility analysis to preserving the current public
  `sol-execbench` options and subcommands.
- Do not add public `sol-execbench` CLI options or subcommands.
- Do not change Pydantic public field names, required fields, or validation
  semantics.
- Do not add fields to trace JSONL.
- Do not replace the eval driver with a hip-execbench pipeline.
- Do not claim CDNA 3 hardware validation.
- Do not introduce the hip-execbench TypeScript/Zod runtime stack.
