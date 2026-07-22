# Development

## Setup and checks

```bash
uv sync --all-groups
uv run --with ruff ruff check .
uv run --with ruff ruff format .
uv run ty check
uv run python scripts/check_coupling.py
uv run python scripts/check_readability.py
uv run pytest tests/ -q
```

Do not add `-n auto`; the repository default is capped because each ROCm test
worker loads a large runtime. Use `-n 0` for subprocess/native tests when
debugging ordering or resource behavior.

## Current CLI smoke commands

```bash
uv run sol-execbench --format json contract evaluator
uv run sol-execbench --format json contract cli
uv run sol-execbench --format json environment doctor
uv run sol-execbench --format json toolchain list
uv run sol-execbench dataset audit problems/local/AMD_AKA/gfx1200
```

One evaluation:

```bash
uv run sol-execbench evaluate tests/sol_execbench/samples/custom_inputs_matmul \
  --solution tests/sol_execbench/samples/custom_inputs_matmul/solution_python.json \
  --trace-output out/dev.trace.jsonl \
  --unsafe-local-execution
```

## Package ownership

| Path | Responsibility |
| --- | --- |
| `src/sol_execbench/cli` | Click surface and outer-process orchestration |
| `src/sol_execbench/core/bench` | candidate correctness/timing and trusted reference service primitives |
| `src/sol_execbench/core/data` | benchmark and trace schemas |
| `src/sol_execbench/core/dataset` | pinned corpus manifest/materialization |
| `src/sol_execbench/core/evidence` | low-level canonical and derived evidence |
| `src/sol_execbench/core/platform` | runtime/hardware capability evidence |
| `src/sol_execbench/core/reports` | presentation and derived summaries |
| `src/sol_execbench/core/scoring` | formula, aggregation and release authority |
| `src/sol_execbench/core/solar_bridge` | only benchmark-to-SOLAR adapter |
| `src/sol_execbench/driver` | staging and generated process templates |
| `src/solar/graph` | operator graph extraction |
| `src/solar/einsum` | strict executable-einsum conversion |
| `src/solar/verification` | conversion proof |
| `src/solar/analysis` | formal resource/lower-bound analysis |

## Boundary changes

Evaluation process changes normally touch all of:

- `driver/problem_packager.py` for staged assets;
- `driver/templates/evaluation_orchestrator.py` for process topology;
- `driver/templates/reference_worker.py` and `core/bench/reference_service.py`
  for trusted reference work;
- `driver/templates/eval_driver.py` and `core/bench/eval_*` for candidate work;
- `tests/sol_execbench/driver` for generated-runtime behavior.

Reference/candidate IPC uses standard-library `Connection.send_bytes` and
`recv_bytes`, JSON control messages and safetensors tensor payloads. Do not add
pickle or a hand-written framing protocol.

SOLAR public stage changes touch `solar/api.py`, the owning stage module and
`tests/solar/test_api.py`. Preserve exact stage codes and atomic fail-closed
publication.

Scoring changes must keep formula, aggregation and official authority separate.
There are no compatibility modules for retired import paths; update all code,
tests and current docs together.

## Quality policy

`scripts/check_coupling.py` scans both packages, cross-package bridge rules,
known layer inversions and orchestration fan-out. `scripts/check_readability.py`
uses the normal non-increasing baseline plus an exact SOLAR debt inventory.
Never enlarge a baseline to accept new debt.

For new generic helpers, search first with `rg`. Reuse the focused primitives
under `core` and do not add a `core.utils` bucket. Use package resources for
large production HIP/C++ source rather than Python string literals.

## Test scope

- schema/logic: focused CPU unit tests;
- generated driver/IPC: `tests/sol_execbench/driver` and reference protocol
  tests with `-n 0` when needed;
- SOLAR stages: `tests/solar`;
- GPU behavior: existing `requires_rocm`, `requires_rdna4`, `requires_cdna3`,
  `cpp` and related markers.

Do not turn unavailable hardware into a broad skip or `xfail`. Hardware claims
require the exact declared device/toolchain and host execution evidence.

## Build and container

```bash
uv build
./scripts/run_docker.sh --build
```

Downloaded data, generated traces, profiler output, caches and proprietary
kernels stay outside commits.
