# Technology Stack

**Project:** SOL ExecBench ROCm Port
**Milestone:** v1.11 Paper Dataset Parity Inventory and ROCm Execution Closure
**Researched:** 2026-05-23
**Scope:** Public dataset acquisition/layout checks, paper parity inventory, per-problem ROCm compatibility classification, small ready-subset execution closure, JSON/JSONL reports, safetensors/custom-input handling, and dataset runner integration.
**Overall confidence:** HIGH for repository integration points and dependency decisions; MEDIUM for exact public dataset counts until the inventory command is run against the live Hugging Face datasets.

## Recommendation

Do not add a new framework dependency for v1.11. The milestone should harden and extend the existing Python package stack:

| Layer | Keep / Add | Decision |
|-------|------------|----------|
| Dataset access | Keep `datasets>=4.8.2`; optionally use `huggingface_hub` only through the existing `hf` CLI path or as a direct helper if already installed transitively. | `scripts/download_solexecbench.py` already uses `datasets.load_dataset("nvidia/SOL-ExecBench", name=<category>, split="train")`; make this auditable and resumable instead of adding a new acquisition system. |
| Dataset categories | Keep local constants for `L1`, `L2`, `Quant`, `FlashInfer-Bench`. | These are already shared conceptually by the downloader and runner. v1.11 should centralize or mirror them deliberately and validate layout for all four. |
| Schema validation | Keep Pydantic v2 models: `Definition`, `Workload`, `Trace`. | Inventory and classification should instantiate existing models and record validation failures, not create loose dict-only validators. |
| JSON/JSONL reports | Keep stdlib `json` plus line-oriented JSONL helpers. | Reports are small metadata artifacts. `pandas`, `pyarrow`, `jsonlines`, SQLite, DuckDB, or dataframe libraries are unnecessary. |
| Safetensors handling | Keep `safetensors>=0.7.0` and existing `load_safetensors()` behavior. | The evaluator already resolves and validates safetensors by path, tensor key, shape, and dtype. v1.11 needs inventory checks for missing blobs and unsupported layouts, not another loader. |
| Custom inputs | Keep `custom_inputs_entrypoint` on `Definition` and `CustomInput` on `Workload`. | Classification should detect custom-input usage and whether the entrypoint exists/validates; execution should continue through the existing eval-driver path. |
| Runner integration | Extend `scripts/run_dataset.py`; do not add a second runner. | The runner already discovers category/problem directories, truncates workloads, emits traces and summary JSON, and can generate AMD-native score reports, AMD SOL v2 sidecars, timing evidence, and SOLAR derivation sidecars. |
| CLI output | Use existing Click/Rich only if adding package CLI commands; otherwise keep scripts with argparse. | v1.11 can be implemented as scripts first. Avoid broad CLI surface changes unless the phase explicitly needs `sol-execbench inventory`. |

The strongest implementation shape is: **download/layout check -> inventory JSONL -> compatibility classification JSONL -> ready subset manifest -> existing dataset runner -> gap report JSON/Markdown**. Each step should be deterministic and file-based so results are auditable without changing canonical trace schemas.

## Recommended Stack

### Core Framework

| Technology | Version | Purpose | Why |
|------------|---------|---------|-----|
| Python | `>=3.12,<3.14` | Inventory scripts, downloader hardening, report generation, runner integration. | Already required by `pyproject.toml`; sufficient for typed dataclasses, pathlib, argparse, and stdlib JSON. |
| Pydantic | `>=2.12.5` | Validate `definition.json`, `workload.jsonl`, and generated report models if new internal models are useful. | Existing public schemas already use Pydantic v2. Use those contracts directly for parity classification. |
| Hugging Face Datasets | `>=4.8.2` | Load `nvidia/SOL-ExecBench` public dataset configs. | Existing downloader uses `datasets.load_dataset` with category configs and `split="train"`. Official docs support `load_dataset(path, name=..., split=...)` for Hub datasets. |
| Safetensors | `>=0.7.0` | Load referenced safetensors blobs at execution time and inspect availability for inventory. | Already in runtime dependencies; existing evaluator uses `safetensors.torch.load_file`. |
| PyTorch ROCm | `torch==2.10.0+rocm7.1` on Linux/Windows | Execute references/solutions and validate dtype/runtime readiness. | Existing ROCm execution baseline. Do not change for inventory work. |
| Triton ROCm | `triton-rocm==3.6.0` on Linux | Existing solution category support. | Keep as-is; v1.11 classifies readiness, it does not expand compiler coverage. |

### Reporting And Files

| Tool | Version | Purpose | Why |
|------|---------|---------|-----|
| stdlib `json` | Python bundled | Write summary JSON, inventory JSONL, classification JSONL, ready subset manifests, and gap reports. | Canonical traces and existing summaries are JSON. This keeps reports dependency-free and easy to diff. |
| `pathlib` | Python bundled | Dataset layout traversal and relative path validation. | Existing scripts already use `Path`; needed for safe local layout checks. |
| `argparse` | Python bundled | New script flags for inventory/classification, or extensions to existing scripts. | Existing dataset scripts use argparse. No need to introduce Click unless exposing a package console command. |
| `hashlib` / stable sorting | Python bundled | Stable identifiers and deterministic report ordering. | Existing runner already uses safe sidecar names; inventory reports should be reproducible. |
| Rich | `>=13.0` | Optional console tables if wired through package CLI. | Already a dependency, but machine-readable JSON/JSONL should remain primary. |

### Existing ROCm Execution Stack

| Technology | Version | Purpose | Why |
|------------|---------|---------|-----|
| `sol-execbench` CLI | project local | Single-problem execution from definition/workload/solution. | `scripts/run_dataset.py` already shells to this CLI and parses strict JSON lines. Keep this as the execution boundary. |
| `scripts/run_dataset.py` | project local | Ready-subset execution closure. | Add manifest/category filters and report stitching here rather than creating a parallel runner. |
| `BenchmarkConfig` | project local | Small-batch execution settings: `warmup_runs`, `iterations`, `lock_clocks`, `seed`. | Existing config file path already supports reproducible quick runs. |
| `rocprofv3` evidence helpers | ROCm toolchain | Optional timing evidence for ready problems. | Already integrated through `--timing-evidence-dir`; v1.11 should preserve it as optional evidence, not make it mandatory for inventory. |

## Recommended Local Additions

### New Or Extended Scripts

| File | Add / Extend | Purpose |
|------|--------------|---------|
| `scripts/download_solexecbench.py` | Extend | Add explicit output root flag, category selection, revision/report metadata, layout verification after download, and clear handling for already-downloaded categories. Keep `datasets.load_dataset`. |
| `scripts/download_data.sh` | Extend lightly | Keep as convenience wrapper. Avoid making `hf` CLI mandatory for the SOL dataset path; reserve it for FlashInfer trace asset download unless replaced by a Python helper. |
| `scripts/run_dataset.py` | Extend | Accept a ready-subset manifest or classification filter, preserve existing `--limit` and `--max-workloads`, and include inventory/classification references in summary JSON. |
| `scripts/inventory_solexecbench.py` or `src/sol_execbench/core/dataset_inventory.py` | Add | Generate machine-readable inventory and compatibility classification without executing GPU workloads. Prefer a core module plus thin script if tests need direct imports. |
| `docs/` parity page | Add or update | Document dataset root layout, acquisition commands, inventory outputs, readiness classes, and claim guardrails. |

### New Internal Models

Use Pydantic only if the model definitions stay small and directly serve validation. Otherwise dataclasses plus `dict` serialization are enough.

| Model | Suggested Fields | Purpose |
|-------|------------------|---------|
| `DatasetProblemInventory` | `category`, `problem`, `definition_path`, `workload_path`, `workload_count`, `input_types`, `dtypes`, `has_reference`, `has_solution_file`, `has_custom_inputs`, `has_safetensors`, `safetensors_refs`, `forward_backward_hint`, `validation_errors` | One JSONL row per problem. |
| `RocmCompatibilityClassification` | `category`, `problem`, `status`, `reasons`, `blocking_inputs`, `blocking_dtypes`, `missing_safetensors`, `custom_input_status`, `runtime_probe_status`, `needs_hardware_evidence` | One JSONL row per problem; source of ready subset manifests and gap reports. |
| `DatasetLayoutReport` | `repo_id`, `categories_expected`, `categories_found`, `problem_counts`, `missing_files`, `unexpected_files`, `flashinfer_trace_root`, `generated_at` | Dataset-level acquisition/layout status. |
| `ReadySubsetManifest` | `problems`, `selection_policy`, `max_workloads`, `generated_from_inventory`, `generated_from_classification` | Input to `scripts/run_dataset.py` for small closure runs. |

Keep these as **derived reports**, not public benchmark schemas. They should not mutate `definition.json`, `workload.jsonl`, `solution.json`, or canonical trace JSONL.

## Classification Logic

The stack should support a deterministic classifier that can run without ROCm hardware, plus an optional runtime probe for ready candidates.

| Status | Detection Inputs | Notes |
|--------|------------------|-------|
| `ready` | Valid `Definition`; all workloads parse; dtypes map through existing dtype helpers; no missing safetensors for selected roots; custom inputs have a valid entrypoint; no known NVIDIA-only import/path markers. | Eligible for small execution closure. |
| `schema_input_blocked` | Pydantic validation errors, missing `definition.json`, missing `workload.jsonl`, bad JSON/JSONL, input names mismatch, custom mixed with non-custom. | Use existing model validation and record exact errors. |
| `dtype_blocked` | Definition/workload requires dtype that current ROCm/PyTorch path cannot generate or execute reliably. | Existing dtype enum includes float8/float4 values; classification must distinguish schema support from runtime support. |
| `custom_input_blocked` | Workload uses `custom`, but `Definition.custom_inputs_entrypoint` is missing, invalid, not defined, or fails a CPU/lightweight syntax check. | Do not execute arbitrary custom input code during pure inventory unless explicitly running a probe. |
| `safetensors_blocked` | Workload references safetensors paths or keys not available under dataset/staging/`FLASHINFER_TRACE_DIR` roots. | Reuse `_resolve_blob_path` semantics for path matching; inspect keys only when files are present. |
| `runtime_blocked` | Small probe via existing runner fails with runtime/compile/timeout errors. | Keep separate from static inventory. |
| `unsupported_nvidia_only_path` | Reference/solution uses CUDA/NVIDIA-only APIs that the ROCm port intentionally does not support. | Static source scan plus execution failure details; avoid overclaiming from source scan alone. |
| `needs_hardware_evidence` | Static checks pass but no ROCm execution trace exists for the problem/category/hardware target. | Important claim guardrail: inventory completion is not validation. |

## Integration Points

| File | v1.11 Work |
|------|------------|
| `pyproject.toml` | Prefer no dependency changes. If a direct FlashInfer trace Python downloader replaces the `hf` CLI, add `huggingface-hub` explicitly with a narrow lower bound instead of relying on transitive installation. |
| `scripts/download_solexecbench.py` | Add flags: `--repo-id`, `--output-dir`, `--category`, `--revision`, `--layout-report`, `--overwrite/--skip-existing`. Preserve defaults for `nvidia/SOL-ExecBench` and the four public categories. |
| `scripts/download_data.sh` | Keep wrapper behavior, but document that the Python downloader handles SOL-ExecBench while `hf download flashinfer-ai/flashinfer-trace --revision 1.0` handles optional trace blobs. |
| `src/sol_execbench/core/data/definition.py` | Reuse `Definition` validation for inventory. Do not loosen public schema unless a live dataset incompatibility is proven and documented. |
| `src/sol_execbench/core/data/workload.py` | Reuse `Workload` and input variants: `random`, `scalar`, `safetensors`, `custom`. Classification should summarize these variants by problem. |
| `src/sol_execbench/core/bench/io.py` | Reuse safetensors path resolution and dtype/shape checks. Consider extracting `_resolve_blob_path` if inventory needs it without importing private helpers. |
| `src/sol_execbench/driver/templates/eval_driver.py` | No major changes expected. Existing safetensors roots are staging dir plus `FLASHINFER_TRACE_DIR`; custom inputs already call `definition.custom_inputs_entrypoint`. |
| `scripts/run_dataset.py` | Add manifest-driven problem selection and preserve sidecar flags: `--amd-score-report`, `--amd-sol-bound-dir`, `--solar-derivation`, `--timing-evidence-dir`. Summary JSON should include skipped/classification metadata when run from a manifest. |
| `docs/definition.md` / `docs/workload.md` | Reference existing schema behavior in gap reports. Only update if inventory discovers paper dataset fields not reflected in docs. |
| `docs/CONFIGURATION.md` | Update if v1.11 adds dataset root variables or changes FlashInfer trace acquisition. Prefer explicit CLI flags over new environment variables. |

## What Not To Add

| Do Not Add | Reason |
|------------|--------|
| `pandas`, `polars`, `duckdb`, `sqlite-utils`, or `pyarrow` | Inventory and gap reports are modest JSON/JSONL artifacts; dataframe/storage engines add unnecessary dependency and packaging surface. |
| `jsonlines` package | JSONL writing is simple and already handled line-by-line elsewhere. stdlib `json` is enough. |
| New public trace fields | Canonical trace JSONL must remain stable. Put parity inventory, readiness, and gap data in derived reports. |
| A new execution runner | `scripts/run_dataset.py` already has the correct execution boundary and sidecar integration. Extend it. |
| A hosted leaderboard, API server, database, or dashboard | Explicitly out of milestone scope and would confuse inventory/closure with public leaderboard equivalence. |
| New ROCm compiler/library dependencies | v1.11 classifies and closes ready subsets; it should not expand HIP/Triton/library support. |
| `transformers`, model-loading libraries, or paper extraction tooling | The original 124-model extraction pipeline is explicitly deferred. Use the public dataset artifacts, not model reconstruction. |
| Automatic execution of arbitrary custom input code during static inventory | Custom input code belongs in the existing isolated eval path or an explicit probe mode. Pure inventory should inspect schema/source structure without side effects. |
| Treating Hugging Face CLI as an implicit package dependency | If `hf` remains required for FlashInfer trace download, document it. If Python code imports `huggingface_hub`, add it explicitly rather than relying on transitive dependencies. |
| v1.10 SOLAR derivation rework | Existing sidecars are context only. v1.11 should route ready runs through existing sidecar flags, not reopen derivation architecture. |

## Installation

No required dependency additions are recommended.

Existing setup remains:

```bash
uv sync --all-groups
```

Optional dataset asset setup remains script-driven:

```bash
uv run python scripts/download_solexecbench.py

# Optional FlashInfer trace blobs when safetensors workloads reference them.
pip install "huggingface-hub[cli]"
hf download flashinfer-ai/flashinfer-trace --repo-type=dataset --revision 1.0 --local-dir data/flashinfer-trace
```

If the phase replaces the shell-level `hf download` with Python code, prefer this single dependency addition:

```toml
dependencies = [
    "huggingface-hub>=1.0",
]
```

Only add it if a Python API is actually imported by repository code.

## Suggested Validation

Static and unit validation:

```bash
uv run pytest \
  tests/sol_execbench/core/bench/test_io.py \
  tests/sol_execbench/test_run_dataset_amd_score.py \
  tests/sol_execbench/test_public_contract_guardrails.py

uv run --with ruff ruff check \
  scripts/download_solexecbench.py \
  scripts/run_dataset.py \
  src/sol_execbench/core/data \
  src/sol_execbench/core/bench/io.py
```

Milestone-specific validation should add focused tests for:

- Downloader category selection and layout reports.
- Inventory JSONL generation from fixture categories.
- Classification statuses for schema errors, missing safetensors, custom input blockers, dtype blockers, and ready problems.
- Manifest-driven `scripts/run_dataset.py` execution with `--max-workloads 1`.
- Gap report guardrails that separate "inventory complete" from "235-problem ROCm validated".

Small ready-subset smoke command after inventory exists:

```bash
uv run scripts/run_dataset.py data/benchmark \
  --category L1 \
  --limit 5 \
  --max-workloads 1 \
  --iterations 1 \
  --warmup-runs 0 \
  --amd-score-report out/v1_11_ready_subset/amd-score.json \
  --amd-sol-bound-dir out/v1_11_ready_subset/amd-sol-v2 \
  --solar-derivation out/v1_11_ready_subset/solar-derivation
```

## Sources

- Repository: `.planning/PROJECT.md` - v1.11 milestone scope, target features, and explicit deferrals.
- Repository: `pyproject.toml` - Python range, ROCm PyTorch/Triton pins, `datasets`, `safetensors`, Pydantic, Click, and Rich dependencies.
- Repository: `scripts/download_solexecbench.py` - current `nvidia/SOL-ExecBench` acquisition path and category list.
- Repository: `scripts/download_data.sh` - current wrapper and `flashinfer-ai/flashinfer-trace` CLI download.
- Repository: `scripts/run_dataset.py` - existing dataset discovery, small batch controls, trace summaries, AMD-native score reports, AMD SOL v2 sidecars, SOLAR derivation sidecars, and timing evidence integration.
- Repository: `src/sol_execbench/core/data/definition.py` - `Definition` schema, dtype enum, reference validation, and custom input entrypoint validation.
- Repository: `src/sol_execbench/core/data/workload.py` - `Workload` schema and input variants: random, scalar, safetensors, custom.
- Repository: `src/sol_execbench/core/data/trace.py` - canonical trace/evaluation schema and status vocabulary.
- Repository: `src/sol_execbench/core/bench/io.py` - safetensors loading, path resolution, dtype/shape checks, custom input generation, and scalar/random input handling.
- Repository: `src/sol_execbench/driver/templates/eval_driver.py` - execution-time safetensors roots and custom input invocation.
- Repository docs: `docs/definition.md`, `docs/workload.md`, `docs/CONFIGURATION.md`, `docs/GETTING-STARTED.md` - public schema and dataset setup behavior.
- Official docs: https://huggingface.co/docs/datasets/loading - `load_dataset` supports loading Hub datasets with named configs and splits. Confidence: HIGH.
- Official docs: https://huggingface.co/docs/safetensors/index - safetensors Torch loading is the supported tensor-file path. Confidence: HIGH.
- Official docs: https://docs.python.org/3/library/json.html - stdlib JSON serialization is sufficient for strict JSON and JSONL report writing. Confidence: HIGH.
