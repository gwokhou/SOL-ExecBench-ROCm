# Feature Landscape

**Domain:** Public SOL-ExecBench dataset parity inventory and ROCm execution closure
**Milestone:** v1.11 Paper Dataset Parity Inventory and ROCm Execution Closure
**Researched:** 2026-05-23
**Overall confidence:** HIGH for repo-local feature recommendations and scope boundaries; MEDIUM for exact upstream dataset field availability until the Hugging Face dataset is inspected in the target environment.

## Scope Decision

v1.11 should make the public `nvidia/SOL-ExecBench` dataset concrete in this ROCm port. The milestone is not about proving every paper result on AMD hardware. It is about giving users and maintainers a trustworthy local surface for acquiring or checking the public dataset layout, counting what is present, classifying each problem's ROCm readiness, running small ready subsets through the existing ROCm runner, and producing explicit gap reports.

The existing repository already provides useful execution and evidence primitives:

- `scripts/download_solexecbench.py` downloads Hugging Face configs `L1`, `L2`, `Quant`, and `FlashInfer-Bench` into `data/benchmark/<category>/<problem>/` with `definition.json`, `reference.py`, and `workload.jsonl`.
- `scripts/run_dataset.py` discovers the same four categories, supports category filters, limits, workload truncation, resumable skips, custom solution selection, timing controls, ROCm timing evidence, AMD-native score reports, AMD SOL v2 sidecars, and SOLAR derivation sidecars.
- Existing docs define the canonical artifact split: traces stay canonical, while AMD score reports, SOL bounds, SOLAR derivation evidence, timing evidence, and baseline reports are derived artifacts.
- Existing tests cover derived score report generation, safe sidecar naming, report generation from skipped passing traces, rerun behavior for failed traces, and source-specific timing evidence.

The v1.11 feature line should therefore add a dataset-management layer around those primitives. Users should be able to answer: "Do I have the expected public dataset shape?", "What problems are present and what fields do they use?", "Which problems can run on this ROCm port now?", "What ready subset actually closed through execution?", and "Which gaps remain without overclaiming paper parity?"

## User-Facing Feature Contract

Users should experience v1.11 as a local audit-and-run workflow:

1. Acquire or point at the public dataset.
2. Verify the dataset has the expected category and per-problem layout.
3. Generate an inventory JSON/Markdown report with counts and problem traits.
4. Generate ROCm readiness classifications with actionable reasons.
5. Run ready subsets in small batches through existing dataset execution.
6. Read a parity gap report that separates completed inventory/closure from blocked or explicitly out-of-scope paper parity items.

The preferred interface can be either new focused scripts or options on existing scripts, but the artifacts should be stable and machine-readable. The most natural fit is to keep `download_solexecbench.py` focused on acquisition, keep `run_dataset.py` focused on execution, and add inventory/report helpers that consume the same on-disk dataset layout.

## Operator-Facing Feature Contract

Operators need deterministic artifacts they can archive with a milestone:

- `layout_report.json`: category presence, problem directories found, required file presence, malformed files, and dataset root metadata.
- `paper_inventory.json`: one entry per discovered problem with category, name, counts, dtypes, workload/input traits, reference and solution availability, and static flags.
- `rocm_readiness.json`: one entry per problem with readiness status, blocking reasons, evidence fields, and suggested next action.
- `ready_subset.json`: selected runnable problem list derived from readiness statuses and optional category/limit filters.
- `execution_closure.json`: joined view of ready subset membership and run results from `summary.json`, trace paths, AMD score report paths, sidecar paths, timing evidence paths, and unresolved execution failures.
- `parity_gap_report.md` and/or `.json`: human-readable and machine-readable summary of inventory completion, runnable closure, readiness blockers, and explicit claim guardrails.

These should be derived artifacts under an output directory such as `out/dataset-parity/` by default. They should not modify `definition.json`, `workload.jsonl`, `solution.json`, canonical trace JSON, or public schemas.

## Table Stakes

Features users and maintainers should expect. Missing any of these leaves v1.11 incomplete.

| Feature | Why Expected | Complexity | Dependencies on Existing Capabilities |
|---------|--------------|------------|---------------------------------------|
| Public dataset acquisition command remains usable | The milestone starts with public dataset acquisition/layout checks; users need a supported path from Hugging Face to local files. | Low | Build on `scripts/download_solexecbench.py`, `datasets.load_dataset`, `REPO_ID = "nvidia/SOL-ExecBench"`, and categories `L1`, `L2`, `Quant`, `FlashInfer-Bench`. |
| Acquisition is idempotent and auditable | Dataset downloads may be interrupted or rerun. Operators need to know whether files were written, skipped, or overwritten. | Medium | Add summary output and optional manifest without changing generated problem files. |
| Layout verification for dataset root | Users may already have a dataset checkout. The tool should validate the existing root instead of requiring a fresh download. | Medium | Reuse `run_dataset.discover_problems()` category semantics and required files `definition.json` plus `workload.jsonl`; additionally check `reference.py` because downloader writes it. |
| Category-level completeness checks | The paper public dataset surface is organized by four categories; absence or misspelling should be visible immediately. | Low | Use the shared category set `L1`, `L2`, `Quant`, `FlashInfer-Bench`. Report present, missing, empty, and extra category-like directories. |
| Per-problem required file checks | A problem is not runnable or inventory-complete without schema/workload/reference data. | Low | Check `definition.json`, `workload.jsonl`, `reference.py`; optionally `solution.json` or named solution files when present. |
| JSON parse and schema-load checks | Layout presence is insufficient if files are malformed or incompatible with current Pydantic models. | Medium | Use `Definition` and `Workload` model loading from `sol_execbench.core.data`. Mark model validation failures as `schema/input blocked`. |
| Workload counting | Inventory must state how many workloads each problem contains and whether `--max-workloads` would be needed for small closure runs. | Low | Read `workload.jsonl` line by line using the existing workload contract. |
| Problem count summaries by category | Paper parity inventory requires counts before execution claims. | Low | Count discovered valid/invalid problems per category and total. Do not hard-code a pass/fail expectation unless official dataset metadata is available at runtime. |
| Inventory includes core identifiers | Operators need stable keys for joins across inventory, readiness, execution, and gap reports. | Low | Include `category`, `problem_name`, `definition.name`, dataset-relative path, and workload UUID list or count. |
| Inventory includes dtype coverage | Dtype support is a likely ROCm blocker, especially for quantized and FlashInfer-derived problems. | Medium | Extract dtypes from definition inputs/outputs and workload descriptors where available. Classify unsupported or unknown dtypes separately from runtime failures. |
| Inventory includes forward/backward indicators | The milestone explicitly asks for forward/backward indicators. | Medium | Infer from problem name, description, axes, input/output names, or reference code markers. Mark as `unknown` when evidence is insufficient rather than guessing. |
| Inventory includes custom input usage | Custom input generation is a first-class compatibility risk. | Medium | Read `custom_inputs_entrypoint` from definitions and workload input descriptor types. |
| Inventory includes safetensors usage | FlashInfer traces may require external safetensors assets and environment lookup. | Medium | Detect safetensors/file-backed descriptors and reference `FLASHINFER_TRACE_DIR` expectation from configuration docs. |
| Inventory includes reference availability | `run_dataset.py` defaults to wrapping `definition.reference`; missing reference code currently causes a skip. | Low | Record `definition.reference` present/non-empty and `reference.py` present/non-empty. |
| Inventory includes solution availability | Some closure runs may use problem-local `solution.py` or `solution.json`; reports should distinguish reference-only closure from candidate-solution closure. | Low | Check common names and any requested `--solution-name`; align with `build_solution_for_problem()`. |
| Readiness status per problem | The milestone explicitly asks for per-problem ROCm compatibility classification. | Medium | Create deterministic status enum and evidence reasons from layout, schema, dtype, custom input, safetensors, solution/reference, and prior execution output. |
| Status: `ready` | A problem is structurally valid and eligible for a small ROCm run using reference or selected solution path. | Medium | Requires valid layout, schema-loadable definition/workloads, usable reference or selected solution, and no known static blockers. |
| Status: `schema/input blocked` | Invalid JSON, schema mismatch, unresolved workload descriptors, or missing required files should block before GPU execution. | Medium | Use model validation errors and layout checks. |
| Status: `dtype blocked` | Unsupported dtype requirements should be visible before spending GPU time. | Medium | Compare inventory dtypes against known ROCm/PyTorch/Triton/HIP support policy in this repo. Keep unknown dtype support conservative. |
| Status: `custom-input blocked` | Problems depending on custom input entrypoints or nonstandard generators may require adapter work. | Medium | Detect `custom_inputs_entrypoint` and unsupported workload input descriptor forms. |
| Status: `runtime blocked` | Structurally valid problems that fail under the ROCm runner need a distinct post-execution status. | Medium | Join `summary.json`, per-problem traces, and saved CLI logs from `run_dataset.py`. |
| Status: `unsupported NVIDIA-only path` | CUDA/NVIDIA library assumptions should not be mislabeled as generic runtime failures. | Medium | Static scan of reference/solution metadata and source text for CUDA-only package/API/library markers. Keep this scoped to evidence present in dataset files. |
| Status: `needs hardware evidence` | Some problems may run structurally but require real hardware/profiler evidence before stronger claims. | Medium | Use trace environment, `--timing-evidence-dir`, AMD score report, SOL bound, and SOLAR sidecar presence to distinguish execution from validated evidence. |
| Reason codes and suggested actions | A status without a reason is not actionable for operators. | Medium | Include stable `reason_codes`, message, evidence path, and `next_action` fields. |
| Ready subset selection | Execution closure should operate on classified ready problems, not arbitrary first-N discovery. | Medium | Produce `ready_subset.json` or support passing inventory/readiness output into `run_dataset.py`; retain existing `--category`, `--limit`, and `--max-workloads` semantics. |
| Small-batch execution closure | The milestone asks ready problems to run in small batches, not full paper-scale validation by default. | Low | Use existing `run_dataset.py --category`, `--limit`, `--max-workloads`, `--iterations`, `--warmup-runs`, `--rerun`, and output behavior. |
| Closure captures canonical traces | Execution closure must preserve the existing trace contract. | Low | Use `run_dataset.py` per-problem `traces.json` outputs; do not add inventory fields to trace JSON. |
| Closure captures `summary.json` | Operators need suite-level pass/fail counts. | Low | Existing runner writes `summary.json`; closure should join against it. |
| Closure can emit AMD-native score report | v1.11 should preserve recent scoring and derivation surfaces for ready subset runs. | Low | Use existing `--amd-score-report`, `--scoring-baseline`, `--amd-sol-bound-dir`, and `--solar-derivation`. |
| Closure can emit timing evidence | Some statuses should distinguish plain execution from profiler-backed evidence. | Low | Use existing `--timing-evidence-dir`, `--timing-tool-version`, and `--gpu-architecture`. |
| Resumable closure behavior | Operators should not rerun passed problems accidentally. | Low | Keep existing skip-passed behavior and `--rerun` override; reports should note when traces came from a previous run. |
| Execution failure classification | Runtime failures should be grouped by actionable class, not just displayed as log snippets. | Medium | Parse trace `evaluation.status`, failure logs, and saved CLI logs. Map to runtime, correctness, timeout, environment, missing asset, or unknown failure buckets. |
| Parity gap report | The milestone requires gap reports distinguishing inventory completion from validation and leaderboard claims. | Medium | Generate Markdown for humans and JSON for automation from layout, inventory, readiness, and closure artifacts. |
| Claim guardrails in every report | Reports must not imply full 235-problem validation, upstream SOLAR parity, B200 equivalence, hosted leaderboard readiness, original 124-model regeneration, or CDNA3 hardware validation. | Low | Reuse wording from `docs/original_parity.md`, `docs/analysis.md`, and `.planning/PROJECT.md`. |
| Tests for inventory and readiness | This is a data/reporting milestone; schema regressions will be subtle without tests. | Medium | Add tests next to `tests/sol_execbench/test_run_dataset_amd_score.py` or new focused files using temporary dataset fixtures. |
| Docs with exact workflow | Users need commands for acquisition, inventory, ready subset run, and gap report interpretation. | Low | Update existing analysis/config/parity docs; keep the command examples local and ROCm-scoped. |

## Inventory Fields

The inventory should be boring, stable, and easy to join with run outputs. Recommended per-problem fields:

| Field | Type | Purpose | Complexity |
|-------|------|---------|------------|
| `schema_version` | string | Version the inventory contract. | Low |
| `dataset_root` | string | Absolute or invocation-relative root inspected. | Low |
| `category` | string | One of `L1`, `L2`, `Quant`, `FlashInfer-Bench`. | Low |
| `problem_name` | string | Directory name and display key. | Low |
| `definition_name` | string or null | Name loaded from `definition.json`. | Low |
| `relative_path` | string | Stable dataset-relative path. | Low |
| `required_files` | object | Presence and parse status for `definition.json`, `workload.jsonl`, `reference.py`. | Low |
| `extra_files` | list | Optional solution/assets discovered without treating them as required. | Low |
| `workload_count` | integer | Number of non-empty workload JSONL records. | Low |
| `workload_uuids_sample` | list | Small sample for debugging and report joins. | Low |
| `input_count` / `output_count` | integer | Basic signature complexity. | Low |
| `input_dtypes` / `output_dtypes` | list | Dtype coverage and blockers. | Medium |
| `axis_names` | list | Shape parameter visibility. | Low |
| `forward_backward` | enum | `forward`, `backward`, `both`, or `unknown`. | Medium |
| `custom_inputs_entrypoint` | string or null | Explicit custom input hook from definition. | Low |
| `custom_input_usage` | boolean/object | Whether workload descriptors require nonstandard generation. | Medium |
| `safetensors_usage` | boolean/object | Whether external safetensors/file-backed assets are referenced. | Medium |
| `flashinfer_trace_dependency` | boolean | Whether `FLASHINFER_TRACE_DIR` is likely required. | Medium |
| `reference_available` | object | Definition and file reference availability. | Low |
| `solution_available` | object | Presence of local `solution.json`, `solution.py`, or requested solution name. | Low |
| `static_source_flags` | list | CUDA-only, file asset, dynamic loader, or unsupported API markers found during static inspection. | Medium |
| `inventory_status` | enum | `complete`, `partial`, or `invalid`. | Low |
| `inventory_errors` | list | Parse/schema/layout errors. | Low |

Do not require every inference field to be perfect in the first implementation. It is better to emit `unknown` with evidence than to classify incorrectly.

## Readiness Status Semantics

Recommended status enum, ordered by when the status is usually discovered:

| Status | Meaning | Typical Evidence | Next Action |
|--------|---------|------------------|-------------|
| `ready` | The problem is layout-valid, schema-valid, has a usable reference or selected solution, and has no known static ROCm blocker. | Valid files, valid `Definition` and `Workload` parse, supported dtypes, no blocking custom inputs/assets. | Include in ready subset execution. |
| `schema/input blocked` | Required files are missing, malformed, schema-invalid, or workload input descriptors cannot be interpreted by current code. | JSON decode errors, Pydantic validation errors, empty workload, missing definition/workload/reference. | Fix downloader/layout adapter or add schema/input compatibility. |
| `dtype blocked` | Problem requires dtype behavior not supported or not safely claimable on current ROCm path. | Definition/workload dtypes outside supported policy; quant dtype unsupported in PyTorch ROCm or local runner. | Add dtype handling or mark as intentionally unsupported. |
| `custom-input blocked` | Problem depends on custom input generation not implemented in the ROCm port. | `custom_inputs_entrypoint`, unsupported workload descriptor type, opaque file-backed generator. | Add a safe custom input adapter or skip with explicit report. |
| `runtime blocked` | Static checks passed, but execution failed, timed out, produced no traces, or failed correctness/performance collection. | `summary.json`, trace statuses, CLI logs, timeout logs. | Debug runner/reference/assets/environment for that problem. |
| `unsupported NVIDIA-only path` | Dataset problem or solution path depends on CUDA/NVIDIA-only APIs or libraries that are out of scope for this ROCm-only fork. | Static source flags for CUDA modules, CUDA C++ language metadata, NVIDIA library-only calls. | Record as parity gap; only pursue if a ROCm replacement is milestone-approved. |
| `needs hardware evidence` | The problem can be structurally classified or may execute, but stronger closure requires trace/profiler/AMD score/SOL sidecar evidence on target hardware. | Missing traces, missing timing evidence, missing SOL/SOLAR sidecars, unknown GPU architecture, unvalidated CDNA3 evidence. | Run ready subset with required evidence flags on supported hardware. |

Status precedence should be deterministic. A practical order is:

```text
layout/schema failure
  -> schema/input blocked
unsupported dtype
  -> dtype blocked
unsupported custom input or required external asset not available
  -> custom-input blocked
NVIDIA-only static path
  -> unsupported NVIDIA-only path
no run evidence yet but structurally runnable
  -> ready
execution attempted and failed
  -> runtime blocked
execution passed but required evidence missing for a stronger claim
  -> needs hardware evidence
execution passed with requested evidence
  -> ready with closure evidence, or closed in execution_closure.json
```

`ready` should mean "ready to attempt local ROCm execution", not "paper parity proven."

## Execution Closure Behavior

Execution closure should be a join across readiness and existing runner outputs, not a second benchmark runner.

| Behavior | Expected Result | Complexity | Existing Dependency |
|----------|-----------------|------------|---------------------|
| Select only ready problems | Avoid spending time on known-blocked problems. | Medium | New readiness artifact plus existing `discover_problems()` style ordering. |
| Preserve category and limit filters | Operators can close one category or small sample at a time. | Low | Existing `--category`, `--limit`, and `--max-workloads`. |
| Use existing reference wrapping by default | Public dataset closure should not require candidate solutions. | Low | Existing `build_reference_solution()` and fallback to `definition.reference`. |
| Support named solution closure | Operators can test problem-local solutions when present. | Low | Existing `--solution-name` behavior. |
| Save traces per problem | Canonical execution evidence remains unchanged. | Low | Existing `out/<category>/<problem>/traces.json`. |
| Save suite summary | Closure has pass/fail counts. | Low | Existing `summary.json`. |
| Save CLI failure logs | Runtime blockers have evidence. | Low | Existing `_save_cli_log()`. |
| Generate optional AMD reports and sidecars | Ready subset closure can include recent v1.9/v1.10 derived evidence. | Low | Existing `--amd-score-report`, `--amd-sol-bound-dir`, `--solar-derivation`. |
| Generate optional timing evidence | Operators can distinguish plain execution from profiler-backed closure. | Low | Existing `--timing-evidence-dir`. |
| Join outputs into `execution_closure.json` | Reports can say exactly which ready problems passed, failed, skipped, or lack evidence. | Medium | New report builder consuming runner outputs. |
| Mark closure scope explicitly | Prevent "small subset passed" from becoming "dataset validated." | Low | Gap report claim guardrails. |

Recommended closure statuses:

| Closure Status | Meaning |
|----------------|---------|
| `not_attempted` | Problem was ready but not included in this closure run. |
| `passed` | All emitted traces passed for the selected workloads. |
| `failed` | At least one trace failed correctness/runtime status. |
| `no_trace` | Runner produced no parseable trace. |
| `skipped_existing_pass` | Existing passing traces were reused by default runner behavior. |
| `blocked_before_run` | Readiness prevented inclusion. |
| `evidence_incomplete` | Execution passed but requested AMD score/SOL/SOLAR/timing evidence is missing or unscored. |

## Parity Gap Reports

The parity gap report is the main user-facing artifact for the milestone. It should answer what was completed without drifting into out-of-scope claims.

Required sections:

| Section | Contents | Complexity |
|---------|----------|------------|
| Dataset Layout Summary | Dataset root, categories present/missing, total discovered problems, invalid layouts. | Low |
| Inventory Summary | Counts by category, dtype coverage, forward/backward indicators, custom input usage, safetensors usage, reference availability, solution availability. | Medium |
| ROCm Readiness Summary | Counts by readiness status and top reason codes. | Medium |
| Ready Subset Closure | Which ready problems were run, workload limits, iterations/warmups, pass/fail/no-trace/skipped counts, output paths. | Medium |
| Evidence Summary | Trace count, summary path, AMD score report path, SOL bound sidecar count, SOLAR derivation sidecar count, timing evidence count. | Medium |
| Gap Table | Problem-level blockers with status, reason, evidence path, and next action. | Medium |
| Claim Boundaries | Explicit statements that inventory completion is not full 235-problem validation, not original 124-model regeneration, not upstream SOLAR parity, not hosted leaderboard readiness, and not CDNA3 hardware validation. | Low |

The Markdown report should be concise enough to read in a PR. The JSON report should preserve the full per-problem records for automation.

## Differentiators

Valuable features that improve operator confidence but should not displace table stakes.

| Feature | Value Proposition | Complexity | Notes |
|---------|-------------------|------------|-------|
| Stable reason-code taxonomy | Makes gap trends measurable across runs and releases. | Medium | Prefer codes such as `missing_reference`, `invalid_workload_json`, `unsupported_dtype_float8`, `custom_inputs_entrypoint`, `missing_flashinfer_trace`, `cuda_only_import`, `cli_no_trace`, `trace_runtime_error`. |
| Markdown plus JSON parity reports | Humans can review PR artifacts while automation can diff JSON. | Low | Generate both from the same data model. |
| Diff mode between two inventories | Helps show milestone progress without rerunning everything. | Medium | Compare previous and current `paper_inventory.json` plus `rocm_readiness.json`. |
| Ready subset manifest consumable by runner | Keeps inventory selection and execution in sync. | Medium | Could be a file of dataset-relative problem paths or JSON records. |
| Evidence completeness score | Useful operator signal for "executed only" versus "executed with trace, timing, AMD score, SOL bound, SOLAR derivation." | Medium | Keep it local and descriptive; do not convert to leaderboard score. |
| FlashInfer asset diagnostics | Saves time by checking `FLASHINFER_TRACE_DIR` and safetensors references before execution. | Medium | Especially useful for `FlashInfer-Bench`; should warn, not hide the dependency. |
| Failure clustering | Makes large gap reports readable by grouping repeated schema/runtime/dtype causes. | Medium | Use deterministic reason codes and first log excerpts. |
| Reproducibility header | Captures command, repo version, Python, ROCm/PyTorch environment when available. | Low | Can reuse trace environment when execution occurs. |
| Minimal fixture dataset for tests | Lets CI validate layout/inventory/readiness/report logic without downloading Hugging Face data. | Low | Use temporary `L1`, `Quant`, and `FlashInfer-Bench` fixtures. |

## Anti-Features

Features to explicitly not build for v1.11.

| Anti-Feature | Why Avoid | What to Do Instead |
|--------------|-----------|-------------------|
| Upstream SOLAR full parity claim | User explicitly excluded upstream SOLAR full parity. The ROCm port has AMD-native derived evidence, not NVIDIA paper-equivalent SOLAR. | Report local SOL/SOLAR sidecar coverage and guard claim language. |
| Original 124-model regeneration | User explicitly excluded original model/subgraph regeneration. It is much larger than dataset inventory. | Inventory the public dataset as acquired; do not recreate paper extraction. |
| Full 235-problem validation as a milestone requirement | User excluded full validation unless environment and runtime budget happen to allow it. Making it table stakes would block the milestone. | Support small ready-subset closure and report unattempted scope honestly. |
| Hosted leaderboard | User explicitly excluded hosted leaderboard. | Produce local reports and derived artifacts only. |
| CDNA3 hardware validation claim | User explicitly excluded CDNA3 hardware validation as in-scope. Existing project state says CDNA3 schema/build/docs support exists but full adapted-suite validation remains deferred. | Mark CDNA3 runs as requiring separate evidence; do not present CDNA3 closure as validated unless a future milestone supplies it. |
| CUDA/NVIDIA runtime restoration | This is a ROCm-only fork, and docs classify NVIDIA runtime compatibility as out of scope. | Classify NVIDIA-only paths and suggest ROCm replacements only when scoped. |
| Mutating public dataset files during inventory | Inventory should be an audit, not a migration that changes downloaded data. | Write derived reports under `out/` or another output directory. |
| Adding inventory fields to canonical traces | Trace JSONL is the stable execution contract. | Keep inventory, readiness, closure, and gap data in separate derived artifacts. |
| Treating reference-solution pass as optimized benchmark parity | Running `definition.reference` proves compatibility/execution, not optimized leaderboard performance. | Label closure as reference-based unless a selected solution/baseline is provided. |
| Treating missing data as success | Unknown forward/backward, unknown dtype, missing optional assets, or uninspected custom inputs should not become `ready` by default. | Emit `unknown`, blocker, or `needs hardware evidence` with reason codes. |
| Hard-failing on extra files or categories | Public dataset packaging may include extras. | Report extras as warnings unless they conflict with required layout. |
| Network dependency in normal tests | CI and local tests should not require Hugging Face access. | Use fixture datasets for tests; keep live acquisition as optional/manual. |

## Feature Dependencies

```text
Dataset acquisition or existing dataset root
  -> layout verification
  -> JSON/Pydantic schema validation
  -> paper_inventory.json
  -> rocm_readiness.json
  -> ready_subset.json
  -> run_dataset.py execution for selected ready problems
  -> summary.json + traces.json + optional derived evidence
  -> execution_closure.json
  -> parity_gap_report.md/json

Definition/workload fields
  -> dtype coverage
  -> forward/backward inference
  -> custom input and safetensors detection
  -> readiness status and reason codes

Existing run_dataset outputs
  -> runtime blocker classification
  -> ready subset closure status
  -> AMD score/SOL/SOLAR/timing evidence completeness
```

## MVP Recommendation

Prioritize:

1. Layout verification and inventory generation for the four public categories, including required file checks, schema/workload parse checks, counts, dtypes, reference availability, custom input, safetensors, and forward/backward indicators.
2. Deterministic readiness classification with the exact statuses requested by the milestone: `ready`, `schema/input blocked`, `dtype blocked`, `custom-input blocked`, `runtime blocked`, `unsupported NVIDIA-only path`, and `needs hardware evidence`.
3. Ready subset manifest generation and small-batch closure using existing `run_dataset.py` behavior, preserving traces, `summary.json`, AMD score report, AMD SOL sidecars, SOLAR derivation sidecars, and timing evidence options.
4. Parity gap report generation that joins inventory, readiness, and execution closure, with explicit claim guardrails.
5. Focused tests with temporary dataset fixtures for layout, inventory fields, readiness precedence, closure joins, and report guardrail wording.

Defer:

- Dataset-wide optimized solution performance campaigns.
- Full paper 235-problem validation.
- Original paper model/subgraph regeneration.
- Upstream SOLAR equivalence.
- Hosted leaderboard service.
- CDNA3 hardware validation claims.

## Suggested Artifact Contracts

These contracts are intentionally derived and can evolve behind schema versions.

```json
{
  "schema_version": "sol_execbench.paper_inventory.v1",
  "dataset_root": "data/benchmark",
  "categories": {
    "L1": {"problems": 0, "invalid": 0},
    "L2": {"problems": 0, "invalid": 0},
    "Quant": {"problems": 0, "invalid": 0},
    "FlashInfer-Bench": {"problems": 0, "invalid": 0}
  },
  "problems": []
}
```

```json
{
  "schema_version": "sol_execbench.rocm_readiness.v1",
  "problems": [
    {
      "category": "L1",
      "problem_name": "example",
      "status": "ready",
      "reason_codes": [],
      "next_action": "include_in_ready_subset"
    }
  ]
}
```

```json
{
  "schema_version": "sol_execbench.execution_closure.v1",
  "scope": {
    "categories": ["L1"],
    "limit": 5,
    "max_workloads": 1,
    "solution_mode": "definition_reference"
  },
  "outputs": {
    "summary": "out/summary.json",
    "amd_score_report": null,
    "amd_sol_bound_dir": null,
    "solar_derivation_dir": null,
    "timing_evidence_dir": null
  },
  "problems": []
}
```

## Sources

- `.planning/PROJECT.md` - v1.11 goal, target features, explicit deferrals, current validation boundaries. Confidence: HIGH.
- `scripts/download_solexecbench.py` - current acquisition path, Hugging Face repo/config names, generated layout. Confidence: HIGH for local script behavior.
- `scripts/run_dataset.py` - category discovery, execution controls, resumable behavior, trace/summary outputs, AMD score/SOL/SOLAR/timing evidence options. Confidence: HIGH.
- `docs/analysis.md` - trace workflow, dataset run outputs, timing evidence, AMD-native score semantics, SOL/SOLAR sidecar semantics, claim boundaries. Confidence: HIGH.
- `docs/original_parity.md` - original public surface mapping, ROCm substitutions, out-of-scope NVIDIA/leaderboard/CDNA validation claims. Confidence: HIGH.
- `docs/CONFIGURATION.md` - environment variables including `FLASHINFER_TRACE_DIR`, benchmark config defaults, CLI defaults, ROCm runtime assumptions. Confidence: HIGH.
- `tests/sol_execbench/test_run_dataset_amd_score.py` - existing test coverage for report generation, sidecar safety, skipped/rerun behavior, timing evidence payloads. Confidence: HIGH.
