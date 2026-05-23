# Project Research Summary

**Project:** SOL ExecBench ROCm Port
**Domain:** v1.11 public paper dataset parity inventory and ROCm execution closure
**Milestone:** v1.11 Paper Dataset Parity Inventory and ROCm Execution Closure
**Researched:** 2026-05-23
**Confidence:** HIGH for repo-local stack, architecture, and scope; MEDIUM for exact upstream dataset counts until the live Hugging Face revision is inventoried and pinned.

## Executive Summary

v1.11 is a dataset-management and evidence-closure milestone for the ROCm-only SOL ExecBench port. The product surface is not a new benchmark runner, leaderboard, or derivation engine; it is an auditable local workflow that acquires or verifies the public `nvidia/SOL-ExecBench` dataset categories, inventories every problem, classifies ROCm readiness, runs bounded ready subsets through the existing ROCm runner, and reports remaining parity gaps without overclaiming.

The recommended approach is to add sidecar-only dataset reports around the existing downloader, Pydantic schemas, `sol-execbench` execution path, and `scripts/run_dataset.py`. Inventory, readiness, closure, and gap data should live in versioned JSON/Markdown artifacts, while canonical `definition.json`, `workload.jsonl`, solution schemas, trace JSONL, AMD SOL v2 sidecars, SOLAR derivation sidecars, and primary CLI behavior remain stable. v1.10 SOLAR derivation is existing context only: v1.11 may route ready subset runs through those existing sidecar flags, but it should not reopen SOLAR derivation architecture.

The main risk is confusing "dataset inventory complete" or "small ready subset passed" with full paper parity. Mitigate this with deterministic readiness statuses, explicit denominators for discovered, blocked, skipped, attempted, passed, and failed problems, source/revision metadata, bounded run profiles, and claim guardrails stating that v1.11 is not full 235-problem ROCm validation, not upstream SOLAR equivalence, not leaderboard equivalence, not original 124-model extraction, and not CDNA3/CDNA4/NVFP4/MXFP4 hardware validation.

## Key Findings

### Recommended Stack

v1.11 should avoid new framework dependencies. The existing Python 3.12+ package, Pydantic v2 schemas, Hugging Face Datasets downloader, safetensors loader, PyTorch ROCm/Triton ROCm execution stack, stdlib JSON/JSONL reporting, and `scripts/run_dataset.py` provide the right base. Add focused core modules and thin scripts rather than dataframe engines, databases, hosted services, or a second runner.

**Core technologies:**
- Python `>=3.12,<3.14`: inventory scripts, report generation, deterministic file traversal, and testable core helpers.
- Pydantic v2: load and validate existing `Definition` and `Workload` contracts for schema/readiness checks.
- `datasets>=4.8.2`: continue loading `nvidia/SOL-ExecBench` configs `L1`, `L2`, `Quant`, and `FlashInfer-Bench`.
- `safetensors>=0.7.0`: inspect and validate referenced tensor blobs without replacing existing execution-time loading behavior.
- PyTorch ROCm, Triton ROCm, HIP/C++, ROCm libraries: keep existing execution support; classify readiness rather than expanding compiler/library scope.
- stdlib `json`, `pathlib`, `argparse`, stable sorting, and hashing: dependency-free JSON/JSONL/Markdown artifacts that are easy to diff and archive.

**Stack additions:**
- Add `src/sol_execbench/core/dataset/` for constants, discovery, inventory, readiness, and report assembly.
- Add `scripts/inventory_solexecbench.py` as a thin CLI for layout checks, inventory, readiness, and gap reports.
- Extend `scripts/download_solexecbench.py` only for idempotent acquisition metadata, category selection, revision/output-root handling, and layout verification.
- Extend `scripts/run_dataset.py` only for optional manifest/readiness filtering and execution-closure reporting, preserving current defaults.

### Expected Features

**Must have (table stakes):**
- Public dataset acquisition/layout contract for `L1`, `L2`, `Quant`, and `FlashInfer-Bench`.
- Idempotent, auditable acquisition or local-root verification with repo/config/split/revision/count metadata.
- Per-category and per-problem layout checks for `definition.json`, `workload.jsonl`, and `reference.py`, with optional solution/file asset detection.
- JSON/Pydantic parse validation for definitions and every workload line before GPU execution.
- Machine-readable paper inventory with problem counts, workload counts, dtypes, input kinds, forward/backward indicators, custom input usage, safetensors usage, reference availability, solution availability, and source/static flags.
- Deterministic ROCm readiness classification: `ready`, `schema_input_blocked`, `dtype_blocked`, `custom_input_blocked`, `runtime_blocked`, `unsupported_nvidia_only_path`, and `needs_hardware_evidence`.
- Stable reason codes, evidence paths, and suggested next actions for every blocked or deferred problem.
- Ready subset manifest generation and bounded execution through existing `scripts/run_dataset.py`.
- Execution closure report joining readiness to canonical traces, `summary.json`, CLI logs, AMD-native score reports, AMD SOL v2 sidecars, SOLAR derivation sidecars, and optional timing evidence.
- Parity gap report in Markdown and JSON with explicit claim guardrails and complete denominators.
- CPU-only fixture tests for layout, inventory fields, readiness precedence, sidecar path safety, closure joins, and report wording.

**Should have (differentiators):**
- Diff mode between inventories/readiness reports to show milestone progress.
- Evidence completeness scoring that distinguishes plain execution from trace, timing, AMD score, AMD SOL, and SOLAR derivation evidence.
- FlashInfer asset diagnostics for `FLASHINFER_TRACE_DIR` and safetensors path/key/shape/dtype checks.
- Failure clustering by stable reason code.
- Reproducibility headers with command, git commit, dataset manifest checksum, Python/ROCm/PyTorch metadata where available.

**Defer / out of scope:**
- Full 235-problem real-hardware validation unless separately authorized and resourced.
- Original 124-model / 7,400-subgraph extraction and curation pipeline.
- Upstream NVlabs/SOLAR equivalence or NVIDIA/B200/leaderboard-equivalent claims.
- Hosted leaderboard, API service, database, dashboard, or remote submission system.
- CDNA3/MI300X, CDNA4, NVFP4, or MXFP4 hardware validation claims.
- CUDA/NVIDIA runtime restoration, CUDA C++/cuBLAS/cuDNN/CUTLASS/CuTe/cuTile support, or new ROCm compiler/library expansions.
- Mutating public dataset files, canonical traces, public schemas, or primary CLI defaults.
- Treating reference-wrapper execution as optimized candidate-solution benchmark parity.

### Architecture Approach

The architecture should be sidecar-only and layered. Pure core helpers inspect dataset files, validate schemas, derive inventory features, classify readiness, and assemble reports. Scripts parse arguments and write artifacts. Execution closure reuses the existing dataset runner and `sol-execbench` subprocess boundary; it should not bypass the primary evaluation path or embed parity state into traces.

**Major components:**
1. `core/dataset/constants.py` - public category set, report schema versions, expected layout constants.
2. `core/dataset/discovery.py` - shared dataset/problem discovery compatible with existing `run_dataset.py` behavior.
3. `core/dataset/inventory.py` - file presence checks, Pydantic validation, workload counting, dtype/input/custom/safetensors/reference/solution feature extraction.
4. `core/dataset/readiness.py` - deterministic status and reason-code classification from inventory plus optional observed execution failures.
5. `core/dataset/reports.py` - acquisition manifest, paper inventory, readiness, execution closure, parity gap report, and Markdown rendering.
6. `scripts/inventory_solexecbench.py` - thin CLI for offline inventory/readiness/gap generation.
7. `scripts/download_solexecbench.py` - existing downloader plus manifest/layout verification, not execution.
8. `scripts/run_dataset.py` - existing execution loop plus optional ready-subset selection and closure report output.

**Primary data flow:**

```text
Dataset acquisition or existing dataset root
  -> layout verification
  -> Definition/Workload validation
  -> paper_inventory.json
  -> rocm_readiness.json
  -> ready_subset.json
  -> existing run_dataset.py execution
  -> traces.json + summary.json + optional AMD/SOL/SOLAR/timing sidecars
  -> execution_closure.json
  -> parity_gap_report.md/json
```

### Critical Pitfalls

1. **Assuming download equals paper dataset parity** - record repo/config/split/revision/counts/checksums/blob availability and separate acquisition status from readiness and execution status.
2. **Letting stale docs override current contracts** - treat existing Pydantic models, trace schemas, CLI behavior, and guardrail tests as source of truth; put v1.11 data in sidecars only.
3. **Collapsing schema/input drift into runtime failure** - preflight every definition and workload with explicit `schema_input_blocked`, `dtype_blocked`, `custom_input_blocked`, `asset_blocked`, and `unsupported_nvidia_only_path` reasons before GPU execution.
4. **Mishandling custom inputs and safetensors** - detect custom entrypoints and safetensors references, verify assets and tensor keys/shapes/dtypes where possible, and never substitute random data for parity runs.
5. **Overstating low-precision dtype support** - distinguish schema-known, input-generation-supported, reference-execution-supported, candidate-execution-supported, and hardware-validated states, especially for FP8/FP4/NVFP4/MXFP4/Quant problems.
6. **Treating skipped or reused traces as invisible success** - include `not_attempted`, `skipped_existing_pass`, missing reference/solution, no trace, failed, and evidence-incomplete states in closure denominators.
7. **Blurring canonical traces with derived artifacts** - keep inventory, readiness, closure, AMD score, AMD SOL, SOLAR, and timing evidence as separate reports or sidecars with stable references.
8. **Letting parity wording overclaim** - add v1.11-specific no-claim wording/tests for full validation, upstream SOLAR, leaderboard equivalence, CDNA3/CDNA4 validation, and low-precision hardware validation.

## Implications for Roadmap

Based on research, suggested phase structure:

### Phase 1: Dataset Contract And Acquisition Metadata
**Rationale:** All later counts and claims depend on a stable local dataset root, category set, source revision, and layout contract.
**Delivers:** Shared category constants, discovery helpers, canonical v1.11 dataset root behavior, acquisition/local-layout manifest, downloader idempotency, category filters, and layout verification.
**Addresses:** Public acquisition command, category completeness, required file checks, source metadata, out-of-git dataset handling.
**Avoids:** Download-success-as-parity, ambiguous `data/benchmark` versus `data/SOL-ExecBench/benchmark`, licensing/redistribution mistakes, stale docs driving contracts.

### Phase 2: Paper Inventory And ROCm Readiness Classification
**Rationale:** The full public surface must be counted and classified before any execution attempt can be interpreted.
**Delivers:** `paper_inventory.json`, `rocm_readiness.json`, reason-code taxonomy, workload-level records, dtype/input/custom/safetensors/reference/solution fields, forward/backward indicators, static NVIDIA-only flags, fixture tests.
**Uses:** Pydantic `Definition`/`Workload`, safetensors helpers, stdlib JSON/JSONL, pure `core/dataset` modules.
**Implements:** Inventory and readiness architecture components.
**Avoids:** Schema drift hidden as runtime failure, custom/safetensors random substitution, dtype overclaims, workload-granularity loss.

### Phase 3: Ready Subset Selection And Bounded Execution Closure
**Rationale:** Closure should run only known-eligible problems under an explicit budget and through the existing benchmark path.
**Delivers:** `ready_subset.json`, optional readiness filtering in `scripts/run_dataset.py`, small-batch run profiles, `execution_closure.json`, trace/summary/CLI-log joins, optional AMD score, AMD SOL v2, SOLAR derivation, and timing-evidence references.
**Uses:** Existing `sol-execbench` CLI, `scripts/run_dataset.py`, existing sidecar flags.
**Avoids:** Paper-scale runtime blowups, blind execution of blocked problems, skipped-results disappearing, reference execution being mislabeled as optimized solution validation.

### Phase 4: Parity Gap Reporting And Evidence Review
**Rationale:** Requirements and roadmap consumers need a single human and machine-readable ledger of what exists, what ran, what passed, and what remains blocked.
**Delivers:** `parity_gap_report.md`, `parity_gap_report.json`, grouped blockers, evidence completeness summaries, per-category counts, per-reason counts, next actions, and artifact references.
**Implements:** Report aggregation and Markdown/JSON contracts.
**Avoids:** Incomplete denominators, weak failure evidence, sidecar path collisions, stale trace reuse without provenance.

### Phase 5: Claim Guardrails, Docs, And Release Closure
**Rationale:** v1.11 introduces "paper dataset parity" and "execution closure" language that can easily be misread as full validation.
**Delivers:** Updated docs with exact workflow, report interpretation, out-of-scope boundaries, v1.11 no-claim tests, public contract guardrails, and examples for acquisition, inventory, readiness, ready subset execution, and gap reports.
**Addresses:** Claim boundaries, user workflow, downstream roadmap clarity.
**Avoids:** Full 235-problem validation claims, upstream SOLAR equivalence claims, leaderboard claims, CDNA3/CDNA4/NVFP4/MXFP4 validation claims, canonical schema drift.

### Phase Ordering Rationale

- Acquisition and layout must come first because every inventory count and readiness reason needs a known dataset root, category set, and source manifest.
- Inventory and readiness must precede execution so schema, dtype, custom-input, safetensors, and NVIDIA-only blockers do not become noisy runtime failures.
- Execution closure should follow readiness and stay bounded by category, limit, workload cap, iterations, timeout, and evidence flags.
- Gap reporting should join all prior artifacts rather than duplicate their data, keeping canonical traces and scoring sidecars independently parseable.
- Docs and guardrails should close the milestone after implementation so examples and no-claim language reflect actual commands and artifacts.

### Research Flags

Phases likely needing deeper research during planning:
- **Phase 1:** Live Hugging Face dataset revision/count handling and whether external safetensors blobs require `huggingface_hub` or documented `hf` CLI setup.
- **Phase 2:** Exact low-precision dtype policy for Quant/FlashInfer records and any live upstream fields that can support forward/backward indicators without inference.
- **Phase 3:** Hardware/runtime evidence profile for the available ROCm environment, including whether closure runs should require `--rerun` or artifact freshness checks.

Phases with standard patterns (skip research-phase unless implementation reveals surprises):
- **Phase 4:** JSON/Markdown aggregation, stable reason-code grouping, and artifact-reference reports are straightforward once prior artifacts exist.
- **Phase 5:** Documentation and guardrail tests follow established project patterns from earlier ROCm parity and SOL/SOLAR milestones.

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Stack | HIGH | Repository already has the necessary dependencies and execution/reporting primitives; research strongly recommends no new framework dependency. Exact `huggingface_hub` need depends on implementation choice for FlashInfer blobs. |
| Features | HIGH | Table stakes match `.planning/PROJECT.md` v1.11 target features and existing script/docs/test surfaces. Exact upstream dataset field availability remains MEDIUM until live inventory runs. |
| Architecture | HIGH | Sidecar-only reporting, pure core plus thin scripts, and existing runner reuse align with current public-contract guardrails and prior milestone patterns. |
| Pitfalls | HIGH | Local risks are well grounded in existing downloader/runner behavior, schema contracts, sidecar history, and guardrail tests. Dataset-shape risks remain MEDIUM until the pinned revision is inspected. |

**Overall confidence:** HIGH for roadmap structure; MEDIUM for exact problem counts, upstream row fields, and blob availability until Phase 1/2 live validation.

### Gaps to Address

- **Pinned dataset revision and expected counts:** Phase 1 should record or require the Hugging Face revision and derive/check counts from that revision rather than hard-coding unverified totals.
- **Public dataset field preservation:** Phase 2 should inspect live rows to confirm whether forward/backward/op-type hints exist upstream or must be derived conservatively.
- **Safetensors and FlashInfer blob acquisition:** Phase 1/2 should decide whether to keep `hf download` as documented setup or add explicit `huggingface_hub` dependency only if Python code imports it.
- **Low-precision readiness policy:** Phase 2 should encode dtype readiness as layered evidence, not a boolean, especially for Quant, FP8, FP4, NVFP4, and MXFP4.
- **Trace freshness for closure claims:** Phase 3 should record git commit, dataset manifest checksum, command args, workload caps, solution hash, and config metadata or require `--rerun` for closure runs.
- **CDNA3/CDNA4 and full validation boundaries:** Phase 5 should ensure every report states these remain deferred unless future evidence explicitly proves them.

## Sources

### Primary (HIGH confidence)
- `.planning/PROJECT.md` - v1.11 goal, target features, explicit deferrals, current RDNA4/CDNA3 validation boundaries, and v1.10 as existing context.
- `.planning/research/STACK.md` - recommended dependency posture, script integration points, stack additions, and what not to add.
- `.planning/research/FEATURES.md` - table stakes, differentiators, anti-features, artifact contracts, and MVP recommendation.
- `.planning/research/ARCHITECTURE.md` - sidecar-only architecture, module layout, JSON contracts, data flow, and build order.
- `.planning/research/PITFALLS.md` - critical/moderate/minor pitfalls, phase warnings, and mitigation strategies.
- `scripts/download_solexecbench.py` - current Hugging Face acquisition behavior and local problem layout.
- `scripts/run_dataset.py` - existing discovery, execution controls, summaries, skips, traces, AMD score, AMD SOL, SOLAR derivation, and timing evidence options.
- `src/sol_execbench/core/data/definition.py` and `src/sol_execbench/core/data/workload.py` - canonical schema contracts used for inventory validation.
- `src/sol_execbench/core/data/trace.py` - canonical trace/evaluation schema that must remain stable.
- `src/sol_execbench/core/bench/io.py` - safetensors, scalar, random, dtype, and custom input handling.
- `tests/sol_execbench/test_public_contract_guardrails.py` and `tests/sol_execbench/test_run_dataset_amd_score.py` - existing guardrail and derived-report behavior to preserve.

### Secondary (MEDIUM confidence)
- Hugging Face Datasets documentation - confirms named-config and split loading patterns, but exact live dataset shape must be checked during implementation.
- Safetensors documentation - confirms supported tensor loading model, but exact dataset blob availability depends on local acquisition setup.

---
*Research completed: 2026-05-23*
*Ready for roadmap: yes*
