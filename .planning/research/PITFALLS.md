# Domain Pitfalls

**Domain:** SOL ExecBench public dataset parity inventory and ROCm execution closure  
**Milestone:** v1.11 Paper Dataset Parity Inventory and ROCm Execution Closure  
**Researched:** 2026-05-23  
**Overall confidence:** HIGH for local integration risks from project docs, scripts, and tests; MEDIUM for public dataset-shape risks until the live Hugging Face dataset revision is inventoried and pinned.

## Scope Boundary

v1.11 should make the public SOL ExecBench benchmark dataset surface concrete in this ROCm port: acquire or verify layout, count problems, classify compatibility, run small ready subsets, emit canonical and derived artifacts, and report parity gaps. It must not claim full 235-problem validation, NVIDIA/B200 or leaderboard equivalence, upstream SOLAR parity, CDNA 3/MI300X validation, CDNA 4 validation, or NVFP4/MXFP4 hardware validation.

The key mistake to avoid is treating "inventory complete" as "benchmark parity validated." The correct closure artifact is a machine-readable status ledger with explicit blocked, skipped, unscored, and ready states.

## Critical Pitfalls

### Pitfall 1: Assuming Dataset Download Equals Paper Dataset Parity

**What goes wrong:** The milestone downloads `nvidia/SOL-ExecBench` and reports success because local folders exist, without proving the dataset revision, category counts, required files, blob assets, and problem identities match the expected public benchmark surface.

**Why it happens:** `scripts/download_solexecbench.py` currently calls `load_dataset(REPO_ID, name=subset, split="train")` for `L1`, `L2`, `Quant`, and `FlashInfer-Bench`, then writes only `definition.json`, `reference.py`, and `workload.jsonl` under `data/benchmark/<subset>/<problem>/`. It does not pin a dataset revision, write source metadata, verify expected counts, download external safetensors blobs, or preserve all row fields.

**Consequences:** The inventory can drift with upstream dataset changes, miss assets needed by workloads, or count a local partial download as paper parity. Later runs fail in ways that look like ROCm incompatibility but are actually acquisition/layout gaps.

**Warning signs:**
- Inventory has category counts but no Hugging Face repo id, revision/commit, split, row count, and acquisition timestamp.
- `data/benchmark` and `data/SOL-ExecBench/benchmark` are both referenced in docs/scripts without a layout alias or migration check.
- Problems with `safetensors` inputs are classified as runtime failures instead of asset-acquisition blocked.
- Download script success is used as evidence for 235-problem validation.

**Prevention strategy:**
- Add a dataset manifest that records source repo, config, split, revision, row count, local output root, and per-problem file checksums.
- Separate acquisition status from ROCm execution status: `downloaded`, `layout_valid`, `missing_blob`, `schema_invalid`, `ready_to_run`.
- Verify category names exactly: `L1`, `L2`, `Quant`, `FlashInfer-Bench`.
- Make count expectations configurable or derived from the pinned revision; fail closed on mismatches.
- Document that public dataset inventory is not paper extraction parity and not full validation.

**Detection:** A clean checkout can run the inventory command and reproduce the same manifest from the same dataset revision; if not, parity inventory is not auditable.

**Future phase:** Phase 1: Dataset Acquisition And Layout Contract.

### Pitfall 2: Letting Stale Documentation Drive Implementation Instead Of Current Code Contracts

**What goes wrong:** v1.11 follows older docs or examples that mention outdated paths, CUDA-era assumptions, or previous milestone claims, while the current Pydantic models and runner enforce different contracts.

**Why it happens:** The repository has accumulated docs across porting milestones. Current code keeps canonical `Definition`, `Workload`, and `Trace` key spaces guarded by tests, while docs still mention historical dataset paths and inherited NVIDIA-origin terms.

**Consequences:** The inventory may classify valid current problems as invalid, or worse, mutate canonical schemas to make stale docs true.

**Warning signs:**
- New inventory fields are added to `definition.json`, `workload.jsonl`, or trace JSONL.
- CLI behavior changes instead of adding a sidecar/report path.
- Docs are updated without tests in `test_public_contract_guardrails.py`.
- README/getting-started paths and script defaults disagree without an explicit compatibility note.

**Prevention strategy:**
- Treat local Pydantic models and guardrail tests as source of truth for public contracts.
- Put inventory, readiness, parity gaps, and derived scoring metadata in sidecars/reports only.
- Add doc guardrails for v1.11 no-claim phrases and canonical schema non-mutation.
- Update docs after code, using generated command examples from the implemented paths.

**Detection:** A diff to `Definition`, `Workload`, `Trace`, primary CLI help, or solution schema is required just to express inventory results. That is a public contract drift warning.

**Future phase:** Phase 1: Contract Baseline; Phase 6: Documentation And Guardrail Closure.

### Pitfall 3: Collapsing Schema Drift Into Runtime Failure

**What goes wrong:** Problems that do not parse current schemas are attempted in the runner and recorded as failed executions, obscuring whether the issue is schema drift, unsupported input descriptors, missing files, or real ROCm execution failure.

**Why it happens:** `scripts/run_dataset.py` discovers directories by `definition.json` and `workload.jsonl`, then later loads definitions and invokes the CLI. It is an execution helper, not a schema-drift inventory tool.

**Consequences:** The milestone produces noisy failure counts and cannot distinguish "public dataset shape changed" from "ROCm cannot run this problem."

**Warning signs:**
- Summary reports only `OK`/`FAIL`.
- Validation errors, missing references, bad custom inputs, and dtype unsupported states appear as generic CLI logs.
- Inventory lacks a stable readiness enum.

**Prevention strategy:**
- Add a preflight inventory pass that parses every `definition.json` and every workload line with current models before any GPU execution.
- Classify blockers before running: `schema_input_blocked`, `dtype_blocked`, `custom_input_blocked`, `asset_blocked`, `unsupported_nvidia_path`, `runtime_blocked`, `needs_hardware_evidence`, `ready`.
- Store first error type, message, file path, and workload UUID without requiring a GPU.
- Keep execution summaries separate from inventory summaries.

**Detection:** A problem with a malformed workload line increments failed execution count instead of a schema-blocked inventory count.

**Future phase:** Phase 2: Machine-Readable Inventory And Readiness Classification.

### Pitfall 4: Mishandling Custom Inputs And Safetensors As Ordinary Random Inputs

**What goes wrong:** Dataset problems requiring `custom_inputs_entrypoint` or safetensors are treated as ready because their schema parses, but execution substitutes random inputs or cannot find blob files.

**Why it happens:** Workloads support `random`, `scalar`, `safetensors`, and `custom`. `Workload` rejects mixed custom and non-custom inputs, and `Definition` validates the named custom entrypoint. `load_safetensors()` resolves relative paths against blob roots and raises on missing files, tensor keys, shapes, or dtypes. The download script does not currently fetch blob assets.

**Consequences:** Correctness failures are misattributed to ROCm. Worse, random substitutes can make a problem appear runnable while not matching the public workload semantics.

**Warning signs:**
- Inventory lists safetensors problems as ready without checking file existence and tensor keys.
- `custom_inputs_entrypoint` is present but the referenced function is not classified separately.
- A workload with `type: custom` is run with reference wrapping but no custom input readiness note.
- FlashInfer trace problems fail only after GPU invocation.

**Prevention strategy:**
- Inventory each input descriptor type per problem and workload.
- For safetensors, verify path resolution against explicit blob roots before execution; record `missing_safetensors`, `missing_tensor_key`, `shape_mismatch`, and `dtype_mismatch`.
- For custom inputs, verify the entrypoint exists and classify whether the current evaluator can invoke it in the reference path.
- Never replace safetensors/custom data with random data for parity runs.

**Detection:** A parity run can pass with a problem whose workload references a missing `.safetensors` path. That indicates the run is not preserving dataset semantics.

**Future phase:** Phase 2: Inventory Classification; Phase 3: Ready-Subset Execution Closure.

### Pitfall 5: Overstating Low-Precision DType Support

**What goes wrong:** FP8, FP4, NVFP4, MXFP4, or quantized problems are classified as ROCm ready because dtype strings parse, even when PyTorch ROCm, AMD hardware, or the benchmark harness cannot validate equivalent semantics.

**Why it happens:** Current dtype mappings include `float8_e4m3fn`, `float8_e5m2`, `float4_e2m1`, and `float4_e2m1fn_x2`. Random generation has special FP8/FP4 paths, and docs explicitly defer NVFP4/MXFP4 validation. Parsing a dtype is not proof that the ROCm stack supports the paper's NVIDIA-oriented low-precision semantics.

**Consequences:** Quant category reports can imply hardware validation or semantic equivalence that v1.11 explicitly must not claim.

**Warning signs:**
- Inventory has a single `dtype_supported: true` flag.
- `float4_e2m1` is treated as interchangeable with packed `float4_e2m1fn_x2`.
- NVFP4/MXFP4 are mentioned without "not validated" or "blocked/pending evidence."
- Quant problems contribute to an AMD-native score report without dtype guard warnings.

**Prevention strategy:**
- Split dtype state into `schema_known`, `input_generation_supported`, `reference_execution_supported`, `candidate_execution_supported`, and `hardware_validation_status`.
- Classify FP8 and FP4/NVFP4/MXFP4 separately; do not merge NVIDIA format names with AMD/PyTorch dtype names.
- For Quant problems, require explicit evidence before `ready`; otherwise use `dtype_blocked` or `needs_hardware_evidence`.
- Keep NVFP4/MXFP4 in no-claim guardrails unless future hardware validation exists.

**Detection:** A report says "Quant ready" without listing dtype families and validation status per problem.

**Future phase:** Phase 2: Readiness Classification; Phase 5: Claim Guardrails.

### Pitfall 6: Treating Skipped Results As Passed Or Invisible

**What goes wrong:** The runner skips already-passed results, missing references, missing solution files, or no-output CLI failures, but the final parity report counts only attempted problems and loses why other problems were skipped.

**Why it happens:** `run_dataset.py` skips existing passed traces unless `--rerun`, continues when no reference exists or a named solution file is missing, and appends failure summaries only for no trace output. This is practical for ad hoc batches but insufficient for parity closure.

**Consequences:** The milestone can undercount blocked problems and overstate closure. Reusing old traces may also mix incompatible code versions, dataset revisions, or config settings.

**Warning signs:**
- Summary denominator is `len(summaries)` rather than total inventory count.
- "Skipping" appears in logs but not in JSON.
- Existing traces are reused without recording code revision, dataset revision, config, and artifact freshness.
- Problems with no reference are absent from output.

**Prevention strategy:**
- Add explicit machine-readable statuses: `not_attempted`, `skipped_existing_pass`, `skipped_missing_reference`, `skipped_missing_solution`, `cli_no_trace`, `failed`, `passed`.
- Include all discovered/inventoried problems in the report denominator.
- Require `--rerun` or artifact freshness checks for closure claims.
- Keep execution closure claims scoped to small ready subsets, with attempted/passed/failed/skipped counts.

**Detection:** The report cannot answer "How many total public problems were found, how many were not attempted, and why?"

**Future phase:** Phase 3: Small Ready-Subset Execution; Phase 4: Parity Gap Reporting.

### Pitfall 7: Sidecar Path Safety Regressions

**What goes wrong:** Problem names, workload UUIDs, or category names from an untrusted dataset are used directly in derived sidecar filenames or output paths, enabling path traversal, collisions, overwritten artifacts, or ambiguous evidence refs.

**Why it happens:** `run_dataset.py` already has `_safe_sidecar_stem()` for AMD SOL and SOLAR derivation sidecars, but other outputs still use category/problem directory names and job-name slices. v1.11 will add inventory and parity sidecars, increasing the number of paths derived from dataset identifiers.

**Consequences:** Reports can overwrite each other, escape intended output roots, or point evidence refs at misleading files.

**Warning signs:**
- New artifact paths concatenate raw `definition.name`, workload UUID, category, or problem directory names.
- Sanitization strips unsafe characters without adding a digest.
- Sidecars for different raw identifiers map to the same filename.
- Relative evidence refs point outside the configured output directory.

**Prevention strategy:**
- Reuse or generalize `_safe_sidecar_stem()` for every derived filename.
- Keep output paths anchored under a resolved output root and verify `is_relative_to()`.
- Add collision tests for names differing only by unsafe characters.
- Store raw identifiers inside the JSON payload; use sanitized plus digest identifiers only for filenames.

**Detection:** A problem named `../x`, `.`, `a/b`, `a:b`, or `a b` can create or reference files outside the output root or collide with another problem.

**Future phase:** Phase 1: Artifact Contract; Phase 4: Report Generation.

### Pitfall 8: Blowing The Runtime Budget With Paper-Scale Execution

**What goes wrong:** v1.11 attempts all workloads for all public problems with default warmups, iterations, and 300-second per-problem timeout, then stalls or produces partial local artifacts that look like a failed validation campaign.

**Why it happens:** The milestone wants execution closure, but it explicitly defers full 235-problem real-hardware validation unless environment and runtime budget are sufficient. `run_dataset.py` has `--limit`, `--max-workloads`, `--iterations`, `--warmup-runs`, and `--timeout`, but closure reporting must make sampled execution intentional.

**Consequences:** Long runs consume GPU time, mask inventory progress, and tempt maintainers to summarize incomplete attempts as validation.

**Warning signs:**
- Default execution plans do not specify `--limit`, `--max-workloads`, reduced iterations, or category filters.
- The parity report lacks an execution profile: workload cap, warmups, iterations, timeout, GPU architecture, clock policy.
- Failed long-running problems are not classified as `runtime_blocked`.

**Prevention strategy:**
- Define a small ready-subset execution profile for v1.11 and record it in every summary.
- Run inventory for all problems, but execute only bounded ready subsets unless explicitly authorized.
- Treat timeout as a classification signal, not just a failure.
- Require separate evidence before claiming anything beyond "small ready subset executed."

**Detection:** A report has pass/fail counts but no run budget metadata or ready-subset selection rule.

**Future phase:** Phase 3: Bounded Execution Closure.

### Pitfall 9: Blurring Canonical Traces With Derived AMD SOL/SOLAR Artifacts

**What goes wrong:** Inventory, readiness, AMD SOL sidecars, SOLAR derivation sidecars, and AMD-native score reports are mixed into canonical traces or treated as required outputs for normal `sol-execbench` execution.

**Why it happens:** v1.10 added sidecar workflows through the dataset runner. It is convenient to add more metadata to trace JSON, but guardrails require canonical trace JSONL to remain unchanged.

**Consequences:** Public contracts drift, downstream tools break, and parity inventory becomes inseparable from optional ROCm analysis.

**Warning signs:**
- Trace JSON contains `coverage_summary`, `derived_evidence_refs`, `inventory_status`, or `readiness`.
- Primary CLI exposes dataset inventory or SOLAR sidecar options.
- AMD-native score report is required for a basic dataset execution pass.

**Prevention strategy:**
- Keep `definition.json`, `workload.jsonl`, solution schemas, primary CLI, and canonical trace JSONL unchanged.
- Emit inventory and parity reports as separate JSON/Markdown artifacts.
- Keep AMD-native and SOLAR artifacts opt-in and derived.
- Extend guardrail tests with v1.11 artifact names.

**Detection:** Existing public contract guardrail tests must be relaxed to make v1.11 pass. That is a blocker unless a public contract change is explicitly approved.

**Future phase:** Phase 1: Contract Baseline; Phase 6: Guardrail Closure.

### Pitfall 10: Claim Guardrails Lag Behind New Parity Language

**What goes wrong:** Docs correctly guarded v1.10 SOLAR derivation claims, but v1.11 introduces new phrases such as "paper dataset parity", "execution closure", and "public benchmark surface" that can imply stronger validation.

**Why it happens:** Existing tests look for v1.9/v1.10 forbidden claims. They may not catch "all 235 problems validated", "paper dataset parity achieved", "leaderboard equivalent", or "upstream SOLAR parity" phrased in v1.11 language.

**Consequences:** Release notes or docs overclaim, especially around public dataset counts and ready-subset execution.

**Warning signs:**
- "Parity" appears without "inventory", "gap report", or "not full validation."
- "Closure" appears without ready-subset denominator and deferred validations.
- CDNA 3 / MI300X language changes from deferred to supported based on schema-only coverage.

**Prevention strategy:**
- Add v1.11-specific no-claim tests.
- Required no-claim phrases should cover: not full 235-problem validation, not NVIDIA/B200/leaderboard equivalence, not upstream SOLAR parity, not CDNA 3/MI300X validation, not CDNA 4 validation, not NVFP4/MXFP4 validation.
- Make reports separate `inventory_complete`, `ready_subset_executed`, and `full_validation_complete` fields; the last should be false for this milestone unless explicitly proven.

**Detection:** A generated report can be quoted as "SOL ExecBench ROCm validates the paper dataset" without contradicting itself nearby.

**Future phase:** Phase 5: Claim Guardrails; Phase 6: Documentation Closure.

## Moderate Pitfalls

### Pitfall 11: Treating Unsupported NVIDIA-Only Solution Paths As ROCm Failures

**What goes wrong:** Original categories or solution metadata for CUDA C++, cuBLAS, cuDNN, CUTLASS, CuTe DSL, or cuTile are attempted as if ROCm should run them directly.

**Prevention strategy:** Classify NVIDIA-only solution paths separately from reference PyTorch readiness. Use existing ROCm dispositions: HIP/C++ and selected ROCm libraries are supported where examples/tests exist; legacy NVIDIA runtimes are out of scope or compatibility-only.

**Warning signs:** `cuda_cpp`, `cuda_cflags`, `cublas`, `cudnn`, `cutlass`, `cute_dsl`, or `cutile` appear in an execution failure bucket instead of `unsupported_nvidia_path`.

**Future phase:** Phase 2: Readiness Classification.

### Pitfall 12: Losing Workload-Level Granularity

**What goes wrong:** A problem is marked ready or blocked based on one workload, hiding mixed states across workloads with different axes, dtypes, safetensors, custom inputs, or tolerances.

**Prevention strategy:** Inventory at problem and workload granularity. Roll up problem status conservatively: a problem is fully ready only if every workload is ready under the selected execution profile.

**Warning signs:** The report has problem counts but no workload counts or UUID-level blocked reasons.

**Future phase:** Phase 2: Inventory Schema.

### Pitfall 13: Not Preserving Tolerance And Numerical Semantics

**What goes wrong:** Readiness or execution closure ignores workload tolerance fields such as `max_atol`, `max_rtol`, `required_matched_ratio`, `max_error_cap`, and `allow_negative_inf`.

**Prevention strategy:** Include tolerance presence and non-default tolerance flags in inventory. Do not normalize or omit tolerances when truncating workloads for small runs.

**Warning signs:** Truncated workload files omit trailing newline or tolerance data, or reports cannot identify problems that rely on special negative-infinity handling.

**Future phase:** Phase 2: Inventory; Phase 3: Execution Closure.

### Pitfall 14: Reusing Old Trace Artifacts Across Code Or Dataset Revisions

**What goes wrong:** `--rerun` defaults to false, so previously passed traces are reused even though code, dataset revision, workload cap, or scoring options changed.

**Prevention strategy:** For closure runs, either require `--rerun` or record artifact freshness keys: git commit, dataset manifest checksum, command args, benchmark config, and solution hash.

**Warning signs:** Report says "skipped existing pass" without trace provenance or freshness comparison.

**Future phase:** Phase 3: Execution Closure.

### Pitfall 15: Confusing Reference Execution With Candidate Solution Validation

**What goes wrong:** Running references through the dataset runner is presented as candidate solution readiness or benchmark competitiveness.

**Prevention strategy:** Label reference-wrapper runs as harness/readiness smoke tests. Candidate solution validation requires explicit solution artifacts and separate claims.

**Warning signs:** Reports use "solution passed" when the solution is `reference_<name>` generated by the runner.

**Future phase:** Phase 3: Execution Closure; Phase 5: Report Guardrails.

## Minor Pitfalls

### Pitfall 16: Non-Deterministic Inventory Ordering

**What goes wrong:** Reports churn because filesystem or dataset iteration order changes.

**Prevention strategy:** Sort categories, problem names, workload UUIDs, and JSON keys. Keep stable status enum order in summaries.

**Future phase:** Phase 2: Inventory Schema.

### Pitfall 17: Ambiguous Path Names In Docs

**What goes wrong:** Users run commands against `data/SOL-ExecBench/benchmark` while the downloader writes `data/benchmark`, or vice versa.

**Prevention strategy:** Choose one canonical v1.11 dataset root or support both with explicit detection and docs. Report the resolved root in every artifact.

**Future phase:** Phase 1: Dataset Layout Contract; Phase 6: Docs.

### Pitfall 18: Weak Failure Logs For CI Or Offline Review

**What goes wrong:** Human-readable CLI logs exist, but JSON reports omit stderr snippets, exception classes, and file paths needed to triage failures later.

**Prevention strategy:** Store concise structured failure evidence in reports: phase, exception type, message, path, workload UUID, and remediation hint.

**Future phase:** Phase 4: Parity Gap Reporting.

### Pitfall 19: Inventory Ignores Licensing And Redistribution Boundaries

**What goes wrong:** Downloaded benchmark assets or generated mirrors are treated as commit-ready project data.

**Prevention strategy:** Keep downloaded datasets and blobs out of git. Record source and license metadata in manifests; do not commit public dataset payloads or proprietary kernels.

**Future phase:** Phase 1: Acquisition Contract; Phase 6: Docs.

## Phase-Specific Warnings

| Phase Topic | Likely Pitfall | Mitigation |
|-------------|----------------|------------|
| Dataset acquisition/layout | Download success mistaken for parity | Pin/record source revision, expected categories, counts, checksums, and blob availability |
| Inventory schema | Runtime failures hide schema drift | Preflight parse all definitions/workloads and classify blockers before GPU execution |
| Custom inputs/safetensors | Missing assets treated as ROCm failure | Verify custom entrypoints and safetensors paths/keys/shapes/dtypes explicitly |
| DType classification | FP8/FP4 parse support overclaimed as hardware validation | Split schema, generation, execution, and hardware-validation states |
| Small execution closure | Full dataset attempted without budget | Bound runs with category/limit/max-workloads/iterations/timeout and report the run profile |
| Derived artifacts | Sidecars leak into canonical contracts | Keep inventory/score/SOLAR/AMD SOL outputs sidecar-only and opt-in |
| Gap reports | Skipped problems disappear | Include every inventoried problem and workload in denominators with explicit statuses |
| Claim guardrails | "Parity" implies full validation | Add v1.11 no-claim tests and report booleans for inventory vs execution vs full validation |

## Sources

- `.planning/PROJECT.md` - v1.11 scope, explicit deferrals, and current validation boundary.
- `docs/internal/solar_derivation_contract.md` - sidecar-only rule and no-claim vocabulary.
- `docs/analysis.md` - dataset runner, timing evidence, AMD-native score, AMD SOL, and SOLAR sidecar semantics.
- `docs/original_parity.md` - original public surface disposition and NVIDIA runtime out-of-scope items.
- `docs/compliance.md` - unsupported NVIDIA runtime features, dependency families, and known gaps.
- `scripts/download_solexecbench.py` - current Hugging Face download and local layout behavior.
- `scripts/run_dataset.py` - current discovery, skip, execution, sidecar, and report behavior.
- `tests/sol_execbench/test_public_contract_guardrails.py` - canonical schema, CLI, sidecar, and claim guardrails.
- `tests/sol_execbench/test_v1_9_validation_closure.py` - validation-claim guardrails and RDNA 4/CDNA 3 boundaries.
