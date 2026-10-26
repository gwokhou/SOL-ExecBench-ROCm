# Project handoff and active follow-ups

Last audited: 2026-08-13 against source revision
`fd7970520e205318e2e63d8a96d5203e1d5972c3`, the live ignored evidence/store,
the remote P0 conformance Release metadata, and the completed start-460 /
start640 rejections described below.

This file records unresolved repository-level work, the decisions that
constrain it, and the evidence that closed the lifecycle production-topology
P0. Generated run state belongs in the diagnostic lifecycle registry; detailed
completed investigations and one-time inventories belong in Git history.

## Current state

- The official gfx1200 score is published: `0.497818327` for the
  `rx9060xt-gfx1200-reference-v2` baseline against the
  `rx9060xt-gfx1200-eager-reference-self-eval` candidate over 43 problems and
  163 workloads. `RELEASE/release-bundle.json` is the publish marker; GitHub
  Release `gfx1200-official-score-v2` is complete and not part of the backlog.
- Current performance-diagnostic contracts: validation corpus v9, performance
  diagnostic v7, calibration v8, inference profile v10, lifecycle run v3,
  BenchmarkConfig v2, reference IPC v2, ROCm event timing v4. Superseded schema
  readers are intentionally absent.
- HEAD implements mandatory pre-collection qualification: an isolated,
  content-bound chain of `qualify-static` (zero-GPU, all 660 design cases),
  `qualify-canary` (per-axis extrema), then `qualify-full` (every workload in
  the selected role), each writing hashed family receipts and binding
  design/contract/collector/source identity. Any drift blocks `collect` before
  its first case. This vocabulary governs every current large-batch GPU
  producer; all qualification timing is non-authoritative.
- Successor history (all terminal, detailed evidence in Git history and the
  lifecycle registry):
  - **start-460** — first full lifecycle; acceptance `accepted=false`
    (median APE 13.0% but P90 61.6%, `restore_wmma_path` recall 0.55). Held-out
    corpus exposed and ineligible for reuse.
  - **Cycle 2 / Cycle 3 (start-220)** — Cycle 2 revealed the working-set
    coordinate change; Cycle 3 froze `M=1032` (beyond the 1024 candidate limit)
    and stopped terminally incomplete. Design
    `2bcfa7fc7feadb39a165b03d6a855d73045b9ae536dabda445a1cdbb1dbc60ee`.
  - **start-280 / start-340 / start-400** — exploratory repairs; start-340 and
    start-400 closed the elementwise capacity gap but stopped pre-verdict
    (`calibration_out_of_range:working_set_bytes`). The conservative reuse unit
    is a family: an exposed case taints its 20-case family; the other families
    reuse by exact artifact identity.
  - **start-640 (`p1-successor-start640-pcie5x16-r1`)** — closed terminal.
    Full lifecycle at `8848d605` via recovery chain `d1ee4324→18535cf→8848d605`;
    acceptance `9f740c5bc9a971e736f0f735161bf823a61e071898b9fcfac835a9c711839327`
    recorded `accepted=false` (model-generalization, same class as start-460).
    No publication/release/tag exists; tag
    `gfx1200-diagnostics-v7-production-v1` must not be created.
- **The start640 coverage failure was root-caused as a cross-design
  distribution shift**, not a model bug: `composite_graph` held-out fell 11/20
  outside the reused development measured range (hard extrapolation), while the
  elementwise family (in-range) reached 100% coverage. The conformal interval is
  a single `solar_lower_bound_ms` point model scaled by `exp(q95)`; the reused
  development `q95` sits far below the held-out P90 residual, and a
  leave-one-out residual over the full development corpus (scheme A) was
  falsified offline. The fix is a single i.i.d.-split design, not a conformal
  code change.
- A fully fresh successor **`p1-successor-start700`** was authored (commit
  `cb35de63`) with capacity-bounded `elementwise`/`transpose` shapes and a
  disjoint 15-neighborhood transformer schedule, reusing the attested
  `d1ee4324` VRAM policy (`dbac7df4`) via the same recovery chain and a fresh
  `source-review-18535cf-to-cb35de63.json`. Design ID
  `410042044bc9c67fa82e048ba49de54662fff27c27a233579658d4cc13f2d1e6`.
- Commit `cbaa3822` drops the rocprofv3 `rocpd` SQLite profiling database from
  counter collection (it was never parsed — the 6-counter model consumes only
  the counter CSV; marker alignment uses the marker CSV; publication omits it).
  Measured effect: collect ~2.5 → ~2.15 min/case (~13%); each new case stops
  writing ~13 MB, and ~3.0 GB of historical `.db.gz` no longer accumulates.
- The start700 qualification chain was re-run at `cbaa3822` (the commit changed
  the bound `source_revision`); all five gates verify (static `6fe2bb7959`,
  canary/full development and held-out). Development collection then produced
  **69 of 440** cases before the operator paused it. **The RDNA4 diagnostic
  production re-release is deliberately on hold**: no frozen development corpus,
  SOLAR, inference profile, held-out collection, or acceptance verdict exists,
  and no publication/release/tag exists. Resumption is
  `collect --role development` from the paused root without re-qualifying.
- The production-topology lifecycle P0 closed on 2026-08-09 (conformance run
  `c96d65d534bdfe51ac23b0fda026721a614fa6f3e03388fa1ca3da533a922096`, GitHub
  Release `diagnostic-lifecycle-p0-conformance-v1`). It proves the mechanism
  but does not itself authorize production acceptance or publication.
- Do not remove `microarchitecture-diagnostics-v7/` or
  `microarchitecture-diagnostics-v7-cycle2/` merely because CAS import
  completed; expanded source-tree retirement needs a reviewed plan.
- Real-device evidence is strongest on one RX 9060 XT/gfx1200 host; multi-GPU
  and CDNA coverage is narrower. Static AMDGPU metadata extraction does not
  parse the CCOB manifest.

## Active backlog

<a id="solar-evolution"></a>

### P1/P2 — SOLAR: BASE-01 closeout; retained stages 1–5 work

Scope clarification received 2026-10-26: the user's answer to the pending scope
question is **BASE-01 closeout only**. BASE-01 CPU collection and reporting are
complete. Its original 627 artifact bindings, later recollection's 664 bindings,
and rules review's 20 bindings were rechecked unchanged. No fresh baseline or
test run is implied by this integrity check. The source-bound results remain
736 passing tests for the original baseline, 821 passing / 1 failing for the
recollection, and 84 passing support tests for the rules review.

Earlier execution followed the stages 1–5 goal record before this answer arrived.
Those code changes and execution receipts remain in the working tree; they have
not been rolled back and are separate from the authorized BASE-01 deliverable.
The stage records below describe completed CPU/fixture work, not authorization
to continue it. This clarification changes documentation only; production code,
GPU experiments, real mapper and R workloads are outside the closeout. The
correction and before/after document identities are retained under
`data/outputs/solar-base01-scope-closeout-20261026/`; earlier seals are preserved
as historical snapshots rather than rewritten to match these documentation edits.

Retained stage implementation record: the updated AGENTS rules were applied to trust
boundaries, owned validated contracts, explicit unsupported/invariant outcomes,
and pre-computation work/output budgets. No quality baseline was raised.
This entry does not re-audit the other state recorded in this file.

The [implementation and acceptance map](docs/internal/solar-stages1-5-implementation.md)
preserves the complete requirements. Stage 0's
[capabilities and assumptions](docs/internal/solar-evolution-stage0.md),
[frozen acceptance samples](docs/internal/solar-evolution-acceptance.md), and
[task specifications](docs/internal/solar-evolution-tasks.md) remain the planning
baseline. Earlier stage checkpoint statements about downstream work describe
those source revisions; this entry is the current status authority.

| Stage | Delivered CPU/fixture scope | Acceptance evidence |
| --- | --- | --- |
| S1 | Structured access/effects, base-coordinate unions, input-storage checks, accounting and sealed access artifacts | [S1 requirement/G01–G10 map](docs/internal/solar-stage1-cpu-acceptance.md); 946 related tests plus 59 final affected tests |
| S2 | Intrinsic counts, resource/rate separation, exact verified-bundle reuse, unfused/ideal-fused/cache-aware views, API/CLI/bridge | [S2 contract](docs/internal/solar-stage2-cpu-checkpoint.md), [32-slot report](docs/internal/solar-stage2-cpu-2026-10-26.md); 1027 related tests and final focused checks |
| S3 | Necessary resource constraints, bounded serial scans, sensitivity, complete point states and identity-bound recovery | [S3 contract](docs/internal/solar-stage3-cpu-checkpoint.md), [32-slot report](docs/internal/solar-stage3-cpu-2026-10-26.md); 1223 related tests plus 316 final affected tests |
| S4 | Explicit finite-domain structural-zero scenarios, bounded demand propagation, separate dense/restricted outputs | [S4 contract](docs/internal/solar-stage4-cpu-checkpoint.md), [32-slot report](docs/internal/solar-stage4-cpu-2026-10-26.md); 1364 related tests plus 67 final affected tests |
| S5 | Budgeted ordered matrix-chain reassociation, unchanged original callable oracle, per-candidate verification/pricing, API/CLI and pinned recovery | [S5 contract](docs/internal/solar-stage5-cpu-checkpoint.md), [32-slot and strict-numerical report](docs/internal/solar-stage5-cpu-2026-10-26.md); final 1445 related tests |

The original [BASE-01 report](docs/internal/solar-base01-cpu-2026-10-23.md) remains
unchanged: 32 F attempts, 30 semantic/counting completions, 2 conversion refusals,
8 mathematical discrepancies, and 736 passing tests. The
[pre-integration snapshot](docs/internal/solar-base01-cpu-2026-10-26.md) is also
historical; its extraction-identity test issue was fixed during storage-contract
integration. Neither snapshot is relabeled as validation of later production code.
All 627 original BASE-01 bindings and 417 protected unrelated source files were
rechecked unchanged at final collection.

S1 corrects F04 to 392/648 B (96/160 unique fp32 input elements plus two scalar
outputs) and F08 to 8/12 VALU operations. S1's 32-slot collection changed only
F04/F08 numerical totals; source reference, trace inputs and verification policy
were preserved. Unknown/effect/copy accesses remain unavailable. S2's collection
retained 20 eligible equal full/reuse results and 12 visible ineligible slots;
reuse never executes the reference or extractor. Missing capacity coverage does
not create a cache-aware result or change the existing formal-bound field.

S3 retained 80 new parameter points, 20 inverse queries and 80 resumed points,
matching independent forward formulas. Nested mapper cleanup handles worker
SIGTERM and owned groups whose leader exits normally; SIGKILL and deliberately
detached tools remain outside that guarantee. Cleanup has separate source-bound
checks rather than a falsely relabeled earlier collection.

S4 retained 24 eligible public analyses, 8 ineligible F07/F08 slots, 20 available
restricted ideal-fused views and 24 declined default IEEE scenarios. Both F06
paths now retain dense 48/70 MAC and explicitly report restricted 30/30 MAC;
torchview triangular conversion is verified rather than routed to the other path.
Eight new/resumed scan points and four inverse queries pass independent formulas.
Four F02 collector errors were recovered separately; all 36 attempts remain.
S4 is sealed in `data/outputs/solar-stages1-5-20261026/demand/closeout.json`.

S5's final `order/check-06` passes 1445 related CPU tests and schema, Ruff, format,
ty, coupling, readability and documentation checks. Source identity
`a225aaa60853091b985fdc5f4db8f7ca17de2b7d603cc483874ac7f963ffb0cc` is identical
across full regression, main collection and recovery. Of 32 F slots, 24 sources
are eligible: 8 matrix-chain search slots and 16 explicit domain refusals;
8 F07/F08 sources remain ineligible. Twelve F search scenarios and two strict
numerical scenarios yield 24 candidates and 24 resumed candidates, including
two retained numerical refusals. F03 independent MACs are 84/70 (D) and 165/224
(Q); model-minimum selection and stable ties are verified separately. A real
four-matrix test covers all five trees. Four F03 collector-only pricing-assertion
errors were recovered without production changes; 36 F worker attempts and all
40 actually generated new candidate artifacts remain available.

All execution evidence stays under ignored
`data/outputs/solar-stages1-5-20261026/`; stage seals retain failed checks and
recovered attempts. S5 is sealed in `order/closeout.json`; its 15 stopped CPU containers were
removed after preserving configuration and exit status. The final S5 report
identifies limitations and evidence locations. No generated benchmark output is
added to tracked files.
R references (16 slots), real mapper and GPU experiments remain `not_run`.
Hypothetical/restricted/search projections confer no performance or publication
authority. Stage 6 was not started; paused production qualification remains paused.

### P1 — start-640 PCIe5.0x16 successor path (closed terminal)

`p1-successor-start640-pcie5x16-r1` completed at `8848d605` and is **closed
terminal evidence**: acceptance `accepted=false` (model-generalization), no
publication/release/tag. Its `accepted=false` held-out corpus must not be
reused; tag `gfx1200-diagnostics-v7-production-v1` must not be created.

### P1 — Build a fully fresh successor after the start-460 rejection

Production publication remains blocked by start-460 `accepted=false`. Tag
`gfx1200-diagnostics-v7-production-v1` stays reserved until a later immutable
run records `accepted=true` and completes external receipt ingestion. None of
the start-460 corpus may be reused; the next inference profile must be fit from
a separately frozen development split and judged on 220 entirely fresh
held-out pairs.

The start640 failure was root-caused as a cross-design distribution shift, so a
single i.i.d.-split design is the fix. That successor is
**`p1-successor-start700`** (design `410042044bc9c67fa82e048ba49de54662fff27c27a233579658d4cc13f2d1e6`,
authored at `cb35de63`, re-qualified at `cbaa3822`). It is **paused by the
operator before acceptance**: qualification complete, development collection
stopped at 69/440, no SOLAR/freeze/inference/held-out/acceptance. Resumption is
`collect --role development` without re-qualifying, then the same sequence:
development collect → SOLAR → freeze → inference profile → held-out collect →
SOLAR → freeze → acceptance once.

The acceptance thresholds remain unchanged: at least 90% interval coverage per
family, median APE at most 15%, P90 APE at most 30%, and at least one enabled
code-changing action with at least 10 held-out positives, 90% precision, and
70% recall.

### P1 — Author and validate a separate MI300X capacity policy

The current total-memory selection admits only gfx1200 8 GiB / 16 GiB classes
and rejects gfx942/MI300X. A separately versioned CDNA3 policy must bind
observed total HBM, exact gfx942 device/software identity, topology/isolation,
a justified probe working set and applicability range, and real MI300X
qualification/calibration receipts. It must fail closed for unknown MI300X
variants and not change existing gfx1200 digest semantics. Runtime free memory,
capacity ratios, simulators, or schema-only tests are insufficient.

Authoritative surfaces:
`src/sol_execbench/core/bench/performance_model/vram_policy.py`,
`calibration.py`, `prediction.py`,
`src/sol_execbench/data/hardware_calibration_probes/diagnostic_microarchitecture.hip`,
`scripts/internal/rdna4/run_rdna4_diagnostic_calibration.py`.

### P1 — Validate the non-formal gfx942 (CDNA3) adaptation on real hardware

The code-level non-formal gfx942 path is authored and verified on gfx1200 only:
`gfx942` now runs materialize → evaluate → `solar analyze` through the
non-formal bridge (`analyze_workload_diagnostic`), but every gfx942-specific
artifact is spec-derived and `inexact`. Formal publication stays gfx1200-only.
Re-derive the following on a real gfx942 (MI300X/MI308X) host before treating
any gfx942 claim as validated:

- Confirm the `gfx942_v1.yaml` counter spellings via
  `rocprofv3-avail -d 0 info --pmc`. The `cache_lds` group reuses RDNA
  spellings as placeholders; CDNA3 exposes different L2/TCP counters. Update the
  manifest and re-bind `counter_semantics_sha256` via
  `build_diagnostic_model_identity(model_version, counter_resource="gfx942_v1.yaml")`.
- Compile and run the CDNA3 MFMA probes and record `V_MFMA_*` ISA evidence via
  `scripts/internal/cdna3/run_cdna3_diagnostic_calibration.py`. Author the
  deferred fp8 (FNUZ encoding) and int8 (32x32x8 output layout) MFMA probes
  on-device.
- Confirm the `MI300X.yaml` roofline profile (304 CUs, 192 GiB HBM, ~5.3 TB/s,
  FP16/BF16/INT8 MFMA peaks, L2/L3 sizes) via `rocminfo` and replace the
  provisional values.
- Derive the `cdna3_total_memory_class.v1` probe working set (currently a
  provisional 8 GiB) with a wave64 `diagnostic_microarchitecture.hip` variant;
  a capacity-ratio/simulator justification is insufficient (see the MI300X
  capacity-policy P1 above).
- Run `uv run pytest -m requires_cdna3 -n 0` and record the full evidence chain.

Authoritative surfaces:
`src/sol_execbench/data/rocprofv3_counters/gfx942_v1.yaml`,
`src/solar/rocm/profiles/MI300X.yaml`,
`src/sol_execbench/core/bench/performance_model/vram_policy.py`,
`scripts/internal/cdna3/run_cdna3_diagnostic_calibration.py`,
`src/sol_execbench/core/solar_bridge/analyzer.py`.

Completion evidence must name the exact GPU, ROCm/PyTorch stack, test set, and
skipped prerequisites; results remain engineering/inexact until a formal
resource-peak calibration receipt is produced.

### P1 — Expand empirical hardware and isolation coverage

- Run `test_real_multi_gpu_candidate_device_switch_is_rejected` with at least
  two visible ROCm GPUs.
- Reacquire the dataset/source for revision `d56fadca` and rerun the six
  historical timeout observations on exact gfx942 (`FlashInfer-Bench/014`,
  `019`, `L2/040`, `L2/055`).
- Validate CDNA4 NVFP4/MXFP4 adaptation on representative hardware.

Completion evidence must name the exact GPU, ROCm/PyTorch stack, test set, and
skipped prerequisites.

### P2 — Close the expanded historical-store archive boundary

`data/store-control-plane-v2-historical-20260809/` is a 7.4 MiB historical P0
store rejected by current readers. `check_non_canonical_artifacts.py` reports
five unmarked v2 run manifests there. Do not migrate or rewrite them. Completion
requires either archiving the tree into `data/cold-archive/` with a reviewed
inventory/digest + deletion approval, or retaining it with a `NON_CANONICAL.md`
marker; the non-canonical check must then pass.

### P2 — Resolve the compressed code-object metadata boundary

Choose one: implement bounded CCOB manifest parsing with exact target-member
selection and real fixtures, or classify full CCOB parsing as unsupported and
keep heuristic gzip/zlib scanning as best-effort. The current scan must not be
described as complete CCOB support.

Authoritative surfaces:
`src/sol_execbench/core/bench/static_kernel/amdgpu_metadata.py`,
`tests/sol_execbench/core/bench/test_amdgpu_metadata.py`,
`docs/user/static_kernel_evidence.md`.

### P2 — Classify the Torchview backward-reference boundary

Four SOLAR comparison cases remain outside the 41-workload denominator because
the forward-only extractor cannot represent backward references. Choose one: a
representation preserving upstream-gradient dependencies (with conversion and
denominator tests), or an exact unsupported reason code + documented denominator
policy that keeps all four cases visible.

Authoritative surfaces: `src/solar/graph/torchview/extraction.py`,
`src/solar/pipeline/`, `scripts/internal/solar/run_cross_path_focus.py`,
`tests/solar/test_pipeline_integration.py`,
`docs/user/CROSS-PATH-COMPARISON.md`.

## Invariants

- Performance diagnostics never change canonical Trace timing, `T_SOL`, SOL
  Score, leaderboard values, or rewards.
- Canonical execution precedes profiler replay. Profiler duration, achieved
  throughput, and measured candidate runtime never become prediction features.
- Evidence identity, schema version, calibration range, artifact hashes, and
  lifecycle parents fail closed. Old-schema compatibility readers are forbidden.
- `L` remains unavailable without an explicitly supplied trusted frontier.
- Partial or ungoverned diagnostics cannot request kernel code changes.
- Tuning and parameter-estimation samples cannot enter held-out acceptance.
- Candidate inputs use per-run entropy and per-invocation trusted-reference
  validation. The candidate process receives neither nonce nor expected output.
- Publication evaluation remains networkless, capability-free, and private-IPC.
- Mutable process evidence stays under ignored `data/outputs/`; immutable
  diagnostic release projections stay under ignored `data/publications/`.
- Current `sol_execbench.*` schema identifiers are defined only in
  `src/sol_execbench/core/integrity/schema_versions.py`; current SOLAR versions
  are defined only in `src/solar/schema_versions.py`.

## Verification before handoff

Run the checks relevant to the changed surface, including:

```bash
uv run python scripts/check_current_docs.py
uv run python scripts/check_schema_versions.py
uv run --with ruff ruff check .
uv run --with ruff ruff format --check .
uv run ty check
uv run python scripts/check_coupling.py
uv run python scripts/check_readability.py
uv run python scripts/check_production_reachability.py
uv run python scripts/check_non_canonical_artifacts.py
uv run python scripts/check_diagnostic_store_consistency.py
uv run python scripts/check_python_reuse.py
uv run pytest tests/
git diff --check
```

Hardware claims additionally require the precisely marked ROCm tests on the
named device. Never treat a skip as passing hardware evidence.

At this audit every command above that was run passed except
`check_non_canonical_artifacts.py`, whose five historical v2 findings are the
explicit expanded-store archive-boundary backlog above.

### Handoff continuity for coding-agent switch

Bootstrap checks before any GPU work resumes:

1. Repository identity and local edit scope:

```bash
git rev-parse --abbrev-ref HEAD
git rev-parse HEAD
git status --short
```

Expected: branch `main`, `HEAD`
`fd7970520e205318e2e63d8a96d5203e1d5972c3`, plus the uncommitted CDNA3
gfx942 non-formal adaptation (see the P1 validation backlog above).

2. No accidental lifecycle run for the active source revision:

```bash
rg -n 'cbaa382289418bda706cabe7df964b24943c927a' data/store/orchestrations data/store/attempts -g '*.json'
```

Expected empty (no `p1-successor-start700` acceptance/publication evidence).

3. Controlled bootstrap state from the output tree:

```bash
ls -1 data/outputs/p1-successor-start700
ls -1 data/outputs/p1-successor-start700/corpus-qualification
ls -1 data/outputs/p1-successor-start700/corpus/cases/point_fit/elementwise | head
```

Expected: five qualification gates (`static`, `development/canary`,
`development/full`, `held_out/canary`, `held_out/full`) and a partially
collected development tree (69/440, no frozen corpus, no SOLAR).

4. Before resuming, run lifecycle admission checks with the reviewed command
contract from `docs/performance-diagnostics.md`:

```bash
uv run sol-execbench --format json diagnostics lifecycle status --run-id <RUN_ID> --store-root data/store
uv run sol-execbench --format json diagnostics lifecycle resume --run-id <RUN_ID> --store-root data/store
uv run sol-execbench --format json diagnostics lifecycle plan ...
uv run sol-execbench --format json diagnostics lifecycle run --plan PLAN.json --store-root data/store
```

5. Keep `source-review-18535cf-to-cb35de63.json` immutable (the start700
recovery preregistration proof). start700 is fully fresh — no
`record-exposure -> freeze-fragment -> compose-held-out` reuse route applies,
and no held-out payload may be composed from any prior `accepted=false` corpus.
