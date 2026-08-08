# Project handoff and active follow-ups

Last audited: 2026-08-09 against `23bb57e7` and the current worktree.

This file records unresolved repository-level work, the decisions that constrain
it, and the evidence that closed the diagnostic-lifecycle P0. Generated run state
belongs in the diagnostic lifecycle registry; detailed completed investigations
and one-time inventories belong in Git history.

## Current state

- The official gfx1200 score is published. The checked-in AKA manifest contains
  45 authored problems: 43 scored, one compatibility sentinel, and one
  target-incompatible problem. The official score is `0.497818327` for the
  `rx9060xt-gfx1200-reference-v2` baseline against the
  `rx9060xt-gfx1200-eager-reference-self-eval` candidate over 43 problems and
  163 workloads. `RELEASE/release-bundle.json` is the repository publish marker;
  GitHub Release `gfx1200-official-score-v2` is complete and is not part of the
  backlog below.
- The current performance-diagnostic contracts are diagnostic corpus v7,
  performance diagnostic v7, BenchmarkConfig v2, reference IPC v2, and ROCm
  event timing v4. Superseded schema readers and migrations are intentionally
  absent.
- The first statistically evaluated gfx1200 v7 cycle failed only the
  preregistered per-family coverage gate. Cycle 2 is immutable source evidence,
  not a repairable acceptance attempt: its held-out reveal exposed a change from
  accumulated hardware traffic to SOLAR semantic bytes as the working-set
  coordinate.
- Cycle 3 has a frozen, pair-disjoint start-220 design. The historical 880-case
  development corpus and its pre-migration artifacts establish feasibility but
  do not authorize current held-out collection, acceptance, or publication.
  Do not remove `microarchitecture-diagnostics-v7/` or
  `microarchitecture-diagnostics-v7-cycle2/` until governed promotion imports
  every reachable artifact and registry reachability proves both trees are dead.
- The immutable diagnostic lifecycle P0 closed on 2026-08-09. The control plane
  now has a complete typed DAG, content-addressed inventories, immutable
  receipts, append-only attempts, descendant invalidation, successor-generation
  handling, fail-closed consistency/GC, hosted publication verification, and an
  externally observed published-release receipt.
- P0 closure used only public development evidence under purpose
  `control_plane_conformance`. It is not a Cycle 3 held-out attempt, production
  acceptance, score release, or leaderboard authority.
- Real-device evidence is strongest on one RX 9060 XT/gfx1200 host. Multi-GPU
  isolation and CDNA-family behavior have contract and unit coverage but
  narrower empirical coverage.
- Static AMDGPU metadata extraction deliberately uses bounded ELF-note and
  gzip/zlib scanning. It does not parse the clang-offload-bundler Compressed Code
  Object Bundle (CCOB) manifest or provide exact member selection.

## P0 closure evidence

The governed data flow is now represented by immutable parent identities rather
than inferred paths:

```text
source collection runs -> source snapshots -> promotion -> development_snapshot
design -> held_out_collection_run -> held_out_snapshot

calibration + development_snapshot -> model_build
calibration + development_snapshot + held_out_snapshot + model_build
  -> acceptance
accepted acceptance + calibration + development_snapshot + model_build
  -> publication -> release_candidate -> published_release
```

Closure is supported by the following concrete evidence:

- Production handlers traverse every stage with pre/post input identity checks,
  exact blob-backed output inventories, stage verification before commit, and
  one legal next action. `status` and `resume` revalidate inputs and descendants;
  acceptance terminality and `accepted=true` publication admission fail closed.
- Frozen evidence cannot be repaired in place. A changed input opens a new
  generation, while per-design locking, reviewed GC plans, and the append-only
  attempt ledger prevent concurrent overwrite, stale publication, or audit loss.
- The conformance run ID is
  `0638897cff2eae889d8fc38fdcb29d1e85dce120c3716d65bda998610ec121cb`.
  Its final publication and release IDs are
  `9a034b2339ca8bb5f24f87e454ed50a2b47c0b886214c2e3ff9423c4cb34040b`
  and
  `f2b7e49441d79731e35700a83109b36129c2d6b848e7e52ec483d9bd0a4c772c`.
  All eight lifecycle stages reverify and report no next stage.
- The deterministic archive contains 440 cases, is 2,348,998 bytes, and has
  SHA-256
  `43072e90c0b501119744a65d726818b58d945a38d4b08254cc15cc2aa502a245`.
  Its attestation SHA-256 is
  `4ed7f95cc384fa047c182cedddf0214ba65713565e35a000d66073a7abd00825`.
- GitHub Release
  [`diagnostic-lifecycle-p0-conformance-v1`](https://github.com/gwokhou/SOL-ExecBench-ROCm/releases/tag/diagnostic-lifecycle-p0-conformance-v1)
  is published at tag target
  `a538171b8045c2c03ac422627cb08461fd0513b0`. Hosted workflow run
  [`31272071579`](https://github.com/gwokhou/SOL-ExecBench-ROCm/actions/runs/31272071579)
  verified the exact two assets, safe extraction, semantic reproduction,
  byte-identical deterministic rebuilding, and release identity before publish.
- `data/store/published-releases/f2b7e49441d79731e35700a83109b36129c2d6b848e7e52ec483d9bd0a4c772c/receipt.json`
  is a v2 immutable round-trip receipt. It binds repository, tag target, GitHub
  Release/asset IDs, exact remote names/sizes/digests, workflow run and attempt,
  publication time, source revision, and the local release candidate/CAS.
- Test-created start-160 objects and obsolete pre-source-revision conformance
  generations were removed only after archival. They remain recoverable from
  `data/cold-archive/test-created-start-160-2026-08-09.tar.zst` and
  `data/cold-archive/pre-source-revision-identity-conformance-2026-08-09.tar.zst`;
  the latter has SHA-256
  `5b567d7bbabd61eff3b35f63a6121d888f734f511e896b650db09e37e86bb5ab`.

The conformance traversal and hosted publication verification are CPU-only. The
full repository test run exercised one short ROCm timing integration using a
4096-by-4096 matrix multiplication, not large-scale GPU computation. That run
reported 2,341 passed, 23 skipped, and one transient variance assertion; the
exact failed test passed on isolated retry. Ruff, formatting, `ty`, architecture
gates, focused lifecycle/release tests, and diagnostic store consistency passed.

## Active backlog

### P1 — Run Cycle 3 held-out acceptance through the governed chain

P0 is closed, but Cycle 3 still must not reuse prior inference or acceptance
artifacts, tune after held-out reveal, change gates, exclude the 24 Cycle 2 cases
that exposed the working-set bug, or reuse a held-out pair.

Required sequence:

1. Re-promote all 880 prior development and revealed held-out cases through a
   governed multi-parent derivation into one blob-backed v7 development snapshot.
   Verify every source before importing it; do not attribute this snapshot to
   the fresh Cycle 3 collection run.
2. Confirm the collection host exactly matches the frozen calibration object:
   RX 9060 XT/gfx1200 GPU `a3ff7590-0000-1000-800f-a29c1cca1511`, BDF
   `0000:03:00.0`, ROCm 7.2.0, compiler
   `HIP version: 7.2.26015-fc0010cf6a`, locked clocks, and `stable_peak` power.
   Any mismatch requires a new governed calibration and inference fit.
3. Fit and freeze a current-policy v7 inference profile from the promoted
   development snapshot. Rebuild every cited case; unavailable hardware
   prediction is a hard failure.
4. Collect and freeze the 220 preregistered start-220 held-out cases—20 for each
   of eleven families—without inspecting partial results.
5. Run acceptance once and commit the terminal verdict. Retain and stop on
   `accepted=false`; only `accepted=true` may create a publication and release.
6. For an accepted verdict, publish through the same lineage and ingest the
   external published-release receipt before treating publication as complete.
7. Use registry reachability to decide whether historical v7 path trees and
   expanded staging directories are reclaimable.

Acceptance requires at least 90% empirical interval coverage per family, median
APE at most 15%, P90 APE at most 30%, and at least one code-changing action with
at least 10 held-out positives, 90% precision, and 70% recall.
`restore_wmma_path` is the only currently supported code-changing candidate;
historical development quality does not substitute for held-out acceptance.

Authoritative surfaces:

- `docs/performance-diagnostics.md`
- `src/sol_execbench/core/bench/performance_model/`
- `scripts/internal/rdna4/build_rdna4_diagnostic_corpora.py`
- `scripts/internal/rdna4/run_rdna4_diagnostic_calibration.py`

### P1 — Expand empirical hardware and isolation coverage

- Run `test_real_multi_gpu_candidate_device_switch_is_rejected` with at least
  two visible ROCm GPUs. A single-device visibility restriction does not meet
  the prerequisite.
- Reacquire the dataset and source context for revision `d56fadca`, then rerun
  the six historical timeout observations on exact gfx942:
  `FlashInfer-Bench/014_gqa_paged_prefill_causal_h32_kv4_d128_ps1` (one),
  `FlashInfer-Bench/019_mla_paged_prefill_causal_h16_ckv512_kpe64_ps1` (three),
  `L2/040_altup_predict_correction_cycle_backward` (one), and
  `L2/055_audio_encoder_conv_positional_layer_stack` (one). These are external
  evidence debt, not members of the current 43/163 score corpus.
- Validate CDNA4 NVFP4/MXFP4 adaptation on representative hardware. Fallback or
  dequantized execution is not CDNA4 evidence.

Completion evidence must name the exact GPU, ROCm/PyTorch stack, test set, and
skipped prerequisites. A skip or generic schema support is not empirical proof.

### P2 — Resolve the compressed code-object metadata boundary

Choose one explicit contract:

1. implement bounded CCOB manifest parsing with exact target-member selection,
   decompression-size limits, malformed-input handling, and real compressed
   bundle fixtures; or
2. classify full CCOB parsing as permanently unsupported and retain heuristic
   gzip/zlib scanning only as best-effort evidence.

The current scan must not be described as complete CCOB support.

Authoritative surfaces:

- `src/sol_execbench/core/bench/static_kernel/amdgpu_metadata.py`
- `tests/sol_execbench/core/bench/test_amdgpu_metadata.py`
- `docs/user/static_kernel_evidence.md`

### P2 — Classify the Torchview backward-reference boundary

Four focused SOLAR comparison cases remain outside the 41-workload denominator
because the forward-only extractor cannot represent their backward references.
Completion requires one of:

- a representation that preserves upstream-gradient dependencies and gradient
  outputs, with focused extraction, conversion, execution-equivalence, and
  cross-path denominator tests; or
- an exact unsupported reason code emitted by the pipeline, a documented
  denominator policy that keeps all four cases visible, and contract tests that
  prevent silent exclusion or reclassification.

Authoritative surfaces:

- `src/solar/graph/torchview/extraction.py`
- `src/solar/pipeline/`
- `scripts/internal/solar/run_cross_path_focus.py`
- `tests/solar/test_pipeline_integration.py`
- `docs/user/CROSS-PATH-COMPARISON.md`

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
