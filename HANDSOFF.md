# Project handoff and active follow-ups

Last audited: 2026-08-09 against `f4453442` and the current worktree.

This file records unresolved repository-level work, the decisions that constrain
it, and the evidence that closed the lifecycle production-topology P0. Generated run
state belongs in the diagnostic lifecycle registry; detailed completed
investigations and one-time inventories belong in Git history.

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
  The start-220 payload is registered as production design
  `2bcfa7fc7feadb39a165b03d6a855d73045b9ae536dabda445a1cdbb1dbc60ee`;
  this records the existing frozen design and does not claim new GPU evidence.
- The production-topology lifecycle P0 closed on 2026-08-09. Historical v7 and
  Cycle 2 evidence was imported into the content-addressed store without GPU
  recomputation, then promoted as an 880-case, three-parent production snapshot
  `892be3337f6bf126c7d432208264a7a93120fa4a328e02afb66e82703c7db18a`.
  A separate production-shaped conformance run proves the two-source promotion
  branch, fresh held-out branch, exact collection-tree inventory, role-separated
  model/acceptance parents, immutable receipts, idempotent resume, successor
  generation rules, and fail-closed consistency/GC.
- P0 closure does not authorize Cycle 3 acceptance or publication. The complete
  end-to-end run described below has purpose `control_plane_conformance`; the
  production registry objects are the three historical source snapshots, their
  promoted development snapshot, and the start-100/start-160/start-220 designs.
  Fresh start-220 GPU collection and the production acceptance verdict remain P1.
- Do not remove `microarchitecture-diagnostics-v7/` or
  `microarchitecture-diagnostics-v7-cycle2/` merely because CAS import
  completed. Expanded source-tree retirement still requires a reviewed path
  retirement plan and explicit deletion approval.
- Real-device evidence is strongest on one RX 9060 XT/gfx1200 host. Multi-GPU
  isolation and CDNA-family behavior have contract and unit coverage but
  narrower empirical coverage.
- Static AMDGPU metadata extraction deliberately uses bounded ELF-note and
  gzip/zlib scanning. It does not parse the clang-offload-bundler Compressed Code
  Object Bundle (CCOB) manifest or provide exact member selection.

## Lifecycle P0 closure evidence

The target production data flow requires the following immutable parent
identities rather than inferred paths:

```text
source collection runs -> source snapshots -> promotion -> development_snapshot
design -> held_out_collection_run -> held_out_snapshot

calibration + development_snapshot -> model_build
calibration + development_snapshot + held_out_snapshot + model_build
  -> acceptance
accepted acceptance + calibration + development_snapshot + model_build
  -> publication -> release_candidate -> published_release
```

The original hosted conformance run proved the shared linear mechanisms. The
2026-08-09 production-shaped closure additionally exercised
`source snapshots -> promotion -> development_snapshot` beside a distinct
`design -> held_out_collection_run -> held_out_snapshot` branch. Closure is
supported by the following concrete evidence:

- The historical source snapshots are
  `dbf39ce807344ddee2ce79b69beb293b971c402cd4c771103e663751d7eef10f`,
  `90d7af07b40047f305e1ab0aae3c66224646bb5d33ad628c7c6f6753394eb2ac`,
  and `1eeeac30a8183b6532f60d254ecdcb8f2ead7cc23ab2205dfc13d29520df3da8`.
  Their promoted 880-case snapshot is
  `892be3337f6bf126c7d432208264a7a93120fa4a328e02afb66e82703c7db18a`;
  its manifest SHA-256 is
  `5729ff90706302cf873659ae7acbcbfcd5722e8b284d207129bfa52bb0bcb756`.
- The final production-shaped conformance run is
  `c96d65d534bdfe51ac23b0fda026721a614fa6f3e03388fa1ca3da533a922096`
  at source revision `f4453442279b557212b7f6ba44d3d87eafc4796d`. Its
  canonical plan ID is
  `356eec29448a96fad0ef32bd1f3da32419733e5294fb08cb0589a87215ff6074`;
  the reviewed plan file SHA-256 is
  `6b2fa68903e04f67130d680d142a28a90456dfea8684620db76bccb60943add8`.
- That run promotes two independent 220-case source snapshots into development
  snapshot
  `4c88532c9f0515cabd2eb2b8eeafe951bfa53277328879a4f954613b5dd84c76`
  and derives held-out snapshot
  `d3621a29da7d83be05aca99723ce1ca1fb595e434aafa5971a7a2e52d027609c`
  only from the fresh conformance collection run. The latter binds an exact
  3,741-file, 18,277,046-byte collection inventory.
- All eight stages are `verified`. A subsequent `status` reported
  `next_stage=null`, and `resume` remained a no-op with exactly one attempt per
  stage. Acceptance
  `c5770933e1401cfdf31aa9a0df08a1269c1e43f0e16ec5ff2b2135c258805141`
  recorded `accepted=true`, publication is
  `6c7fe7d17f9b77d12b6d05c8f4a456414276f3c55a0eeb72ff9e3a629cf48c7f`,
  and the local release candidate is
  `5e4eb276dc7ace676fbf8b647d3d3ffdff8033695065db63bbb62bfce01eaa5e`.
- The final consistency check accepted `data/store`. Its GC dry-run observed
  37,949 blobs totalling 23,239,103,276 bytes; 37,913 blobs totalling
  23,229,476,339 bytes were reachable. The 36 unreferenced blobs remain intact:
  no GC apply or expanded-tree deletion was performed.

- The conformance traversal executes every linear stage with pre/post input
  identity checks, blob-backed persisted stage outputs, verification before
  commit, and one legal next action. `status` and `resume` revalidate inputs and
  descendants; acceptance terminality and `accepted=true` publication admission
  fail closed.
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

The production-shaped conformance traversal and CAS adoption are CPU and storage
operations; they did not launch fresh GPU collection or refit Cycle 3 inference.
The final repository test run reported 2,355 passed and 23 skipped. Ruff,
formatting, `ty`, architecture gates, focused lifecycle/release tests, and the
diagnostic store consistency checker passed.

## Active backlog

### P1 — Run Cycle 3 held-out acceptance through the governed chain

The production-topology P0 is closed; Cycle 3 must
not reuse prior inference or acceptance artifacts, tune after held-out reveal,
change gates, exclude the 24 Cycle 2 cases that exposed the working-set bug, or
reuse a held-out pair.

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
