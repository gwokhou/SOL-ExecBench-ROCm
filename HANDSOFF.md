# Project handoff and active follow-ups

Last audited: 2026-08-08 against `98dd100c` and the current worktree.

This file records only unresolved repository-level work, the decisions that
constrain it, and its completion criteria. Generated run state belongs in the
diagnostic lifecycle registry; completed investigations and one-time artifact
inventories belong in Git history.

## Current state

- The official gfx1200 score is published. The checked-in AKA manifest contains
  45 authored problems: 43 scored, one compatibility sentinel, and one
  target-incompatible problem. The official score is `0.497818327` for the
  `rx9060xt-gfx1200-reference-v2` baseline against the
  `rx9060xt-gfx1200-eager-reference-self-eval` candidate over 43 problems and
  163 workloads. `RELEASE/release-bundle.json` is the repository publish marker;
  GitHub Release `gfx1200-official-score-v2` contains the verified
  `gfx1200-official-score-release.tar.zst` asset. This release is complete and
  is not part of the backlog below.
- The current performance-diagnostic contracts are diagnostic corpus v7,
  performance diagnostic v7, BenchmarkConfig v2, reference IPC v2, and ROCm
  event timing v4. Superseded schema readers and migrations are intentionally
  absent.
- The first statistically evaluated gfx1200 v7 cycle failed only the
  preregistered per-family coverage gate. Cycle 2 cannot be accepted: after
  held-out reveal, hardware prediction changed from using accumulated hardware
  traffic to SOLAR semantic bytes as the working-set coordinate. Cycle 2 is
  source evidence, not a repairable acceptance attempt.
- Cycle 3 has a frozen, pair-disjoint start-220 design. The existing 880-case
  development corpus, inference profile, publication projection, and archive
  are pre-migration v6 artifacts. They establish historical CPU feasibility but
  are not current v7 lifecycle objects and cannot authorize held-out collection,
  acceptance, publication, or release.
- Lifecycle models, a SHA-256 blob store, receipts, run/status/resume commands,
  retention/GC policy, release packaging, and a least-privilege draft-release
  workflow exist. They are building blocks, not yet an authoritative production
  control plane: the P0 gaps below prevent a fresh generation from having one
  complete, re-verifiable immutable lineage.
- The identity foundation closed on 2026-08-08. Every stage identity now
  recomputes from its manifest inputs (single source of truth), calibration
  and corpus-snapshot promotion are first-class identity-bearing objects,
  GPU fingerprints must be complete whenever hardware is bound, and the
  consistency gate verifies blob content against its digest and rejects a
  stored stage_id that no longer matches its recomputed identity. Acceptance
  still derives its stage_id outside the identity family, the runtime layer
  has not yet routed every handler through its identity function or filled
  complete hardware inputs, and collection-run/corpus-snapshot parent sets
  are not yet populated by the authoring scripts, so the P0 runtime-semantic
  gaps below still stand.
- The reviewed 2026-08-07 retirement plan was executed. Superseded v3/v6 and
  unreferenced Orojenesis roots were cold-archived and reclaimed. Do not remove
  `microarchitecture-diagnostics-v7/` or
  `microarchitecture-diagnostics-v7-cycle2/` until current v7 promotion has
  imported every reachable artifact and registry reachability proves the path
  trees are no longer live.
- Real-device evidence is strongest on one RX 9060 XT/gfx1200 host. Multi-GPU
  isolation and CDNA-family behavior have contract and unit coverage but
  narrower empirical coverage.
- Static AMDGPU metadata extraction deliberately uses bounded ELF-note and
  gzip/zlib scanning. It does not parse the clang-offload-bundler Compressed
  Code Object Bundle (CCOB) manifest or provide exact member selection.

## Active backlog

### P0 — Finish the immutable diagnostic lifecycle

Do not start fresh Cycle 3 held-out collection until this section is complete.
The required control plane is a DAG, not a single path-shaped chain:

```text
source collection runs -> source snapshots -> promotion -> development_snapshot
design -> held_out_collection_run -> held_out_snapshot

calibration + development_snapshot -> model_build
calibration + development_snapshot + held_out_snapshot + model_build
  -> acceptance
accepted acceptance + calibration + development_snapshot + model_build
  -> publication -> release_candidate -> published_release
```

Each arrow must be represented by an immutable parent identity, not inferred
from a path, filename, directory, or prose. Development and held-out snapshots
are distinct lifecycle objects. Promotion is a governed multi-parent derivation,
and calibration is a first-class immutable input rather than an unowned file.

#### Lifecycle topology and authority

1. Add first-class calibration and promotion/derivation identities. A model
   build must cite the calibration object and promoted development snapshot it
   consumed; a held-out acceptance must cite every input it reads directly,
   including calibration, development, model-build, and held-out identities.
2. Keep development and held-out snapshots role-specific. Do not collapse both
   roles into one synthetic snapshot identity or attribute the 880 historical
   development cases to the fresh Cycle 3 held-out collection run.
3. Separate a locally verified release candidate from an externally published
   release. Packaging does not prove that any GitHub tag or asset was published.
4. Every behavior-changing input must be identity-bearing: source revision,
   producer/tool versions, calibration and gate policy, timeout or collection
   policy, GPU/software identity, and exact command parameters. `unknown`, a
   mutable path, or an unhashed descriptive value is not an authoritative
   production identity.

#### Registry production and identity

1. Produce and persist the current lifecycle manifest for every DAG node. The
   production path currently creates only a subset of the declared manifest
   families; calibration, promotion, model-build, acceptance, publication, and
   external-publication lifecycle manifests need real writers.
2. Populate every manifest's complete parent set. Snapshot and release objects
   must not use empty parents when they consume a collection run or publication,
   and a multi-input stage must not cite only its immediate linear predecessor.
3. Import each governed output into the blob store before recording it in a
   manifest, run-state object, or receipt. A receipt digest used as a parent
   identity must itself be durably resolvable under the same storage contract.
4. Make the registry object, receipt, and run-state representations agree on
   stage identity and inventory. There must be one canonical current generation
   and one legal next action.

#### Verification and resume semantics

1. Capture and verify all input identities before a handler has side effects,
   then recompute them after execution. Only unchanged inputs may be committed
   into the receipt.
2. Import and verify the exact output inventory, call the stage verifier, and
   only then stamp `receipt_verified` and transition the stage to `verified`.
   Writers must not self-assert verification merely because execution returned.
3. On `status` and `resume`, recompute each stage's current input identities and
   compare them with the receipt. Rechecking only output files is insufficient.
4. Verify collection evidence by an exact typed inventory; the continued
   existence of `cases/` is not collection verification.
5. If a stage or any input drifts, invalidate every descendant. A rebuilt stage
   may retain its descendants only when its verified identity is unchanged;
   otherwise all dependent stages must be rebuilt in order.
6. Preserve an append-only attempt ledger with stable failure codes and bounded
   captured output. Do not overwrite the only receipt, discard the error, reset
   the lifetime attempt count on every `resume`, or lose evidence of a partially
   completed acceptance attempt.
7. Serialize mutation per design/generation with a lock or compare-and-swap
   protocol. Concurrent `run`, `resume`, successor creation, status transitions,
   and GC must not create two current generations or overwrite one another.
8. Add real-handler tests for calibration/corpus drift, collection mutation,
   missing blobs, changed parent receipts, descendant invalidation, and resumed
   semantic equivalence. State-machine tests using fake handlers are not enough.

#### Acceptance finality and publication admission

1. Treat technical completion separately from the statistical verdict. A
   well-formed rejected acceptance is a verified artifact with `accepted=false`,
   not a failed or publishable stage.
2. Require an exact, immutable `accepted=true` acceptance parent before creating
   a publication. Stage status `verified` alone is insufficient, and neither
   publication nor release may proceed from a missing, rejected, superseded, or
   drifted verdict.
3. A completed acceptance verdict is terminal for its exact model-build and
   held-out-snapshot identities. Operational resume may finish an interrupted
   attempt only with unchanged inputs and preserved attempt history; it must not
   create a second acceptance opportunity for the same generation.
4. If Cycle 3 is rejected, record and retain the verdict, stop publication, and
   open a new design/model generation with fresh held-out pairs. Revealed Cycle 3
   cases may become future development evidence but can never be held out again.

#### Successor generations and frozen data

1. Keep the unconditional refusal to overwrite a frozen corpus or force-rewrite
   frozen held-out evidence.
2. Make the supported successor path executable end to end:
   `new generation -> fresh root -> collect -> freeze -> new snapshot`.
   Generation selection must use the registry rather than always deriving
   generation one or defaulting blindly to generation two.
3. Bind `freeze` to the selected collection generation. A successor snapshot
   must never be attributed to generation one.
4. Remove the unreachable static-identity repair implementation after confirming
   no production caller remains. Repair after freeze is a new generation, not a
   hidden mutation stage.

#### Store consistency and test isolation

1. Make the consistency gate inspect any populated registry. A store containing
   manifests but no `blobs/` directory is inconsistent, not absent.
2. Verify every blob's content against its digest, not merely that a regular file
   exists. Ensure orchestrator outputs and receipt parents satisfy the same
   checks that CI applies to lifecycle manifests.
3. Make reachability and GC fail closed on any unreadable, misplaced, unknown, or
   inconsistent registry object. GC must first pass the store consistency gate,
   apply the declared retention window, and bind deletion to one reviewed plan.
4. Prevent references from being added between the final reachability check and
   deletion through the same store transaction or lock. Define how superseded
   run-state and receipt objects stop retaining blobs forever without erasing
   their audit history.
5. Isolate every test with `SOL_EXECBENCH_DIAGNOSTIC_STORE` or an injected store
   root. Tests must not write pytest-root manifests into the workspace
   `data/store`.
6. After the tests are isolated, review and remove the current test-created
   start-160 design object from the local store. This is a scoped data cleanup
   and still requires the normal deletion approval.

#### Release and external-publication closure

1. Write a release-candidate manifest that cites the publication parent and
   binds the archive, attestation, source revision, and exact asset inventory.
2. After GitHub publication, ingest a new immutable published-release receipt
   containing the repository, tag and target commit, GitHub release and asset
   IDs, exact asset names/sizes/digests, workflow run identity, publication time,
   and downloaded round-trip verification result.
3. Make the hosted workflow reject extra or ambiguous archives, malformed or
   mismatched attestations, a tag/source-revision mismatch, and any unexpected
   asset inventory. Editing a draft to non-draft is not registry completion.
4. Keep remote publication least-privilege and one-way: the GPU runner produces
   evidence with read-only repository access; only the hosted publishing job may
   publish, and it must never mutate an existing lifecycle object in place.

#### P0 completion criteria

P0 is complete only when:

- a production-like generation traverses the complete DAG with real handlers;
- calibration, promotion, role-specific snapshots, and local versus external
  release state are represented explicitly in the DAG;
- every stage has a registry manifest, complete parent set, receipt, and
  blob-backed exact inventory;
- receipt verification happens only after pre/post input identity checks and
  output verification, with append-only attempt history;
- changing an input or parent causes `status` to fail the stage and all affected
  descendants;
- `resume` rebuilds from the first invalid stage and never preserves a stale
  acceptance, publication, or release;
- `accepted=false` is retained as a terminal verdict and hard-blocks publication;
- frozen same-generation mutation fails, while a new generation can be
  collected and frozen successfully;
- the consistency checker accepts the completed store and rejects missing or
  mutated blobs and every malformed registry object;
- concurrent writers and GC cannot create duplicate current generations, lose
  attempt history, or delete a newly referenced blob;
- a published GitHub Release has a registry receipt that round-trips every exact
  remote asset to its local release candidate; and
- GC explains the retention decision for every blob without relying on path
  naming or `HANDSOFF.md`.

Authoritative surfaces:

- `src/sol_execbench/core/bench/performance_model/lifecycle/`
- `src/sol_execbench/core/bench/performance_model/release/`
- `src/sol_execbench/core/integrity/`
- `src/sol_execbench/cli/commands/diagnostics.py`
- `scripts/internal/rdna4/build_rdna4_diagnostic_corpora.py`
- `scripts/check_diagnostic_store_consistency.py`
- `.github/workflows/diagnostic-release.yml`
- `docs/performance-diagnostics.md`

### P1 — Run Cycle 3 held-out acceptance through the governed chain

Cycle 3 must not reuse either prior inference or acceptance artifact, tune after
held-out reveal, change gates, exclude the 24 Cycle 2 cases that exposed the
working-set bug, or reuse a held-out pair.

Required sequence after P0 closes:

1. Re-promote all 880 prior development and revealed held-out cases through a
   governed multi-parent derivation into one blob-backed v7 development
   snapshot. Verify every source snapshot and original artifact before importing
   it; do not attribute this derived snapshot to the fresh Cycle 3 collection
   run or depend on the two historical path trees afterward.
2. Confirm the collection host exactly matches the frozen calibration object:
   RX 9060 XT/gfx1200 GPU
   `a3ff7590-0000-1000-800f-a29c1cca1511`, BDF `0000:03:00.0`, ROCm 7.2.0,
   compiler `HIP version: 7.2.26015-fc0010cf6a`, locked clocks, and
   `stable_peak` power. Any mismatch requires a new governed calibration and
   inference fit.
3. Fit and freeze a current-policy v7 inference profile from the promoted
   development snapshot. Rebuild and verify every cited case; unavailable
   hardware prediction is a hard failure.
4. Collect and freeze the 220 preregistered start-220 held-out cases—20 for each
   of eleven families—without inspecting partial results.
5. Run acceptance once and commit the terminal verdict. If `accepted=false`,
   retain the rejection and stop. Only `accepted=true` may create the publication
   projection and local release candidate.
6. For an accepted verdict, create and verify the publication projection,
   deterministic archive, attestation, lifecycle release candidate, and draft
   GitHub Release through the same lineage. Publish only after hosted
   verification, then ingest and verify the external published-release receipt.
7. After registry-backed remote round-trip verification, use registry
   reachability to decide whether the old v7 path trees and expanded publication
   staging directory are reclaimable.

Acceptance requires:

- at least 90% empirical interval coverage for every family;
- median APE at most 15% and P90 APE at most 30%;
- at least one code-changing action metric; and
- for every enabled action, at least 10 held-out positives, at least 90%
  precision, and at least 70% recall.

`restore_wmma_path` is the only currently supported code-changing candidate.
Its historical development precision/recall does not substitute for held-out
acceptance.

Authoritative surfaces:

- `docs/performance-diagnostics.md`
- `src/sol_execbench/core/bench/performance_model/`
- `scripts/internal/rdna4/build_rdna4_diagnostic_corpora.py`
- `scripts/internal/rdna4/run_rdna4_diagnostic_calibration.py`

### P1 — Expand empirical hardware and isolation coverage

- Run `test_real_multi_gpu_candidate_device_switch_is_rejected` with at least
  two visible ROCm GPUs. A single-device visibility restriction does not meet
  the prerequisite.
- Reacquire the archived dataset and source context for revision `d56fadca`,
  then rerun the six historical timeout observations on exact gfx942:
  `FlashInfer-Bench/014_gqa_paged_prefill_causal_h32_kv4_d128_ps1` (one),
  `FlashInfer-Bench/019_mla_paged_prefill_causal_h16_ckv512_kpe64_ps1` (three),
  `L2/040_altup_predict_correction_cycle_backward` (one), and
  `L2/055_audio_encoder_conv_positional_layer_stack` (one). These are historical
  external-evidence debt, not members of the current 43/163 score corpus.
- Validate CDNA4 NVFP4/MXFP4 adaptation on representative hardware. Fallback or
  dequantized execution is not CDNA4 evidence.

Completion evidence must name the exact GPU, ROCm/PyTorch stack, test set, and
skipped prerequisites. A skip or generic schema support is not empirical proof.

### P2 — Resolve the compressed code-object metadata boundary

Choose one explicit contract:

1. implement bounded CCOB manifest parsing with exact target-member selection,
   decompression-size limits, malformed-input handling, and real compressed
   bundle fixtures; or
2. classify full CCOB parsing as permanently unsupported, remove the follow-up
   claim, and retain heuristic gzip/zlib scanning only as best-effort evidence.

The current embedded-zlib scan must not be described as complete CCOB support.

Authoritative surfaces:

- `src/sol_execbench/core/bench/static_kernel/amdgpu_metadata.py`
- `tests/sol_execbench/core/bench/test_amdgpu_metadata.py`
- `docs/user/static_kernel_evidence.md`

### P2 — Classify the Torchview backward-reference boundary

Four focused SOLAR comparison cases remain outside the 41-workload denominator
because the forward-only extractor cannot represent their backward references.
Either add a representation with focused equivalence tests or declare the cases
unsupported with a stable reason code and denominator policy. Do not leave them
as an unowned exception in a completed investigation summary.

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
  lifecycle parents fail closed. Old-schema compatibility readers are not
  allowed.
- `L` remains unavailable without an explicitly supplied trusted frontier.
- Partial or ungoverned diagnostics cannot request kernel code changes.
- Tuning and parameter-estimation samples cannot enter held-out acceptance.
- Candidate inputs use per-run entropy and per-invocation trusted-reference
  validation. The candidate process receives neither nonce nor expected output.
- Publication evaluation remains networkless, capability-free, and private-IPC.
- Mutable process evidence stays under ignored `data/outputs/`; immutable
  diagnostic release projections stay under ignored `data/publications/`.
  Neither is committed to Git.
- Current `sol_execbench.*` schema identifiers are defined only in
  `src/sol_execbench/core/integrity/schema_versions.py`; current SOLAR string and
  numeric artifact versions are defined only in `src/solar/schema_versions.py`.

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
