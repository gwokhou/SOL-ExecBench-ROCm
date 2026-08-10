# Project handoff and active follow-ups

Last audited: 2026-08-10 against `7424243e`, the live ignored evidence/store,
the remote P0 Release metadata, and the start-340 lifecycle acceptance
hard-failure (coverage, not a harness defect).

This file records unresolved repository-level work, the decisions that
constrain it, and the evidence that closed the lifecycle production-topology
P0. Generated run state belongs in the diagnostic lifecycle registry; detailed
completed investigations and one-time inventories belong in Git history.

## Current state

- The official gfx1200 score is published. The checked-in AKA manifest contains
  45 authored problems: 43 scored, one compatibility sentinel, and one
  target-incompatible problem. The official score is `0.497818327` for the
  `rx9060xt-gfx1200-reference-v2` baseline against the
  `rx9060xt-gfx1200-eager-reference-self-eval` candidate over 43 problems and
  163 workloads. `RELEASE/release-bundle.json` is the repository publish marker;
  GitHub Release `gfx1200-official-score-v2` is complete and is not part of the
  backlog below.
- The current performance-diagnostic contracts include diagnostic validation
  corpus v9, performance diagnostic v7, diagnostic calibration v8, diagnostic
  inference profile v10, lifecycle run v3, BenchmarkConfig v2, reference IPC
  v2, and ROCm event timing v4. Superseded schema readers and migrations are
  intentionally absent.
- The first statistically evaluated gfx1200 v7 cycle failed only the
  preregistered per-family coverage gate. Cycle 2 is immutable source evidence,
  not a repairable acceptance attempt: its held-out reveal exposed a change from
  accumulated hardware traffic to SOLAR semantic bytes as the working-set
  coordinate.
- Cycle 3 has a frozen, pair-disjoint start-220 design. Its production
  collection is now terminally incomplete: the first 181 held-out cases
  returned, then `held_out-transformer_block-01` failed before timing because
  the frozen design requested `M=1032` while the packaged candidate contract
  permits sequence lengths only through 1024. No held-out corpus was frozen,
  no acceptance verdict exists, and no publication is authorized. The
  start-220 payload remains registered as production design
  `2bcfa7fc7feadb39a165b03d6a855d73045b9ae536dabda445a1cdbb1dbc60ee`;
  this records the immutable failed design and does not authorize repair or
  reuse of its now-revealed held-out pairs.
- HEAD implements mandatory pre-collection qualification. Counter
  collection now requires an isolated, content-bound chain of three verified
  gates: zero-GPU validation of all 660 frozen design cases and every prepared
  problem artifact; risk-first per-axis-extrema canaries; then minimal-profile
  correctness qualification of every workload in the selected role. Family
  receipts are written after each success, all incidental qualification timing
  is explicitly non-authoritative, and any gate/input/collector hash drift
  blocks `collect` before its first case. Existing collection evidence is
  resumable only when the complete manifest and all cited artifacts verify as
  `available` with the current definition/workload/solution identity; partial
  evidence is a hard error.
- The same mandatory stage vocabulary now governs every current large batch GPU
  producer: `qualify-static`, `qualify-canary`, and `qualify-full`. Release
  baseline/candidate timing, full-corpus SOLAR construction, AKA tolerance
  calibration, the 82-analysis SOLAR cross-path focus, diagnostic calibration,
  and future resource-peak calibration all verify the complete content-bound
  chain before their first formal item. The shared gate schema records the task,
  subject, runner, configuration, source, exact item denominator, parent gate,
  and hashed receipts; all qualification timing remains non-authoritative.
  The historical resource-peak v3 producer remains byte-frozen because the real
  checked-in audit pins its whole-file SHA-256. New runs use
  `run_qualified_rdna4_resource_peak_calibration.py`; changing the historical
  script or rewriting its audit digest is forbidden.
- The user accepts the 181 completed Cycle 3 cases only as a degraded
  exploratory observation. They remain ineligible for freeze, production
  acceptance, inference fitting, or publication. Any optional completion of
  the remaining 19 transformer and 20 concurrent observations must use a
  separately labelled exploratory root with valid replacement transformer
  shapes; it must not amend the frozen start-220 design or create production
  lifecycle authority.
- The first isolated repair draft, start 280 / design
  `a28e0e604d532ecb5f7703cf9cdc586f0a68ec178b2caa4549a8f350802ca956`,
  passed only the zero-GPU static gate. It used a synthetic odd transformer
  sweep and was superseded before canary, full qualification, or collection;
  its old gate is additionally invalid after the collector changed. Retain it
  only as abandoned process evidence.
- The current representative exploratory repair is the isolated start-340
  design `20f86f6c7a0ed6a084a6ff59120acdb1f90dad8f4c8dc4e805ce1c48472ca5f8`.
  Its 60 transformer shapes are unique, bounded by 1024, and organized into
  audited neighborhoods around short contexts, BERT/GPT boundaries, and
  vision-token anchors while remaining disjoint from start 100/160/220/280.
  Design SHA-256 is
  `29270b044c2adb06dad462a7e7ee8ba7f3bbe7cd5df537fc4eb2a942f4a2512e`.
  Its zero-GPU static gate covers all 660 cases, is explicitly
  non-authoritative for performance. After the qualification runner was bound
  to the hardened Docker entrypoint and explicit non-authoritative
  `not_tested`-target smoke routing, the current static gate SHA-256 is
  `45db95164c401221da1733c1de5d9724d4177c3eb7546d306ee89e79f067aa9c`.
  The real RX 9060 XT/gfx1200 canary passed all 34 extrema across all 11
  families in container isolation; every Trace is `PASSED`, and the canary
  gate SHA-256 is
  `675dd03390112e05fb60f5470976227bc42fe7e1013575b50e5b529cf0547204`.
  Full qualification then passed all 220 held-out workloads—20 in each of the
  11 families—with every Trace reporting `PASSED`, `gfx1200`, and container
  isolation. Its historical gate SHA-256 is
  `963d4197965a3f7a01af756066f319b943fef3a145db4a663e6fef0a37048b45`.
  Both GPU gates remain `performance_authority=false`; no rocprof or counter
  collection has run for this repair. These three gates bind source revision
  `f89b94c248edb6a5e2601828e9aa728fde6d87ea` and collector SHA-256
  `47d4b21b7d1040460ead27236acc21d05a08b585200438fda6e802ce4c52b372`.
  HEAD now hashes the collector as
  `4850a4cd953d7fb7df02fad460cb55cb65eda7e21ab5c9b15bf9dfed82a93975`;
  the production entrypoint therefore rejects the old static gate with
  `qualification gate identity drift`. The old gates are retained real GPU
  process evidence, but they no longer admit collection. Any future use of
  start-340 requires a new isolated current-HEAD static/canary/full chain.
- A new isolated current-HEAD start-340 chain was produced at revision
  `7424243e`, which first closes two lifecycle harness gaps that had blocked
  the production plan: `load_collection_gpu_identity` now resolves blob-backed
  held-out corpus references through the lifecycle blob store (it previously
  required path-backed evidence, but `freeze` records immutable corpora as
  blob-backed), and `inventory_regular_tree` now sorts by relative-path string
  to match the run-state validator (path-object ordering disagreed with string
  ordering for a regular file that prefixes a sibling directory, e.g.
  `solar.log` beside `solar/`). With those fixes, the fresh chain passed:
  static gate `73acd0d15ce8ab7568ad3e65b768e5b683d91c3bf90aea016e0aa01b8dfa1f1f`,
  canary gate `c40c2051eb2fcac715ff57fffcdb48b0eb2ab83086b446fada05d7b166d845e1`
  (34 extrema, all `PASSED`), and full gate
  `291a2814fc6fc58e719100232c959a3465d2b062e8628d121130c73ba9f2db6a` (220
  held-out workloads, all `PASSED`), binding collector
  `4850a4cd953d7fb7df02fad460cb55cb65eda7e21ab5c9b15bf9dfed82a93975`. All 220
  held-out cases were then collected in container isolation (one transient
  partial counter pass on `held_out-transformer_block-17` was re-collected to
  `available`), SOLAR manifests generated, and the corpus frozen (held-out
  corpus snapshot `65754cb308612387bf72dce0ab44d3c9c03d9d681a6b08afae105c2a02047425`).
  A current-schema calibration pair was produced on the same host.
- Lifecycle run
  `9128fa06bef3acaa55b592deb5888b1e0a6e625769c5991d4a9f43694bb84c44`
  (generation 2, development snapshot `892be333`, design `634dee74`) executed
  DESIGN, CALIBRATION, COLLECTION_RUN, CORPUS_SNAPSHOT, and MODEL_BUILD, all
  `verified` (held-out snapshot
  `d33a3dd0bc8ef8688f955ab4336e10e4e6d5fde20370f396e4a280682a600642`).
  ACCEPTANCE then hard-failed after three attempts with
  `calibration_out_of_range:working_set_bytes` on `held_out-elementwise`.
  This is an honest, terminal coverage finding, not a repairable harness
  defect: the start-340 representative elementwise neighborhoods have working
  sets of roughly 145–389 MB, while `calibration.py:139` hardcodes the VRAM
  throughput applicability at `[64 MiB, 256 MiB]`. The promoted development
  snapshot derives from the start-100/160/220 corpus, whose elementwise cases
  are far smaller, so the development-fit calibration does not cover the
  start-340 held-out elementwise regime and the model cannot predict those
  cases. The held-out acceptance surfaced this generalization gap exactly as
  designed (`unavailable hardware prediction is a hard failure`). Widening the
  calibration applicability, re-collecting, or excluding the revealed
  elementwise cases would constitute post-reveal tuning and is refused; the
  start-340 successor is therefore retained as terminal coverage-hard-failure
  process evidence and is not acceptance-viable against the development
  calibration. A future viable attempt requires a new governed successor whose
  working-set range a separately developed, pre-reveal calibration policy
  covers, with a fresh held-out design; it must not reuse the start-340 pairs.
- HEAD `7424243e` records the two lifecycle harness fixes with non-mocked
  regression tests (a 220-case blob-backed corpus for the identity loader and
  an inventory ordering test for the file/directory prefix collision). The full
  test suite (2403 passed, 23 skipped) and the repository gates pass except
  the standing `check_non_canonical_artifacts.py` five-v2-historical boundary.
- The production-topology lifecycle P0 closed on 2026-08-09. Historical v7 and
  Cycle 2 evidence was imported into the content-addressed store without GPU
  recomputation, then promoted as an 880-case, three-parent production snapshot
  `892be3337f6bf126c7d432208264a7a93120fa4a328e02afb66e82703c7db18a`.
  A separate production-shaped conformance run proves the two-source promotion
  branch, fresh held-out branch, exact collection-tree inventory, role-separated
  model/acceptance parents, immutable receipts, idempotent resume, successor
  generation rules, and fail-closed consistency/GC.
- HEAD now binds full PCIe topology in all new GPU performance evidence. The
  runtime collector records the ordered root-port/bridge/GPU path with each
  link's BDF, current/max speed, and current/max width, then derives the
  effective bottleneck. On the audited RX 9060 XT host the production collector
  returned
  `0000:00:01.1 -> 0000:01:00.0 -> 0000:02:00.0 -> 0000:03:00.0`,
  all at 32.0 GT/s, with the root port's x8 link as the effective PCIe 5.0 x8
  path. Calibration and replay require stable pre/post topology; production
  plan authoring requires the calibration profile/audit and every held-out
  performance manifest to share that exact identity. The topology is bound into
  the reviewed plan and collection-run ID and is rechecked during collection
  adoption/resume. Older endpoint-only evidence remains readable historical
  evidence but is ineligible for a new production plan; the current host probe
  does not retroactively prove its collection-time topology.
- Active-store conformance run
  `2311f13ea8f7216e44e9514fda41fd09c2682efdfab8fb5641367bddafab1bc5`
  remains interrupted at `corpus_snapshot=running` and is retained as process
  evidence; it was not resumed. Status selection now returns that first
  non-verified stage (`next_stage=corpus_snapshot`) instead of incorrectly
  jumping to `model_build`. The completed conformance authority remains run
  `c96d65d534bdfe51ac23b0fda026721a614fa6f3e03388fa1ca3da533a922096`.
- P0 closure did not authorize Cycle 3 acceptance or publication. The complete
  end-to-end run described below has purpose `control_plane_conformance`; the
  production registry objects are the three historical source snapshots, their
  promoted development snapshot, and the start-100/start-160/start-220 designs.
  The failed start-220 collection is retained process evidence. A new
  pair-disjoint successor design and collection are required before a
  production acceptance verdict can exist.
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
- All eight stages are `verified`. At this audit, the same handler-backed
  verification path used by `sol-execbench diagnostics lifecycle status`
  again returned `next_stage=null` with exactly one attempt per stage.
  Acceptance
  `c5770933e1401cfdf31aa9a0df08a1269c1e43f0e16ec5ff2b2135c258805141`
  recorded `accepted=true`, publication is
  `6c7fe7d17f9b77d12b6d05c8f4a456414276f3c55a0eeb72ff9e3a629cf48c7f`,
  and the local release candidate is
  `5e4eb276dc7ace676fbf8b647d3d3ffdff8033695065db63bbb62bfce01eaa5e`.
- The current consistency checker accepts `data/store`. A new GC dry-run at
  this audit observed
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
- The earlier hosted linear conformance run ID is
  `0638897cff2eae889d8fc38fdcb29d1e85dce120c3716d65bda998610ec121cb`.
  Its final publication and release IDs are
  `9a034b2339ca8bb5f24f87e454ed50a2b47c0b886214c2e3ff9423c4cb34040b`
  and
  `f2b7e49441d79731e35700a83109b36129c2d6b848e7e52ec483d9bd0a4c772c`.
  It has been moved out of the active store to
  `data/store-control-plane-v2-historical-20260809/`. Its v2 run/corpus objects
  are historical evidence and are intentionally rejected by the current v3/v9
  readers; it must not be described as a currently reverified active-store run.
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
  A live GitHub API check at this audit confirmed the annotated tag target,
  completed/successful workflow attempt, and the two uploaded assets with the
  exact 2,348,998-byte archive and 1,142-byte attestation digests above.
- The retained v2 immutable round-trip receipt is under
  `data/store-control-plane-v2-historical-20260809/published-releases/` for
  release ID
  `f2b7e49441d79731e35700a83109b36129c2d6b848e7e52ec483d9bd0a4c772c`.
  This receipt binds repository, tag target, GitHub Release/asset IDs, exact
  remote names/sizes/digests, workflow run and attempt, publication time, source
  revision, and the historical local release candidate/CAS. It is not a member
  of the active `data/store` production-shaped run.
- Test-created start-160 objects and obsolete pre-source-revision conformance
  generations were removed only after archival. They remain recoverable from
  `data/cold-archive/test-created-start-160-2026-08-09.tar.zst` and
  `data/cold-archive/pre-source-revision-identity-conformance-2026-08-09.tar.zst`;
  the latter has SHA-256
  `5b567d7bbabd61eff3b35f63a6121d888f734f511e896b650db09e37e86bb5ab`.

The production-shaped conformance traversal and CAS adoption are CPU and
storage operations; they did not launch fresh GPU collection or refit Cycle 3
inference. At HEAD, the batch-GPU qualification change passed the full test
suite plus Ruff, formatting, `ty`, architecture, schema, readability, reuse,
production-reachability, and documentation gates. This audit additionally
reverified the active lifecycle status, store consistency, and GC dry-run; it
did not run GPU qualification, collection, GC apply, or deletion.

## Active backlog

### P1 — Open a successor after terminal Cycle 3 collection failure

Status (2026-08-10, HEAD `7424243e`): a full isolated successor attempt on the
representative start-340 design reached a terminal coverage hard-failure and
is NOT acceptance-viable. The lifecycle run
`9128fa06bef3acaa55b592deb5888b1e0a6e625769c5991d4a9f43694bb84c44` verified
DESIGN/CALIBRATION/COLLECTION_RUN/CORPUS_SNAPSHOT/MODEL_BUILD, then
ACCEPTANCE hard-failed with `calibration_out_of_range:working_set_bytes`:
start-340 representative elementwise working sets (≈145–389 MB) exceed the
hardcoded VRAM applicability `[64 MiB, 256 MiB]` (`calibration.py:139`), and
the development-derived calibration does not cover them. This is an honest
held-out generalization finding; forcing acceptance by widening the
calibration, re-collecting, or excluding the revealed cases is refused as
post-reveal tuning. start-340 is retained as terminal process evidence. The
item remains open for a NEW governed successor whose working-set range a
separately developed pre-reveal calibration policy covers (fresh held-out
pairs; do not reuse start-340's revealed pairs). See the "Current state"
start-340 entries above for the full chain and the two lifecycle harness
fixes shipped at `7424243e`.

The production-topology P0 is closed, but Cycle 3 cannot reach acceptance. Its
failed case is canonical Trace `478a2d6990f188c89f719d8b5215f9d8ade32d3707a7cd6680640df304611772`
with collection-log SHA-256
`fb40ebd6a9df4148dc56406d002b5fdba1ecf94b57d4bdda8e60a8af66c13236`
and partial performance-evidence SHA-256
`489a02851671e968a4db959c4d4905f0d06d7bd9a2192ffa5c1b459b3735c9bf`.
The failure is deterministic, not a retryable sandbox or profiler incident.
HEAD rejects any such design before GPU collection and has
focused regression coverage; it also implements the mandatory three-gate
qualification chain described above. The gates have passed CPU/static and
simulated-evaluator regression tests. Real static, canary, and full-role
qualification receipts exist for the isolated representative start-340 repair.
Two earlier canary attempts stopped before GPU: direct host execution was
rejected for lacking isolation, then the Docker matrix rejected an unvalidated
`not_tested` target. Their logs remain intact. The successful historical runs
used the repository's explicit non-authoritative smoke route and grant no
benchmark, score, acceptance, or publication authority. Because HEAD changed
the bound collector and source revision, the existing three-gate chain now
fails closed and admits no collection. No exploratory collection has started;
this does not mutate the frozen start-220 payload.

Do not force, skip, repair, or resume the Cycle 3 tree. The next production
attempt must not reuse prior inference or acceptance artifacts, tune after
held-out reveal, change gates, exclude revealed failures, or reuse a held-out
pair.

Required sequence:

1. Reverify and reuse the existing blob-backed 880-case development snapshot
   `892be3337f6bf126c7d432208264a7a93120fa4a328e02afb66e82703c7db18a`
   and its three source parents. Current production plan authoring accepts this
   immutable object through `--development-snapshot-id`; do not repeat
   promotion unless a source or governing policy actually changes, and never
   attribute this snapshot to the fresh held-out collection run.
2. Confirm the collection host exactly matches the frozen calibration object:
   RX 9060 XT/gfx1200 GPU `a3ff7590-0000-1000-800f-a29c1cca1511`, BDF
   `0000:03:00.0`, ROCm 7.2.0, compiler
   `HIP version: 7.2.26015-fc0010cf6a`, the complete ordered PCIe path and
   effective link, locked clocks, and `stable_peak` power. Every collected case
   must bind the same pre/post topology. Any mismatch requires a new governed
   calibration and inference fit.
3. Fit and freeze a current-schema v10 inference profile for the v7 corpus from
   the promoted development snapshot. Rebuild every cited case; unavailable
   hardware prediction is a hard failure.
4. Register a fresh successor design whose complete generated workload set is
   statically checked against every packaged candidate contract before freeze.
   Complete `qualify-static`, `qualify-canary --role held_out`, and
   `qualify-full --role held_out` into an isolated qualification root. Verify
   the parent/hash chain, then collect and freeze 220 new held-out cases—20 for
   each of eleven families—without reusing a Cycle 3 pair or inspecting partial
   performance results.
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

- `docs/user/GPU-QUALIFICATION.md`
- `src/sol_execbench/core/bench/batch_gpu_qualification.py`
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

### P2 — Close the expanded historical-store archive boundary

`data/store-control-plane-v2-historical-20260809/` is a 7.4 MiB expanded
historical P0 store, not the active lifecycle registry. Current readers require
lifecycle run v3 and validation corpus v9, so current store verification
correctly rejects its v2-era run/corpus objects. The repository-wide
`check_non_canonical_artifacts.py` gate currently reports five unmarked v2 run
manifests under this tree. Do not migrate or rewrite those immutable historical
objects merely to make current readers accept them.

Completion requires one reviewed boundary:

1. archive the exact expanded tree into `data/cold-archive/` with a verified
   inventory and digest, then use a path retirement plan and explicit deletion
   approval before removing the expanded copy; or
2. retain the expanded tree with a `NON_CANONICAL.md` marker that records its
   identity, immutable historical purpose, schema boundary, inventory/digest,
   and recovery/retention decision.

After either route, the repository-wide non-canonical artifact check must pass
without adding superseded schema readers or weakening the current registry.

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

At this audit every command above that was run passed except
`check_non_canonical_artifacts.py`, whose five historical v2 findings are the
explicit expanded-store archive-boundary backlog above. Do not report the full
handoff verification set green until that boundary is closed.
