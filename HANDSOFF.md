# Project handoff and active follow-ups

Last audited: 2026-08-13 against source revision
`cbaa382289418bda706cabe7df964b24943c927a`, the live ignored evidence/store,
the remote P0 conformance Release metadata, both pre-verdict exposure
boundaries, and the completed start-460 rejection described below.

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
- The start-460 successor closed the indexed-read applicability precondition
  without weakening any gate. Its final calibration SHA-256 is
  `9573c1235ec59752f2454d9b19c0690a9ed59f9fa002c9b1726d8c2ce2377b3a`;
  the seven-family replacement fragment is
  `d4ed38e00dd5b200743fcd7b63e2a6f9083859ae5a6d9c6d0228b9683f743ff3`.
  The production composer replaced 140 exposed cases and reused 80 unaffected
  cases, producing held-out corpus
  `627fe009fe16dd19784406062ab2d488b89d872ca22c719abae8e57e9c32466a`.
  Lifecycle run
  `753d91a311eb86622a049ef4b4e7ac15f8dbb16340da3561b2c345f3d05f7147`
  completed a real 220-case verdict with `accepted=false`. Median APE was
  13.02%, but P90 APE was 61.65%; several families missed 90% empirical
  coverage, and `restore_wmma_path` recall was 0.55 against the unchanged 0.70
  gate. Publication correctly refused the terminal rejection, so no local
  publication, release candidate, GitHub production Release, tag, or
  published-release receipt exists. The complete rejected held-out corpus is
  now exposed and is ineligible for reuse in another acceptance run.
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
  ACCEPTANCE stopped before a verdict after three historical retry attempts,
  each on `calibration_out_of_range:working_set_bytes`. This is now represented
  as `precondition_failed`, not `accepted=false`: no acceptance result or
  aggregate/error/action metric was written. The durable exposure receipt
  records that `held_out-elementwise-00` was evaluated before the stop and that
  `held_out-elementwise-01` plus the unavailable-reason code were released.
  Receipt SHA-256 is
  `1eb56872a70776f5770fffcdc8157ef428032dc7551df3c4695a6b9478effff3`;
  it is registered under `data/store/acceptance-exposures/` and retained by GC.
  The cause remains real: start-340 elementwise working sets are roughly
  145–389 MB, outside the `[64 MiB, 256 MiB]` calibration applicability.
  However, this does not logically invalidate raw counters from all eleven
  families. Reuse is now decided from the exact exposure receipt and an exact,
  reviewed source diff. The conservative statistical unit is a family: replace
  all 20 elementwise cases, while the 200 cases in the ten unexposed and
  source-unaffected families may be reused by exact artifact identity.
- The capacity-governed start-400 successor closed that elementwise gap on the
  real RX 9060 XT/gfx1200 host without changing acceptance thresholds. Frozen
  VRAM policy SHA-256
  `99dff937b6aad9fec415af52abd488eb567766b6753b811bd33b8e3f81646639`
  classifies observed total physical memory `17095983104` bytes as the 16 GiB
  class and binds a `[64 MiB, 512 MiB]` applicability interval; runtime free
  memory is not an input. Design
  `fa543e236dea1163adb0b4dfd3588d0e28ac8648ae50b9b129dccdaa565171bb`
  passed static, 34-case canary, and 220-case full qualification. All 20 fresh
  elementwise cases were formally collected and received SOLAR manifests. The
  exact reviewed composition replaced those 20 and reused 200 unaffected
  start-340 cases; held-out SHA-256 is
  `bf727aa62dda511ab988b21416a1e08867be166faaad2208e21f6f461d1a9fa7`
  and the revision-bound reuse-manifest SHA-256 is
  `1e96962b0b15a012bfd2ac6ba259f40abf8c0712b39f0a2065ff9029be9e8cdf`.
  The development-only inference profile was frozen before this held-out
  collection and reproduced byte-identically at model build with SHA-256
  `f0b3dd95341cf85745b086584e34fd304f09c14c9e3adbeb8cefa5847512caa4`.
- A new successor attempt `p1-successor-start640-pcie5x16-r1` was produced on
  PCIe 5.0 x16 at source revision
  `8848d6050b4a46e08c86acaf37333f33d90c8076`. The source transition is
  documented as
  `source-review-18535cf-to-8848d60.json` (`14` paths, git patch SHA-256
  `a5e2c7ed16b003494b6857c9daeb72bb09dc933016d856c23b185f15bb0f7cb7`),
  and the same revision was used by all qualification gates in
  `corpus-qualification-8848d60`: static (660 cases), canary (38 cases), and
  held-out full (220 cases), all with collector SHA-256
  `1733783495c1e6108502f0e48fb1ac35e16ee9ad24272d0e7f248770a24e763c`.
  `p1-successor-start640-pcie5x16-r1` has since completed a full production
  lifecycle at revision `8848d605`. The 20 held-out `elementwise` cases were
  collected with counter+replay artifacts (`PASSED`) and SOLAR-analyzed
  (`analyzed`, `publication_eligible`). An `elementwise` fragment was frozen
  and composed with the 200 unaffected families reused by exact identity from
  `p1-successor-start580-pcie5x16-r1/composed-held-out` (replace-family
  `elementwise`, exposure receipt = start580 acceptance `e8486e19` that
  released `held_out-elementwise-05`). Because the design is fresh (universe
  640) while calibration/inference/policy are reused, design registration used
  the recovery path (`preregister_rdna4_recovery.py`): design
  `be1fed469b7a26d68c1351a65a8b8fb52aa06a37d552daff576032b128c41c03` at
  `8848d605` reuses the attested `d1ee4324` VRAM policy (`dbac7df4`) via the
  prior source-transition attestation `d1ee4324→18535cf` (continuous chain
  `d1ee4324→18535cf→8848d605`, policy/calibration UNCHANGED). Lifecycle run
  `367696957bf65d48c269aa19822fe40f5fde614519a6c14f3f10361e7fcabadf` verified
  DESIGN, CALIBRATION, COLLECTION_RUN, CORPUS_SNAPSHOT, and MODEL_BUILD
  (model_build succeeded, so the reused `d1ee4324` development elementwise is
  admissible under the `8848d605` replay-completeness admission). ACCEPTANCE
  manifest
  `9f740c5bc9a971e736f0f735161bf823a61e071898b9fcfac835a9c711839327` recorded
  `accepted=false` — a model-generalization quality-gate failure (same class
  as start-460), not a precondition failure. The run was timeout-interrupted
  during the terminal publication attempts, but the immutable acceptance
  verdict is persisted, so this is closed terminal evidence: no local
  publication, release candidate, GitHub production Release, tag, or
  published-release receipt exists, and tag
  `gfx1200-diagnostics-v7-production-v1` must not be created. Per the reuse
  rule, none of this `accepted=false` held-out corpus may be reused in another
  acceptance; the next successor must use a separately frozen development
  split and 220 entirely fresh held-out pairs.
- The start640 coverage failure was root-caused as a cross-design
  distribution shift, not a model bug or a reusable-route defect. Offline
  analysis of the frozen start520 development corpus against the start640
  held-out metrics showed that `composite_graph` held-out cases fell 11/20
  outside the reused development measured range (hard extrapolation), while
  `indexed_read` and `reduction_norm` were in-range but systematically
  cleaner than their reused development residuals. The elementwise family —
  whose held-out cases stayed inside the development range — reached 100%
  coverage, confirming the diagnosis. The conformal interval is a single
  `solar_lower_bound_ms` linear point model scaled by `exp(q95)`; the reused
  development `q95` (composite_graph 0.041, indexed_read 0.024) is far below
  the held-out P90 residual (0.057 and 0.035 respectively), and a
  leave-one-out residual over the full development corpus (scheme A) does not
  close the gap — it was falsified offline. The fix is therefore a single
  design whose development and held-out splits are i.i.d., not a conformal
  code change.
- A fully fresh successor `p1-successor-start700` was authored at commit
  `cb35de63` (Author start700 fresh successor design shapes): capacity-bounded
  `elementwise`/`transpose` shape functions (`M` starting at 23000 / 33000),
  a disjoint 15-neighborhood transformer realism schedule (60 sequences all
  bounded by 1024 and disjoint from the 313 historical sequences), and the
  successor dispatch. Recovery preregistration reused the attested
  `d1ee4324` VRAM policy (`dbac7df4`) via the same `d1ee4324→18535cf→8848d605`
  chain, with a fresh `source-review-18535cf-to-cb35de63.json` (15 paths).
  Design SHA-256 is
  `b6401661ab53d14e41c441559df36edc4236484f60baedc9384a31bc7d4cac28`
  and design ID is
  `410042044bc9c67fa82e048ba49de54662fff27c27a233579658d4cc13f2d1e6`.
- Commit `cbaa3822` (Drop rocpd output from diagnostic counter collection)
  removes the rocprofv3 `rocpd` SQLite profiling database from counter
  collection. The 134.5 MiB raw database per pass was never parsed — the
  6-counter semantic model consumes only the counter CSV, marker alignment
  uses the marker CSV, `rocpd` was only existence-checked, and publication
  omits it. The counter job now emits `csv` only; the `counter_pass_rocpd_count`
  check and `_compress_rocpd` path are removed. `rocpd` is not in the required
  evidence-artifact set, so `evidence_manifest` verification is unchanged.
  Measured effect: collect drops from ~2.5 min/case to ~2.15 min/case (~13%),
  because rocprofv3 process start + PMC hardware initialization, not the
  rocpd write (gzip level 1 is ~118 ms), dominates per-case time. Every new
  case stops writing ~13 MB; historical rocpd totals ~3.0 GB (4308 `.db.gz`).
- The start700 qualification chain was re-run at `cbaa3822` (the commit
  changed `source_revision`, which the gates bind, so the prior `cb35de63`
  gates drifted). All five gates verify: static
  `6fe2bb79592b23b3`, canary development `0dc26ebcc6ee1dd6`, full development
  `9beb78d3fa9f753b`, canary held-out `4dd61d8ce42f83e6`, and full held-out
  `79a90b49b50c250b`. Development collection then produced 69 of 440 cases
  before it was paused by the operator; no development corpus was frozen, no
  SOLAR manifests were built, no inference profile was fit, no held-out
  collection began, and no acceptance verdict exists. The RDNA4 diagnostic
  production re-release is deliberately paused: the operator does not want to
  spend the ~30 h of full fresh GPU collection now, and no publication,
  release candidate, GitHub Release, tag, or published-release receipt exists.
  The paused state is resumable — qualification is complete at `cbaa3822`, so
  `collect --role development` can be restarted without re-qualifying.
- Production lifecycle generation 3 is run
  `70f28d3986cd7275a7ae7c62921202da6b0b123fd7a533edfbddacdab10ab7f0`
  with plan ID
  `f3419a343dfd680d25dc9f8127d5c6a244b427ceff930cdf119a72f3add4cee4`.
  DESIGN, CALIBRATION, COLLECTION_RUN, CORPUS_SNAPSHOT, and MODEL_BUILD are
  verified; held-out snapshot is
  `7934902da596447a06b9f35fac709e28b872fef219d8af097824e244e83aa860`
  and model build is
  `d89d5e6bf6be390362b11ed1dbe1af3bbf1d40c3c06e79f2983f69cccdec7865`.
  ACCEPTANCE stopped before a verdict at `held_out-indexed_read-14` with
  `calibration_out_of_range:indexed_read`. No acceptance result, metric,
  publication, tag, GitHub production Release, or published-release receipt
  exists. A mistaken overlapping operator resume created two exposure receipts
  for the same case/reason and prefix; both are retained, with SHA-256
  `33d6ecc64c2c49dbbaf69259c33a286a7418f3bc41c36cdf1ca12cc4016b3766`
  and `eeb77c226f20b01664511b39ccdb382d3373b9ed94e5f1717c54b6b69ae0fc69`.
  Neither contains metric fields or a verdict, and the run must not be resumed
  for its remaining attempt.
- `7424243e` records the two lifecycle harness fixes with non-mocked
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
- P0 closure did not itself authorize production acceptance or publication.
  The complete end-to-end run described below has purpose
  `control_plane_conformance`; it remains proof of the mechanism, not a score or
  production diagnostic release. The later start-340 and start-400 production
  attempts are retained separately. Start-400 reached model build but stopped
  pre-verdict as recorded above, so no production publication authority exists.
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

### P1 — start-640 PCIe5.0x16 successor path (closed terminal)

`p1-successor-start640-pcie5x16-r1` completed its full production lifecycle at
`8848d6050b4a46e08c86acaf37333f33d90c8076` and is **closed terminal evidence**,
no longer active. The previously listed next sequence was executed end-to-end:
the source-transition recovery chain (`d1ee4324→18535cf→8848d605`) proved
policy/calibration continuity; the held-out corpus was recomposed (20 fresh
`elementwise` + 200 reused by exact identity); SOLAR and acceptance ran.
ACCEPTANCE recorded `accepted=false` (a model-generalization quality-gate
failure, same class as start-460; see Current state above for the immutable
design/run/acceptance IDs). This is closed evidence: no publication, release
candidate, GitHub Release, tag, or published-release receipt exists, and tag
`gfx1200-diagnostics-v7-production-v1` must not be created. The next
production successor must use a separately frozen development split and 220
entirely fresh held-out pairs; it must not reuse this `accepted=false` corpus.

### P1 — Build a fully fresh successor after the start-460 rejection

The indexed-read calibration gap is closed, but production publication remains
blocked by the completed start-460 `accepted=false` verdict. This is now a
model-generalization failure, not a missing-prediction precondition. Do not
resume the exhausted publication stage, refit against the rejected held-out
corpus, narrow cases, disable the action, or change acceptance thresholds.
Tag `gfx1200-diagnostics-v7-production-v1` remains reserved and must not be
created until a later immutable run records `accepted=true` and completes
external receipt ingestion.

Because acceptance released aggregate, family, error, and action metrics for
all 220 cases, none of the start-460 final corpus may be reused in another
acceptance. The next inference profile must be fit only from a separately
frozen development split, and the next verdict must use 220 entirely fresh
held-out pairs. The authored start-520 policy provides one 660-case universe:
point-fit, conformal, and held-out splits are fixed before collection;
elementwise uses a capacity-bounded stratification inside 64–512 MiB; and the
transformer family uses a fourth audited neighborhood schedule disjoint from
starts 100/160/220/280/340/400/460.

The first start-520 execution root, `p1-successor-start520-r1`, froze design
`a90288e73ef11d0e871cbe07d47c4652bcab4c9ccb0c17f60da7d3fd1d81c51c`
at source revision `d1ee4324907aa91172abce737b437a7929c119ca`. It also produced the
source-bound VRAM policy
`dbac7df435d95352e4819b2fd6f0b98142867a789690e60fd39fa7f11f6496ab`
and final calibration
`297939b560a2f68e65710e1444af1a21a64d23e7a601cd11d6f62c087cd3490a`.
Its development static and canary gates passed. During full qualification,
eight family receipts passed, but `indexed_update` twice completed only 31 of
40 correct evaluations before the CLI's implicit 300-second whole-batch
timeout. No development counter collection or start-520 held-out qualification
or collection began.

HEAD therefore makes the qualification batch timeout explicit at 900 seconds;
it does not reduce the 40-case family batch, change correctness, make
qualification timing authoritative, or alter any acceptance threshold. The
300-to-900 change affects only the monotonic whole-batch qualification
watchdog: compilation retains its separate timeout, the GPU-lock wait remains
capped at 60 seconds, and any evaluation that completed before 300 seconds has
the same result. It does not causally invalidate the r1 policy, calibration,
design, prepared problems, raw collection, SOLAR, inference, acceptance, or
publication semantics.

The earlier blanket source-revision invalidation was therefore correct as a
syntactic identity rule but unnecessarily broad as an experimental rule. The
r1 policy and r2 policy have identical behavior after removing only
`source_revision` and `created_at`; their 660-case design projections and every
prepared problem file are identical. The r1 calibration profile and audit are
not claimed to be byte-identical to r2—the independently compiled probe binary,
code object, and disassembly hashes differ—but the r1 bundle still passes the
current production loader, exact profile/audit linkage, complete GPU identity,
and PCIe-topology checks. The performance-independent recovery rule therefore
selects the earliest complete, frozen, pre-collection r1 policy, calibration,
design, and prepared problems as canonical.

The redundant `p1-successor-start520-r2` rebuild remains immutable process
evidence. Its development qualification completed 440/440 correct evaluations
under source revision `b3fa42e44fa960ac8d59621e1bb32a3fc3f4fb7c`; formal
collection was then stopped at a case boundary after 14 complete point-fit
elementwise cases and before case 15. All 14 pass the current production
performance-evidence verifier; none is silently admitted into r1 and none is
deleted or repaired in place.

The separate source-transition control plane does not modify or re-export the
collector. Before any GPU write resumes, it must bind the exact Git name-status
diff and binary patch digest from r1 to the final source revision; classify each
changed path by lifecycle stage; prove unchanged VRAM-policy, design, prepared-
problem, and raw-collection AST projections; and independently validate the r1
calibration. Rebinding r2 raw cases additionally requires exact source/target
design-case equality, a complete available production manifest, an exact
regular-file tree inventory, non-overwriting staged copy, post-copy production
verification, and an immutable receipt. Because no start-520 held-out outcome
was accessed, the preregistered start-520 universe remains eligible and must not
change in response to the development timeout.

Required sequence:

1. Retain generations 3 and start-460 generation 1, their exposure/rejection
   records, and every verified predecessor stage as immutable process evidence.
2. Freeze the final governance source revision and author/verify the exact
   stage-scoped source-transition attestation. Keep the frozen r1 policy,
   calibration, design, and prepared problems; do not regenerate them merely
   because the qualification watchdog changed.
3. Under a new r1 qualification root, rerun the exact static/canary/full
   qualification chain for all 440 development workloads at the final source
   revision. Qualification timing remains non-authoritative.
4. Only after those gates verify, rebind the 14 complete r2 raw cases into empty
   r1 case directories through the transition tool and verify its immutable
   receipt. Resume r1 collection at case 15; never overwrite either root.
5. Collect all 440 point-fit/conformal cases in container isolation, build all
   440 SOLAR manifests, freeze the development corpus, promote it through the
   registry, and author the inference profile without reading future held-out
   evidence.
6. Freeze that inference profile before qualifying or collecting start-520
   held-out. Then qualify and collect all 220 fresh held-out cases and their
   SOLAR manifests; do not compose or reuse any prior acceptance case.
7. Run acceptance once. Preserve `accepted=false` or another precondition
   failure as terminal evidence. Only `accepted=true` may proceed through local
   publication/release, draft GitHub Release verification, hosted workflow
   publication, and published-release receipt ingestion.

The acceptance thresholds remain unchanged: at least 90% interval coverage per
family, median APE at most 15%, P90 APE at most 30%, and at least one enabled
code-changing action with at least 10 held-out positives, 90% precision, and
70% recall. Historical development quality does not substitute for held-out
acceptance.

The start640 coverage failure was root-caused as a cross-design distribution
shift (see Current state), so a single i.i.d.-split design, not a conformal
code change, is the fix. That fully fresh successor is `p1-successor-start700`
(design ID `410042044bc9c67fa82e048ba49de54662fff27c27a233579658d4cc13f2d1e6`,
authored at `cb35de63`, re-qualified at `cbaa3822`). It is **paused by the
operator before acceptance**: qualification is complete (five gates bind
`cbaa3822`), development collection stopped at 69/440, and no SOLAR, freeze,
inference, held-out collection, or acceptance verdict exists. The operator
does not currently want to spend the ~30 h of full fresh GPU collection, so
the RDNA4 diagnostic production re-release is on hold. Resumption is
`collect --role development` from the paused root without re-qualifying;
whenever the operator resumes, the remaining sequence is the same Required
sequence above (development collect → SOLAR → freeze → inference → held-out
collect → SOLAR → freeze → acceptance once).

### P1 — Author and validate a separate MI300X capacity policy

The current total-memory selection algorithm intentionally admits only
gfx1200 8 GiB and 16 GiB classes. It rejects gfx942/MI300X rather than scaling
the 512 MiB RDNA4 probe tier to a 192 GB accelerator. That fail-closed behavior
is correct, but it leaves MI300X as explicit external hardware debt; current
gfx1200 evidence is not CDNA3 evidence.

Closure requires a separately versioned CDNA3 policy authored before any
held-out design. It must bind observed total physical HBM, the exact gfx942
device and software identity, topology/isolation, a physically justified probe
working set and applicability range, and real MI300X qualification/calibration
receipts. Runtime free memory, a simple capacity ratio, simulator results, or
schema-only tests are insufficient. The policy must fail closed for unknown
MI300X capacity/configuration variants and must not change the existing gfx1200
digest semantics or release boundary.

Authoritative surfaces:

- `src/sol_execbench/core/bench/performance_model/vram_policy.py`
- `src/sol_execbench/core/bench/performance_model/calibration.py`
- `src/sol_execbench/core/bench/performance_model/prediction.py`
- `src/sol_execbench/data/hardware_calibration_probes/diagnostic_microarchitecture.hip`
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

### Handoff continuity for coding-agent switch

This handoff is sufficient for a new agent to continue, provided the following
bootstrap checks are re-run before any GPU work resumes:

1. Confirm repository identity and local edit scope:

```bash
git rev-parse --abbrev-ref HEAD
git rev-parse HEAD
git status --short
```

The expected state for this transition is branch `main`, `HEAD`
`cbaa382289418bda706cabe7df964b24943c927a`, and only `HANDSOFF.md` modified.
Any additional changed work should be staged deliberately and reconciled before
GPU writes.

2. Confirm there is no accidental lifecycle run for the active source revision:

```bash
rg -n 'cbaa382289418bda706cabe7df964b24943c927a' data/store/orchestrations data/store/attempts -g '*.json'
```

Expected result is empty (no current-`data/store` lifecycle acceptance/publication
evidence yet for `p1-successor-start700`).

3. Confirm the controlled bootstrap state from the output tree:

```bash
ls -1 data/outputs/p1-successor-start700
ls -1 data/outputs/p1-successor-start700/corpus-qualification
ls -1 data/outputs/p1-successor-start700/corpus/cases/point_fit/elementwise | head
```

These should align with the paused layout recorded above: five qualification
gates present (`static`, `development/canary`, `development/full`,
`held_out/canary`, `held_out/full`) and a partially collected development tree
(69 of 440 cases, no frozen corpus, no SOLAR manifests).

4. Before resuming, run lifecycle admission checks with the reviewed command
contract from `docs/performance-diagnostics.md`:

```bash
uv run sol-execbench --format json diagnostics lifecycle status --run-id <RUN_ID> --store-root data/store
uv run sol-execbench --format json diagnostics lifecycle resume --run-id <RUN_ID> --store-root data/store
uv run sol-execbench --format json diagnostics lifecycle plan ... 
uv run sol-execbench --format json diagnostics lifecycle run --plan PLAN.json --store-root data/store
```

`<RUN_ID>` is only used when a concrete chain exists. For this path there is none
yet; the agent should first execute `plan` and move to `run` from a freshly
authored transition artifact.

5. Keep the `source-review-18535cf-to-cb35de63.json` gate as immutable
   (the start700 recovery preregistration proof). The start700 successor is
   fully fresh: its development and held-out splits come from one i.i.d.
   design, so no `record-exposure -> freeze-fragment -> compose-held-out`
   reuse route applies. No held-out payload may be composed from any prior
   `accepted=false` corpus, and the 200-case family reuse route of start640 is
   not part of this path.
