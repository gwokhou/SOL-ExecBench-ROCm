# Project handoff and active follow-ups

Last audited: 2026-08-08 against source revision `0d380e75` and the current
worktree; the gfx1200 official-score publication was rechecked in that state.

This file is the single backlog for repository-level work that remains useful.
Completed investigations, superseded contracts, acceptance attempts, and
one-time readiness snapshots belong in Git history. Public documentation must
describe the current contract rather than duplicate this backlog.

## Current state

- The checked-in AKA manifest contains 45 authored problems: 43 scored, one
  compatibility sentinel, and one target-incompatible problem. The official
  gfx1200 score is published: 0.497818327 for the
  `rx9060xt-gfx1200-reference-v2` baseline against the
  `rx9060xt-gfx1200-eager-reference-self-eval` candidate, covering the 43 scored
  problems and 163 workloads. The manifest is cutover to `status: available`
  (`content_addressed_publisher_v1`); the repository publish marker is
  `RELEASE/release-bundle.json`, and the distributed evidence archive is the
  GitHub Release `gfx1200-official-score-v2` (asset
  `gfx1200-official-score-release.tar.zst`, verified through
  `score release-verify` under a content-identical rebuild contract). The
  pre-publication source-evidence root `data/outputs/p0-release-36e44fb/` is now
  superseded archival evidence; its per-statement digests are not the published
  bundle identity.
- The current performance-diagnostic family is v7, with BenchmarkConfig v2,
  reference IPC v2, and ROCm event timing v4. There are no compatibility
  readers for the superseded timing or diagnostic schemas.
- The last statistically evaluated gfx1200 v7 cycle completed calibration, 440
  development cases, frozen inference, and 220 pair-disjoint held-out cases. It
  failed the preregistered coverage gates: median APE 3.303951%, P90 APE
  15.186039%, `composite_graph` coverage 0.70, and `concurrent_graph` coverage
  0.85, against a required per-family coverage of 0.90.
- Cycle 2 promoted all 660 prior development and revealed held-out cases to
  development, preregistered the fresh universe beginning at 160, froze the
  inference profile, and collected all 220 fresh held-out cases with both
  SOLAR and performance evidence. The prior 660-case corpus is under the
  ignored `data/outputs/microarchitecture-diagnostics-v7/` root and the Cycle 2
  inference and held-out corpus are under the ignored
  `data/outputs/microarchitecture-diagnostics-v7-cycle2/` root. Their
  development and held-out corpus SHA-256 values are respectively
  `a0a3a5b24eb620d76ade646d64a7219dc6c53b6cd31f66e7ef6a64dd17ef7a7a`
  and `3fe7797229d180d0f824f14866b6aaaca5cc4940050960fbf896045f9f5a2b4b`.
  Acceptance was input-invalid, not a statistical rejection: four
  `elementwise` and all twenty `transformer_block` cases selected a calibration
  tier using accumulated hardware traffic as the working-set coordinate and
  exceeded the registered range.
- Hardware prediction now keeps counter-derived traffic as the byte amount but
  uses SOLAR semantic bytes as the working-set coordinate. This changes the
  model policy identity after held-out reveal, so Cycle 2 cannot be repaired or
  accepted retrospectively; its ignored artifacts are source evidence only.
- Cycle 3 CPU preparation is complete. Governed cross-root promotion produced
  the ignored 880-case `data/outputs/promoted-development-cycle3.json`;
  production inference fitting rebuilt every cited case successfully and wrote
  the frozen inference profile, which enables `restore_wmma_path` from 80
  development positives and 800 negatives with 1.0 precision and 1.0 recall;
  all other code-changing actions remain disabled. The start-220 design and
  CPU preflight are frozen under the ignored preregistered root. A governed,
  directory-isolated publication projection exists under
  `data/publications/microarchitecture-diagnostics-v7-cycle3/` with 880 cases
  and 74,253,001 bytes excluding the self-describing manifest; the production
  verifier accepts it, and the deterministic zstd archive is 6,116,405 bytes.
  These pre-migration artifacts are the archival v6 generation; their exact
  digests live in the lifecycle registry, the release attestation
  (`sol_execbench.diagnostic_release_attestation.v1`), and the
  registry-generated `diagnostics lifecycle status --run
  data/store/runs/<collection_run_id>/run.json` output, not duplicated in
  this handoff. They are local diagnostic artifacts, not publisher authority
  or an official score release.
- A governed release packager now packages a verified publication into a
  deterministic zstd archive, a release attestation
  (`sol_execbench.diagnostic_release_attestation.v1`), and an immutable release
  manifest under the lifecycle store: `diagnostics release package` /
  `diagnostics release verify`. A GitHub-hosted `diagnostic-release.yml`
  workflow verifies the archive checksum and publication and then publishes a
  draft GitHub Release; the self-hosted `rdna4-hardware.yml` workflow stays
  `contents: read`. The current `microarchitecture-diagnostics-v7-cycle3.tar.zst`
  is the first candidate this flow can package and round-trip verify. The
  immutable lifecycle contracts, store layout, and identity chain live under
  `src/sol_execbench/core/bench/performance_model/lifecycle/`.
- Promotion now targets the local SHA-256 blob store
  (`data/store/blobs/sha256/<digest>` by default, overridable with
  `SOL_EXECBENCH_DIAGNOSTIC_STORE`): `promote` imports every cited artifact into
  the write-once store and emits blob-backed corpus references
  (`CorpusArtifactReference` with `blob_backed: true`), so a promoted corpus
  depends on no historical physical path tree. The corpus schema is now
  `diagnostic_validation_corpus.v7`. The pre-migration Cycle 3 artifacts on disk
  (the path-rebased `data/outputs/promoted-development-cycle3.json` and the
  projected corpus inside `data/publications/.../publication.json`) are v6 and
  therefore archival-only records for the current toolchain; the Phase 1 release
  archive preserves the publication tree, and the real Cycle 3 collection work
  re-promotes and re-projects into the v7 world. The old v6 schema identifier is
  removed with no compatibility reader.
- A lifecycle orchestrator now automates the monotonic chain with
  verification-based status: `diagnostics lifecycle run --design <design.json>
  [--stages ...] [--store-root ...] [--max-attempts N]`, `diagnostics
  lifecycle status --run <run.json>`, and `diagnostics lifecycle resume --run
  <run.json>`. The orchestrator owns the DAG, bounded attempts, typed receipts,
  legal transitions, and an atomic per-generation run-state object
  (`data/store/runs/<collection_run_id>/run.json`,
  `sol_execbench.diagnostic_lifecycle_run.v1`). Each stage delegates to a thin
  handler: the CPU stages (model build, acceptance, publication, release) call
  the existing production functions, while the collection stages validate the
  operator-collected evidence and frozen corpora. Status and resume re-verify
  every recorded stage through the handler; a missing receipt or drifted input
  is re-executed, never reported complete by file existence. The real GPU
  collection itself remains operator-run through the corpus authoring script.
- The registry is now the run truth. `diagnostics lifecycle status` writes a
  registry-generated `status.json` under
  `data/store/runs/<collection_run_id>/` with the current generation, every
  stage's status and produced stage identity, the parent chain, and the next
  legal action. `scripts/check_diagnostic_store_consistency.py` fails CI when
  a lifecycle manifest's stage, `stage_id`, directory placement, or referenced
  blob disagrees with the store layout; it runs in the code-quality
  architecture-guardrails step.
- Candidate inputs now use per-run entropy and per-invocation trusted-reference
  validation. The candidate process does not receive the nonce or expected
  outputs. Publication runs additionally use the networkless, capability-free,
  private-IPC Docker boundary described in the evaluator contract.
- The focused SOLAR comparison completed all 82 path analyses for 41 unique
  workloads. Coverage is complete on both extraction paths: 27 workloads match
  with legitimate dialect/decomposition differences and 14 remain different
  because of normalization differences. Review found no external-reference-I/O
  defect and no remaining resource-model-bug classification; raw
  fusion/intermediate and mandatory-work differences are classified as dialect,
  decomposition, or normalization effects rather than silently treated as
  equal. The ignored report is
  `data/outputs/solar-cross-path-focus-cycle2-c84869e/path-comparison.json`,
  SHA-256
  `7329cf39ad86937fd19a88a9b9ee39c9597ebf33ff244a2d660588c341765b60`.
  Four additional Torchview failures remain outside this denominator because
  their backward references cannot be represented by the forward-only
  extractor.
- Real-device evidence is strongest on one RX 9060 XT/gfx1200 host. Multi-GPU
  isolation and CDNA-family behavior have contract/unit coverage but narrower
  empirical coverage.
- The RGA `planned` registry route has been removed. The deliberate static
  endpoint is the bounded `llvm-objdump`/`readelf`/`roc-objdump`, ELF-note, and
  pinned ISA coverage documented by Static Kernel Evidence.

## Active backlog

### P0 — Establish an immutable diagnostic data lifecycle and automated flow

The repository has strong artifact-level integrity but no unified lifecycle
control plane. Individual designs, corpora, calibration profiles, inference
profiles, acceptance results, and publication projections are typed and
content-addressed. The repository does not yet have one machine-readable object
that records which immutable generation is current, which stage produced it,
its complete parent chain, whether it may still be mutated, its retention
class, or the only legal next stage. Directory names, file existence, manual
command order, and this handoff still carry too much operational state.

The current effective flow is:

```text
preregister
  -> prepare
  -> SOLAR / GPU collect
  -> freeze corpus
  -> promote prior development plus revealed held-out
  -> fit inference
  -> collect and freeze fresh held-out
  -> acceptance
  -> publication projection
  -> deterministic archive
  -> GitHub Release
```

The publication end of this flow is now governed: it fully verifies the source
tree, projects only reproducible model inputs, refits inference, requires exact
semantic equivalence, writes an exact inventory, and keeps process and release
directories disjoint. The preceding stages remain a manually orchestrated
script pipeline, and archive creation, checksum publication, and GitHub Release
upload remain manual. The existing `RDNA4 Hardware` workflow performs a
different job: it writes 30-day Actions artifacts under read-only repository
permissions and is not a durable diagnostic publication workflow.

#### Lifecycle gaps that must close before fresh Cycle 3 held-out collection

1. **Frozen generations are not completely immutable.** The corpus authoring
   script guards one `collect --force` path after held-out freeze, but the
   normal interface still represents confirmed overwrite as valid behavior.
   `solar --force`, `repair-static-identity`, and a subsequent `freeze` can
   replace artifacts or corpus declarations within the same filesystem
   generation. A frozen held-out generation must never be modified. Any
   recollection, repair, source-policy change, or identity correction must
   create a new `collection_run_id` and `corpus_snapshot_id`; the prior
   generation remains immutable and becomes `superseded`.
2. **Stage completion is inferred from file existence.** The lifecycle
   orchestrator now closes this gap at the chain level: `status` and `resume`
   re-verify a typed receipt, every input identity, and the exact output
   inventory, and re-execute any stage whose receipt is missing or whose
   inputs or outputs drifted. Within the corpus authoring script, per-case
   evidence is still adopted through the collection-run handler rather than
   typed per-case receipts; the chain-level contract is authoritative.
3. **Promotion is content-addressed but the on-disk corpus is path-coupled.**
   The `promote` command now imports artifacts into the content-addressed blob
   store and emits blob-backed v7 references, so new promotions extend no
   historical path tree. The pre-migration on-disk Cycle 3 corpus
   (`data/outputs/promoted-development-cycle3.json`) is v6 and path-rebased;
   it is archival-only until the real Cycle 3 work re-promotes it, at which
   point the two roughly 22 GB roots become governed GC candidates.
4. **There is no monotonic lifecycle registry.** Introduce one current
   `DiagnosticLifecycleManifest` family whose immutable objects form the chain
   `design_id -> collection_run_id -> corpus_snapshot_id -> model_build_id ->
   acceptance_id -> publication_id -> release_id`. Each object must bind its
   parent digests, source revision, producer version, policy hashes, GPU and
   software identity when applicable, stage status, exact inventory, and
   retention class. Human aliases such as `cycle3` may point to an ID but may
   not define identity.
5. **There is no lifecycle orchestrator.** Now delivered: one resumable DAG
   entry point, `diagnostics lifecycle run/status/resume`, owns dependencies,
   attempts, bounded retries, typed stage receipts, legal monotonic
   transitions, and an atomic per-generation run-state object. Low-level
   stages remain independently testable; the real GPU collection stays
   operator-run and is adopted by the orchestrator's collection handler.
6. **There is no executable retention or garbage-collection policy.** Now
   delivered: `diagnostics lifecycle gc` operates only from registry
   reachability (every lifecycle manifest, run-state object, and typed
   receipt), defaults to a dry-run plan that explains every retained and
   reclaimable blob with its retention class and reason, and refuses any
   deletion of a blob reachable from a frozen snapshot, acceptance,
   publication, or release. `--delete` re-verifies reachability immediately
   before removal and refuses the entire operation if a planned blob became
   reachable. The ignored output tree still mixes governed evidence,
   superseded releases, debug experiments, caches, and temporary probes;
   retiring the audited v3/v6 and Orojenesis roots still requires an explicit,
   reviewed GC run over the resolved targets.
7. **Packaging and publication are only partially automated.** Add a governed
   packager that emits the archive, checksum, and release attestation from one
   verified publication manifest. A separate GitHub-hosted release job may
   create a draft release after tag/revision and checksum verification. The
   self-hosted GPU runner must remain a collection producer and must not receive
   durable `contents: write` release authority.
8. **Operational state is duplicated in prose.** Corpus hashes, current stage,
   archive size, and the next legal action must come from the lifecycle
   registry and a generated status command. `HANDSOFF.md` should retain human
   decisions, external blockers, risks, and authorization points rather than
   serve as the run database.

#### Target storage and retention model

Use immutable manifests over a replaceable storage backend. A local first
implementation may use:

```text
data/store/blobs/sha256/<digest>
data/store/runs/<collection_run_id>/
data/store/snapshots/<corpus_snapshot_id>/manifest.json
data/store/publication-registry/<publication_id>/manifest.json
data/store/releases/<release_id>/manifest.json
```

The blob key, not a mutable path, is the durable identity. The same contracts
must later support an object-store backend without changing corpus or release
semantics. Assign every object one closed retention class:

- **cache**: reproducible and unreferenced; deletable at any time;
- **debug**: bounded short retention and never admissible as governed input;
- **process evidence**: hot while its generation is active, then cold after a
  successor is accepted and a grace period expires;
- **frozen source evidence**: retained while reachable from a governed corpus,
  model, acceptance, or unreleased publication;
- **publication/release**: retained durably with its external archive digest
  and release attestation.

ROCPD databases and nested Orojenesis output must not return to the compact
GitHub artifact, but they must not be treated as disposable merely because the
projection omits them. They move from hot process storage to cold source-audit
storage until registry policy proves them unreachable and past retention.

#### Flows and data that should be retired

Retire the following behaviors after callers are migrated:

- using `cycleN` directories or filenames as the primary identity;
- overwriting any frozen held-out generation, even behind a confirmation flag;
- using `repair-static-identity` as a normal post-freeze lifecycle stage;
- treating an existing output filename as proof that a stage completed;
- manually maintaining the preregister/prepare/collect/freeze/fit/accept chain;
- manually creating tar archives and copying their hashes into prose;
- keeping `HANDSOFF.md` as the only record of current run state;
- the unreferenced standalone
  `scripts/internal/rdna4/verify_rdna4_diagnostic_acceptance.py` wrapper after
  migration to the stronger production acceptance authoring/verifier path
  (**retired: removed in commit d93406f2's follow-up; the production
  `diagnostics accept-performance-model` path and
  `verify_diagnostic_acceptance` are authoritative, and stale `scripts/*.pyc`
  caches were cleaned**).

The local ignored data audit measured `data/outputs/` at approximately 40.8 GB.
The following retirement candidates were reclaimed on 2026-08-07 per the
reviewed plan `docs/ops/retirement-plan-2026-08-07.md` (explicit approval
recorded; dry-run was 17.57 GB over nine targets, all registry-unreachable;
v3/v6 source evidence plus the non-canonical calibration/evidence files were
cold-archived to `data/cold-archive/superseded-source-evidence-2026-08-07.tar.zst`
and byte-verified before deletion). `data/outputs/` is now approximately
24 GB:

- **reclaimed**: `microarchitecture-diagnostics-v3/` (8.0 GB),
  `microarchitecture-diagnostics-v6/` (6.6 GB),
  `orojenesis-reproducible-9d17c17/` (1.4 GB), `data/calibration/`,
  `data/local-evidence/`, and `p0-release-*` attempts other than
  `p0-release-36e44fb/` (~695 MB);
- still present (not in scope): reproducible caches, counter probes, smoke
  output, and directories named as debug/fix experiments that remain
  unreferenced; future candidates are re-measured with
  `sol-execbench --format json diagnostics lifecycle retirement-plan`.

The retired v3/v6 roots and the unreferenced Orojenesis root alone account for
roughly 16 GB. Do not delete `microarchitecture-diagnostics-v7/` or
`microarchitecture-diagnostics-v7-cycle2/` yet: the promoted Cycle 3 source
corpus still reaches them by path. First import their reachable artifacts into
the blob store, emit and verify a replacement corpus snapshot, and prove that
the old paths have no registry references. After a diagnostic publication is
uploaded and round-trip verified, its local expanded directory is also staging
rather than a second permanent copy; retain the durable archive/release object
according to policy.

No data was removed by this audit. The reviewed retirement plan now exists:
`docs/ops/retirement-plan-2026-08-07.md` supplies the exact dry-run inventory,
byte totals (17.57 GB across nine targets), reachability proof (all targets
are registry-unreachable; the store holds only one test-root design manifest
and an empty blob store), cold-archive decision, and the explicit approval
checklist. The inventory is regenerable through the lifecycle toolchain:
`sol-execbench --format json diagnostics lifecycle retirement-plan` (or the
thin `scripts/plan_diagnostic_retirement.py` wrapper), which is audit-only and
never deletes or moves data. Deletion of the resolved targets still requires
explicit approval per the plan's checklist, including a cold copy of the v3/v6
source evidence first.

#### Implementation order and completion criteria

1. Remove every same-generation mutation path for frozen held-out data and add
   tests proving that recollection or repair requires a new generation ID.
   **(complete: Phases 0-3)**
2. Define the lifecycle manifest, stage receipt, retention enum, and legal
   monotonic transitions in production code. **(complete: Phases 0-3)**
3. Introduce the local SHA-256 blob store and migrate promotion to immutable
   blob references without compatibility re-exports or multi-version readers.
   **(complete: Phase 4)**
4. Add lifecycle `run`, `status`, and `resume` orchestration; make status
   verification-based rather than existence-based. **(complete: Phase 5)**
5. Add registry-driven `gc --dry-run`, then retire only the explicitly
   unreachable legacy/debug/cache roots. **(complete: Phase 6; the reviewed
   retirement plan at `docs/ops/retirement-plan-2026-08-07.md` was approved
   and executed on 2026-08-07 — the nine targets (17.57 GB) were reclaimed
   after a byte-verified cold archive of the v3/v6 source evidence, with the
   inventory regenerable via
   `sol-execbench --format json diagnostics lifecycle retirement-plan`)**
6. Add governed archive/checksum/attestation creation and a least-privilege
   draft GitHub Release workflow. **(complete: Phases 1-2)**
7. Generate current-cycle status from the registry and remove duplicated
   hashes and one-time run snapshots from this handoff once Git history retains
   them. **(complete: Phase 7; `diagnostics lifecycle status` emits
   registry-generated status.json, the Cycle 3 hash enumerations are replaced
   by registry pointers, and the store consistency gate runs in CI)**

Completion requires a fresh diagnostic generation to move from preregistration
through release using one immutable lineage, with an interrupted run resuming
only after receipt verification, a frozen held-out overwrite failing
unconditionally, promotion independent of historical physical paths, release
creation consuming only a verified publication object, and GC proving the
reachability and retention decision for every candidate before deletion.

Authoritative surfaces:

- `src/sol_execbench/core/bench/performance_model/`
- `src/sol_execbench/core/integrity/`
- `scripts/internal/rdna4/build_rdna4_diagnostic_corpora.py`
- `src/sol_execbench/cli/commands/diagnostics.py`
- `.github/workflows/rdna4-hardware.yml`
- `docs/performance-diagnostics.md`

### P1 — Complete Cycle 3 held-out collection and acceptance

The 880-case development corpus, start-220 pair-disjoint universe, current-policy
inference profile, and action thresholds are frozen. Do not tune against either
revealed held-out set or reuse either prior inference/acceptance artifact. The
next governed step is to collect and freeze the fresh 220 held-out cases without
reading partial results, then run the same acceptance gates. Changing gates,
excluding the 24 exposed Cycle 2 cases, or reusing held-out pairs is not an
acceptable fix.

Cycle 2 GPU collection itself is complete: every one of the eleven families has
20 content-addressed SOLAR manifests and 20 performance-evidence manifests.
The working-set/traffic feature asymmetry is fixed in production with a focused
regression test, but that post-reveal code change invalidates the Cycle 2 frozen
identity.

Cycle 3 CPU gates are closed. The governed `promote` stage now accepts the two
source corpora beneath one explicit common `--root`, verifies every source
artifact against its original corpus root, imports each cited artifact into the
immutable lifecycle blob store, and emits the 880-case development corpus with
blob-backed v7 references. The existing `fit-performance-inference` command is
the prediction gate: it rebuilds all 880 cases, fails on any unavailable
hardware prediction, and writes the versioned inference profile bound to the
corpus, calibration, audit, and model-policy hashes. A separate
prediction-preflight command is neither required nor used.
The governed publication stage additionally verifies the complete source tree,
then projects model inputs into a separate `data/publications/` tree, omits raw
ROCPD and nested Orojenesis artifacts, sanitizes private static-evidence paths,
and proves inference equivalence by refitting from the projected corpus. Its
exact-inventory verifier is the distribution gate; the two 22 GB process roots
are not GitHub release inputs.

The frozen calibration is reusable only on its exact recorded identity: RX 9060
XT/gfx1200 GPU `a3ff7590-0000-1000-800f-a29c1cca1511` at BDF
`0000:03:00.0`, ROCm 7.2.0, compiler identity
`HIP version: 7.2.26015-fc0010cf6a`, locked clocks,
and `stable_peak` power. A different collection identity requires a new governed
calibration and a new inference fit before held-out collection. Do not collect
or inspect the preregistered start-220 held-out cases until those frozen inputs
match the collection host.

Completion requires all eleven families to meet at least 90% empirical interval
coverage, median APE at most 15%, and P90 APE at most 30%. At least one
code-changing action metric must exist; every enabled action must have at least
10 held-out positives, at least 90% precision, and at least 70% recall. Agent
feedback may enable a code-changing action only after rebuilding and verifying
every cited source.

Authoritative surfaces:

- `docs/performance-diagnostics.md`
- `src/sol_execbench/core/bench/performance_model/`
- `scripts/internal/rdna4/build_rdna4_diagnostic_corpora.py`
- `scripts/internal/rdna4/run_rdna4_diagnostic_calibration.py`

### P1 — Expand empirical hardware and isolation coverage

The executable tests and exact skip boundaries are now present. The remaining
hardware queue is:

- Multi-GPU isolation: run
  `test_real_multi_gpu_candidate_device_switch_is_rejected` with at least two
  visible ROCm GPUs. `HIP_VISIBLE_DEVICES=0` or any equivalent single-device
  restriction does not satisfy the prerequisite.
- Historical gfx942 timeouts: reacquire the archived dataset and source context
  for revision `d56fadca`, then rerun on exact gfx942. The unresolved cases are
  `FlashInfer-Bench/014_gqa_paged_prefill_causal_h32_kv4_d128_ps1` (one),
  `FlashInfer-Bench/019_mla_paged_prefill_causal_h16_ckv512_kpe64_ps1` (three),
  `L2/040_altup_predict_correction_cycle_backward` (one), and
  `L2/055_audio_encoder_conv_positional_layer_stack` (one). They are historical
  external-evidence debt, not members of the current 43/163 corpus; do not
  resurrect a retired runner against the current manifest.
- CDNA4 NVFP4/MXFP4 adaptation remains blocked on representative hardware and
  must not be claimed from fallback or dequantized execution.

Completion evidence must identify the exact GPU, ROCm/PyTorch stack, test set,
and skipped prerequisites; generic schema support is insufficient.

### P2 — Resolve the compressed code-object metadata boundary

Static AMDGPU metadata extraction currently scans bounded gzip/zlib variants of
clang-offload-bundler Compressed Code Object Bundles (CCOBs). It does not parse
the CCOB manifest, even though the implementation describes full manifest
parsing as a documented follow-up.

Either implement bounded CCOB manifest parsing with exact target selection,
decompression limits, malformed-input handling, and real compressed-bundle
fixtures, or explicitly classify full CCOB parsing as unsupported and remove
the follow-up claim. Heuristic embedded-zlib scanning must not be described as
complete CCOB coverage.

Authoritative surfaces:

- `src/sol_execbench/core/bench/static_kernel/amdgpu_metadata.py`
- `tests/sol_execbench/core/bench/test_amdgpu_metadata.py`
- `docs/user/static_kernel_evidence.md`

## Invariants

- Performance diagnostics never change canonical Trace timing, `T_SOL`, SOL
  Score, leaderboard values, or rewards.
- Canonical execution precedes profiler replay. Profiler duration, achieved
  throughput, and the measured candidate runtime never become prediction
  features.
- Evidence identity, schema version, calibration range, and artifact hashes
  fail closed. Old-schema compatibility readers are not allowed.
- `L` remains unavailable without an explicitly supplied trusted frontier.
- Partial or ungoverned diagnostics cannot request kernel code changes.
- Tuning and parameter-estimation samples cannot enter held-out acceptance.
- Mutable process evidence stays under ignored `data/outputs/`; immutable
  diagnostic release projections stay under ignored `data/publications/`.
  Neither is committed to Git.
- Current `sol_execbench.*` schema identifiers are defined only in
  `src/sol_execbench/core/integrity/schema_versions.py`; current SOLAR string
  and numeric artifact versions are defined only in
  `src/solar/schema_versions.py`.

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
uv run python scripts/check_python_reuse.py
uv run pytest tests/
git diff --check
```

Hardware claims additionally require the precisely marked ROCm tests on the
named device. Never treat a skip as passing hardware evidence.
