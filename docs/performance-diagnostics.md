# gfx1200 performance diagnostics

Performance diagnostics are diagnostic-only. They never change canonical Trace
timing, `T_SOL`, SOL Score, leaderboard values, or rewards.

## Collect one governed evidence bundle

Counter mode accepts exactly one workload and requires persisted static
evidence:

```bash
sol-execbench --format json evaluate PROBLEM_DIR \
  --solution SOLUTION.json \
  --workload-uuid WORKLOAD_UUID \
  --profile rocprofv3-counters \
  --static-evidence auto \
  --trace-output TRACE.jsonl
```

The unprofiled canonical run executes first. Only after it succeeds does
rocprofv3 replay that workload in fail-safe counter passes. Replay stdout and
timing never become canonical timing. Controlled same-process/same-GPU
multi-queue replay may establish overlap topology; profiler timestamp deltas
never become predicted durations.

Every selected counter group must first pass an exact
`rocprofv3-avail -d 0 pmc-check ...` invocation. A rejected or timed-out group
fails closed before replay. The combined command/output digest is stored in
the counter provenance alongside availability, profiler, configuration, and
application hashes.

Alongside the normal Trace/profile/static artifacts, the command writes:

- `TRACE.jsonl.performance-timing.json`: the exact canonical trial/iteration
  samples and a deterministic 10,000-replicate hierarchical-bootstrap interval.
- `TRACE.jsonl.performance-access.json`: canonical-input-bound locality and
  collision summaries for INT32/INT64 index tensors; raw indices are omitted.
- `TRACE.jsonl.performance-replay.json`: exact input hash, process executable,
  10-warmup/5-evidence ROCTx markers, cache policy, pre/post AMD SMI telemetry,
  and cross-pass dispatch sequence identity.
- `TRACE.jsonl.performance-evidence.json`: a root manifest binding definition,
  workload, solution, compile command/compiler, code objects, GPU/ROCm/clock
  identity, timing, static ISA, counter CSV, ROCPD, and counter provenance by
  SHA-256.

HIP/C++ candidates with inspectable code objects can produce complete hardware
diagnostics. Other candidate forms remain explicitly partial.

ROCm 7.2 containers set `ROCPROF_TMPDIR=/tmp`. This avoids the upstream
rocprofv3 ring-buffer failure caused by constructing profiler temporary paths
from an unsuitable container working directory.

## Freeze inference and build the v7 diagnostic

Development and held-out corpora contain only labels and content-addressed
evidence/SOLAR references. They cannot contain supplied predictions. For each
supported family, development contains at least 20 point-fit cases followed by
20 independent conformal-calibration cases per family (440 development cases
across eleven families).
Held-out contains at least 20 cases per family (220 total). All three phases
must come from a bounded, collection-time preregistered and stratified shape
universe, and their workload/candidate pair IDs must be disjoint.

Freeze that universe before preparing problem templates or collecting any
case. The concrete start at 160 below records the completed Cycle 2 design; it
is not the next fresh universe:

```bash
uv run python scripts/internal/rdna4/build_rdna4_diagnostic_corpora.py \
  preregister --root CORPUS_ROOT --universe-start 160

uv run python scripts/internal/rdna4/build_rdna4_diagnostic_corpora.py \
  prepare --root CORPUS_ROOT

uv run python scripts/internal/rdna4/preflight_rdna4_diagnostic_corpus.py \
  --corpus-root CORPUS_ROOT --output CORPUS_ROOT/preflight.json
```

The universe start is explicit only at preregistration. `prepare`, `solar`,
`collect`, and `freeze` regenerate their case set from that exact typed frozen
design; an existing mismatched design is never overwritten. `prepare` loads
the eleven definitions and HIP sources from installed package resources, so it
does not depend on an ignored smoke directory. The versioned preflight output
validates all 660 authored workloads and produces the deterministic 33-batch
collection plan without accessing a GPU.

For the completed Cycle 2 preparation, the previous development and held-out
corpora lived beneath one root and were promoted in that exact order:

```bash
uv run python scripts/internal/rdna4/build_rdna4_diagnostic_corpora.py \
  promote --root PREVIOUS_CORPUS_ROOT \
  --source-corpus PREVIOUS_CORPUS_ROOT/development.json \
  --source-corpus PREVIOUS_CORPUS_ROOT/held_out.json \
  --output PREVIOUS_CORPUS_ROOT/promoted-development-cycle2.json
```

Promotion does not copy large evidence. The command requires both source corpus
files and every referenced evidence/SOLAR artifact to remain beneath one
explicit common `--root`. It validates each reference against its source corpus
directory, verifies SHA-256 values, rebases references under the common root,
preserves development-before-held-out ordering, and refuses to overwrite an
existing output. The new preregistered universe is then reserved for fresh
held-out collection; it is not read while fitting from the promoted development
corpus.

### Cycle 3 and exploratory-repair boundary

Cycle 2 revealed its 220 held-out pairs and was input-invalid after a
working-set-coordinate model-policy fix. Cycle 3 must therefore combine the
existing 660-case promoted development corpus with those 220 revealed pairs,
for 880 development cases. The historical fresh universe beginning at 220 was
then frozen, but its collection stopped after 181 successful held-out cases:
the next transformer workload was `1032x768`, beyond the packaged candidate's
sequence limit of 1024. That design is immutable failure evidence and must not
be repaired, resumed, accepted, or published.

Create the combined corpus from the common ignored evidence root; do not copy
paths into a hand-authored corpus:

```bash
uv run python scripts/internal/rdna4/build_rdna4_diagnostic_corpora.py \
  promote --root data/outputs \
  --source-corpus \
  data/outputs/microarchitecture-diagnostics-v7/preregistered-corpus/promoted-development-cycle2.json \
  --source-corpus \
  data/outputs/microarchitecture-diagnostics-v7-cycle2/preregistered-corpus/held_out.json \
  --output data/outputs/promoted-development-cycle3.json
```

Promotion imports every cited artifact into the immutable lifecycle blob store
(`data/store/blobs/sha256/<digest>` by default; override with the
`SOL_EXECBENCH_DIAGNOSTIC_STORE` environment variable) and emits
blob-backed corpus references, so the promoted corpus depends on no historical
physical path tree. The old source roots stay readable until a governed GC
proves them unreachable.

The first zero-GPU repair draft at start 280 was executable and pair-disjoint,
but used a synthetic odd-length transformer sweep. It was superseded before any
GPU execution because executable-domain validity alone does not establish
real-model representativeness.

The replacement exploratory design begins at 340. Its fixed hidden width 768
matches GPT-2's `n_embd=768` and BERT-Base's 768-wide encoder. Its explicit
sequence neighborhoods cover short contexts, BERT's 512 boundary, GPT-2's 1024
boundary, and vision-token anchors around 197/257/577. Exact previously revealed
shapes are excluded; adjacent values preserve boundary sensitivity without
claiming a new held-out pair. The anchors derive from the official
[GPT-2 implementation](https://github.com/openai/gpt-2/blob/master/src/model.py),
[BERT implementation](https://github.com/google-research/bert/blob/master/modeling.py),
and [Google ViT repository](https://github.com/google-research/vision_transformer).

Materialize this repair only in an isolated exploratory store and run all three
qualification gates before any counter collection:

```bash
export SOL_EXECBENCH_DIAGNOSTIC_STORE=\
data/outputs/microarchitecture-diagnostics-v7-cycle3-exploratory-realistic/store
uv run python scripts/internal/rdna4/build_rdna4_diagnostic_corpora.py \
  preregister \
  --root \
  data/outputs/microarchitecture-diagnostics-v7-cycle3-exploratory-realistic/corpus \
  --universe-start 340
uv run python scripts/internal/rdna4/build_rdna4_diagnostic_corpora.py \
  prepare \
  --root \
  data/outputs/microarchitecture-diagnostics-v7-cycle3-exploratory-realistic/corpus
uv run python scripts/internal/rdna4/build_rdna4_diagnostic_corpora.py \
  qualify-static \
  --root \
  data/outputs/microarchitecture-diagnostics-v7-cycle3-exploratory-realistic/corpus \
  --qualification-root \
  data/outputs/microarchitecture-diagnostics-v7-cycle3-exploratory-realistic/qualification-v3-smoke
uv run python scripts/internal/rdna4/build_rdna4_diagnostic_corpora.py \
  qualify-canary --role held_out \
  --root \
  data/outputs/microarchitecture-diagnostics-v7-cycle3-exploratory-realistic/corpus \
  --qualification-root \
  data/outputs/microarchitecture-diagnostics-v7-cycle3-exploratory-realistic/qualification-v3-smoke
uv run python scripts/internal/rdna4/build_rdna4_diagnostic_corpora.py \
  qualify-full --role held_out \
  --root \
  data/outputs/microarchitecture-diagnostics-v7-cycle3-exploratory-realistic/corpus \
  --qualification-root \
  data/outputs/microarchitecture-diagnostics-v7-cycle3-exploratory-realistic/qualification-v3-smoke
```

`qualify-static` is zero-GPU. Canary and full qualification use the canonical
evaluator with a minimal non-authoritative timing configuration and no profiler.
Every receipt binds the design, prepared problems, evaluator traces, collector,
configuration, and parent gate. `collect` refuses to start its first profiler
case unless the complete chain verifies. Combining 181 historical observations
with 39 replacement observations is exploratory synthesis only; it cannot
produce a frozen corpus, inference fit, acceptance verdict, or publication.
The current real canary contains 34 `PASSED` gfx1200/container Trace rows across
all eleven families. It used the wrapper's explicit
`--allow-untested-target-smoke` route and remains non-authoritative. Full-role
qualification also passed all 220 held-out workloads, 20 per family, with the
same gfx1200/container identity. Its gate SHA-256 is
`963d4197965a3f7a01af756066f319b943fef3a145db4a663e6fef0a37048b45`.
The three gates now admit an explicitly selected exploratory counter
collection, but they do not make its output eligible for freeze, inference,
acceptance, or publication.

### Publish frozen diagnostic inputs

Process evidence and release artifacts use disjoint roots. Keep raw collection,
ROCPD databases, Orojenesis search output, and other mutable intermediates under
`data/outputs/`. Build the immutable publication tree under
`data/publications/`; the command rejects an output located in or above any
input directory and refuses to overwrite an existing tree.

```bash
uv run sol-execbench --format json diagnostics \
  build-publication-projection \
  --development-corpus data/outputs/promoted-development-cycle3.json \
  --calibration-profile \
  data/outputs/microarchitecture-diagnostics-v7/calibration/gfx1200-diagnostic-v7.json \
  --source-inference-profile \
  data/outputs/microarchitecture-diagnostics-v7-cycle3-inference.json \
  --output \
  data/publications/microarchitecture-diagnostics-v7-cycle3

uv run sol-execbench --format json diagnostics \
  verify-publication-projection \
  --manifest \
  data/publications/microarchitecture-diagnostics-v7-cycle3/publication.json
```

The builder first verifies every artifact cited by the source corpus, including
the large ROCPD and nested Orojenesis inputs. It then retains the canonical
trace, timing, access, replay, profile summary, counter CSV/provenance, compact
static evidence, and formal top-level SOLAR artifacts. ROCPD databases and
nested Orojenesis output are omitted. Static evidence is rewritten to remove
source paths, tool commands/output, warnings, and nested artifact paths. The
projected corpus is content-addressed, its inference profile is refitted, and
the build fails unless that profile is exactly equivalent to the frozen source
profile apart from the new corpus digest.

`publication.json` records the exact regular-file inventory, hashes, byte
counts, source-corpus digest, projection policy, and diagnostic-only authority.
The verifier rejects extra files, symlinks, hash drift, restored raw evidence,
case-identity drift, or a non-reproducible inference profile. It requires only
the unpacked publication tree; the 22 GB process roots are not distribution
dependencies.

For the current 880-case Cycle 3 input, the governed inventory is 74,253,001
bytes, excluding the self-describing `publication.json`. Package that archive
with the governed release packager, which re-verifies the publication, creates
the deterministic zstd archive, and writes the release attestation; it never
commits generated evidence to Git:

```bash
sol-execbench --format json diagnostics release package \
  --manifest \
  data/publications/microarchitecture-diagnostics-v7-cycle3/publication.json \
  --archive-output \
  data/publications/microarchitecture-diagnostics-v7-cycle3.tar.zst \
  --attestation-output \
  data/publications/microarchitecture-diagnostics-v7-cycle3.attestation.json \
  --source-revision <source revision>
```

After download, verify the externally published archive SHA-256 and unpack it
with `diagnostics release verify`, which re-runs
`verify-publication-projection` on the unpacked tree before you consume
`development.json`, `calibration/profile.json`, or `inference.json`. This
publication remains diagnostic-only and cannot authorize an official score or
leaderboard result. See `docs/user/diagnostic-release.md` for the full contract
and the draft-first GitHub Release workflow.

Only after those CPU gates pass and inference plus action thresholds are frozen
may the operator collect or inspect new held-out cases. A wholly new corpus is
220 cases; a governed successor to a pre-verdict failure may instead collect
only exposure- or diff-affected families under the reuse contract below. The
exact local readiness evidence and remaining GPU work are tracked in
`HANDSOFF.md`.

The current corpus contract derives each pair ID from the evidence-bound
workload SHA-256 and candidate SHA-256. Authoring re-derives that identity,
checks the declared family against the built diagnostic, and rejects reused
evidence manifests across development and held-out data. A caller-provided
`independent=true` assertion is not accepted.

```bash
sol-execbench --format json diagnostics fit-performance-inference \
  --development-corpus DEVELOPMENT.json \
  --calibration-profile gfx1200-diagnostic-calibration.json \
  --output gfx1200-diagnostic-inference.json
```

This fits the family point models only from the point-fit phase, freezes 95%
split-conformal expansion factors only from the following independent
conformal-calibration phase, and fits deterministic action thresholds before
held-out data is read. A code-changing action is enabled only with at least 10
development positives, at least 10 development negatives, at least 90%
precision, and at least 70% recall. The current reduction point model has a
separate outer-row slope for each calibrated width; unsupported widths fail
closed.

```bash
sol-execbench --format json diagnostics performance \
  --evidence-manifest TRACE.jsonl.performance-evidence.json \
  --solar-manifest SOLAR_REQUEST/manifest.yaml \
  --calibration-profile gfx1200-diagnostic-calibration.json \
  --inference-profile gfx1200-diagnostic-inference.json \
  --output TRACE.performance-diagnostic.json
```

A trusted frontier is optional:

```text
--frontier-trace FRONTIER.jsonl
```

The command only consumes manifest-bound identities; GPU/compiler/power
identity cannot be supplied manually. The SOLAR manifest must cite an eligible
analysis for the exact `definition:workload_uuid`. The current narrow admission
set is:

- contiguous FP32/BF16 elementwise graphs;
- 2D contiguous, out-of-place FP16/BF16/FP32 transpose;
- last-axis sum/mean/RMSNorm/LayerNorm with BF16/FP32 input and FP32
  accumulation;
- contiguous last-axis Softmax/LogSoftmax and 2D class-index CrossEntropy;
- single-axis gather/index-select/embedding and bounded indexed overwrite or
  FP32 atomic-add updates using trusted access summaries;
- FP16 or FP32 GEMM/BMM, including calibrated strided batches;
- bounded, exact, acyclic primitive DAGs, the preregistered MiniGPT
  FP32/C=768/8-head/S≤1024 graph, and controlled concurrent DAGs.

Unsupported semantics, missing fusion regions, hash or identity mismatch,
counter-pass misalignment, missing queue identity, or unverified overlap scope produces
`partial`/`unavailable` reason codes. No representative shape, achieved-rate,
profiler-duration, or measured-runtime fallback is permitted.

The output contract is `sol_execbench.performance_diagnostic.v7` using model
`gfx1200_diagnostic.v7`. It contains `T_pred(IR)`, `T_pred(HW)`, the canonical
measured confidence interval, optional trusted frontier, uncertainty-aware
`L/C/R`, bounded attribution, and stable action codes.

## Govern Agent feedback

Code-changing recommendations require a current diagnostic and an accepted
held-out model report:

```bash
sol-execbench --format json diagnostics agent-feedback \
  --performance-diagnostic TRACE.performance-diagnostic.json \
  --evidence-manifest TRACE.jsonl.performance-evidence.json \
  --acceptance ACCEPTANCE.json \
  --acceptance-manifest ACCEPTANCE-MANIFEST.json \
  --development-corpus DEVELOPMENT.json \
  --held-out-corpus HELD_OUT.json \
  --calibration-profile CALIBRATION.json \
  --inference-profile INFERENCE.json \
  --output TRACE.performance-agent-feedback.json
```

Partial diagnostics may only produce reprofile/model-gap actions. They cannot
request a kernel change. If `--acceptance` is omitted, or a matching report
records a failed verdict, the output is still generated but contains only
those safe actions. Supplying `--acceptance` requires all five source inputs;
the command rebuilds every held-out case from the cited corpus evidence and
rejects any measurement, prediction, action, identity, or aggregate drift.
Identity or hash mismatch is an input error.

## Calibration and acceptance

```bash
uv run python scripts/internal/rdna4/run_rdna4_diagnostic_calibration.py \
  qualify-static --output CALIBRATION.json --gpu-id GPU_UUID \
  --qualification-root data/outputs/diagnostic-calibration-qualification \
  --estimation-batches 5
uv run python scripts/internal/rdna4/run_rdna4_diagnostic_calibration.py \
  qualify-canary --output CALIBRATION.json --gpu-id GPU_UUID \
  --qualification-root data/outputs/diagnostic-calibration-qualification \
  --estimation-batches 5
uv run python scripts/internal/rdna4/run_rdna4_diagnostic_calibration.py \
  qualify-full --output CALIBRATION.json --gpu-id GPU_UUID \
  --qualification-root data/outputs/diagnostic-calibration-qualification \
  --estimation-batches 5
uv run python scripts/internal/rdna4/run_rdna4_diagnostic_calibration.py \
  run \
  --output CALIBRATION.json --gpu-id GPU_UUID \
  --qualification-root data/outputs/diagnostic-calibration-qualification \
  --estimation-batches 5

sol-execbench --format json diagnostics accept-performance-model \
  --development-corpus DEVELOPMENT.json \
  --held-out-corpus HELD_OUT.json \
  --calibration-profile CALIBRATION.json \
  --inference-profile INFERENCE.json \
  --manifest-output ACCEPTANCE-MANIFEST.json \
  --output ACCEPTANCE.json
```

Calibration uses a frozen two-phase protocol: tuning first, then at least five
fresh parameter-estimation processes. Calibration and replay audits include
stable pre/post GPU identity, clock, temperature, power, and foreign-process
observations. Each GPU observation also records the ordered PCIe path from the
CPU root port through every bridge to the GPU endpoint. Every link binds its
BDF, current/max speed, and current/max width; the narrowest negotiated link is
stored as the effective path. Missing pre/post topology makes new performance
evidence partial and blocks production calibration, while any link drift makes
the evidence inconsistent. A current host probe cannot retroactively establish
the topology of older endpoint-only evidence.

Inference authoring separately requires at least 20 point-fit and 20
conformal-calibration cases per family. Acceptance requires at least 20 held-out
cases per family (220 total), at least 90% empirical interval coverage in every
family, median absolute percentage error at most 15%, P90 at most 30%, and at
least one enabled code-changing action metric. Every enabled action requires at
least 10 held-out positives, at least 90% precision, and at least 70% recall.

### Pre-verdict exposure and case reuse

An unavailable prediction stops acceptance before a verdict. It writes no
acceptance result and releases no metric fields. The lifecycle records a typed
`precondition_failed` exposure receipt containing only the evaluated case-ID
prefix, the stopping case/family, and reason codes. This is distinct from a
completed `accepted=false` verdict. The receipt is imported into CAS and the
immutable `acceptance-exposures/<run-id>/` registry, so store consistency and GC
retain the failure boundary.

Historical attempts can be currentized without rerunning acceptance:

```bash
uv run python scripts/internal/rdna4/manage_rdna4_diagnostic_reuse.py \
  record-exposure \
  --root SOURCE_COLLECTION_ROOT \
  --attempt data/store/attempts/RUN_ID/acceptance/0001.json \
  --held-out-corpus SOURCE_COLLECTION_ROOT/held_out.json \
  --released-case-id held_out-elementwise-01 \
  --reason-code calibration_out_of_range:working_set_bytes \
  --source-revision COLLECTION_REVISION \
  --output EXPOSURE.json
```

Freshness is exposure- and impact-scoped, not age-scoped. The current policy
uses a family as the smallest statistical replacement unit: an exposed case
taints its 20-case family. A source change that affects raw collection or
derived diagnostics must list every affected family; those families are also
replaced. Unaffected families may reuse exact evidence identities even across a
version change. Any omitted diff path, reused pair in a replacement family,
hash drift, wrong family count, or unreviewed affected family fails closed.

After a separately frozen successor design has passed its required
qualification gates, collect counters and build SOLAR only for each replacement
family, then freeze a typed fragment:

```bash
uv run python scripts/internal/rdna4/manage_rdna4_diagnostic_reuse.py \
  freeze-fragment \
  --root REPLACEMENT_COLLECTION_ROOT \
  --family elementwise \
  --output REPLACEMENT_COLLECTION_ROOT/elementwise-fragment.json
```

The impact review is a sorted JSON list that exactly matches
`git diff --name-status --find-renames BASE TARGET`. Each entry records
`path`, optional `previous_path`, `change`, both impact booleans, exact
`affected_families`, and a rationale. A documentation-only path has both
booleans false and an empty family list; a change affecting only elementwise
raw collection names `affected_families: ["elementwise"]`.

Compose the final 220-case corpus without rewriting either source:

```bash
uv run python scripts/internal/rdna4/manage_rdna4_diagnostic_reuse.py \
  compose-held-out \
  --root REPLACEMENT_COLLECTION_ROOT \
  --source-corpus SOURCE_COLLECTION_ROOT/held_out.json \
  --replacement-fragment REPLACEMENT_COLLECTION_ROOT/elementwise-fragment.json \
  --exposure-receipt EXPOSURE.json \
  --impact-review IMPACT_REVIEW.json \
  --base-source-revision BASE \
  --target-source-revision TARGET \
  --replace-family elementwise \
  --output COMPOSED_COLLECTION_ROOT
```

The output bundle contains canonical copies of the source corpus, replacement
fragment, and exposure receipt, plus `held_out.json` and
`case-reuse-manifest.json`. Lifecycle plan authoring and collection adoption
reverify the whole bundle and require the selected lifecycle design to be the
exact design cited by the replacement fragment. With one tainted family,
exactly 20 cases are fresh and 200 are reused. This saves formal counter
collection; it does not waive the forward calibration decision, qualification
prerequisites, exact GPU identity, or the one-shot acceptance thresholds.

The overlap surface stores measured `resource_mix` points, not broad bins.
Prediction uses piecewise-linear interpolation only inside the measured
convex hull and for the exact calibrated concurrency count; it fails closed
outside that domain.

## Eleven-family hardware smoke

Before collecting the full validation corpora, verify one complete,
content-addressed case from each supported family. Set
`SOL_EXECBENCH_DIAGNOSTIC_SMOKE_JSON` to a root-confined configuration:

```json
{
  "schema_version": "diagnostic_smoke_test.v1",
  "calibration_profile": "calibration/gfx1200-diagnostic-v7.json",
  "cases": [
    {
      "workload_kind": "elementwise",
      "evidence_manifest": "elementwise/performance-evidence.json",
      "solar_manifest": "elementwise/solar/manifest.yaml"
    },
    {
      "workload_kind": "transpose",
      "evidence_manifest": "transpose/performance-evidence.json",
      "solar_manifest": "transpose/solar/manifest.yaml"
    },
    {
      "workload_kind": "reduction_norm",
      "evidence_manifest": "reduction/performance-evidence.json",
      "solar_manifest": "reduction/solar/manifest.yaml"
    },
    {
      "workload_kind": "matmul",
      "evidence_manifest": "matmul/performance-evidence.json",
      "solar_manifest": "matmul/solar/manifest.yaml"
    },
    {
      "workload_kind": "softmax",
      "evidence_manifest": "softmax/performance-evidence.json",
      "solar_manifest": "softmax/solar/manifest.yaml"
    },
    {
      "workload_kind": "cross_entropy",
      "evidence_manifest": "cross-entropy/performance-evidence.json",
      "solar_manifest": "cross-entropy/solar/manifest.yaml"
    },
    {
      "workload_kind": "indexed_read",
      "evidence_manifest": "indexed-read/performance-evidence.json",
      "solar_manifest": "indexed-read/solar/manifest.yaml"
    },
    {
      "workload_kind": "indexed_update",
      "evidence_manifest": "indexed-update/performance-evidence.json",
      "solar_manifest": "indexed-update/solar/manifest.yaml"
    },
    {
      "workload_kind": "composite_graph",
      "evidence_manifest": "composite/performance-evidence.json",
      "solar_manifest": "composite/solar/manifest.yaml"
    },
    {
      "workload_kind": "transformer_block",
      "evidence_manifest": "transformer/performance-evidence.json",
      "solar_manifest": "transformer/solar/manifest.yaml"
    },
    {
      "workload_kind": "concurrent_graph",
      "evidence_manifest": "concurrent/performance-evidence.json",
      "solar_manifest": "concurrent/solar/manifest.yaml"
    }
  ]
}
```

Then run:

```bash
uv run pytest \
  tests/sol_execbench/core/bench/test_rdna4_performance_diagnostics_smoke.py
```

The smoke requires available IR and HW predictions plus available `C` and `R`
for all eleven cases. A missing configuration skips the optional hardware test
and is not evidence that the smoke passed.

## Lifecycle orchestration

The lifecycle accepts one immutable, reviewable plan rather than a loose set of
stage flags:

```bash
uv run sol-execbench --format json diagnostics lifecycle plan \
  --design-id <design_id> \
  --development-snapshot-id <promoted_development_snapshot_id> \
  --collection-root <operator_collected_tree> \
  --held-out-corpus <operator_collected_tree>/held_out.json \
  --calibration-profile <calibration_profile.json> \
  --calibration-audit <calibration_profile.audit.json> \
  --output-root <lifecycle_output_root> \
  --model-version <model_version> \
  --max-attempts 3 \
  --store-root data/store \
  --output PLAN.json

uv run sol-execbench --format json diagnostics lifecycle run \
  --plan PLAN.json \
  --store-root data/store
```

`PLAN.json` uses the current `sol_execbench.diagnostic_lifecycle_plan` schema
and binds the registered design, promoted development snapshot, exact collection
tree, held-out corpus, calibration profile and audit, output root, source
revision, evidence purpose, model version, and bounded attempt count. The design,
development snapshot, and plan purposes must match. The command also verifies
both registry identities and that the selected source revision matches the
current `src/`, `scripts/`, `pyproject.toml`, and `uv.lock` state. Plan creation
is therefore a governed authoring step; do not replace it with hand-translated
legacy command flags. For production, plan authoring additionally requires one
complete PCIe-aware GPU identity shared exactly by the calibration profile,
calibration audit, and every performance-evidence manifest referenced by the
held-out corpus. That identity participates in both the plan ID and collection
run ID, and the collection handler rechecks it before adoption and on resume.

`run` executes `design -> calibration -> collection_run -> corpus_snapshot ->
model_build -> acceptance -> publication -> release` in monotonic order while
enforcing the declared multi-parent dependencies. Here `release` is the local
release candidate; externally observed publication is recorded separately by
the published-release receipt flow. The collection handler adopts an already
operator-collected evidence tree and its frozen role corpora; it does not run
GPU collection itself. CPU stages execute fitting, acceptance, projection, and
packaging.

Each verified stage has a typed receipt and immutable registry manifest. Mutable
orchestration state is stored at
`data/store/orchestrations/<collection_run_id>/run.json`, with receipts beneath
the same orchestration directory and append-only attempts under
`data/store/attempts/<collection_run_id>/`.

Status and resume are verification-based, never existence-based:

```bash
uv run sol-execbench --format json diagnostics lifecycle status \
  --run-id <collection_run_id> \
  --store-root data/store
uv run sol-execbench --format json diagnostics lifecycle resume \
  --run-id <collection_run_id> \
  --store-root data/store
```

`status` re-verifies every recorded stage through its handler and reports the
first stage that is not `verified` or `superseded`. This includes a persisted
`running` stage left by an interrupted process; it is never skipped in favor of
a later stage. `resume` re-verifies each completed stage, re-executes any stage
whose receipt is missing or whose inputs or outputs drifted, and continues from
the first incomplete stage. A run that exhausts its attempt budget on a stage
is recorded `failed` and can be resumed after the operator fixes the input.

## Registry-driven blob GC

Blob retention is decided only by registry reachability, never by directory
layout:

```bash
uv run sol-execbench --format json diagnostics lifecycle gc plan \
  --store-root data/store \
  --output data/outputs/diagnostic-gc-plan.json
```

The 24-hour plan binds the store root, registry snapshot, reachability result,
and every blob's retention decision. A blob referenced by any retained
lifecycle manifest, run-state object, typed receipt, or published-release
receipt remains reachable. Review the persisted plan before applying it:

```bash
uv run sol-execbench --format json diagnostics lifecycle gc apply \
  --plan data/outputs/diagnostic-gc-plan.json
```

`apply` derives the exact store root from the reviewed plan. It refuses an
expired plan or any registry/reachability change since planning and deletes
only the blob identities recorded by that plan.

Blob GC does not delete expanded process-evidence trees under `data/outputs/`.
Audit path-root retirement separately with:

```bash
uv run sol-execbench --format json diagnostics lifecycle retirement-plan \
  --store-root data/store
```

That command is dry-run only. Any material path deletion still requires exact
target resolution, reachability proof, review, and explicit approval. In
particular, retain the current v7 and Cycle 2 roots until governed promotion
and registry reachability prove them dead.
