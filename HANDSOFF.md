# gfx1200 Performance Diagnostics Handoff

## Current state

Handoff date: 2026-07-31.

`main` is in sync with `origin/main` at `321978f1`. The v6 semantic-range
expansion has been merged into `main` and pushed:

```text
321978f1 Enforce acyclic architecture domains
f124b1d2 Tighten schema and reuse governance
3cc00558 Expand gfx1200 performance diagnostics
13085b42 Harden performance diagnostic governance
fe95530e Implement governed gfx1200 performance diagnostics
```

The v6 expansion no longer needs a separate worktree. The leftover
`.worktrees/task4-live-test` (`codex/task4-live-test`, `806c9878`) is marked
prunable and may be removed once confirmed unneeded. Do not reset `main`.

The software implementation, full test suite, static checks, real gfx1200
calibration, and the eleven-family smoke are complete. The 660-case corpus
design was frozen before collection; on 2026-08-01 it was re-frozen once to
correct a graph-family shape universe that produced duplicate workload
UUIDs (see "Pilot findings"). Smoke was re-verified green on real gfx1200
hardware on 2026-07-31 and again after the authoring fixes on 2026-08-01; its
content hashes are recorded in "Completed evidence" below.

The remaining blocking outcome is the v6 hardware statistical acceptance:

```text
11-family smoke
-> 440 development cases
-> frozen inference
-> 220 held-out cases
-> acceptance
-> accepted Agent feedback smoke
```

Do not describe v6 as hardware-accepted until this sequence completes.

Status on 2026-08-02: **v6 is hardware-accepted.** The 440 development cases
are frozen (`development.json` SHA `f8e2b07d...`). A conformal-policy defect
exposed by the first held-out run was root-caused and fixed (see "Conformal
score collapse"), inference refitted (`inference.json` SHA `d216fe86...`), all
220 held-out cases re-collected fresh, and acceptance passed
(`acceptance.json` SHA `91e65dd7...`, `accepted: true`, every family `>= 0.90`).
The accepted `agent-feedback` smoke passed (`feedback.json` SHA `6bfc5093...`).

A follow-up reward-hacking audit of the whole harness ran on 2026-08-02; its
conclusions are recorded under "Reward-hacking audit". The live gameable
vectors it confirmed are confined to the in-process candidate-execution
boundary and the diagnostic governance layer, not the kernel SOL score.

The 33-case timing pilot ran on 2026-08-01. A first pass exposed authoring
defects in six families; the corpus was then re-preregistered (graph-family
shape universe corrected) and the authored definitions and solutions were
fixed, after which the pilot re-passed all eleven families end-to-end with
zero failures. The full 660-case run may now begin. See "Pilot findings" and
"Runtime and batching" below.

## Boundaries that must not change

- Performance diagnostics remain diagnostic-only.
- `T_SOL`, canonical Trace timing, SOL Score, leaderboard values, and rewards
  remain unchanged.
- Canonical execution precedes profiler replay.
- Profiler duration, timestamp delta, achieved throughput, and the same
  candidate's measured runtime never become prediction components.
- Timestamps may establish verified dispatch topology only.
- Evidence identity, schema version, calibration range, and artifact hashes
  fail closed.
- `L` stays unavailable without an explicitly supplied trusted frontier.
- Partial or ungoverned diagnostics cannot request kernel code changes.
- Tuning and parameter-estimation samples cannot enter development or
  held-out acceptance.
- The current hardware target is RX 9060 XT/gfx1200 with the ROCm 7.2
  compatible toolchain.
- Generated evidence under `data/outputs/` is ignored and must not be
  committed.

Current schema names and versions are canonical only in
`src/sol_execbench/core/integrity/schema_versions.py`. The performance
diagnostic model is `sol_execbench.performance_diagnostic.v6`, surrounded by
`sol_execbench.diagnostic_calibration.v6`,
`sol_execbench.diagnostic_calibration_audit.v6`,
`sol_execbench.diagnostic_inference_profile.v8`, and
`sol_execbench.agent_feedback.v6`. Do not add old-schema compatibility
readers.

Reuse `core.data.definition_models.DType` for integer indices. Deterministic
field-order JSON uses `atomic_write_json_value(..., sort_keys=False)`.

## Implemented scope

The model supports eleven validation families:

1. elementwise;
2. transpose;
3. reduction, RMSNorm, and LayerNorm;
4. FP16/FP32 GEMM and BMM;
5. Softmax and LogSoftmax;
6. class-index CrossEntropy;
7. gather, index-select, and embedding;
8. indexed overwrite and FP32 atomic add;
9. exact acyclic primitive graphs of at most 32 nodes;
10. preregistered MiniGPT FP32/C=768/8-head/S<=1024 blocks;
11. controlled concurrent DAGs.

Indexed workloads use a canonical-input-bound access sidecar containing only
de-identified INT32/INT64 locality and collision summaries. Raw indices are
prohibited.

Sequential dispatches are summed. Controlled overlap requires verified
process/GPU/lane/marker scope and uses a duration-free precedence DAG.
Timestamp distances are not prediction inputs.

The overlap calibration stores eleven measured `resource_mix` points:

```text
0.000, 0.108, 0.195, 0.327, 0.492, 0.660,
0.795, 0.886, 0.939, 0.969, 1.000
```

Prediction interpolates only between adjacent measured points for the exact
calibrated concurrency count. It does not claim wide ranges such as
`resource_mix=0.67:1`.

Semantic and hardware descriptor dispatch use `functools.singledispatch`.
Closed string operation vocabularies use mapping tables.

## Completed evidence

### Calibration

A locked 3-batch tuning plus 5-batch independent parameter-estimation run
completed on the real gfx1200 device. On 2026-08-01 the calibration was
re-run once: the `atomic_update` probe was switched to a canonical
random-index distribution (uniform-with-replacement `[0, output_count)`)
matching the indexed_update corpus semantics, and the mid-collision
`max_multiplicity` cell was widened to `3:16`. This removed a coverage gap
that otherwise left `conformal-indexed_update-06` (duplicate_fraction ~0.368,
maximum_multiplicity 9) with no available HW prediction, which failed
`fit-performance-inference` closed. The measurement coordinates are
self-consistent with the declared cells: the random probe measures
`duplicate_fraction ~ 1 - 1/e` with a multiplicity tail that grows
logarithmically with `output_count`, so no synthetic point is invented (see
"Calibration coverage gap" below).

```text
data/outputs/microarchitecture-diagnostics-v6/calibration/
  gfx1200-diagnostic-v6.json
  gfx1200-diagnostic-v6.audit.json
```

```text
profile 9a92662a0d23f9256235e889784415aaf58aa2286cb7d990c8298c8a89ab7c76
audit   dc22b590ca2704426db8613babc4184c24e99514578aa89a3bed282487e0c1fc
```

The profile strictly reloads as v6 and contains 37 scalar parameters, four
multidimensional surfaces, and eleven overlap points. The probe covers the
new indexed-read, atomic, FP32 matrix, residency, and overlap behavior.
Residency is measured rather than represented by placeholder constants.

### Corpus design

The design was frozen before any v6 corpus case was collected:

```text
data/outputs/microarchitecture-diagnostics-v6/
  preregistered-corpus/design.json
```

```text
SHA256 eee743ed7ab876f9aaea267d688a318a659454b29f75290f721d55e1700c1ed4
```

The 2026-08-01 re-registration replaced the original design
(`45cae9e0...`) because the graph-family `_shape` used only 16 distinct M
values for 20 cases per phase, duplicating 16 workload UUIDs per family
(see "Pilot findings"). The re-frozen design gives every graph-family case a
distinct M; the other eight families' case entries are byte-identical, so
their already-collected pilot evidence remains valid.

It defines 60 cases per family:

```text
point-fit development       220
conformal development       220
development total           440
held-out                    220
total                       660
```

The separate `preregister` stage never overwrites a mismatch and later stages
require the exact frozen design. The one 2026-08-01 re-registration occurred
before any blocked-family evidence existed; the design is now frozen for the
full collection.

### Verification

The current worktree passed:

```bash
uv run pytest tests/
uv run ty check
uv run python scripts/check_coupling.py
uv run python scripts/check_readability.py
uv run --with ruff ruff check .
uv run --with ruff ruff format --check .
uv run python scripts/check_current_docs.py
uv run python scripts/check_schema_versions.py
git diff --check
```

The HIP probe compiled for gfx1200 and all new modes ran on the real device.

### Eleven-family smoke

Re-verified green on real gfx1200 hardware under the current `main` HEAD on
2026-07-31, and again on 2026-08-01 after the softmax and cross_entropy
authoring fixes (which now also cover the wider corpus shapes):

```text
test_real_gfx1200_diagnostics_cover_all_supported_families  1 passed in 2.33s
```

Artifacts under `data/outputs/microarchitecture-diagnostics-v6/smoke/`
(composite, concurrent, cross_entropy, elementwise, indexed_read,
indexed_update, matmul, reduction, softmax, transformer, transpose) plus
`diagnostic-smoke.json` indexing all eleven.

```text
diagnostic-smoke.json (index)  SHA256 126debcd604b28ef57ffeeedc527797c1128d4ce1201e1b07b0d05c237ab24c6
smoke/ tree (613 files)        SHA256 3086fde8cc5c09c3e7258ac106146bf39afe2efe51d142df8aac8592ab927cb8
```

The 2026-08-01 tree hash reflects the softmax solution generalization
(width up to 1024) and the cross_entropy definition/solution/workload
alignment to the M/N axis convention.

Every family passed with `AVAILABLE` status: correct SOLAR descriptor and
family classification, current evidence, IR/HW predictions, `C` and `R`, and
scope-verified concurrent scheduling.

### 33-case timing pilot (2026-08-01)

Three held-out cases per family ran through SOLAR plus GPU collect
(`--unsafe-local-execution --lock-clocks --profile rocprofv3-counters
--static-evidence auto`), mirroring the governed corpus commands. Driver and
results are under `data/outputs/` (ignored, not committed):

```text
data/outputs/microarchitecture-diagnostics-v6/pilot/run_pilot.py
data/outputs/microarchitecture-diagnostics-v6/pilot/timing.json
```

A first pass measured five families before six were blocked by corpus
defects; after the re-preregistration and authoring fixes below, a second
pass measured the remaining six. All 33 cases resolved (18 collected fresh,
15 re-skipped as already collected) with zero failures. Merged per-family
collect wall time:

```text
family            n   P50 (s)   P90 (s)
elementwise       2   55.1      55.3
transpose         3   50.0      56.6
reduction_norm    3   45.9      48.2
matmul            3   46.2      84.3
indexed_read      3   51.7      52.6
softmax           3   43.0      43.1
cross_entropy     3   43.2      43.6
indexed_update    3   58.0      60.1
composite_graph    3   43.2      43.9
transformer_block  3   45.8      46.1
concurrent_graph  3   46.9      46.9
```

`elementwise` shows `n=2` because its first case was already collected by an
earlier interrupted attempt. SOLAR is cheap (roughly 3 s/case, 8-10 s for
transformer blocks). All eleven families collect end-to-end.

## Pilot findings: corpus defects, now resolved

The pilot is the first exercise of the full case universe, and it exposed
authoring defects that a one-case-per-family smoke cannot see. Six of eleven
families initially failed closed; all defects were in the authored corpus
(design, definition, solution, workload), not in the pilot driver or the
profiler. All were fixed on 2026-08-01 and the pilot re-passed all eleven
families with zero failures.

### 1. Graph families reused workload UUIDs (fixed by re-preregistration)

`composite_graph`, `transformer_block`, and `concurrent_graph` each had 60
cases but only 44 distinct `workload_uuid`s; 16 UUIDs were reused by two
cases within the same phase because the shape universe produced only 16
distinct M values for 20 cases per phase. This broke collect
("workload UUIDs must be unique within a problem") and SOLAR ("workload UUID
must match exactly once").

Fix: `_shape` now derives M as `32 + 8 * (global_index - UNIVERSE_START)`,
giving every graph-family case a distinct M (32..504). The corpus was
re-preregistered (design `eee743ed...`, see "Corpus design"); the affected
held-out case dirs were cleared before the re-run.

### 2. `indexed_update` output-name mismatch (fixed in the generator)

`definition.json` declares output `result`, while the generated workload
checks covered `output`, so `_validate_output_inventory` failed with
`missing=['result'], extra=['output']`. Fix: `_workload` now threads the
definition's declared output tensor name into the numeric check, so the
check covers `result`.

### 3. `softmax` solution was width-128 only (fixed in the solution)

The bundled `diagnostic_block_softmax_f32` solution hard-coded a 128-wide row
block (`TORCH_CHECK(input.size(1) == 128, ...)`). Fix: the kernel now accepts
any width up to 1024 (the corpus maximum width is 544) using a power-of-two
block with `-INFINITY`/zero padding so non-power-of-two widths reduce
correctly.

### 4. `cross_entropy` definition/solution/workload mismatches (fixed)

Three linked problems were fixed:

- The definition used axis tokens `B`/`C` while the workloads provide only
  `{M, N}`, so SOLAR died with `KeyError: 'B'`. The definition now uses
  `M`/`N` (the corpus-wide convention).
- The generated workload named the inputs `logits`/`target` and the check
  `output`; they now match the definition's `predictions`/`targets` and
  `loss`.
- The solution hard-coded `256x128` (`TORCH_CHECK(..., == 256 && == 128)`)
  and a fixed 256-thread single-block reduction. It was rewritten to one
  block per row with an atomic row-sum, supporting rows up to the corpus
  maximum (2672) and classes up to 1024.

### Working families

`elementwise`, `transpose`, `reduction_norm`, `matmul`, `indexed_read`
collected cleanly in the first pass; their design entries, problem files,
and workload UUIDs are byte-identical after the re-registration, so their
evidence remains valid. The six fixed families collected cleanly in the
second pass. All changes are in the authored corpus and the generator
(`scripts/internal/rdna4/build_rdna4_diagnostic_corpora.py`), not in `main`
runtime behavior.

## Calibration coverage gap (found at freeze/fit, fixed)

After the full 440-case development collection and freeze, the first
`fit-performance-inference` run failed closed with
`validation case lacks an available HW prediction` for exactly one case:
`conformal-indexed_update-06` reported `duplicate_fraction 0.3636` and
`maximum_multiplicity 9`.

Root cause: the indexed_update corpus draws `indices` with shape `[M]`
uniformly from `[0, M)`, so `duplicate_fraction` is fixed at
`1 - 1/e ~ 0.368` for every M while `maximum_multiplicity` grows only
logarithmically with M (measured 6.5-7.6 average, tail to 9 across the
corpus M range 13824-21248). The calibration `atomic_update` surface's
mid-collision cell covered `max_multiplicity 3:8`; multiplicity 9 only
existed in the high-collision band, whose `collision_fraction` does not
contain 0.368. Case 06 therefore landed in a coverage gap and `cell()` had no
matching cell, so the prediction was correctly refused (fail-closed).

The fix (2026-08-01) re-ran calibration with the `atomic_update` probe
rewritten to use the canonical random-index distribution (uniform
with-replacement `[0, output_count)`, matching corpus semantics), and widened
the mid-collision cell to `max_multiplicity 3:16`. The measured coordinates
are now self-consistent with the declared cells: the random probe measures
`duplicate_fraction ~ 0.368` and a multiplicity tail 8-11 at
`output_count = 1<<20`, both inside `[0.250001, 0.85] x [3, 16]`, and no
sample leaks into the `17+` cell. This is honest coverage, not a synthetic
point: the throughput measured for the random distribution (about 1.85M
item/ms) lies between the no-collision (~3.87M) and high-collision (~0.45M)
measurements, consistent with physical behavior. The new profile hash is
`9a92662a...` (see "Calibration" in "Completed evidence"). No corpus case,
design, or collected evidence changed.

## Conformal score collapse (found at held-out acceptance, fixed)

The first held-out acceptance run failed with
`family_empirical_coverage_below_90_percent`: `indexed_read` and
`composite_graph` each covered `17/20 = 0.85`, below the `>= 90%` gate. All
other gates passed (median APE 1.23%, P90 APE 14.94%, enabled-action precision
and recall 1.0). The violations were not evidence artifacts: `indexed_read`
deviations were `+2.5-3.4%` run-to-run noise (re-measuring `M=15360` gave
`123.7/123.9/131.4 us`), and `composite_graph` has a real, reproducible
grid-scheduling cliff at `gridDim == 0 (mod 128)` (`M=128/256/384` measure
`40-43 us` against `30-34 us` neighbors; the torch reference is smooth; a
2-rows-per-block variant moves the cliff to the new grid size, proving it is
gridDim-driven, not shape-noise).

Root cause: `_conformal_score` measured only the log-excess beyond the base
prediction interval and floored the result at zero. Whenever every development
conformal point lands inside the base interval, every score is `0.0`, the
frozen `q95` collapses to `0.0`, and `apply_conformal_interval` expands the
interval by `exp(q95) == 1.0` — leaving only the base band, which under-covers
on held-out data with higher residual scatter. This is the documented "zero
quantile" degenerate-calibration failure of conformal prediction.

The fix (2026-08-02) changed `_conformal_score` to the standard split-conformal
conformity score, the raw absolute log-residual
`abs(log(measured_ms / point))`, so `q95` is the `(n + 1)`-corrected order
statistic of the true point-model residual magnitude and never collapses. The
interval formula `[point * lower_ratio / factor, point * upper_ratio * factor]`
is unchanged; only the score that drives `factor` was corrected. Re-fitting
inference and re-running acceptance against the invalidated held-out evidence
as a dry run gives `accepted: true` with all eleven families `>= 90%`
(`indexed_read` and every family except `composite_graph` at `1.0`;
`composite_graph` at exactly `0.90`, its two `M == 0 (mod 128)` grid-cliff
cases remaining outside any reasonable interval). No kernel, calibration
profile, corpus design, or evidence was changed; changing the solution kernel
to dodge the grid cliff would be reward-hacking (it alters the measured object,
hides a real hardware scheduling phenomenon, and is informed by held-out
results), so it was explicitly rejected.

The conformal-policy change invalidated the held-out run per the boundary
below, so a fresh independent held-out collection was required. All 220
preregistered cases were re-collected on 2026-08-02 (zero failures, evidence
`220 == 220`, old evidence backed up under `heldout/evidence.invalidated.bak/`),
`held_out.json` refrozen, and acceptance re-run against the fixed inference
profile:

```text
held_out.json    SHA256 cb79abddf645cd6c9674...
inference.json   SHA256 d216fe86402c09082f51...  (fixed conformal scoring)
acceptance.json  SHA256 91e65dd7796b2a1bd3b0...
acceptance-manifest.json SHA256 6bee2978539ed147a524...
agent-feedback.json SHA256 6bfc5093328d203f1b8b...
```

Acceptance: **`accepted: true`**, `median APE 1.67%`, `P90 APE 15.15%`.
Per-family interval coverage: `elementwise 0.95`, `cross_entropy 0.90`,
`composite_graph 0.90`, every other family `1.00`. The two `composite_graph`
violations remain its `gridDim == 0 (mod 128)` cliff cases (`M=128/256`); an
interval wide enough to absorb a reproducible `+30%` scheduling cliff would
destroy the family's diagnostic value, so the honest outcome is exactly `0.90`
at the gate. Enabled safe action `restore_wmma_path` (20 positives, precision
and recall 1.0). The accepted `agent-feedback` smoke produced
`feedback_generated` with `enabled_performance_actions = ["restore_wmma_path"]`
and `performance_acceptance_status = "accepted"`.

References — the fix follows industry-standard split-conformal practice:

- Vovk, Gammerman, Shafer, *Algorithmic Learning in a Random World* (raw
  conformity score = absolute residual).
- Berkeley StatLearn split-conformal notes, "prediction intervals":
  `https://www.stat.berkeley.edu/~ryantibs/statlearn-s24/lectures/conformal.pdf`
  (zero-quantile collapse when calibration residuals are degenerate).
- jammi_ai `conformal.rs` (`(n + 1)` rank correction is required for the
  finite-sample guarantee, not cosmetic):
  `https://docs.rs/jammi-ai/0.32.0/src/jammi_ai/predict/conformal.rs.html`
- tsbootstrap adaptive-conformal tutorial (static quantiles fail silently under
  calibration-test drift; ACI/NexCP adapt online):
  `https://tsbootstrap.readthedocs.io/en/latest/tutorials/adaptive_drift_aci_nexcp.html`
- Microbenchmark-driven analytical GPU performance modeling (occupancy/wave
  quantization as first-class model terms):
  `https://arxiv.org/html/2605.04178v1`
- GB300 wave-quantization capstone (grid sizes at machine capacity trigger
  measurable gaps):
  `https://github.com/cfregly/ai-performance-engineering/blob/71772268/code/docs/gb300-capstone-wave-quant.md`
- Berkeley trustworthy-benchmarks audit (the scored object must be isolated
  from the scoring; do not alter the measured kernel to make results pass):
  `https://rdi.berkeley.edu/blog/trustworthy-benchmarks/`

## Reward-hacking audit (2026-08-02, findings recorded)

After the conformal fix was accepted, a systematic reward-hacking audit was run
over the whole harness. Nine lens-based finder agents (official scoring,
conformal model, acceptance policy, reward-hack detector, bench timing,
kernel/eval driver, calibration probes, collection pipeline, AKA tolerance)
produced 41 suspicions, each of which was then adversarially verified by an
independent agent reading the actual code; 31 were refuted and 10 confirmed
(4 high, 3 medium, 3 low by corrected severity). This section records the
authoritative statement of what the harness defends and where it is currently
gameable. The three previously documented reward-hacking patterns (conformal
q95 collapse, kernel-variant gaming, probe fabrication) were the seed patterns
for the hunt and are not re-listed here.

### Cleared by the audit (refuted)

- **Official SOL Score layer** (4 attacks refuted): loosened SOL bound, forged
  trace latency, forged `source_revision`, and corpus-pin timing gaps cannot
  reach the score — every scored value derives from measured mean
  latency/correctness, the scored object never touches the publisher-owned
  content-addressed bundle, and the official scorer is SHA-pinned and
  fail-closed.
- **Conformal model** (4 refuted): the point-model floor, interval clamping,
  and family re-split suspicions are not gameable — the fixed raw-residual
  score plus the measured-coverage acceptance gate make a collapsed `q95`
  harder to pass.
- **Calibration probes** (6 refuted): WMMA synthetic tiles, FP32 serial-chain
  probes, counter `x1024` rescale, indexed-address aliasing, residency
  normalization, and overlap clamping all feed only non-authoritative
  diagnostic sidecars, never the score.
- **Bench timing** (4 refuted): clock-lock mode-label check, reference-latency
  zero, GPU-contention gap, and profiler-CSV warmup are measurement-robustness
  nits with no scored-object agency.
- **AKA tolerance** (4 refuted): the 99%-matched-ratio skip, unbounded
  calibration margin, reference-self-variation, and equivalence-crosscheck
  suspicions fail on arithmetic or on the author-only boundary.

### Confirmed live gameable vectors (4 high)

All four are in the in-process untrusted-execution boundary: the eval
subprocess runs candidate code with full privileges and the guard is a narrow,
single-shot, `id()`-based static + runtime detector.

1. `driver/templates/eval_driver.py:34` — full trace forgery. `os.dup(1)`
   preserves the real stdout on fd 3; candidate code runs in-process, so
   `os.write(3, forged JSONL)` + `os._exit(0)` emits fabricated PASSED traces
   that the CLI parses as Traces with no UUID/count/provenance check — a
   candidate can score an arbitrary fabricated speedup without ever running a
   kernel.
2. `core/bench/reward_hack/runtime.py:58` — `check_monkey_patch` id-compares
   `torch.cuda.Event.elapsed_time` once before timing and is never re-run, so a
   patch installed on invocation >= 11 persists through every `elapsed_time`
   read and yields fabricated ~0.001 ms latency with correct outputs.
3. `core/bench/eval_workload_execution.py:96` — timing hardcodes
   `round_index=9`, reusing byte-identical inputs from correctness round 9, so
   a correct-but-slow kernel can memoize its round-9 output and return it
   during timing; the value-based timing validator passes because the cached
   output IS correct.
4. `core/bench/reward_hack/models.py:135` — the output-cache static detector is
   `.py`-only and name-based, so HIP/C++ sources get no output-cache review and
   a native solution can hold a process-lifetime input-hash-to-output map.

### Confirmed integrity gaps (3 medium, 3 low)

- `core/bench/performance_model/acceptance.py:235` (medium) — zero enabled
  code-changing actions yields an empty `action_metrics` set that trivially
  satisfies the validator, so a model with no held-out action-quality evidence
  is certified accepted (vacuous action policy).
- `core/bench/performance_model/acceptance.py:217` (medium) — acceptance scores
  self-declared manifest fields; the verifier never re-derives `measured_ms`
  from the cited evidence and `manifest_sha256` is a self-checksum (a live
  demonstration with `measured_ms == predicted_ms` yields `accepted: true`
  indistinguishable from genuine).
- `scripts/internal/aka_author_seed.py:3298` (medium) — the MoE topk routing
  check admits a materially different expert set on up to 5% of tokens
  (`max_mismatch_ratio = 0.05` is the live materialized config; `weight_atol`
  is effectively a no-op; `float32_contract=False` skips the static check).
- `core/bench/performance_model/acceptance.py:170` (low) — the acceptance-result
  validator checks shape not substance; coverage/action metrics are trusted
  fields never recomputed from cited cases.
- `scripts/internal/rdna4/build_rdna4_diagnostic_corpora.py:446` (low) —
  `--force` deletes and re-collects held-out evidence with no re-run guard,
  enabling rerun-until-acceptance (requires the corpus operator to act against
  its own trust gate).
- `core/bench/reward_hack/runtime.py:302` (low) — the integrity snapshot stores
  only `id()` integers with no strong reference, so CPython address reuse can
  defeat function-identity checks.

Blast radius note: the acceptance/governance findings (medium/low) gate only the
diagnostic agent-feedback flow, not kernel SOL scores.

### Known blindspots (open follow-ups)

The audit explicitly did not cover: the `.so` dlopen path (`build_ext.py`
constructor/static-init runs after the single pre-timing check, re-opening
finding 2 for HIP/C++), CLI orchestration robustness (orphaned children writing
stdout, staging/symlink/safetensors), the remaining output-check semantic
families beyond elementwise and MoE routing, in-process device/stream switching,
seed/input determinism as a precomputable-answer surface, downstream reachability
of the acceptance gates, a full sweep of committed workload.jsonl tolerance
parameters, and the diagnostic prediction evidence chain end-to-end. A follow-up
blindspot audit ran against all of these on 2026-08-02; its findings are
recorded under "Blindspot follow-up audit" below.

### Blindspot follow-up audit (2026-08-02, findings recorded)

The follow-up ran nine focused lens agents over the blindspot list above and
adversarially verified 38 suspicions: 20 confirmed, 18 refuted. **The verdict
materially changes the first audit's picture**: every one of the first audit's
four confirmed high vectors is widened or re-opened, and the two most-relied-upon
runtime checks are themselves defeatable.

The single most important non-result: **acceptance-reachability confirms the
accepted verdict has exactly one in-repo consumer (the agent-feedback sidecar)
and no in-repo scored/action-application path** — the first audit's medium/low
acceptance findings do NOT escalate, and the primary SOL-score authority remains
canonical trace JSONL.

New confirmed vectors (grouped):

- **`runtime.py:40` dlopen-b1** (high) — `_ELAPSED_TIME_ADDR` is a plain mutable
  module global captured before `load_user_function` (eval_driver.py:146 snapshot
  precedes :154 dlopen); a C++ `__attribute__((constructor))` can replace
  `torch.cuda.Event.elapsed_time` and rewrite the baseline id, permanently
  neutralizing `check_monkey_patch` for every workload. Re-opens first-audit
  vector 2 for HIP/C++ deterministically.
- **`runtime.py:322` static-b2** (high) — the integrity snapshot is a plain
  mutable dict in `__main__` globals, so a candidate reaching
  `sys.modules['__main__']` can patch `timing.time_runnable` and rewrite the
  snapshot entry to the patched id; preflight/per-iteration/post checks all pass
  patched-vs-patched while `_resolve_timer` reads the fake — forged latency on a
  genuinely correct kernel.
- **Static-review BLOCK corpus bypassable five independent ways** (high) —
  `static-b1` import-root (`__builtins__.__import__`), `static-b6` marshal
  bytecode via `import types`/`types.FunctionType`, `static-b7` getattr/
  `os.__dict__`/`os.__getattribute__` restoring `os.system`/`os.popen`,
  `static-b8` `os.write/dup/fdopen/open` + `mmap`/`fcntl` absent from every rule
  set (the exact trace-forgery primitives), and `seed-seed3` value-match
  precompute invisible to the `.py`-only output-cache regex. Process spawn, file
  I/O, native loading, and arbitrary bytecode exec are all reachable.
- **`staging.py:59` cli-c1** (high) — candidate sources are staged last with no
  collision guard and the SourceFile validator rejects only absolute paths and
  `..`, so a source named `reference_definition.json`/`workload.jsonl`/`config.json`
  silently overwrites the trusted files and the candidate sets its own ground
  truth and near-zero reference latency.
- **`cli/evaluation/runtime.py:176` cli-c2** (high) — the eval subprocess fails
  only on `returncode != 0 AND empty stdout`; a crashed/early-exiting child that
  emitted any parseable trace is scored as full success (returncode stored, never
  read; exit code derived purely from `all_traces_passed`). A 2/10 partial run
  reports `2/2 passed`.
- **`core/process/subprocesses.py:249` cli-c3** (high) — a 5s post-leader-exit
  drain window keeps parsing the stdout pipe; a forked child inheriting the
  preserved `_real_stdout_fd` can inject forged traces after the leader computed
  them, decoupling what ran from what is scored.
- **`eval_driver.py:106` device-b3** (high, medium confidence) — the GPU is never
  pinned (`SOL_EXECBENCH_DEVICE` passthrough, no `torch.cuda.set_device`), and
  `torch.cuda.synchronize()` syncs only the current device; on a 2-GPU host a
  `set_device(1)` candidate is timed as ~0 on idle device 0 while its correct
  device-1 output passes validation. Not empirically reproduced on the
  single-GPU gfx1200 target.
- **`memory_pool.py:158` seed-seed2** (high) — `ShiftingMemoryPoolAllocator`
  shifts `data_ptr` but re-presents byte-identical VALUES ~50x per trial, so a
  value-keyed (torch.equal/checksum) cache collapses latency inside a single
  trial with no round detection and survives per-trial randomization fixes.
- **`custom_inputs.py:86` seed-seed1** (medium) — the input seed is fully
  deterministic and reproducible across runs (`definition.name`/`workload.uuid`/
  `row_index`/fixed seed/`round_index=9`, no per-run entropy), enabling offline
  precompute-and-lookup kernels.
- **`correctness.py:179` correctness-b1 / artifacts-b4** (medium) —
  `required_matched_ratio=0.99` with `max_error_cap=None` on all 344 committed
  numeric checks lets a DPS kernel skip one whole output slice (127/128 = 0.9922)
  and still be graded CORRECT.
- **`validation_corpus.py:109` prediction-b2** (medium, author-only) — held-out
  disjointness depends on the author-declared `candidate_sha256`, never
  re-derived from solution bytes.
- **`cli/commands/diagnostics.py:328` prediction-b3** (medium, author-only) —
  `_acceptance_admission` trusts a self-authored `DiagnosticAcceptanceResult`
  JSON; `manifest_sha256` is never verified in the agent-feedback path.

Notable refuted suspicions (raise confidence in the confirmed set): stream
relocation and hidden-stream timing collapse, `functools.cache` cross-phase
output caching, atexit/subinterpreter exec, signal-based mid-loop patch,
`NORMALIZED_MAX` slice-gaming, vacuous AKA calibration, the fp8 sentinel cast,
and timing-protocol self-declaration.

Remaining blindspots after this pass: no live multi-GPU ROCm host to test
device-b3; no end-to-end .so compile+dlopen adversarial test on real torch; the
static-obfuscation family is open-ended (the five bypasses are a representative
sample, not a closed set); the external autonomous-agent consumer of the
agent-feedback sidecar is outside this repo; and the eval-path trace-count guard
remains absent (only the separate release scoring path has it).

Recommended remediation priority: (1) make `_ELAPSED_TIME_ADDR` and the
integrity snapshot immutable/guarded, or re-capture after dlopen; (2) close the
five static-bypass families and add an HIP/C++ content gate; (3) collision-guard
staging and gate returncode/workload count in the evaluate path; (4) pin the
device before candidate code runs.

### P0/P1 remediation applied (2026-08-02)

The in-process reward-hacking vectors above were hardened. No canonical timing,
calibration, corpus, inference, acceptance, or agent-feedback artifact changed,
so the frozen v6 evidence hashes remain valid; the GPU corpus does not need
re-collection (legitimate diagnostic kernels never touch the blocked
primitives). Verification: `ty check`, `ruff check`, `ruff format --check`,
`check_coupling`, `check_readability`, `check_schema_versions`,
`check_current_docs`, `git diff --check`, and the CPU test sweep all pass.

- **dlopen-b1 / static-b2 / runtime.py:302 (P0):** the `id()`-in-a-mutable-
  global pattern is replaced by a shared sealed base `_Sealed` and two
  immutable holders — `_SealedReference` (for `torch.cuda.Event.elapsed_time`)
  and `_IntegritySnapshot` (for the eval call graph). They keep **strong
  references** and compare with identity (`is`), defeating both CPython address
  reuse and the `__main__` rewrite of the stored reference. The eval driver now
  calls `verify_timing_function_intact()` immediately after candidate
  `load_user_function` so a native `__attribute__((constructor))` patch is
  caught before any workload is timed.
- **eval_driver.py:34 / cli-c2 / cli-c3 (P0/P1):** the evaluate path now
  enforces an `expected_trace_count` + clean-exit (`returncode == 0`) guard
  (`EvaluationRuntimeIncomplete`), closing trace forgery, partial-run scoring
  (2/10 → 2/2), and post-leader-exit drain injection. The in-process write/exit
  primitives a forger needs are blocked statically (below).
- **static-b1/b6/b7/b8 + HIP/C++ gate (P1):** the AST and regex review now block
  `os.write/dup/dup2/dup3/fdopen/open/_exit/abort`, `import mmap`/`fcntl`,
  `types.FunctionType/CodeType/MethodType`, `__builtins__`, `os.__dict__` /
  `os.__getattribute__` (and `getattr(os, "__dict__")`), bare `system(`/`popen(`/`
  fork(` in HIP/C++, plus `import mmap`/`fcntl`. Legitimate `data_ptr<>` library
  calls still pass.
- **cli-c1 / staging.py:59 (P1):** `SourceFile` rejects reserved harness
  filenames (`definition.json`, `reference_definition.json`, `workload.jsonl`,
  `solution.json`, `config.json`) at the staging root; `stage_solution_sources`
  re-checks as defense in depth. A candidate can no longer overwrite trusted
  ground truth or the benchmark config.
- **seed-seed2 (P1, partial):** the value-keyed output-cache construction
  surface is closed statically — cache POPULATION writes (`_cache[key] = out`) are
  now flagged in addition to initialization, and the unbounded
  `functools.cache`/`cached_property` decorators join `lru_cache`. Residual: the
  within-trial identical input values that a compute-once-per-value cache could
  still collapse cannot be varied without breaking the single-input/expected
  validator or the canonical-timing boundary; the full fix (reference-per-
  iteration validation) is deferred as boundary-constrained.

Open follow-ups unchanged: device-b3 (pin device; needs a multi-GPU host to
reproduce), the static-obfuscation tail (the blocked families are a
representative sample, not a closed set), and the eval-path residual above.

## Remaining work

### 1. ✅ Eleven-case hardware smoke (done)

One problem/solution pair per family is authored and staged under
`data/outputs/microarchitecture-diagnostics-v6/smoke/` (composite, concurrent,
cross_entropy, elementwise, indexed_read, indexed_update, matmul, reduction,
softmax, transformer, transpose), with `diagnostic-smoke.json` indexing all
eleven. Re-run on real gfx1200 hardware under the current `main` HEAD passed
green on 2026-07-31; hashes recorded in "Completed evidence".

Special requirements still apply to the existing artifacts:

- MiniGPT (transformer) must expose FP32/C=768/8-head/S<=1024 semantics.
- Concurrent cases must emit controlled lane identity and marker scope.
- Indexed cases must exercise trusted summaries without retaining raw indices.

### 2. ✅ Pass the eleven-family hardware smoke (done)

For every case require:

- correct SOLAR descriptor and family classification;
- current timing/access/replay/static/counter evidence;
- available IR and HW predictions;
- available `C` and `R`;
- scope-verified concurrent scheduling;
- no profiler-duration or achieved-rate dependency.

All eleven passed on 2026-07-31; the long collection may begin.

### 3. ✅ Collect development (done)

All 440 development cases (20 point-fit + 20 conformal per family) were
collected on 2026-08-01 through the resumable driver under
`data/outputs/microarchitecture-diagnostics-v6/development/` (ignored). Every
case resolved with `available` evidence whose `workload_uuid` matches the
frozen design; the softmax kernel was re-collected after a shared-memory race
fix (an unsynchronized `values[0]` read that could corrupt a row's softmax
normalization nondeterministically). `status.jsonl` shows 440/440 resolved
with zero active failures.

Freeze development and fit inference completed:

```text
development.json  SHA256 f8e2b07d6091c9021d8b99e7264d93a1f691ec0244abdba787d9a9ddf824b4fa
inference.json    SHA256 a3ebd554fefcfec121d6559405a984e148f1ac55b0f03ce450fcdc30325ab164
```

`fit-performance-inference` (against the re-run calibration
`9a92662a...`) fitted 11 conformal intervals over all 440 development cases;
`indexed_update` is covered now that its mid-collision multiplicity tail is
inside the calibration surface. Held-out results must not change calibration,
features, conformal policy, or action thresholds. A required change invalidates
the held-out run.

### 4. ✅ Collect held-out and accept (done)

Collect 20 pair-disjoint cases per family, 220 total. Reject any pair reused
from development and any tuning, parameter-estimation, or conformal sample.

```bash
uv run sol-execbench --format json diagnostics accept-performance-model \
  --development-corpus DEVELOPMENT.json \
  --held-out-corpus HELD_OUT.json \
  --calibration-profile CALIBRATION.json \
  --inference-profile INFERENCE.json \
  --manifest-output ACCEPTANCE-MANIFEST.json \
  --output ACCEPTANCE.json
```

Required gates:

```text
median absolute percentage error <= 15%
P90 absolute percentage error    <= 30%
per-family interval coverage     >= 90%
enabled action precision         >= 90%
enabled action recall            >= 70%
```

Every enabled code-changing action also needs at least ten held-out positives.
If `reduce_atomic_contention` or `restore_fused_attention_path` should be
enabled, add explicit positive/negative candidate variants and gold labels.

Do not weaken a gate to make a run pass. Fix the model or evidence, refreeze,
and collect a new independent held-out set.

Accepted 2026-08-02 with the fixed conformal scoring (see "Conformal score
collapse"): `accepted: true`, `case_count 220`, `median APE 1.67%`,
`P90 APE 15.15%`; family coverage `elementwise 0.95`, `cross_entropy 0.90`,
`composite_graph 0.90`, all others `1.00`; enabled safe action
`restore_wmma_path` (`positive_support 20`, precision `1.0`, recall `1.0`).

### 5. ✅ Close the Agent loop (done)

Only an acceptance result for the exact calibration and inference hashes may
authorize code-changing feedback.

```bash
uv run sol-execbench --format json diagnostics agent-feedback \
  --performance-diagnostic DIAGNOSTIC.json \
  --evidence-manifest EVIDENCE.json \
  --acceptance ACCEPTANCE.json \
  --output FEEDBACK.json
```

Record the smoke, development, held-out, inference, acceptance, and feedback
hashes plus every family/action statistic in this file.

Final v6 hardware-acceptance hashes and statistics (2026-08-02):

```text
development.json          f8e2b07d6091c9021d8b99e7264d93a1f691ec0244abdba787d9a9ddf824b4fa
calibration profile       9a92662a0d23f9256235e889784415aaf58aa2286cb7d990c8298c8a89ab7c76
inference.json            d216fe86402c09082f51d38b1bb8ba3c9b997c0d57830b0f34d6e376786d461dc
held_out.json             cb79abddf645cd6c9674eaa2d85c90e3e4fcf52a1170a57f4f44167a0f9958090
acceptance-manifest.json  6bee2978539ed147a52474b50b61af8b638c5c46f7e6b0ffc9d5337a01721236
acceptance.json           91e65dd7796b2a1bd3b04a2093f28b1ba6ea37beb15ce70e962e35866bb21cf8
agent-feedback.json       6bfc5093328d203f1b8b17c22c1e34f431eb8dcccec6e59bfc29ee2a540ea564
```

```text
accepted: true          case_count: 220
median APE: 1.67%       P90 APE: 15.15%
family coverage: elementwise 0.95, cross_entropy 0.90, composite_graph 0.90,
                 all other eight families 1.00
enabled safe action: restore_wmma_path (positive_support 20, precision 1.0,
                     recall 1.0)
agent-feedback: feedback_generated, enabled_performance_actions
                ["restore_wmma_path"], performance_acceptance_status accepted
```

## Runtime and batching

GPU evidence collection is the dominant cost. The 2026-08-01 pilot measured
per-family collect wall time for all eleven families (two-pass merged table
in "Completed evidence"). Applying `sum(family P50 x 60)` /
`sum(family P90 x 60)` to the full table:

```text
development (440 cases):     expected ~5.9 h,  conservative ~6.5 h
held-out    (220 cases):     expected ~2.9 h,  conservative ~3.2 h
full corpus (660 cases):     expected ~8.8 h,  conservative ~9.7 h
```

SOLAR analysis adds roughly 3 s/case (8-10 s for transformer blocks), about
0.5 hour. These are pilot-based projections from three cases per family, not
full measurements; re-measure during the early batches of the full run.

Before the full run:

0. run the CPU-only corpus and resume preflight before occupying the GPU;

   ```bash
   uv run python scripts/internal/rdna4/preflight_rdna4_diagnostic_corpus.py \
     --corpus-root data/outputs/microarchitecture-diagnostics-v6/preregistered-corpus
   ```

1. complete the eleven-case smoke (done, hashes recorded and re-verified
   after the authoring fixes);
2. run a 33-case pilot, three cases per family (done, all eleven families
   collect end-to-end);
3. record per-family P50 and P90 wall time (done, all eleven families);
4. estimate per the formulas above (full eleven-family estimate above);

   ```text
   expected = sum(family P50 x 60)
   conservative = sum(family P90 x 60)
   ```

Use 33 recoverable batches: eleven families x three phases x twenty cases.
Retry only the failed bounded batch. Do not run concurrent GPU collectors.

Recommended order:

```text
prepare templates
-> 11-case smoke
-> development SOLAR
-> development GPU collection
-> freeze development
-> fit and freeze inference
-> held-out SOLAR
-> held-out GPU collection
-> freeze held-out
-> acceptance
-> Agent feedback smoke
-> record final hashes and statistics
```

## Key locations

```text
docs/performance-diagnostics.md
    user workflow and current contract

src/sol_execbench/core/bench/performance_model/
    contracts, prediction, calibration, access, scheduling, and acceptance

src/sol_execbench/core/solar_bridge/performance.py
    validated semantic boundary

scripts/internal/rdna4/run_rdna4_diagnostic_calibration.py
    locked two-phase calibration

scripts/internal/rdna4/build_rdna4_diagnostic_corpora.py
    preregistration and resumable corpus authoring

tests/sol_execbench/core/bench/test_rdna4_performance_diagnostics_smoke.py
    eleven-family hardware smoke
```

Deferred scope is limited to arbitrary Transformer/control-flow graphs,
non-FP32 atomics, uncontrolled overlap, cross-architecture calibration, and
training-reward changes.

Before changing code, re-read `AGENTS.md` and
`/home/guohao/.codex/RTK.md`. Prefix repository shell commands with `rtk`.
