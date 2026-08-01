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
completed on the real gfx1200 device.

```text
data/outputs/microarchitecture-diagnostics-v6/calibration/
  gfx1200-diagnostic-v6.json
  gfx1200-diagnostic-v6.audit.json
```

```text
profile 063f3759ec542442d82e50ff9b29635aaf27955527022e9f20a1be6c1bf6a092
audit   42e889052772ce90c771d385e7441a5f00eb53436436c6f8e76d5852741f0ffb
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

### 3. Collect development

Collect 20 point-fit and 20 conformal cases per family, 440 total. Work in
recoverable `family x phase x 20` batches and validate each batch
immediately. The pilot no longer blocks collection: all eleven families
collect end-to-end on the re-frozen design.

GPU collection must remain serial. SOLAR may use bounded parallelism only
after memory behavior is verified.

Freeze development and fit inference:

```bash
uv run sol-execbench --format json diagnostics fit-performance-inference \
  --development-corpus DEVELOPMENT.json \
  --calibration-profile CALIBRATION.json \
  --output INFERENCE.json
```

Record the inference SHA256. Held-out results must not change calibration,
features, conformal policy, or action thresholds. A required change invalidates
the held-out run.

### 4. Collect held-out and accept

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

### 5. Close the Agent loop

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
