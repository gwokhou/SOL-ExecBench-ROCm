# gfx1200 Performance Diagnostics Handoff

## Current state

Handoff date: 2026-07-31.

`main` is two commits ahead of `origin/main`:

```text
13085b42 Harden performance diagnostic governance
fe95530e Implement governed gfx1200 performance diagnostics
```

The worktree contains the active v6 semantic-range expansion. Do not discard
or reset it.

The software implementation, full test suite, static checks, real gfx1200
calibration, and immutable 660-case corpus design are complete.

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
`src/sol_execbench/core/integrity/schema_versions.py`. The model is
`gfx1200_diagnostic.v6`; do not add old-schema compatibility readers.

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
SHA256 45cae9e06a4c247e452f9ec5401701b4979a1bec47e20bb0818443ba8d06cd5c
```

It defines 60 cases per family:

```text
point-fit development       220
conformal development       220
development total           440
held-out                    220
total                       660
```

The separate `preregister` stage is immutable. Later stages require the exact
design and never overwrite a mismatch.

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

## Remaining work

### 1. Prepare eleven smoke cases

Create one runnable problem/solution pair per family. The likely missing
templates are Softmax, CrossEntropy, indexed read/update, composite, MiniGPT,
and concurrent DAG.

Special requirements:

- MiniGPT must expose FP32/C=768/8-head/S<=1024 semantics.
- Concurrent cases must emit controlled lane identity and marker scope.
- Indexed cases must exercise trusted summaries without retaining raw indices.

### 2. Pass the eleven-family hardware smoke

For every case require:

- correct SOLAR descriptor and family classification;
- current timing/access/replay/static/counter evidence;
- available IR and HW predictions;
- available `C` and `R`;
- scope-verified concurrent scheduling;
- no profiler-duration or achieved-rate dependency.

Do not begin the long collection until all eleven pass.

### 3. Collect development

Collect 20 point-fit and 20 conformal cases per family, 440 total. Work in
recoverable `family x phase x 20` batches and validate each batch immediately.

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

GPU evidence collection is the dominant cost:

```text
estimated wall time ~= per-case P50 x 660
```

Illustrative totals:

```text
2 minutes/case   about 22 hours
5 minutes/case   about 55 hours
10 minutes/case  about 110 hours
```

Development is approximately 15-73 hours and held-out 7-37 hours under those
assumptions. The other potentially expensive stage is 660-case SOLAR analysis,
especially MiniGPT and composite graphs.

Before the full run:

1. complete the eleven-case smoke;
2. run a 33-case pilot, three cases per family;
3. record per-family P50 and P90 wall time;
4. estimate:

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
