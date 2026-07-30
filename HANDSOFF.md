# gfx1200 Performance Diagnostics Handoff

## Repository state

Handoff date: 2026-07-30

Branch `main` is three DCO-signed commits ahead of `origin/main`:

```text
079fdcb5 Implement gfx1200 diagnostic feedback loop
54ef60d6 Harden performance diagnostic evidence validation
16e11690 Implement microarchitecture diagnostics
```

The upstream base is:

```text
1bd7e798 Document microarchitecture diagnostics plan
```

The worktree was clean before this handoff update. `HANDSOFF.md` is the only
intended uncommitted change after the update. The three implementation commits
have not been pushed.

The earlier SOLAR dual-path commits were rebased. Their current hashes are:

```text
b5c92aee Add fail-closed SOLAR path comparison
83c42d75 Enable fixed dual-path SOLAR analysis
501676fe Expand AKA workload and correctness contracts
```

Do not use the obsolete pre-rebase hashes previously recorded in this file.

## Current implementation

The new path is a diagnostic-only gfx1200 performance model. It does not change
canonical Trace timing, `T_SOL`, SOL Score, leaderboard values, or rewards.
The model and artifact contracts are:

```text
model: gfx1200_diagnostic.v3
diagnostic: sol_execbench.performance_diagnostic.v3
calibration: sol_execbench.diagnostic_calibration.v3
evidence manifest: sol_execbench.performance_evidence_manifest.v2
timing evidence: sol_execbench.performance_timing_evidence.v2
acceptance: sol_execbench.diagnostic_acceptance.v2
```

### Governed evidence collection

Evaluation now has an explicit single-workload counter mode:

```bash
uv run sol-execbench --format json evaluate PROBLEM_DIR \
  --solution SOLUTION.json \
  --workload-uuid WORKLOAD_UUID \
  --profile rocprofv3-counters \
  --static-evidence auto \
  --output TRACE.jsonl
```

The unprofiled canonical run happens first. Counter collection is a later
diagnostic replay and cannot replace canonical timing. The workflow writes:

```text
TRACE.jsonl.performance-timing.json
TRACE.jsonl.performance-evidence.json
```

The timing sidecar binds the exact trial/iteration samples and a deterministic
10,000-replicate hierarchical-bootstrap interval. The evidence manifest binds
the definition, workload, solution, compile command/compiler, code objects,
GPU/ROCm/clock identity, Trace, timing sidecar, static ISA, counter CSV, ROCPD,
and counter provenance by SHA-256.

The rocprofv3 counter path:

- selects counters from the versioned gfx1200 manifest;
- checks availability through `rocprofv3-avail`;
- preserves raw CSV and ROCPD evidence;
- aligns passes by workload, candidate, kernel, launch geometry, queue, and
  iteration;
- rejects missing queue identity, multi-queue execution, overlap, incomplete
  passes, counter mismatch, and candidate/code-object drift;
- never uses profiler duration or achieved throughput as a prediction input.

HIP/C++ candidates with inspectable code objects can produce complete hardware
evidence. Candidate forms without a content-bound code object remain partial.

### Prediction and attribution

The admitted semantic families are intentionally narrow:

- contiguous FP32/BF16 elementwise graphs;
- out-of-place 2D FP16/BF16/FP32 transpose;
- last-axis sum, mean, and RMSNorm with BF16/FP32 input and FP32 accumulation;
- contiguous FP16 GEMM/BMM with FP32 accumulation and output.

`T_pred(IR)` consumes verified SOLAR work and fusion regions.
`T_pred(HW)` consumes actual dispatch decomposition, ISA/resource footprint,
dynamic counters, and the compatible calibration profile. Unsupported
semantics, missing evidence, identity drift, calibration range misses, and
overlap return explicit `partial`/`unavailable` reason codes.

The diagnostic reports:

```text
L = T_frontier / T_SOL
C = T_pred(HW) / T_pred(IR)
R = T_measured / T_pred(HW)
```

`L` is unavailable unless the caller supplies a trusted frontier Trace.
The scoring baseline is never substituted for a frontier. Ratio and action
selection account for prediction intervals and canonical timing noise.
Ratios materially below one are treated as identity/model contradictions, not
as evidence that a candidate exceeded the model.

Build a diagnostic with:

```bash
uv run sol-execbench --format json diagnostics performance \
  --evidence-manifest TRACE.jsonl.performance-evidence.json \
  --solar-manifest SOLAR_REQUEST/manifest.yaml \
  --calibration-profile CALIBRATION.json \
  --output TRACE.performance-diagnostic.json
```

`--frontier-trace FRONTIER.jsonl` is optional.

### Calibration, acceptance, and Agent feedback

The calibration workflow uses a frozen two-phase protocol: tuning followed by
at least five fresh parameter-estimation processes. Its audit binds GPU UUID,
BDF, gfx target, ROCm, compiler, code object, clock/power state, frozen
configuration, and all input evidence hashes.

The independent acceptance contract requires at least twenty non-tuning
held-out cases in each of the four supported families, at least 80 cases total.
Development uses the same minimum and is pair-disjoint from held-out data. It
accepts the model only when:

```text
median absolute percentage error <= 15%
P90 absolute percentage error    <= 30%
per-family interval coverage     >= 90%
enabled action precision         >= 90%
enabled action recall            >= 70%
```

Agent feedback requires the exact accepted calibration identity and profile
hash, a current diagnostic, and its exact evidence manifest:

```bash
uv run sol-execbench --format json diagnostics agent-feedback \
  --performance-diagnostic TRACE.performance-diagnostic.json \
  --evidence-manifest TRACE.jsonl.performance-evidence.json \
  --acceptance ACCEPTANCE.json \
  --output TRACE.performance-agent-feedback.json
```

Partial or ungoverned diagnostics can request new evidence or report a model
gap, but cannot recommend a kernel code change. Stable actions cover launch
bound search, dispatch reduction, WMMA restoration, excess traffic,
coalescing, LDS/barrier pressure, missing counters, and model gaps.

## Verification completed

The following focused CPU/contract tests were rerun on the current `HEAD`
during this handoff update and passed:

```bash
uv run pytest tests/sol_execbench/core/bench/performance_model
uv run pytest \
  tests/sol_execbench/cli/commands/test_diagnostics_performance.py
uv run pytest \
  tests/sol_execbench/cli/evaluation/test_runtime.py \
  tests/sol_execbench/cli/evaluation/test_compilation.py
uv run pytest tests/sol_execbench/core/bench/test_agent_feedback.py
uv run pytest \
  tests/sol_execbench/core/bench/test_rdna4_performance_model_acceptance.py
uv run pytest tests/sol_execbench/cli/sidecars/test_profile.py
uv run pytest tests/sol_execbench/core/bench/test_staged_evaluation.py
```

A governed gfx1200 calibration was collected on the RX 9060 XT and strict
current-schema reload succeeded:

```text
data/outputs/gfx1200-diagnostic-v3.json
data/outputs/gfx1200-diagnostic-v3.audit.json
```

The audit binds GPU UUID/BDF, ROCm/compiler/code-object identity, STABLE_PEAK
pre/post state, temperature, and foreign-process observations. The repository
counter orchestrator was also run against the real device. Its four independent
passes selected `SQ_WAVES_sum`, memory traffic, cache hit/miss, and LDS conflict
percentage; all passes produced CSV and ROCPD artifacts, and the orchestrator
reported complete coverage.

ROCm 7.2 has an upstream rocprofv3 ring-buffer defect when its temporary path is
derived from an unsuitable container working directory. The runtime now sets
`ROCPROF_TMPDIR` to a writable controlled directory. With that setting, the
repository parser aligned all 168 probe dispatches across the four real passes.

No independent 20-per-family development and held-out corpora are present under
`data/`, so no accepted held-out artifact was produced. Until those 160
content-addressed cases are collected and pass the frozen gates, the model is
hardware-calibrated and counter-validated but not hardware-accepted.

The following static and quality gates were also rerun and passed:

```bash
uv run --no-sync ty check
uv run --no-sync python scripts/check_coupling.py
uv run --no-sync python scripts/check_readability.py
uv run --no-sync python scripts/check_production_reachability.py
uv run --no-sync python scripts/check_current_docs.py
uv run --with ruff ruff check .
uv run --with ruff ruff format --check .
git diff --check
```

The schema-version gate and the full `tests/` suite now pass. Package coverage
also passes the repository's line and branch policy.

## Next work

### P0: Collect and pass independent held-out acceptance

Prepare at least twenty independent cases for each of elementwise, transpose,
reduction/norm, and matmul. For every case:

1. Produce a canonical single-workload Trace and governed counter evidence.
2. Produce the exact eligible SOLAR manifest.
3. Build the v3 diagnostic against the frozen calibration and inference policy.
4. Record only labels and content-addressed evidence references in the public
   corpus; the authoring command derives predictions and measured timing.
5. Prove that the workload/candidate was not used for tuning or parameter
   estimation.

Then run:

```bash
uv run sol-execbench --format json diagnostics accept-performance-model \
  --development-corpus DEVELOPMENT.json \
  --held-out-corpus HELD_OUT.json \
  --calibration-profile CALIBRATION.json \
  --inference-profile INFERENCE.json \
  --manifest-output ACCEPTANCE-MANIFEST.json \
  --output ACCEPTANCE.json
```

Do not weaken the 15% median, 30% P90, coverage, independence, or attribution
requirements to make the first run pass. Investigate model or evidence defects,
repeat calibration when justified, refreeze, and collect a new independent
acceptance set.

Only an accepted result for the exact calibration profile authorizes
code-changing Agent feedback.

### P1: Run full repository gates

After hardware-facing fixes, run:

```bash
uv run pytest tests/
uv run --no-sync ty check
uv run --no-sync python scripts/check_coupling.py
uv run --no-sync python scripts/check_readability.py
uv run --no-sync python scripts/check_production_reachability.py
uv run --no-sync python scripts/check_current_docs.py
uv run --no-sync python scripts/check_schema_versions.py
uv run --with ruff ruff check .
uv run --with ruff ruff format --check .
git diff --check
```

Do not raise readability, coupling, or quality baselines.

### P1: Complete the SOLAR release evidence

The diagnostic work does not unblock official scoring. Current policy remains:

```yaml
official_scoring:
  status: unavailable
  baseline_id: rx9060xt-gfx1200-reference-v2
  reason_code: baseline_v2_release_evidence_pending
```

The previous development audit reported MakeFX readiness at 163/163 and
Torchview readiness at 159/163. The four Torchview failures are the explicit
backward references in `instruction2triton/rmsnorm_bwd`; a forward-only trace
is not a valid fix. These results were not rerun on the diagnostic `HEAD` and
must not be presented as release evidence.

The repository-owned comparison covers 32 dual-ready workloads and reports
agreement in external identity, model I/O, mandatory work, limiting resource,
and formal bound. Internal fusion/intermediate accounting differs because of
the two dialect decompositions. The nine later Torchview coverage fixes have
not been added to a reviewed 41-workload comparison.

Rerun the complete audits and comparison on the exact release source:

```bash
uv run sol-execbench solar corpus-audit \
  /tmp/solar-corpus-audit-makefx-release \
  --backend make_fx_aten \
  --device cuda:0

uv run sol-execbench solar corpus-audit \
  /tmp/solar-corpus-audit-torchview-release \
  --backend torchview_extended_einsum \
  --device cuda:0

uv run sol-execbench solar compare-paths \
  /tmp/solar-corpus-audit-makefx-release \
  /tmp/solar-corpus-audit-torchview-release \
  --output /tmp/solar-path-comparison-release.json
```

Follow `docs/user/RELEASE-SCORING.md` for reproducible Orojenesis, canonical
baseline, SOLAR release, statement construction, bundle assembly, and official
verification. Do not change the policy until reviewed content-addressed release
evidence exists.

## Important invariants

- Performance diagnostics remain diagnostic-only.
- Canonical execution happens before profiler replay.
- Canonical timing comes only from the unprofiled Trace.
- Profiler duration, achieved rate, and the candidate's measured runtime never
  enter `T_pred(IR)` or `T_pred(HW)`.
- Evidence identity and hashes fail closed; do not add guessed fallbacks.
- Multi-queue execution and overlapping dispatches remain unsupported until a
  reviewed overlap model exists.
- An unavailable frontier keeps `L` unavailable.
- Partial diagnostics cannot request kernel code changes.
- Tuning or parameter-estimation samples cannot enter held-out acceptance.
- Keep `torchview_extended_einsum` as the default SOLAR path.
- Do not restore `--extractor`, automatic path fallback, mixed release roots, or
  the retired extended-einsum MakeFX conversion.
- Unknown SOLAR operations and unclassified resource work remain errors.
- Do not commit GPU evidence, benchmark outputs, downloaded data, kernels,
  tokens, or proprietary inputs.
- GPU conclusions require a bounded host retry when the sandbox hides required
  devices or runtime resources.
- Use DCO signing for later commits:

  ```bash
  git commit -s -m "Imperative summary"
  ```

## Key locations

```text
docs/performance-diagnostics.md
    user workflow and current diagnostic contract

microarchitecture_diagnostics_plan.md
    scope, design decisions, and deferred work

src/sol_execbench/core/bench/performance_model/
    contracts, prediction, attribution, calibration, acceptance, governance,
    timing evidence, and evidence manifest

src/sol_execbench/core/bench/rocm_profiler/
    counter discovery, collection, parsing, and pass alignment

src/sol_execbench/cli/sidecars/performance.py
    evaluation-time timing/evidence sidecar construction

src/sol_execbench/core/solar_bridge/performance.py
    validated SOLAR-to-diagnostic boundary

src/sol_execbench/cli/commands/diagnostics.py
    performance diagnostic and governed Agent-feedback commands

scripts/internal/rdna4/run_rdna4_diagnostic_calibration.py
scripts/internal/rdna4/verify_rdna4_diagnostic_acceptance.py
    host calibration and held-out acceptance entry points

src/sol_execbench/data/rocprofv3_counters/gfx1200_v1.yaml
    versioned gfx1200 counter groups

src/sol_execbench/data/hardware_calibration_probes/diagnostic_microarchitecture.hip
    packaged calibration probe source

docs/user/RELEASE-SCORING.md
docs/user/CROSS-PATH-COMPARISON.md
    outstanding SOLAR publication workflow and comparison contract
```

Before changing code, re-read `AGENTS.md` and
`/home/guohao/.codex/RTK.md`. Prefix repository shell commands with `rtk`.
