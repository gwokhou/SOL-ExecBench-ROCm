# Project handoff and active follow-ups

Last audited: 2026-08-05; CPU readiness updated through current HEAD.

This file is the single backlog for repository-level work that remains useful.
Completed investigations, superseded contracts, acceptance attempts, and
one-time readiness snapshots belong in Git history. Public documentation must
describe the current contract rather than duplicate this backlog.

## Current state

- The checked-in AKA manifest contains 45 authored problems: 43 scored, one
  compatibility sentinel, and one target-incompatible problem. The official
  `rx9060xt-gfx1200-reference-v2` score remains unavailable because its
  publisher-authored release evidence has not been published.
- The current performance-diagnostic family is v7, with BenchmarkConfig v2,
  reference IPC v2, and ROCm event timing v4. There are no compatibility
  readers for the superseded timing or diagnostic schemas.
- The last governed gfx1200 v7 cycle completed calibration, 440 development
  cases, frozen inference, and 220 pair-disjoint held-out cases. Its acceptance
  result was deliberately rejected, not incomplete: median APE 3.303951%, P90
  APE 15.186039%, `composite_graph` coverage 0.70, and `concurrent_graph`
  coverage 0.85. The required per-family coverage is 0.90.
- That local cycle is no longer current admission authority. Its recorded
  `prediction.py` and `inference.py` policy hashes differ from the current tree
  after later contract/helper consolidation. The ignored artifacts remain
  useful only as governed source evidence for a newly built cycle.
- Candidate inputs now use per-run entropy and per-invocation trusted-reference
  validation. The candidate process does not receive the nonce or expected
  outputs. Publication runs additionally use the networkless, capability-free,
  private-IPC Docker boundary described in the evaluator contract.
- The current focused SOLAR readiness set is 41/41 on both extraction paths.
  The last repository-owned accounting comparison covers only the earlier 32
  dual-ready workloads. Four additional Torchview failures are backward
  references whose gradient dependencies cannot be represented by the
  forward-only Torchview extractor.
- Real-device evidence is strongest on one RX 9060 XT/gfx1200 host. Multi-GPU
  isolation and CDNA-family behavior have contract/unit coverage but narrower
  empirical coverage.
- The RGA `planned` registry route has been removed. The deliberate static
  endpoint is the bounded `llvm-objdump`/`readelf`/`roc-objdump`, ELF-note, and
  pinned ISA coverage documented by Static Kernel Evidence.

## Active backlog

### P0 — Publish the official gfx1200 v2 scoring baseline

The formula, verifier, release builder, corpus pin, and baseline identity exist,
but the manifest correctly fails closed with
`baseline_v2_release_evidence_pending`.

CPU preparation is closed: the release builder deterministically validates and
plans the exact 43-problem/163-workload denominator. The remaining work is the
GPU measurement and publisher-authorized release operation; a locally built
plan or bundle is not publication evidence.

Completion requires a publisher-authored, content-addressed baseline bundle for
the exact 43-problem/163-workload scored denominator, locked-clock measurement
evidence, pinned SOLAR manifests, publisher release statements, and successful
verification through the documented release-scoring workflow. Local diagnostic
or validation sidecars cannot authorize this score.

Authoritative surfaces:

- `problems/AMD_AKA/manifest.yaml`
- `src/sol_execbench/core/scoring/`
- `docs/SCORING-V3.md`
- `docs/user/RELEASE-SCORING.md`

### P1 — Start a new governed performance-diagnostic cycle

Do not tune against the revealed v7 held-out set or reuse its frozen inference
and acceptance artifacts. Rebuild diagnostics from the source evidence under
the current policy identity, promote the revealed cases to development,
preregister a fresh pair-disjoint held-out universe, freeze the model and action
thresholds before reading it, then recollect and run the same acceptance gates.
The immediate modeling gaps are the under-covered `composite_graph` and
`concurrent_graph` families; changing gates or reusing held-out pairs is not an
acceptable fix.

CPU preparation is closed. The authoring command now preregisters an explicit
fresh universe, regenerates every later stage from the typed frozen design,
loads all eleven templates from package resources, emits a versioned CPU
preflight artifact, and content-verifies promotion of the prior development
then held-out corpora. The documented cycle-2 start is 160, disjoint from the
prior 100--159 universe.

Remaining work requires governed GPU time: collect fresh SOLAR and performance
evidence, freeze the current-policy model before reading the new held-out set,
then run acceptance. The prior rejected acceptance remains evidence, not a
model or threshold input.

Completion requires all eleven families to meet at least 90% empirical interval
coverage, median APE at most 15%, P90 APE at most 30%, and every enabled action
to meet its support, precision, and recall gates. Agent feedback may enable a
code-changing action only after rebuilding and verifying every cited source.

Authoritative surfaces:

- `docs/performance-diagnostics.md`
- `src/sol_execbench/core/bench/performance_model/`
- `scripts/internal/rdna4/build_rdna4_diagnostic_corpora.py`
- `scripts/internal/rdna4/run_rdna4_diagnostic_calibration.py`

### P1 — Refresh the dual-path SOLAR accounting comparison

Regenerate both extraction roots for the current 41-workload dual-ready set and
run `sol-execbench solar compare-paths`. Publish a repository-owned comparison
whose coverage denominator is 41, with verified artifact hashes and conversion
attestations. Do not generalize the older 32-workload accounting result.

The four backward-reference Torchview cases are an explicit extraction boundary,
not part of this 41-workload refresh. Supporting them would require a separately
approved backward-graph contract rather than weakening the fail-closed result.

CPU preparation is closed. The focused runner's `--check` mode validates the
exact eight scored problems and 41 unique workloads.
The same runner fixes both IR paths, supports per-workload resume, requires all
82 worker results to be formal publications, and invokes the hash-verifying
comparator only after full success. Remaining work is the 82-analysis GPU run
and publication of its generated comparison.

Authoritative surfaces:

- `docs/user/CROSS-PATH-COMPARISON.md`
- `src/sol_execbench/core/solar_bridge/path_comparison.py`
- `src/sol_execbench/core/solar_bridge/corpus_readiness.py`

### P1 — Expand empirical hardware and isolation coverage

The executable tests and exact skip boundaries are now present. The remaining
hardware queue is:

- Multi-GPU isolation: run
  `test_real_multi_gpu_candidate_device_switch_is_rejected` with at least two
  visible ROCm GPUs. `HIP_VISIBLE_DEVICES=0` or any equivalent single-device
  restriction does not satisfy the prerequisite.
- Native constructor defense: run
  `test_native_constructor_elapsed_time_patch_is_detected` with a visible ROCm
  GPU, ROCm development headers, and the real HIP/PyTorch extension toolchain.
  The test links a `.hip` translation unit and restores the sealed method after
  the adversarial constructor check.
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
- Generated evidence under `data/outputs/` remains ignored and uncommitted.
- Current schema identifiers are defined only in
  `src/sol_execbench/core/integrity/schema_versions.py`.

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
uv run pytest tests/
git diff --check
```

Hardware claims additionally require the precisely marked ROCm tests on the
named device. Never treat a skip as passing hardware evidence.
