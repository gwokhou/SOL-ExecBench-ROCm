# Project handoff and active follow-ups

Last audited: 2026-08-06; CPU and local-evidence readiness updated through
current HEAD.

This file is the single backlog for repository-level work that remains useful.
Completed investigations, superseded contracts, acceptance attempts, and
one-time readiness snapshots belong in Git history. Public documentation must
describe the current contract rather than duplicate this backlog.

## Current state

- The checked-in AKA manifest contains 45 authored problems: 43 scored, one
  compatibility sentinel, and one target-incompatible problem. The official
  `rx9060xt-gfx1200-reference-v2` score remains unavailable because its
  publisher-authored release evidence has not been published.
- A complete local gfx1200 v2 publication candidate now exists at source
  revision `36e44fb6`: all 163 locked-clock baseline records pass, and the
  current verifier accepts both the 163-workload baseline statement and the
  163-workload `make_fx_aten` SOLAR statement. These ignored local artifacts
  are not publisher authority and are not an official release.
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
  SOLAR and performance evidence. Acceptance was input-invalid, not a
  statistical rejection: four `elementwise` and all twenty `transformer_block`
  cases selected a calibration tier using accumulated hardware traffic as the
  working-set coordinate and exceeded the registered range.
- Hardware prediction now keeps counter-derived traffic as the byte amount but
  uses SOLAR semantic bytes as the working-set coordinate. This changes the
  model policy identity after held-out reveal, so Cycle 2 cannot be repaired or
  accepted retrospectively; its ignored artifacts are source evidence only.
- Candidate inputs now use per-run entropy and per-invocation trusted-reference
  validation. The candidate process does not receive the nonce or expected
  outputs. Publication runs additionally use the networkless, capability-free,
  private-IPC Docker boundary described in the evaluator contract.
- The focused SOLAR comparison completed all 82 path analyses for 41 unique
  workloads. Coverage is complete on both extraction paths: 27 workloads match
  with legitimate dialect/decomposition differences and 14 remain different
  because of normalization differences. There are no external-I/O or fused-I/O
  accounting mismatches and no remaining resource-model-bug classification.
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

### P0 — Publish the official gfx1200 v2 scoring baseline

The formula, verifier, release builder, corpus pin, and baseline identity exist,
but the manifest correctly fails closed with
`baseline_v2_release_evidence_pending`.

CPU preparation and the local GPU candidate are closed. At source revision
`36e44fb6`, the exact 43-problem/163-workload baseline run is complete with all
records passing under locked clocks, and the current verifier accepts its
baseline and 163-workload formal-SOLAR statements. The bundled corpus hash is
identical to the current manifest.

The remaining work is the publisher-authorized release operation. The existing
candidate is ignored local evidence, binds a revision before the current HEAD,
and cannot authorize the manifest. A publisher must either adopt that exact
revision and its verified artifacts consistently or rebuild the release under
the revision it intends to publish; a locally built plan, statement, or bundle
is not publication evidence.

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

### P1 — Start Cycle 3 after the Cycle 2 input invalidation

Do not tune against either revealed held-out set or reuse either frozen
inference/acceptance artifact. Promote the 220 revealed Cycle 2 pairs to
development, preregister a fresh pair-disjoint universe, and rebuild the model
under the current policy identity. Freeze the model and action thresholds
before collecting or reading the new held-out evidence, then run the same
acceptance gates. Changing gates, excluding the 24 exposed cases, or reusing
held-out pairs is not an acceptable fix.

Cycle 2 GPU collection itself is complete: every one of the eleven families has
20 content-addressed SOLAR manifests and 20 performance-evidence manifests.
The working-set/traffic feature asymmetry is fixed in production with a focused
regression test, but that post-reveal code change invalidates the Cycle 2 frozen
identity. A CPU preflight should first prove every newly generated development
case produces an available prediction under the corrected feature contract;
only then is governed GPU time justified for the fresh 220-case held-out set.

Completion requires all eleven families to meet at least 90% empirical interval
coverage, median APE at most 15%, P90 APE at most 30%, and every enabled action
to meet its support, precision, and recall gates. Agent feedback may enable a
code-changing action only after rebuilding and verifying every cited source.

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
