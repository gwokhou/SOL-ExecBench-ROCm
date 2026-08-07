# Project handoff and active follow-ups

Last audited: 2026-08-07 against source revision `d0d07e0c` and the current
worktree; CPU and local-evidence readiness were rechecked in that state.

This file is the single backlog for repository-level work that remains useful.
Completed investigations, superseded contracts, acceptance attempts, and
one-time readiness snapshots belong in Git history. Public documentation must
describe the current contract rather than duplicate this backlog.

## Current state

- The checked-in AKA manifest contains 45 authored problems: 43 scored, one
  compatibility sentinel, and one target-incompatible problem. The official
  `rx9060xt-gfx1200-reference-v2` score remains unavailable because its
  publisher-authored release evidence has not been published.
- Local gfx1200 v2 baseline and SOLAR publication inputs exist under the ignored
  `data/outputs/p0-release-36e44fb/` root at source revision `36e44fb6`: all 163
  locked-clock baseline records pass, and the current verifier accepts both the
  163-workload baseline statement and the 163-workload `make_fx_aten` SOLAR
  statement. The root has no candidate statement or assembled
  `release-bundle.json`; it is not a complete official-score release and is not
  publisher authority. The baseline-statement SHA-256 is
  `b6da525d0bf493476ee7df44f7f3691df3cb9ca81a70700f35062a70c96ef92b`
  and the SOLAR-statement SHA-256 is
  `4c85d5e223ebd6f84e52a2d7f21da2fdf776f0cebbcc371b09b0f3fe7b8e6d9b`.
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
  the ignored 880-case `data/outputs/promoted-development-cycle3.json`, SHA-256
  `a7f9407fbf2dd72f83b5779defcab4fcccef2c355bfa67ddad363e351fc0cc99`.
  Production inference fitting rebuilt every cited case successfully and wrote
  `data/outputs/microarchitecture-diagnostics-v7-cycle3-inference.json`,
  SHA-256
  `f437d50946a97e28c0d954f2210e7b2d770537f2c5175909b9348cf531248085`.
  The frozen profile enables `restore_wmma_path` from 80 development positives
  and 800 negatives with 1.0 precision and 1.0 recall; all other code-changing
  actions remain disabled.
  The profile binds calibration
  `e9ba5e76bda2843cbac213f1404ca9f197942b476612dc26bbf6ec50273920d9`
  and audit
  `fce918aa953aafee1fbe5a496b69f32cb46753b56a8de6ca5b94f9907d41a004`.
  The start-220 design and CPU preflight are frozen under the ignored
  `data/outputs/microarchitecture-diagnostics-v7-cycle3/preregistered-corpus/`
  root with design SHA-256
  `59167a6f0acb8c8e2754f01d9e89873f2cd0bc66724a3c1c82bd126b01770c26`
  and preflight SHA-256
  `f83243a56bc33d0c9926cef3aba9f37de925eff594fc06bb7a9af977cb7df834`.
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

### P0 — Authorize gfx1200 v2 scoring and publish its first release

The formula, verifier, release builder, corpus pin, and baseline identity exist,
but the manifest correctly fails closed with
`baseline_v2_release_evidence_pending`.

CPU preparation and the local GPU source-evidence run are closed. At source
revision `36e44fb6`, the exact 43-problem/163-workload baseline run is complete
with all records passing under locked clocks, and the current verifier accepts
its baseline and 163-workload formal-SOLAR statements. The bundled corpus hash
is identical to the current pending manifest.

The remaining work is a publisher-authorized release rebuild and repository
cutover. The existing local root binds the pending manifest and revision
`36e44fb6`. Authorizing scoring changes that manifest and its repository pin,
so the existing baseline and SOLAR statements cannot be copied unchanged into
the final bundle. They are source evidence for the publication rebuild, not
adoptable final statements.

The current release policy is candidate-specific: `release-bundle.json` must
contain baseline, candidate, and SOLAR statements. A publisher must choose the
candidate being scored, prepare the final `official_scoring` manifest contract
(`status: available`, `content_addressed_publisher_v1`, and the canonical three
required-evidence values), update `OFFICIAL_CORPUS_MANIFEST_SHA256`, and rebuild
baseline, candidate, and SOLAR evidence from one clean publication revision
against that exact manifest. The verified complete bundle and every referenced
artifact must then be distributed under `RELEASE/`, with pending-state docs and
tests updated in the same publication change. If baseline-only publication is
desired instead, it requires a separately reviewed contract; the current
official-score API does not represent it.

Completion requires a publisher-authored, content-addressed official release
bundle for the exact 43-problem/163-workload scored denominator, including the
selected candidate, locked-clock baseline and candidate measurement evidence,
pinned SOLAR manifests, publisher release statements, and successful
verification through the documented release-scoring workflow. Local diagnostic
or validation sidecars cannot authorize this score.

Authoritative surfaces:

- `problems/AMD_AKA/manifest.yaml`
- `src/sol_execbench/core/scoring/`
- `docs/SCORING-V3.md`
- `docs/user/RELEASE-SCORING.md`

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
artifact against its original corpus root, rebases references under the common
root, and emits the 880-case development corpus. The existing
`fit-performance-inference` command is the prediction gate: it rebuilds all 880
cases, fails on any unavailable hardware prediction, and writes the versioned
inference profile bound to the corpus, calibration, audit, and model-policy
hashes. A separate prediction-preflight command is neither required nor used.

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
- Generated evidence under `data/outputs/` remains ignored and uncommitted.
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
