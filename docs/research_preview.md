# Research Preview Evidence Package

This document is the v1.26 research preview entry point for SOL ExecBench ROCm.
It explains how to interpret the current prerelease evidence without promoting
it to paper parity, upstream SOLAR parity, leaderboard authority, hard-sandbox
authority, or completed MI300X/CDNA4 hardware validation.

## Methodology

The project preserves SOL ExecBench benchmark semantics where practical while
porting the execution stack to AMD ROCm. The primary methodology is:

1. Keep benchmark definitions, workloads, solution metadata, correctness
   checks, reward-hack checks, and Trace JSONL semantics stable.
2. Run representative ROCm paths through the local harness and selected
   examples.
3. Attach explicit sidecar evidence for environment, profiling, static build
   artifacts, compatibility Matrix rows, dataset closure, consistency,
   claim-upgrade, trust summary, release validation, artifact bundle, and
   readiness gates.
4. Keep every sidecar's authority boundary visible so reviewers can separate
   canonical benchmark output from diagnostic or provisional evidence.

## Attribution And Provenance

This preview is for the Apache-2.0 SOL ExecBench ROCm port. The upstream
project is NVIDIA SOL-ExecBench; retained or derivative upstream files preserve
applicable NVIDIA notices, while independent ROCm work uses this project's own
attribution. See `docs/provenance.md` and `provenance.toml`.

The SOL-ExecBench paper is the benchmark and methodology citation. It does not
make every independent ROCm implementation file NVIDIA-owned. This preview does
not imply NVIDIA or AMD endorsement.

## Benchmark Scope

The research preview is scoped to the ROCm port's current benchmark harness,
schemas, examples, release validation, and bounded dataset-slice workflows. It
does not claim full 235-problem paper-scale validation. A curated or bounded
slice can support engineering review only when its denominator, commands,
closure artifact, and known gaps are recorded.

Trace JSONL is the canonical run artifact. Definition, workload, and solution
schemas remain the benchmark contract. Derived reports help interpret evidence
but do not replace canonical traces.

## Hardware Scope

| Hardware scope | Research preview status | Interpretation |
| --- | --- | --- |
| RDNA4 `gfx1200` | v1.35 same-source rerun evidence exists under `out/rdna4-v135-rerun-20260611/`: execution closure has `derived_evidence_missing=0`; profiler timing batch produced 121 replacement timing sidecars and 121 workload manifests with 0 remaining resume targets; rebuilt coverage has 88 full profiler-backed problems, 28 partial profiler-backed problems, 0 fallback timing problems, 0 profiler-blocked problems, and 73 ready-missing profiler timing problems; rebuilt consistency has 0 findings; the full prerelease artifact bundle reports `overall_status=passed`. | Bounded RDNA4 evidence for the recorded host, commands, and artifacts only. Full profiler-backed timing coverage remains false, timing remains non-authoritative, and the result is not paper parity, upstream SOLAR parity, hosted leaderboard authority, CDNA3/MI300X validation, CDNA4 validation, or broader AMD hardware validation. |
| MI300X GPU under CDNA3 `gfx942` | Infrastructure evidence with known blockers. | Current CDNA3 evidence was recorded on MI308X (`gfx942`), not MI300X. MI300X and MI308X are sibling GPU products under CDNA3 with different hardware configurations, even though they share the `gfx942` code path. |
| CDNA4 | Unavailable. | CDNA4 validation is unavailable because suitable hardware is not currently accessible. |
| Docker/container ROCm user-space | Diagnostic compatibility evidence where recorded. | Container user-space evidence is not native-host validation. |

## Evidence Surfaces

| Evidence surface | Authority class | Use in research preview | Not a claim for |
| --- | --- | --- | --- |
| Trace JSONL | canonical | Primary per-workload run artifact. | Paper parity or leaderboard authority. |
| Release validation output | diagnostic-only | CPU-safe prerelease validation summary and optional smoke evidence. | Full benchmark validation. |
| Prerelease artifact bundle | diagnostic-only | Review package with manifest, checksums, transcripts, and authority classes. | New benchmark authority. |
| Prerelease readiness report | diagnostic-only | Gate report for missing evidence, claim-boundary regressions, and known gaps. | Hosted release approval or leaderboard policy. |
| Bounded dataset slice | provisional | Scoped representative execution when denominator and closure are recorded. | Full 235-problem paper-scale validation. |
| AMD SOL / AMD score sidecars | derived diagnostic evidence | Local AMD-native interpretation when required sidecars and eligibility state exist. | Upstream SOLAR parity or NVIDIA B200 equivalence. |
| Environment, profile, static, Matrix, consistency, claim-upgrade, and trust-summary sidecars | diagnostic-only | Reproducibility and review context. | Correctness, timing, score, paper-parity, hardware-validation, or leaderboard authority. |
| Full MI300X validation | blocked | Future exact-hardware evidence target; current MI308X (`gfx942`) evidence records pytest pass, full-dataset execution, expected Quant skips, and timeout blockers. | Completed MI300X hardware validation today. |
| CDNA4 validation | unavailable | Cannot currently be collected. | Validated CDNA4 support. |

## SOL/AMD-Derived Versus Upstream SOLAR

AMD-native SOL and score reports are derived from ROCm traces and AMD-side
bound artifacts. They can support local interpretation when the required trace,
bound, hardware model, eligibility, and warning fields are present.

They are not upstream SOLAR parity. Upstream SOLAR parity would require a
side-by-side comparison against upstream SOLAR outputs for the scoped dataset
and operator families. This preview also does not claim NVIDIA B200
equivalence, official leaderboard equivalence, or paper-scale validation.

## Representative Command Traceability

| Step | Representative command | Expected artifacts | Authority class |
| --- | --- | --- | --- |
| First-run trace | `uv run sol-execbench --format json evaluate tests/sol_execbench/samples/rmsnorm --solution tests/sol_execbench/samples/rmsnorm/solution_triton.json --trace-output <out>/rmsnorm.trace.jsonl` | canonical trace in the selected output directory | canonical |
| Release validation | `UV_CACHE_DIR=/tmp/uv-cache uv run scripts/internal/release/release_candidate_validation.py --output-dir out/release_candidate_validation` | release-candidate validation summaries | diagnostic-only |
| Artifact bundle | `UV_CACHE_DIR=/tmp/uv-cache uv run scripts/internal/release/build_prerelease_artifact_bundle.py --version v1.26.0-rc1 --output-dir out/prerelease_artifact_bundle/v1.26.0-rc1` | bundle manifest report, Markdown summary, checksum file, transcripts, and release readiness diagnostics in the selected output directory | diagnostic-only |
| Readiness gate | `UV_CACHE_DIR=/tmp/uv-cache uv run scripts/internal/release/check_prerelease_readiness.py --bundle-dir out/prerelease_artifact_bundle/v1.26.0-rc1 --output-dir out/prerelease_readiness/v1.26.0-rc1` | readiness summary outputs in the selected output directory | diagnostic-only |
| Bounded dataset slice | `UV_CACHE_DIR=/tmp/uv-cache uv run scripts/internal/release/release_candidate_validation.py --output-dir out/release_candidate_validation --include-dataset-slice --dataset-dir data/SOL-ExecBench/benchmark --dataset-limit 5` | execution closure, trust summary when inputs exist, known-gap records | provisional |

## Limitations

- No full 235-problem paper-scale validation is claimed.
- No upstream SOLAR parity is claimed.
- No NVIDIA B200 equivalence is claimed.
- No hosted leaderboard readiness is claimed.
- No hard multi-tenant sandbox authority is claimed.
- RDNA4 timing is not authoritative until full profiler-backed timing coverage
  and a release policy for benchmark-grade timing authority are both satisfied.
- RDNA4 ready-missing, partial-profiler, readiness-blocked, and
  current-device-OOM-blocked rows remain visible denominator limitations.
- No native-host ROCm validation is inferred from Docker/container user-space
  evidence.
- Full MI300X validation under CDNA3 remains blocked by known timeout shards and
  missing benchmark-grade timing/score evidence. Current operational CDNA3
  validation infrastructure evidence was recorded on MI308X, not MI300X.
- CDNA4 validation is unavailable because suitable hardware is not currently
  accessible.
