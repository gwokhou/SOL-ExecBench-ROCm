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
| RDNA4 `gfx1200` | Recorded prerelease evidence exists where archived commands and artifacts support the scope. | Engineering-prerelease evidence for the recorded host and commands only. |
| MI300X on CDNA3 `gfx942` | Deferred. | MI300X is the concrete CDNA3 hardware target, not a separate architecture target. Full MI300X validation requires complete real-hardware evidence. |
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
| Full MI300X validation | deferred | Future real-hardware evidence target. | Completed CDNA3 hardware validation today. |
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
| First-run trace | `uv run sol-execbench tests/sol_execbench/samples/rmsnorm --solution tests/sol_execbench/samples/rmsnorm/solution_triton.json --json -o out/researcher/rmsnorm.trace.jsonl` | `out/researcher/rmsnorm.trace.jsonl` | canonical |
| Release validation | `UV_CACHE_DIR=/tmp/uv-cache uv run scripts/release_candidate_validation.py --output-dir out/release_candidate_validation` | `release_candidate_validation.json`, `release_candidate_validation.md` | diagnostic-only |
| Artifact bundle | `UV_CACHE_DIR=/tmp/uv-cache uv run scripts/build_prerelease_artifact_bundle.py --version v1.26.0-rc1 --output-dir out/prerelease_artifact_bundle/v1.26.0-rc1` | bundle manifest, Markdown, `SHA256SUMS`, transcripts, release validation outputs, environment evidence | diagnostic-only |
| Readiness gate | `UV_CACHE_DIR=/tmp/uv-cache uv run scripts/check_prerelease_readiness.py --bundle-dir out/prerelease_artifact_bundle/v1.26.0-rc1 --output-dir out/prerelease_readiness/v1.26.0-rc1` | `prerelease_readiness.json`, `prerelease_readiness.md` | diagnostic-only |
| Bounded dataset slice | `UV_CACHE_DIR=/tmp/uv-cache uv run scripts/release_candidate_validation.py --output-dir out/release_candidate_validation --include-dataset-slice --dataset-dir data/SOL-ExecBench/benchmark --dataset-limit 5` | execution closure, trust summary when inputs exist, known-gap records | provisional |

## Limitations

- No full 235-problem paper-scale validation is claimed.
- No upstream SOLAR parity is claimed.
- No hosted leaderboard readiness is claimed.
- No hard multi-tenant sandbox authority is claimed.
- No native-host ROCm validation is inferred from Docker/container user-space
  evidence.
- Full MI300X validation on the CDNA3 `gfx942` target remains deferred without
  a complete real-hardware evidence chain.
- CDNA4 validation is unavailable because suitable hardware is not currently
  accessible.
