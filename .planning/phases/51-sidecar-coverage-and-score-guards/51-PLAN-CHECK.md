# Phase 51 Plan Check: Sidecar Coverage And Score Guards

**Checked:** 2026-05-23  
**Verdict:** PASS  
**Plans checked:** 3  
**Blockers:** 0  
**Warnings:** 0  

## Phase Goal

Users can rely on SOLAR sidecars and AMD-native score reports to separate
scored, degraded, and unscored derivation evidence without manual
interpretation.

## Requirement Coverage

| Requirement | Covering Plans | Verdict |
| --- | --- | --- |
| REPORT-01 | 51-01, 51-03 | Covered |
| REPORT-02 | 51-01, 51-03 | Covered |
| REPORT-03 | 51-02, 51-03 | Covered |
| TEST-03 | 51-01, 51-03 | Covered |

REPORT-01 is covered by Plan 51-01's coverage summary and aggregate sidecar
model, with Plan 51-03 adding parser and regression closure. The planned fields
include family counts, status counts, extraction provenance, missing patterns,
unsupported patterns, degraded nodes, estimated nodes, and unsupported node IDs.

REPORT-02 is covered by Plan 51-01's explicit aggregate status contract and
strict parser support. The plans preserve the required parseable `scored`,
`degraded`, and `unscored` state vocabulary and specify `unscored > degraded >
scored` precedence.

REPORT-03 is covered by Plan 51-02's AMD-native score guard integration and Plan
51-03's regression tests. The planned semantics are clear: explicit unscored
SOLAR aggregate evidence forces `AmdNativeScore.score` to `None`; degraded
SOLAR evidence preserves warnings and still allows numeric scoring when
existing AMD-native numeric inputs are complete; missing sidecar data remains
neutral.

TEST-03 is covered by Plan 51-01's sidecar parser and round-trip tests, then
closed by Plan 51-03's explicit matrix for every new machine-verifiable field,
including malformed payloads, missing fields, unknown fields, invalid statuses,
non-boolean `score_eligible`, malformed counts, malformed node lists, provenance
refs, compatibility normalization, and deterministic ordering.

## Decision And Boundary Compliance

| Check | Verdict |
| --- | --- |
| Strict parser with legacy normalization | Pass |
| Internal `solar_derivation` evidence refs only without public drift | Pass |
| Public schema boundary preserved | Pass |
| Primary CLI boundary preserved | Pass |
| Canonical trace JSONL boundary preserved | Pass |
| Score guard semantics clear | Pass |
| Tests concrete enough for all new fields | Pass |

Plan 51-01 explicitly accepts only the old Phase 48-50 exact top-level sidecar
shape or the new Phase 51 exact shape, recomputing coverage for legacy payloads
while continuing to reject unknown top-level and nested keys.

Plan 51-02 allows internal SOLAR sidecar or aggregate inputs for score guards,
but it also requires that public score `evidence_refs` do not include
`solar_derivation`, `coverage_summary`, `aggregate_status`, `formula_evidence`,
`byte_evidence`, or `bound_evidence`. Plan 51-03 reinforces this with public
contract guardrails across canonical models, primary CLI help, trace JSONL, and
score evidence refs.

The plans do not add public CLI options, dataset-runner output, documentation,
claim-boundary docs, new dependencies, real-hardware validation, paper-scale
extraction, hosted leaderboard claims, NVIDIA equivalence claims, candidate
solution execution, or a second derivation graph.

## Plan Structure

| Plan | Wave | Depends On | Tasks | Files | Status |
| --- | --- | --- | ---: | ---: | --- |
| 51-01 | 1 | none | 3 | 2 | Valid |
| 51-02 | 2 | 51-01 | 3 | 3 | Valid |
| 51-03 | 3 | 51-01, 51-02 | 3 | 4 | Valid |

All tasks have files, actions, automated verification, and done criteria.
Dependencies are valid and acyclic. Task count and file count are within GSD
scope thresholds.

## Nyquist Compliance

| Task | Plan | Wave | Automated Command | Status |
| --- | --- | --- | --- | --- |
| 51-01-01 | 51-01 | 1 | `uv run pytest tests/sol_execbench/test_solar_derivation_evidence.py -k "coverage or aggregate or round_trip" -n 0 -x` | Pass |
| 51-01-02 | 51-01 | 1 | `uv run pytest tests/sol_execbench/test_solar_derivation_evidence.py -k "coverage or aggregate or deterministic" -n 0 -x` | Pass |
| 51-01-03 | 51-01 | 1 | `uv run pytest tests/sol_execbench/test_solar_derivation_evidence.py -k "coverage or aggregate or parser or unknown or malformed or round_trip" -n 0 -x` | Pass |
| 51-02-01 | 51-02 | 2 | `uv run pytest tests/sol_execbench/test_amd_native_score.py -k "solar or degraded or unscored or workload_score" -n 0 -x` | Pass |
| 51-02-02 | 51-02 | 2 | `uv run pytest tests/sol_execbench/test_amd_native_score.py -k "trace or suite or solar or degraded or unscored" -n 0 -x` | Pass |
| 51-02-03 | 51-02 | 2 | `uv run pytest tests/sol_execbench/test_amd_native_score.py tests/sol_execbench/test_public_contract_guardrails.py -k "solar or score or degraded or unscored or evidence_refs" -n 0 -x` | Pass |
| 51-03-01 | 51-03 | 3 | `uv run pytest tests/sol_execbench/test_solar_derivation_evidence.py -k "coverage or aggregate or parser or malformed or unknown or round_trip or deterministic" -n 0 -x` | Pass |
| 51-03-02 | 51-03 | 3 | `uv run pytest tests/sol_execbench/test_public_contract_guardrails.py tests/sol_execbench/test_amd_native_score.py tests/sol_execbench/test_amd_sol_v2.py -k "solar or coverage or aggregate or score or cli or trace or schema" -n 0 -x` | Pass |
| 51-03-03 | 51-03 | 3 | Full Phase 51 pytest gate plus Ruff over touched files | Pass |

Sampling: every implementation task has automated verification. No watch mode,
long sleeps, missing `VALIDATION.md`, or broken Wave 0 dependency was found.

## Pattern And Architecture Compliance

The plans follow the Phase 51 research responsibility map:

- SOLAR coverage aggregation remains in Python scoring internals.
- Aggregate scored/degraded/unscored semantics are owned by scoring internals
  and consumed by AMD score guards.
- AMD-native score suppression and warning preservation stay in `amd_score.py`.
- Public boundary enforcement remains in tests.

The plans also follow `51-PATTERNS.md`: coverage dataclasses and parsers extend
`solar_derivation.py` using frozen dataclasses, deterministic `to_dict()`, and
exact-key parser helpers; AMD score behavior extends existing warning and
`score=None` guard patterns; tests extend existing parser, AMD score, AMD SOL
v2, and public guardrail files.

## Issues

No blockers or warnings found.

```yaml
issues: []
```

## Recommendation

Proceed to Phase 51 execution. The plan set is sufficient to achieve the phase
goal and the mapped requirements.
