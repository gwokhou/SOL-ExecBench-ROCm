# SOLAR Derivation Contract

**Date:** 2026-05-23
**Scope:** v1.10 sidecar-only SOLAR derivation fixture contract
**Validation target:** Phase 47 golden fixture matrix for later extractor and modeling phases

## Claim Boundary

This contract supports the v1.10 ROCm-port claim that maintainers have
paper-aligned automatic SOLAR derivation evidence for this ROCm port through a
sidecar-only source of truth for fixture expectations, extractor evidence,
formula evidence, coverage, and score-eligibility refs.

This contract is not paper-scale dataset extraction, not the original
124-model / 235-problem paper extraction, not hosted leaderboard readiness, not
NVIDIA Blackwell/B200 equivalence, and not new real-hardware validation. It also
does not claim full upstream SOLAR equivalence, paper benchmark parity, MI300X
validation, CDNA 3 validation, CDNA 4 validation, or new NVFP4/MXFP4
validation.

No-claim phrases for guardrails:

- not paper-scale dataset extraction
- not original 124-model / 235-problem extraction
- not hosted leaderboard readiness
- not NVIDIA Blackwell/B200 equivalence
- not new real-hardware validation
- not upstream SOLAR parity
- not CDNA 3 / MI300X / CDNA 4 validation
- not NVFP4/MXFP4 validation

## Derivation Source Boundary

SOLAR derivation sidecars are derived from canonical `Definition` and
`Workload` inputs plus reference-visible and static evidence available to the
ROCm port's analyzer. The sidecar may record formula evidence, hardware-model
refs, `coverage_summary`, `aggregate_status`, warnings, and score eligibility
refs when those values can be justified from those static inputs.

The derivation boundary excludes `candidate_solution_execution`, candidate
solution execution, submitted kernel behavior, and timed benchmark results as
sources for formula evidence. Timing and candidate results may appear in
separate trace or AMD-native score contexts, but they must not be used to create
SOLAR formula evidence.

## Sidecar-Only Artifact Rule

SOLAR derivation evidence belongs only in derived sidecars or explicitly opted
in reports. The sidecar contract is internal and must not mutate:

- canonical `definition.json`;
- canonical `workload.jsonl`;
- canonical trace JSONL;
- public solution schemas;
- primary `sol-execbench` CLI behavior or default output.

Later phases may emit derivation sidecars next to benchmark outputs, but the
canonical benchmark inputs and traces remain the public contract. Fixture
expectations in this document describe sidecar content that downstream tests can
consume; they are not public schema additions.

## Family Vocabulary

The fixture matrix uses the following exact family identifiers:

- `attention`
- `moe`
- `convolution`
- `ssm_mamba`
- `embedding_positional`
- `linear_projection`

Every positive fixture must name one of these values as `expected_family` and
must include enough `expected_subroles` and `required_evidence` for a later
extractor to prove that the case is not taxonomy-only.

## State Vocabulary

Per-family derivation confidence uses these exact values:

- `supported`
- `inexact`
- `unsupported`

Aggregate fixture status uses these exact values:

- `scored`
- `degraded`
- `unscored`

The expected state is contractual evidence, not an exception expectation.
Degraded and unsupported or negative cases are valid fixtures when they parse,
declare their missing evidence, and expose stable warning prefixes.

## Fixture Classes

Each required family must have at least these classes represented in the golden
matrix:

- `positive`: enough evidence exists for `expected_confidence: supported` and
  `expected_status: scored`.
- `degraded`: the family is recognizable, but missing or dynamic evidence makes
  the result `expected_confidence: inexact` and `expected_status: degraded`.
- `unsupported` or `negative`: the case is intentionally unscored because the
  operator is unsupported, taxonomy-only, dynamically unbounded, or missing
  essential metadata.

Unsupported and negative fixtures are not invalid JSON and must not rely on
exception expectations. They use `expected_status`, `missing_evidence`,
`warning_prefixes`, and `degradation_rationale` to define behavior.

## Fixture JSON Schema

Fixture files live under
`tests/sol_execbench/fixtures/solar_derivation/*.json`. Each file is one
machine-readable contract case with this shape:

```json
{
  "case_id": "attention_positive_dense_qkv",
  "family": "attention",
  "fixture_class": "positive",
  "negative_category": null,
  "description": "Dense self-attention with explicit Q/K/V, softmax, PV, and output projection.",
  "source_kind": "reference_snippet",
  "reference": "def run(q, k, v, w_o): ...",
  "workload_axes": {
    "B": 2,
    "S": 16,
    "H": 4,
    "D": 32
  },
  "expectation": {
    "expected_family": "attention",
    "expected_subroles": [
      "q_projection",
      "k_projection",
      "v_projection",
      "qk_scores",
      "softmax",
      "pv_aggregation",
      "output_projection"
    ],
    "expected_confidence": "supported",
    "expected_status": "scored",
    "required_evidence": [
      "shape:batch",
      "shape:sequence_q",
      "shape:sequence_k",
      "shape:head_dim"
    ],
    "missing_evidence": [],
    "warning_prefixes": [],
    "degradation_rationale": null
  },
  "scope_boundary": {
    "paper_scale_dataset": false,
    "hosted_leaderboard_ready": false,
    "nvidia_blackwell_b200_equivalence": false,
    "real_hardware_validation": false
  }
}
```

Required top-level fields:

- `case_id`
- `family`
- `fixture_class`
- `negative_category`
- `description`
- `source_kind`
- `reference`
- `workload_axes`
- `expectation`
- `scope_boundary`

Required expectation fields:

- `expected_family`
- `expected_subroles`
- `expected_confidence`
- `expected_status`
- `required_evidence`
- `missing_evidence`
- `warning_prefixes`
- `degradation_rationale`

Required scope-boundary fields:

- `paper_scale_dataset`
- `hosted_leaderboard_ready`
- `nvidia_blackwell_b200_equivalence`
- `real_hardware_validation`

All scope-boundary values must be `false` for Phase 47 fixtures.

## Stable Warning Prefixes

Fixtures assert warning prefixes rather than full warning text. The stable
prefix vocabulary is:

- `graph_warning:`
- `estimate_warning:`
- `inexact_operator:`
- `unsupported_operator:`
- `aggregate_degraded:`
- `aggregate_unscored:`

Positive fixtures usually have an empty `warning_prefixes` list. Degraded
fixtures must include at least one prefix that explains why the aggregate status
is `degraded`. Unsupported or negative fixtures must include at least one prefix
that explains why the aggregate status is `unscored`.

## Golden Fixture Matrix Inventory

The Phase 47 matrix must cover the following family and state combinations:

| Family | Positive class | Degraded class | Unsupported or negative class |
|--------|----------------|----------------|-------------------------------|
| `attention` | explicit Q/K/V, score, softmax, and PV subroles | partial mask, dynamic sequence evidence, or incomplete projection metadata | unsupported dynamic axes or missing attention structure |
| `moe` | router, expert projection, top-k routing, and combine subroles | dynamic routing or partial expert-count evidence | taxonomy-only MoE label without derivable router/expert evidence |
| `convolution` | input, weight, stride, padding, dilation, and output evidence | missing padding, grouped-shape ambiguity, or partial layout evidence | unsupported dynamic kernel or non-static convolution metadata |
| `ssm_mamba` | scan, state update, projection, and gating evidence | missing recurrence or partial state-size evidence | unsupported custom scan with no derivable recurrence contract |
| `embedding_positional` | token or positional lookup evidence with index and table shapes | dynamic indices or partial table-shape evidence | missing metadata that prevents table/index derivation |
| `linear_projection` | matrix multiply or affine projection evidence with input and output shapes | missing shape, rank, or bias evidence | missing metadata that prevents projection derivation |

The negative and degradation categories must cover:

- dynamic behavior;
- partial evidence;
- unsupported operators;
- taxonomy-only labels;
- missing metadata.

## Degradation And Negative Behavior

Valid degraded, unsupported, and negative fixtures are successful contract cases.
They must parse and carry enough expectation evidence for later phases to
produce deterministic sidecar results.

Degraded cases use:

- `expected_confidence: inexact`
- `expected_status: degraded`
- non-empty `missing_evidence`
- at least one warning prefix, commonly `inexact_operator:`,
  `estimate_warning:`, or `aggregate_degraded:`
- a non-empty `degradation_rationale`

Unsupported or negative cases use:

- `expected_confidence: unsupported`
- `expected_status: unscored`
- non-empty `missing_evidence`
- at least one warning prefix, commonly `unsupported_operator:`,
  `graph_warning:`, or `aggregate_unscored:`
- a non-empty `degradation_rationale`

Fixtures must not encode degraded or negative behavior as expected exceptions.
Schema-invalid JSON may fail loader validation, but valid negative fixtures are
ordinary sidecar expectation cases.

## Downstream Phase Consumption

Phase 48 should consume this contract to build extraction-side evidence without
changing canonical benchmark inputs or primary CLI behavior.

Phase 49 should use the positive fixture expectations for high-confidence
family modeling and should preserve `supported` / `scored` semantics.

Phase 50 should use degraded and unsupported or negative fixture expectations
for complex family modeling, including MoE and SSM/Mamba cases, and should
preserve deterministic `inexact`, `unsupported`, `degraded`, and `unscored`
behavior.

Phase 51 should wire sidecar coverage and score guards to the same stable
warning prefixes and aggregate status vocabulary. Score integration must remain
opt-in and sidecar-backed unless a later plan explicitly changes the public
contract.

Phase 52 may run dataset-level workflows against the implemented sidecars, but
this document remains an internal contract and does not itself provide
paper-scale extraction, original 124-model / 235-problem extraction, hosted
leaderboard readiness, NVIDIA Blackwell/B200 equivalence, upstream SOLAR parity,
or new real-hardware validation.
