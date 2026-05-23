# Research Summary: v1.10 Paper-Aligned SOLAR Derivation

**Milestone:** v1.10 paper-aligned SOLAR automatic derivation  
**Researched:** 2026-05-23  
**Scope:** SOLAR derivation only; no dataset extraction, new hardware validation, hosted leaderboard, or NVIDIA paper-equivalence claims.

## Decision Summary

v1.10 should extend the existing AMD-native derived SOL/SOLAR sidecar pipeline instead of changing canonical benchmark execution. The repository already has the correct architectural boundary:

```text
Definition + Workload + AmdHardwareModel
  -> BoundGraph
  -> OperatorWorkEstimate
  -> AMD SOL sidecar
  -> guarded AMD-native score/report
```

The milestone should make that derivation path substantially more paper-aligned by adding automatic extraction, family-specific formula evidence, coverage evidence, and score eligibility guards for the paper-relevant operation families. It should preserve canonical `definition.json`, `workload.jsonl`, trace JSONL, solution schemas, and primary CLI behavior.

## Recommended Technical Shape

- Add focused internal scoring modules rather than a new framework dependency.
- Keep `build_bound_graph()` and `estimate_bound_work()` as compatibility facades.
- Use existing Python/torch tooling: `ast`, best-effort `torch.fx.symbolic_trace`, FX shape propagation, and explicit fallback warnings.
- Add small local helpers for symbolic shape evidence, pattern recognition, family formulas, and coverage gates.
- Keep derivation evidence in sidecars and opt-in reports; do not mutate canonical benchmark artifacts.
- Avoid ONNX, MLIR, Dynamo, `sympy`, `networkx`, or dataset-extraction machinery in this milestone.

## Required Feature Families

The researched table stakes are:

- Attention: Q/K/V, QK, scaling/masking, softmax, PV, output projection, and explicit degradation when axes or mask semantics are missing.
- MoE: routing, top-k, expert projections, dispatch/combine movement, and conservative inexact defaults.
- Convolution: 1D/2D/3D, grouped/depthwise metadata, stride, padding, dilation, and output spatial formulas.
- SSM/Mamba: projection, depthwise convolution, scan/state update, gating, and output projection with conservative degraded evidence.
- Embedding/positional: gather/index traffic, positional transforms, and memory-bound evidence.
- Linear projection: first-class semantic family while reusing GEMM-compatible formulas when dimensions are explicit.
- Compound grouping: family-level grouping and subrole evidence without hiding unsupported child work.

## Modeling Requirements

Each promoted family needs more than taxonomy:

- family-specific formula kind, formula text, and formula inputs;
- dtype-aware read, write, intermediate, movement, and total byte evidence;
- tensor shape, dtype, semantic axis, and extraction-source provenance;
- deterministic confidence rules for supported, inexact, and unsupported states;
- per-op compute, memory, limiting-resource, and SOL-bound evidence;
- machine-verifiable warnings and rationales for degradation.

## Reporting And Guardrails

The sidecar/report layer should expose:

- family-aware coverage and extraction provenance;
- skipped, missing, unsupported, degraded, and estimated node evidence;
- aggregate states of `scored`, `degraded`, and `unscored`;
- `None` score behavior for unscored SOLAR evidence;
- retained warnings for degraded and provisional evidence;
- explicit claim boundaries: AMD-native derived evidence, not paper benchmark parity or hardware validation.

## Minimum Validation Gates

- Golden fixtures for attention, MoE, convolution, SSM/Mamba, embedding/positional, and linear projection.
- Negative/degradation fixtures for dynamic, partial, unsupported, and taxonomy-only cases.
- Sidecar parse/serialize round-trip tests for any new evidence fields.
- Score guard tests proving unsupported derivation cannot produce a normal SOL Score.
- Public contract guardrails proving canonical schemas, trace JSONL, and primary CLI behavior remain unchanged.
- Documentation and static checks preventing leaderboard, B200, paper-equivalent, or new-hardware-validation overclaims.

## Recommended Phase Shape

1. Derivation contract and golden fixture matrix.
2. Extraction frontend, semantic shape provenance, and pattern infrastructure.
3. High-confidence family modeling: linear projection, convolution, embedding/positional, explicit attention.
4. Degraded complex family modeling: MoE and SSM/Mamba.
5. Sidecar coverage, aggregate bound, and score guard integration.
6. Dataset-runner integration, documentation, and public contract closure.

## Explicit Exclusions

- Original paper-scale 124-model / 235-problem extraction.
- MI300X, CDNA3, CDNA4, NVFP4, or MXFP4 real-hardware validation.
- Hosted leaderboard or submission service.
- NVIDIA Blackwell/B200 equivalence.
- Importing or executing submitted solution code for derivation.
- Changing canonical public benchmark schemas or primary CLI defaults.

## Sources

- `.planning/research/STACK.md`
- `.planning/research/FEATURES.md`
- `.planning/research/ARCHITECTURE.md`
- `.planning/research/PITFALLS.md`
- arXiv 2603.19173 abstract, submitted 2026-03-19: https://arxiv.org/abs/2603.19173
