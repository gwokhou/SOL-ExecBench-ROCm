# Phase 134: NVIDIA/Blackwell Low-Precision ROCm Equivalence - Context

**Gathered:** 2026-06-04
**Status:** Ready for planning
**Mode:** Autonomous smart discuss

<domain>
## Phase Boundary

Implement CPU-safe ROCm compatibility abstractions for NVIDIA/Blackwell
low-precision dataset semantics. This phase may provide complete importable
code paths for migrated definitions, but it must not claim real CDNA4 hardware
validation, performance authority, paper parity, or NVIDIA Blackwell/B200
equivalence.

</domain>

<decisions>
## Implementation Decisions

### Scope
- Add project-owned low-precision compatibility helpers under the dataset
  package so migrated dataset tooling can import them without requiring CDNA4
  hardware.
- Preserve public string format names and explicit packing/unpacking semantics
  for NVFP4/MXFP4-like E2M1 payloads.
- Carry scale metadata in structured models and deterministic evidence records
  instead of treating compatibility as validation.
- Keep all real hardware, performance, score, and leaderboard claims false or
  explicitly deferred.

### Readiness Integration
- Reuse Phase 133 readiness classes and blocker vocabulary.
- Add an explicit compatibility evidence marker for Blackwell/NVFP4 workloads
  while keeping `needs_hardware_evidence` status until real CDNA4 validation
  exists.

### Tests
- Use CPU-safe PyTorch tensors and mocked/no-hardware evidence.
- Cover quantize/pack/unpack/dequantize round trips, shape validation,
  fallback/evidence markers, and readiness blocker reporting.

</decisions>

<code_context>
## Existing Code Insights

- `src/sol_execbench/core/bench/io.py` already contains an internal
  `_cast_to_fp4x2` helper for random input generation, but it returns
  `torch.float4_e2m1fn_x2` and is tied to execution input generation.
- `src/sol_execbench/core/dataset/readiness.py` classifies Quant, NVFP4, MXFP4,
  Blackwell, and float4 workloads as `nvfp4_blackwell_specific` with
  `needs_hardware_evidence`.
- `src/sol_execbench/core/dataset/__init__.py` exports dataset sidecar models
  and is the right place for public dataset compatibility exports.
- `tests/sol_execbench/test_dataset_inventory_readiness.py` already contains
  synthetic Blackwell/NVFP4 readiness tests.

</code_context>

<deferred>
## Deferred Ideas

- Real CDNA4 execution and performance validation.
- Dataset runner consumption of low-precision compatibility evidence beyond
  static readiness classification.
- Redistribution of NVIDIA/SOL-ExecBench original or derivative dataset
  content.

</deferred>
