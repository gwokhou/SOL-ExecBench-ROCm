# Project Research: Features

**Project:** SOL ExecBench ROCm Port
**Milestone:** v1.6 AMD SOLAR Coverage, Live Profiler Timing, and Scoring Workflow
**Researched:** 2026-05-22
**Confidence:** HIGH

## Paper Baseline

The original SOL-ExecBench paper frames the benchmark around three properties:

- analytically derived hardware Speed-of-Light bounds from a SOLAR pipeline;
- SOL Score as progress from a release-defined baseline toward the SOL bound;
- robust benchmark execution with isolation, clock discipline, cache handling,
  and reward-hack defenses.

For this ROCm port, v1.6 should improve the AMD-native equivalents of the first
two properties while preserving the already migrated harness semantics.

## Feature Categories

### AMD SOLAR Coverage

**Table stakes**

- Add more operation analyzers beyond v1.5 matmul and broad elementwise
  detection.
- Preserve per-op confidence: supported, inexact, unsupported.
- Emit coverage summaries so unsupported operations do not look like complete
  SOL evidence.
- Keep bound artifacts derived and auditable.

**Good v1.6 targets**

- Reductions and normalization patterns used by RMSNorm/layernorm.
- Softmax and attention-like score/probability patterns.
- Shape/view/transpose/broadcast operations as data-movement or zero-FLOP nodes.
- Elementwise activation families with clearer FLOP and byte rationale.

### Live Timing

**Table stakes**

- Execute benchmark subprocesses through `rocprofv3` when policy selects
  profiler-backed timing.
- Parse generated timing output into evidence artifacts.
- Label backend, activity domain, aggregation rule, fallback reason, tool
  version, GPU architecture, and parsed rows.
- Preserve source-specific semantics:
  HIP native and Triton are primarily kernel-activity timing; PyTorch needs
  operator attribution; mixed workloads need explicit fallback or split evidence.

**Anti-feature**

- A single unified timer口径 is not acceptable if it hides the difference between
  HIP kernel activity, Triton generated-kernel activity, and PyTorch operator
  attribution.

### Score Workflow

**Table stakes**

- Generate AMD-native workload and suite score reports from existing trace
  JSONL plus derived timing and SOL-bound artifacts.
- Expose reports through dataset runner or additive CLI paths.
- Keep score reports separate from canonical trace JSONL.
- Carry evidence references and warnings for incomplete inputs, unsupported
  operators, unvalidated hardware models, and CDNA3 no-claim status.

### Compatibility

**Table stakes**

- Existing `sol-execbench` invocation semantics continue to work.
- Existing trace JSONL and public Pydantic schemas remain unchanged.
- New outputs are opt-in, additive, or separate files.
- Contract tests fail if canonical traces gain new fields or CLI defaults change.

## Deferred

- Real CDNA3 `gfx94*` full-suite validation.
- NVIDIA B200/SOLAR/leaderboard equivalence.
- Full upstream SOLAR parity, especially Blackwell-specific datatype and
  hardware-feature modeling.

## Sources

- SOL-ExecBench paper: https://arxiv.org/abs/2603.19173
- ROCprofiler-SDK `rocprofv3` docs: https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/docs-7.0.1/how-to/using-rocprofv3.html
