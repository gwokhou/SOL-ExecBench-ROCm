# Project Research: Pitfalls for v1.8

## Pitfalls

| Pitfall | Risk | Prevention |
| --- | --- | --- |
| Declaring support when only schema accepts the category | Repeats the candidate-category gap v1.8 is meant to close | Require runnable example, dependency test, docs, and RDNA 4 E2E evidence before marking supported |
| Silent PyTorch fallback inside a library solution | Public examples overclaim library replacement | Tests inspect solution language and source patterns for real library includes/API calls |
| Missing ROCm package headers in Docker | Examples compile locally for one developer but fail in CI/container | Add dependency smoke tests with explicit missing-header/missing-library messages |
| rocWMMA architecture mismatch | rocWMMA example may not run on unsupported GPUs | Scope v1.8 validation to RDNA 4 and document CDNA 3/CDNA 4 deferral separately |
| MIOpen API shape limitations | Softmax or convolution descriptors may not match arbitrary benchmark shapes | Choose a small supported example and document operation-specific constraints |
| CK template complexity | A full CK example can become too large or fragile | Start with minimal GEMM/fused epilogue and keep requirements focused on runnable support, not peak performance |
| Changing public schema to solve build convenience | Breaks SOL ExecBench compatibility guarantees | Use existing `compile_options` first; add internal helpers only if required |
| Performance claims without measurement discipline | Users infer SOL or leaderboard-quality support from examples | Keep v1.8 about support completeness; require separate profiler evidence for performance claims |
| CDNA 3/CDNA 4 validation creep | Milestone completion becomes blocked by unavailable hardware | Tests and docs must state RDNA 4-only acceptance and deferred CDNA validation |

## Phase Ownership

- Build plumbing phase owns dependency diagnostics and compile metadata risks.
- Per-library phases own API correctness and example E2E risks.
- Closure phase owns overclaiming, compatibility cleanup, and RDNA 4 validation
  evidence.
