# RDNA4 Denominator Policy

This document defines the RDNA4 `gfx1200` 16GB validation denominator policy for
the current benchmark evidence chain. It is a claim-boundary document, not a
new benchmark schema and not a hardware validation upgrade by itself.

## Scope

- Architecture/device class: RDNA4 `gfx1200` on the recorded 16GB host.
- Problem denominator: the migrated 235 SOL ExecBench problems.
- Evidence surfaces: readiness reports, trace JSONL, profiler timing sidecars,
  score records, AMD SOL/SOLAR derivation sidecars, and exclusion ledgers.

This policy does not apply to CDNA3/MI300X, CDNA4, NVIDIA B200, upstream SOLAR
equivalence, hosted leaderboard authority, or full paper-parity claims.

## Status Classes

| Status | Counts in 235 denominator | Counts as profiler-backed timing | Claim meaning |
| --- | --- | --- | --- |
| `profiler_backed` | yes | yes | The problem has complete `rocprofv3` kernel activity timing for every expected included workload. |
| `timing_fallback` | yes | no | Timing exists, but it is PyTorch/device-event or another non-authoritative fallback surface. |
| `ready_missing_profiler_timing` | yes | no | The problem is ready but lacks profiler-backed timing evidence. |
| `partial_profiler_backed` | yes | no | Some `rocprofv3` evidence exists, but not every expected workload has passing profiler-backed timing. |
| `profiler_blocked` | yes | no | A profiler replacement attempt exists but did not produce usable complete kernel activity timing. |
| `reference_oom_blocked` | yes | no | The current 16GB RDNA4 host cannot produce passing reference/input-generation traces for one or more workloads because of HIP OOM. |
| `readiness_blocked` | yes | no | Static readiness or required assets block execution under the current ROCm dataset setup. |

The key policy decision is that blocker statuses remain visible inside the
235-problem denominator. They are not silently excluded, but they also do not
count as successful profiler-backed timing or full validation pass evidence.

## Current Device Memory Boundary

`reference_oom_blocked` means the recorded evidence shows current-device HIP OOM
in reference execution, input generation, a staged reference-equivalent user
function path, or profiler-closure correctness/error-stat computation. For the
current 16GB RDNA4 host, these workloads are accounted as memory-footprint
blockers.

This status is intentionally different from:

- `profiler_blocked`, where the profiler itself or the profiling workflow fails
  to produce usable kernel activity evidence.
- `partial_profiler_backed`, where profiler rows exist but complete workload
  coverage is missing.
- `readiness_blocked`, where static readiness, unsupported format, or missing
  assets prevent normal execution.

Larger-memory AMD hardware may resolve some `reference_oom_blocked` workloads,
but doing so would create a new hardware evidence boundary. It must not be
folded back into the 16GB RDNA4 claim without recording the exact hardware,
commands, traces, timing sidecars, and denominator mapping.

Detailed blocker classes may include:

- `reference_oom_blocked`
- `gen_inputs_oom_blocked`
- `user_solution_oom`
- `memory_oom_with_profiler_gap`
- `profiler_closure_oom_blocked`

`profiler_closure_oom_blocked` is accepted for the current 16GB RDNA4 host when
`rocprofv3` replacement reaches correctness or error-stat computation and then
fails with HIP OOM. It is accounted in the denominator, but it is not passing
profiler-backed timing.

## Claim Rules

Allowed wording:

- "The RDNA4 235-problem denominator is accounted with explicit blocker
  classes."
- "`reference_oom_blocked` problems are current-device memory blockers on the
  recorded 16GB RDNA4 host."
- "Profiler-backed timing coverage is limited to problems classified
  `profiler_backed`."

Disallowed wording:

- "`reference_oom_blocked` problems passed RDNA4 validation."
- "Partial or blocked profiler evidence is complete profiler-backed timing."
- "The RDNA4 evidence proves authoritative timing without clock-lock evidence."
- "The RDNA4 evidence upgrades CDNA3/MI300X, CDNA4, NVIDIA B200, upstream
  SOLAR, paper-parity, or leaderboard claims."

## Required Evidence For Future Upgrade

To move a `reference_oom_blocked` problem into an included passing/timing class,
the project must record:

- Exact AMD hardware model, architecture, memory size, ROCm/HIP/PyTorch
  versions, and command line.
- Passing trace JSONL for every affected workload.
- Complete `rocprofv3` timing sidecars if the claim involves profiler-backed
  timing.
- Clock-lock/reset evidence if the claim involves benchmark-grade timing
  authority.
- Updated coverage and blocker ledgers with checksums.

Until those artifacts exist, `reference_oom_blocked` remains accounted but not
passing.
