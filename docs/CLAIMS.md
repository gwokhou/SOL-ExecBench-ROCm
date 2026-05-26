# Claims And Evidence Boundaries

This project is an AMD ROCm port of SOL ExecBench. It preserves the benchmark
shape where practical, but it is not the NVIDIA B200 paper implementation and
does not claim official leaderboard equivalence.

## What Can Be Claimed Today

| Claim level | Allowed claim | Required evidence |
| --- | --- | --- |
| ROCm-port evidence | The CLI, schemas, isolated evaluation, correctness checks, reward-hack checks, trace JSONL, and selected examples run through ROCm-specific paths. | Passing tests for the touched surface, canonical trace examples, and ROCm environment documentation. |
| Runtime evidence | A run was executed in a recorded ROCm environment. | Canonical trace JSONL plus optional `traces.jsonl.environment.json` or `sol-execbench doctor --json` output. |
| Profiling evidence | A run produced diagnostic `rocprofv3` artifacts. | `--profile rocprofv3`, `traces.jsonl.profile.json`, registered artifact paths, and profiler status metadata. |
| Toolchain routing evidence | The project selected or rejected ROCm tools for a requested evidence level based on registry entries and bounded probes. | `sol-execbench toolchain --json`, registry source refs, selected tool, fallback, status, and reason code. |
| Static Kernel Evidence | A HIP/C++ run produced diagnostic static build artifacts and bounded routed extractor outputs. | `--static-evidence auto`, `<trace>.static-evidence.json`, persisted artifact paths, `llvm-objdump` / `readelf` tool-run records, and diagnostic-only authority flags. |
| AMD-native-derived evidence | A score or bound report was derived from ROCm traces and AMD-side bound artifacts. | Canonical traces, AMD SOL bound sidecars, hardware model refs, score eligibility state, and warnings. |
| Research-preview evidence | A curated benchmark slice was executed or audited under documented limits. | Slice definition, command transcript or expected commands, artifact list, pass/fail/skip/unavailable accounting, and known gaps. |

## What Must Not Be Claimed Yet

- NVIDIA B200, Blackwell, or official leaderboard parity.
- Upstream NVlabs/SOLAR equivalence.
- Full 124-model extraction or full 235-problem paper validation.
- CDNA 3 / MI300X or CDNA 4 hardware validation without archived full-suite
  evidence from that hardware class.
- NVFP4 or MXFP4 validation without suitable AMD hardware evidence.
- `rocprofv3` profiling as correctness or score authority.
- Toolchain routing as correctness, performance, static-kernel, paper-parity,
  or leaderboard authority.
- Static Kernel Evidence as correctness authority, performance authority,
  timing authority, score authority, paper-parity authority, or leaderboard
  authority.
  In short: Static Kernel Evidence as correctness, performance, timing, score,
  paper-parity, or leaderboard authority is not allowed.
- Static Kernel Evidence as CDNA 3, CDNA 4, Triton cache, RGA-rich resource,
  or paper-scale static coverage unless direct evidence is archived.
- Curated-slice results as paper-level benchmark results.

## Claim Upgrade Rules

| Desired upgraded claim | Evidence required before wording can change |
| --- | --- |
| Full ROCm paper validation | Complete 235-problem denominator accounting, execution closure, trace artifacts, failure analysis, score artifacts, and reproducible commands. |
| CDNA 3 hardware validation | Full adapted suite on real `gfx94*` hardware, environment sidecars, clock policy evidence, trace artifacts, and documented failures or skips. |
| Upstream SOLAR parity | A side-by-side comparison against upstream SOLAR outputs for the scoped dataset and operator families. |
| Leaderboard readiness | Stable submission format, hosted or reproducible scoring policy, hardware policy, anti-cheat policy, and release-defined baselines. |
| Static kernel evidence | Trace-adjacent static sidecars, persisted current-build artifacts, routed extractor records, bounded raw outputs, and documented diagnostic-only interpretation rules. |

## Reporting Language

Use precise wording:

- Say "ROCm-port evidence" for ordinary benchmark behavior on AMD runtime paths.
- Say "AMD-native-derived score" only when the required sidecars and eligibility
  state exist.
- Say "curated research preview" for bounded representative slices.
- Say "Static Kernel Evidence" only for diagnostic static-analysis sidecars and
  persisted artifacts.
- Say "unscored" or "unavailable" when evidence is missing.

Avoid ambiguous wording:

- Do not say "paper parity" for a curated slice.
- Do not say "SOLAR equivalent" for local AMD-derived sidecars.
- Do not say "hardware validated" for schema/build support alone.
- Do not say "score authority" for profiling sidecars.
- Do not say "correctness authority", "performance authority", "timing
  authority", "score authority", "paper-parity authority", or "leaderboard
  authority" for static evidence sidecars.
