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
- Curated-slice results as paper-level benchmark results.

## Claim Upgrade Rules

| Desired upgraded claim | Evidence required before wording can change |
| --- | --- |
| Full ROCm paper validation | Complete 235-problem denominator accounting, execution closure, trace artifacts, failure analysis, score artifacts, and reproducible commands. |
| CDNA 3 hardware validation | Full adapted suite on real `gfx94*` hardware, environment sidecars, clock policy evidence, trace artifacts, and documented failures or skips. |
| Upstream SOLAR parity | A side-by-side comparison against upstream SOLAR outputs for the scoped dataset and operator families. |
| Leaderboard readiness | Stable submission format, hosted or reproducible scoring policy, hardware policy, anti-cheat policy, and release-defined baselines. |
| Static kernel evidence | RGA/code-object/GPUOpen ISA artifacts linked to solution builds and interpreted by documented classification rules. |

## Reporting Language

Use precise wording:

- Say "ROCm-port evidence" for ordinary benchmark behavior on AMD runtime paths.
- Say "AMD-native-derived score" only when the required sidecars and eligibility
  state exist.
- Say "curated research preview" for bounded representative slices.
- Say "unscored" or "unavailable" when evidence is missing.

Avoid ambiguous wording:

- Do not say "paper parity" for a curated slice.
- Do not say "SOLAR equivalent" for local AMD-derived sidecars.
- Do not say "hardware validated" for schema/build support alone.
- Do not say "score authority" for profiling sidecars.

