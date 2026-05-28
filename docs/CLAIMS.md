# Claims And Evidence Boundaries

This project is an AMD ROCm port of SOL ExecBench. It preserves the benchmark
shape where practical, but it is not the NVIDIA B200 paper implementation and
does not claim official leaderboard equivalence.

## What Can Be Claimed Today

| Claim level | Allowed claim | Required evidence |
| --- | --- | --- |
| ROCm-port evidence | The CLI, schemas, isolated evaluation, correctness checks, reward-hack checks, trace JSONL, and selected examples run through ROCm-specific paths. | Passing tests for the touched surface, canonical trace examples, and ROCm environment documentation. |
| Runtime evidence | A run was executed in a recorded ROCm environment. | Canonical trace JSONL plus optional per-trace environment sidecar, for example `<trace>.environment.json`, or `sol-execbench doctor --json` output. |
| Profiling evidence | A run produced diagnostic `rocprofv3` artifacts. | `--profile rocprofv3`, per-trace profile sidecar such as `<trace>.profile.json`, registered artifact paths, and profiler status metadata. |
| Toolchain routing evidence | The project selected or rejected ROCm tools for a requested evidence level based on registry entries and bounded probes. | `sol-execbench toolchain --json`, registry source refs, selected tool, fallback, status, and reason code. |
| Static Kernel Evidence | A HIP/C++ run produced diagnostic static build artifacts and bounded routed extractor outputs. | `--static-evidence auto`, `<trace>.static-evidence.json`, persisted artifact paths, `llvm-objdump` / `readelf` tool-run records, and diagnostic-only authority flags. |
| AMD-native-derived evidence | A score or bound report was derived from ROCm traces and AMD-side bound artifacts. | Canonical traces, AMD SOL bound sidecars, hardware model refs, score eligibility state, and warnings. |
| Research-preview evidence | A curated benchmark slice was executed or audited under documented limits. | Slice definition, command transcript or expected commands, artifact list, pass/fail/skip/unavailable accounting, and known gaps. |

Docker Matrix Entries may claim **container ROCm user-space validated on recorded host driver/devices** only; a Docker row is not native host ROCm validated without direct native-host validation evidence.

## ROCm Compatibility Matrix

The ROCm Compatibility Matrix is diagnostic sidecar evidence. It does not
change canonical trace JSONL, correctness semantics, timing semantics, scoring
schemas, benchmark defaults, or benchmark exit semantics.

Each Matrix Entry has two different kinds of data:

- **Target/requested values** identify the selected compatibility target, such
  as the Target id, requested ROCm user-space version, Docker image repository
  and tag, PyTorch ROCm wheel target, validation scope, and intended `gfx*`
  architecture.
- **Observed evidence** records what probes found at runtime, separated into
  host, container, Python dependency, dependency policy, toolchain, and GPU
  scopes.

Target identity is required because observed values only become meaningful when
compared with what was requested. A ROCm 7.1 PyTorch wheel observed while the
Target requested ROCm 7.2 is a compatibility mismatch, not a benchmark
correctness failure. Those mismatches are represented with bounded diagnostic
statuses such as `mixed_version`, `pytorch_wheel_unavailable`,
`runtime_unavailable`, or `not_tested`.

Docker Matrix Entries validate **container ROCm user-space on recorded host
driver/devices**. They do not prove native host ROCm validation. Native host
validation requires direct native-host evidence for the requested host ROCm
stack and must not be inferred from Docker image selection or container
execution alone.

As of the 2026-05-29 live checks, the local Docker rows for
`sol-execbench:rocm-7.0.2-complete`, `sol-execbench:rocm-7.1.1-complete`, and
`sol-execbench:rocm-7.2-complete` have container ROCm user-space evidence on the
recorded RDNA 4 `gfx1200` host driver/devices. The 7.0.2 and 7.2 rows were
recorded through `./scripts/run_docker.sh --record-container-validation`, which
checks target-container dependency evidence before writing `container_validated`
sidecars. ROCm 7.0.2 remains unlocked performance evidence with
`CLOCKS_LOCKED=0`; ROCm 7.1.1 and 7.2 completed with `CLOCKS_LOCKED=1`. These
remain Docker container evidence, not native-host ROCm hardware validation
upgrades.

Illegal mixed-version Targets are blocked by default before benchmark
execution. The explicit mixed-version debug override may allow bounded probes or
smoke diagnostics only; it cannot create `container_validated` or
`host_validated` entries, score authority, paper-parity authority, or
leaderboard authority.

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
- Docker Matrix Entries as native host ROCm validation without direct
  native-host evidence.
- Mixed-version debug override output as clean validation, score authority,
  paper-parity authority, or leaderboard authority.

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
