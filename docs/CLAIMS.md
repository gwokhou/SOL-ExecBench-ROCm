# Claims And Evidence Boundaries

This project is an AMD ROCm port of SOL ExecBench. It preserves the benchmark
shape where practical, but it is not the NVIDIA B200 paper implementation and
does not claim official leaderboard equivalence.

## What Can Be Claimed Today

For the centralized v1.19 guide to evidence surfaces, generation commands, and
claim boundaries, see `docs/v1_19_evidence_guide.md`. v1.19 evidence has no
full 235-problem paper validation, no upstream SOLAR parity, no score authority,
no leaderboard readiness, no CDNA3-family validation, including MI300X, and no CDNA4 validation,
no native-host ROCm Matrix validation, and no new-hardware validation.

For v1.20 evidence-quality gates, see
`docs/v1_20_evidence_quality_guide.md`. v1.20 consistency, stability,
claim-upgrade, and trust-summary reports are local diagnostic sidecars. They do
not add full 235-problem paper validation, CDNA3-family validation, including
MI300X (`gfx942`), CDNA4 validation, native-host Matrix authority, hosted
leaderboard readiness, upstream SOLAR parity, score authority, or new-hardware
validation.

v1.21 reduces codebase debt and boundary ambiguity through helper extraction,
CPU-safe guardrail tests, and clearer documentation. It does not add hard sandboxing,
multi-tenant safety, CDNA3-family validation, including MI300X (`gfx942`),
paper-scale SOLAR parity, hosted leaderboard authority, or one-for-one native
ROCm replacement proof for every former NVIDIA library category. In particular,
v1.21 does not provide hosted leaderboard authority.

For v1.25 engineering-prerelease release notes, see
`docs/v1_25_release_notes.md`. The canonical run artifact is Trace JSONL.
Environment, profile, static, Matrix, closure, consistency,
claim-upgrade, trust-summary, and release-candidate validation outputs are
diagnostic-only sidecar evidence unless a narrower document gives them a more
limited role. Bounded dataset slices and support-matrix rows are provisional
prerelease evidence, not paper parity, upstream SOLAR parity, score authority,
leaderboard readiness, hard-sandbox authority, native-host validation from
Docker/container evidence, full validation of the MI300X GPU model under
CDNA3, or CDNA4 validation.

| Claim level | Allowed claim | Required evidence |
| --- | --- | --- |
| ROCm-port evidence | The CLI, schemas, isolated evaluation, correctness checks, reward-hack checks, trace JSONL, and selected examples run through ROCm-specific paths. | Passing tests for the touched surface, canonical trace examples, and ROCm environment documentation. |
| Runtime evidence | A run was executed in a recorded ROCm environment. | Canonical trace JSONL plus optional per-trace environment sidecar, for example `<trace>.environment.json`, or `sol-execbench doctor --json` output. |
| Profiling evidence | A run produced diagnostic `rocprofv3` artifacts. | `--profile rocprofv3`, per-trace profile sidecar such as `<trace>.profile.json`, registered artifact paths, and profiler status metadata. |
| Toolchain routing evidence | The project selected or rejected ROCm tools for a requested evidence level based on registry entries and bounded probes. | `sol-execbench toolchain --json`, registry source refs, selected tool, fallback, status, and reason code. |
| Static Kernel Evidence | A HIP/C++ run produced diagnostic static build artifacts and bounded routed extractor outputs. | `--static-evidence auto`, `<trace>.static-evidence.json`, persisted artifact paths, `llvm-objdump` / `readelf` tool-run records, and diagnostic-only authority flags. |
| v1.19 sidecar evidence | Execution closure, paper denominator, Matrix schema export, Matrix semantic diff, and AMD bound sanity artifacts were generated or inspected under bounded report semantics. | `docs/v1_19_evidence_guide.md`, explicit sidecar/report paths, relative source refs, checksums, false authority fields, and focused CPU-safe docs/contract guardrails. |
| v1.20 evidence-quality review | Consistency, stability, claim-upgrade, and trust-summary artifacts were generated or inspected under bounded report semantics. | `docs/v1_20_evidence_quality_guide.md`, source refs, checksums, false authority fields, and CPU-safe docs/contract guardrails. |
| v1.21 debt-reduction evidence | Dataset, evaluator, analysis, SOLAR/static evidence, and boundary tests were refactored or hardened under stable public schemas. | Focused helper tests, unchanged sidecar schemas, `CONCERNS.md` status categories, and explicit non-claim wording. |
| AMD-native-derived evidence | A score or bound report was derived from ROCm traces and AMD-side bound artifacts. | Canonical traces, AMD SOL bound sidecars, hardware model refs, score eligibility state, and warnings. |
| Research-preview evidence | A curated benchmark slice was executed or audited under documented limits. | Slice definition, command transcript or expected commands, artifact list, pass/fail/skip/unavailable accounting, and known gaps. |
| CDNA3/gfx942 infrastructure evidence | Real MI308X (`gfx942`) runs exercised the adapted pytest suite and full dataset validation path. | Archived environment details, pytest pass counts, dataset summary, per-problem traces, expected Quant NVFP4/MXFP4 skips, and documented timeout blockers. This is not full MI300X hardware validation because MI308X and MI300X have different hardware configurations despite sharing the `gfx942` code path. |

Docker Matrix Entries may claim **container ROCm user-space validated on recorded host driver/devices** only; a Docker row is not native host ROCm validated without direct native-host validation evidence.

For v1.25 engineering-prerelease support wording:

- RDNA 4 evidence is engineering-prerelease evidence only when backed by the
  recorded artifacts and commands for that host/scope.
- Docker/container ROCm user-space evidence is not native-host validation.
- MI300X and MI308X are sibling GPU products under the CDNA3 architecture
  family and share the `gfx942` code path. Current CDNA3/gfx942 validation
  infrastructure evidence was recorded on MI308X, not MI300X, so full-suite
  MI300X validation remains blocked until timeout, clock-lock, timing, score,
  FP8, low-precision, and exact-hardware evidence boundaries are resolved.
- CDNA4 validation is unavailable because suitable hardware is not currently
  accessible.

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
- Full CDNA3-family hardware validation, including MI300X (`gfx942`), or CDNA4
  hardware validation without archived full-suite evidence and accepted
  timeout/skip boundaries from that hardware class.
- CDNA4 validation while suitable CDNA4 hardware is not currently accessible.
- NVFP4 or MXFP4 validation without suitable AMD hardware evidence.
- NVFP4/MXFP4 Quant benchmark ROCm adaptation or hardware validation while
  CDNA4-class hardware is unavailable; CDNA3 expected skips are not validation
  or performance evidence.
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
- v1.19 sidecar/report evidence as full paper validation, upstream SOLAR parity,
  score authority, leaderboard readiness, native-host ROCm Matrix validation,
  CDNA3-family validation, including MI300X (`gfx942`), CDNA4 validation, or
  new-hardware validation.
- v1.21 helper extraction, guardrail tests, or documentation updates as hard
  sandboxing, multi-tenant safety, paper-scale SOLAR parity, CDNA3-family
  validation including MI300X (`gfx942`), or hosted leaderboard readiness.

## Claim Upgrade Rules

| Desired upgraded claim | Evidence required before wording can change |
| --- | --- |
| Full ROCm paper validation | Complete 235-problem denominator accounting, execution closure, trace artifacts, failure analysis, score artifacts, and reproducible commands. |
| CDNA 3 hardware validation | Full adapted suite on real `gfx94*` hardware, environment sidecars, clock policy evidence, trace artifacts, timing/score artifacts where claimed, and documented accepted failures or skips. |
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
