# Release Candidate Validation

This guide defines the bounded engineering prerelease validation path for SOL
ExecBench ROCm. It helps maintainers collect reviewable evidence before tagging
a release candidate without upgrading that evidence into paper parity,
leaderboard readiness, hard-sandbox authority, or new hardware validation.
For the v1.25 release-claim boundary and artifact authority table, see
`docs/v1_25_release_notes.md`.

## Command

Run the CPU-safe prerelease validation path:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run scripts/release_candidate_validation.py \
  --output-dir out/release_candidate_validation
```

The command writes:

- `release_candidate_validation.json`
- `release_candidate_validation.md`

The JSON and Markdown summaries classify every check as `blocking`,
`deferred`, or `diagnostic-only`, and every non-passing or unavailable check has
an explicit next action.

## CPU-Safe Validation

The default command runs a bounded CPU-safe pytest set covering evaluator
contracts, closure, consistency, claim-upgrade, and trust-summary behavior. A
failing CPU-safe check is a **blocking** prerelease issue because it does not
depend on live ROCm hardware.

The wrapper composes existing tests; it does not change Trace, Definition,
Workload, Solution, correctness, timing, score, or evaluator contract schemas.

## Optional ROCm And Docker Smoke

On a ROCm-capable host, add runtime smoke evidence:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run scripts/release_candidate_validation.py \
  --output-dir out/release_candidate_validation \
  --include-rocm-smoke
```

On a Docker-capable ROCm setup, add container smoke evidence:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run scripts/release_candidate_validation.py \
  --output-dir out/release_candidate_validation \
  --include-docker-smoke
```

These checks may be `passed`, `failed`, `skipped`, or `unavailable` depending on
the host. Missing ROCm devices, unavailable Docker, or skipped hardware-marked
tests should normally be classified as **deferred** or **diagnostic-only**, not
as proof that the engineering prerelease is unusable.

Clock-policy environment values such as `SOL_EXECBENCH_CLOCKS_LOCKED`,
`SOL_EXECBENCH_GPU_CLK_MHZ`, and `SOL_EXECBENCH_DRAM_CLK_MHZ` are recorded when
present. They are evidence fields, not timing authority by themselves.

## Bounded Dataset Slice

If local benchmark assets are present, add a bounded dataset slice:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run scripts/release_candidate_validation.py \
  --output-dir out/release_candidate_validation \
  --include-dataset-slice \
  --dataset-dir data/SOL-ExecBench/benchmark \
  --dataset-limit 5
```

The dataset limit is required and must be positive. This path is intentionally
bounded engineering prerelease evidence. It is not full 235-problem paper
validation and must not be presented as paper parity.

The wrapper records expected trace, execution-closure, trust-summary, and
known-gap artifact paths where available. Missing assets or readiness blockers
should be reviewed as deferred evidence with concrete next actions.

## Failure Classification

Use these classifications when reviewing the summary:

| Classification | Meaning | Typical next action |
|----------------|---------|---------------------|
| `blocking` | Must be fixed before tagging a release candidate. | Fix the failing CPU-safe check or command. |
| `deferred` | Requires external hardware, dataset assets, Docker, or other evidence outside this bounded run. | Record the gap and rerun in the right environment when available. |
| `diagnostic-only` | Useful context but not release authority. | Review the sidecar, but do not treat it as correctness, timing, score, paper-parity, or leaderboard authority. |

## Claim Boundaries

This validation path supports an **engineering prerelease** claim only. It
should be interpreted alongside the support matrix in `docs/rocm.md`. In
particular, Docker/container user-space evidence remains distinct from
native-host validation. CDNA3/gfx942 validation infrastructure evidence exists
from MI308X (`gfx942`) runs, but full MI300X validation on the CDNA3 `gfx942`
target remains blocked until timeout, clock-lock, timing, score, FP8,
low-precision, and exact-hardware evidence boundaries are resolved. CDNA4
validation is unavailable because suitable hardware is not currently
accessible.

It does not provide:

- full 235-problem paper validation
- upstream SOLAR parity
- hosted leaderboard readiness
- hard sandbox or multi-tenant adversarial execution
- CDNA4 validation, because suitable hardware is not currently accessible
- full MI300X validation on the CDNA3 `gfx942` target while timeout,
  clock-lock, timing, score, FP8, or low-precision evidence boundaries remain
  unresolved
- native-host validation from Docker/container user-space evidence alone

Trace JSONL remains the canonical run artifact. Profile, static, environment,
Matrix, closure, consistency, claim-upgrade, trust-summary, and release
candidate validation summaries are diagnostic-only sidecar evidence. Bounded
dataset slices and support-matrix rows are provisional prerelease evidence, not
paper, score, leaderboard, sandbox, native-host, full MI300X validation on the
CDNA3 `gfx942` target, or CDNA4 validation authority.

Command stdout and stderr are streamed to temporary files while checks run.
Only bounded redacted tails are copied into the validation JSON, and temporary
stream files are removed after each check. Redaction is stateful across stream
chunks so credential-like values split at chunk boundaries are not exposed in
the recorded tail.
