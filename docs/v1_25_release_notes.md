# v1.25 Engineering Prerelease Notes

v1.25 is an engineering prerelease for the SOL ExecBench ROCm port. It packages
the current ROCm execution path, prerelease validation wrapper, support matrix,
and claim-boundary documentation so external users can install, run, and review
the project without stronger research or service claims.

## Shipped Capability

- ROCm-port benchmark execution remains centered on the existing CLI, schemas,
  correctness checks, trace JSONL output, timing policy, and selected ROCm
  examples.
- Maintainers can run bounded release-candidate validation through
  `scripts/release_candidate_validation.py`.
- Users can read the engineering prerelease support matrix in `docs/rocm.md`
  and the claim policy in `docs/CLAIMS.md`.

## Validation Evidence

- CPU-safe release-candidate checks can produce recorded pass/fail summaries.
- Optional ROCm and Docker smoke checks can record environment and clock-policy
  evidence when the host supports them.
- Bounded dataset slices can produce trace, closure, trust, and known-gap
  artifacts when local benchmark assets are present.

These are engineering-prerelease evidence surfaces. They do not create paper
parity, upstream SOLAR parity, score authority, leaderboard readiness,
hard-sandbox authority, native-host validation from container evidence, new
hardware validation, MI300X/CDNA3 full-suite validation, or CDNA4 validation.

## Artifact Authority

| Artifact or evidence surface | v1.25 authority class | Interpretation |
| --- | --- | --- |
| Trace JSONL | canonical | The canonical run artifact for benchmark traces. |
| Environment, profile, static, Matrix, closure, consistency, claim-upgrade, trust-summary, and release-candidate validation outputs | diagnostic-only sidecar | Reviewable evidence and diagnostics only; not correctness, timing, score, paper parity, leaderboard, or hardware-validation authority. |
| Bounded dataset slices and prerelease support-matrix rows | provisional prerelease evidence | Useful for engineering prerelease review within the recorded scope; not full 235-problem paper validation. |
| Full 235-problem paper validation, upstream SOLAR parity, hosted leaderboard readiness, and hard-sandbox authority | deferred | Outside this prerelease milestone. |
| MI300X/CDNA3 full-suite validation | deferred | Requires a complete real-hardware evidence chain on the concrete CDNA3 target, MI300X (`gfx942`). |
| CDNA4 validation | unavailable | CDNA4 validation is unavailable because suitable hardware is not currently accessible. |

## Support Boundaries

- RDNA 4 evidence is engineering-prerelease evidence only where recorded
  artifacts and commands support that scope.
- Docker/container ROCm user-space evidence is not native-host validation.
- MI300X is the concrete CDNA3 hardware target represented by `gfx942`; schema
  and build readiness are not hardware validation.
- CDNA4 validation is unavailable because suitable hardware is not currently
  accessible.

## Known Limitations

- No full 235-problem paper-scale validation is claimed.
- No upstream SOLAR parity or NVIDIA B200 equivalence is claimed.
- No hosted leaderboard readiness or remote submission workflow is claimed.
- No hard multi-tenant sandbox or adversarial execution isolation is claimed.
- No native-host ROCm validation should be inferred from Docker/container
  user-space evidence alone.

## Public Entry Points

- `docs/GETTING-STARTED.md` for first-run setup.
- `docs/rocm.md` for ROCm support and hardware boundaries.
- `docs/CLAIMS.md` for allowed and forbidden claim language.
- `docs/release_candidate_validation.md` for prerelease validation commands.
- `docs/RESEARCHER-GUIDE.md` for research workflow interpretation.
