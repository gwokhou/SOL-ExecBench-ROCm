# Public release status

Historical v1.x prerelease automation is not part of the v3 command surface.
The current tree intentionally exposes no artifact-bundle or readiness scripts
that can publish an official score.

For v3, use these sources of truth:

- `problems/RX_9060_XT/manifest.yaml` for pinned corpus and authority status;
- `sol-execbench --format json contract evaluator` for ownership;
- `docs/user/RELEASE-SCORING.md` for evidence required by a future release;
- `sol_execbench.core.scoring.official_authority` for the fail-closed gate.

The checked-in manifest declares `release_authority_not_published`. A release
must not claim an official score, paper parity, upstream SOLAR parity, hosted
leaderboard authority or broad hardware validation until the required evidence
is implemented, pinned and independently verified.
