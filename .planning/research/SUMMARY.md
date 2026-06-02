# Project Research Summary

## Milestone

v1.27 Copyright Provenance Cleanup

## Stack Additions

- Keep Apache-2.0 as the repository license.
- Continue using SPDX file tags.
- Add a lightweight provenance policy and classification artifact.
- Prefer deterministic tests and prerelease gates over full REUSE certification
  for this milestone.

## Feature Table Stakes

- Classify active files by provenance.
- Fix SPDX/copyright headers based on provenance.
- Preserve NVIDIA notices for retained or derivative upstream files.
- Attribute independent ROCm work to this project instead of NVIDIA.
- Update compliance, attribution, paper citation, and non-endorsement docs.
- Refactor residue and release-readiness gates to prevent future blanket
  header drift.

## Watch Out For

- Do not rewrite git history for ordinary blanket header correction.
- Do not remove NVIDIA notices from derivative files.
- Do not leave NVIDIA-only notices on clearly independent ROCm work.
- Do not turn this milestone into full legal review or relicensing.
- Do not conflate paper citation with file-level copyright ownership.

## Sources

- Apache License 2.0: https://www.apache.org/licenses/LICENSE-2.0
- SPDX file tags: https://spdx.github.io/spdx-spec/v2.3/file-tags/
- REUSE specification 3.3: https://reuse.software/spec-3.3/
- NVIDIA SOL-ExecBench repository: https://github.com/NVIDIA/SOL-ExecBench
- SOL-ExecBench paper: https://arxiv.org/abs/2603.19173
