# Project Research: Stack

## Milestone

v1.27 Copyright Provenance Cleanup

## Question

What stack additions or process changes are needed to clean up copyright and
provenance metadata for the ROCm port without changing runtime behavior?

## Findings

Apache-2.0 remains the correct project license. The relevant operational
requirements are not relicensing work; they are attribution hygiene:

- keep a copy of Apache-2.0 in the distribution;
- retain upstream copyright, patent, trademark, and attribution notices that
  pertain to retained or derivative upstream material;
- make modified upstream files visibly modified;
- include NOTICE material when the upstream distribution includes relevant
  NOTICE text;
- add this project's own attribution for independent modifications or new work
  without implying that added attribution changes the license.

SPDX file tags are appropriate for this repository because active source and
tests already use `SPDX-License-Identifier` and `SPDX-FileCopyrightText`.
SPDX permits multiple copyright tags in a file, which matches the needed
distinction between upstream-derived files and project modifications.

REUSE-style conventions are useful but should be adopted lightly. Full REUSE
conformance may require `LICENSES/` and `REUSE.toml` work that is not necessary
for this milestone. A project-level provenance manifest plus deterministic
tests is a better immediate fit.

## Sources

- Apache License 2.0: https://www.apache.org/licenses/LICENSE-2.0
- SPDX file tags: https://spdx.github.io/spdx-spec/v2.3/file-tags/
- REUSE specification 3.3: https://reuse.software/spec-3.3/

## Recommended Stack Changes

- Add or update a provenance document such as `docs/provenance.md` or
  `COPYRIGHTS.md`.
- Keep `LICENSE` as Apache-2.0.
- Review `THIRD_PARTY_NOTICES.txt` only for obvious stale or missing notices;
  do not turn this milestone into a full dependency legal audit.
- Add a small package-owned or script-owned provenance check if the existing
  residue audit cannot express file classifications cleanly.

## Non-Goals

- Rewriting git history for earlier blanket headers.
- Changing the project license.
- Achieving full REUSE certification unless it falls out naturally from the
  narrower cleanup.
