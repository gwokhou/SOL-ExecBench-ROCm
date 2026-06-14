# Internal Scripts

Scripts in this directory are repository-maintenance, validation-closure, release
review, and report-generation helpers. They are not the primary user-facing entry
points.

Subdirectories:

- `rdna4/`: RDNA4 profiler timing, coverage, closure, and validation evidence helpers.
- `release/`: prerelease validation, bundle, and readiness helpers.
- `reports/`: report, schema, matrix diff, and trust-summary generators.

User-facing scripts remain in `scripts/`, including dataset download, dataset
execution, Docker environment setup, dataset inspection, and ROCm clock sudoers
setup.
