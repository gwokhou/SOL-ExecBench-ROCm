# Internal Scripts

Scripts in this directory support repository maintenance, validation and
historical reproduction. They are not primary user-facing entry points;
retaining a campaign helper does not authorize rerunning its experiment.

Subdirectories:

- `upgrade/`: bounded stack-upgrade probes and strict paired benefit analysis;
  engineering-only checkpoint and prerequisites are documented in its README.
- `rdna4/`: bounded clock behavior checks, rocprofv3 overhead calibration, and
  exact RX 9060 XT/gfx1200 content-addressed validation. Local and manual
  self-hosted bundles are engineering evidence and never release authority.
- `cdna3/`: explicit-device validation helpers; hardware evidence is required
  before treating a target as qualified.
- `solar/`: shared BASE-01 samples/oracles and bounded SOLAR experiment helpers.
  The samples and workers are imported by current tests; retain them even when
  the original campaign is complete. Current E/C execution follows the frozen
  [next-iteration plan](../../docs/internal/solar-next-iteration-plan-2026-11-07.md).
- `reports/`: current compatibility-matrix JSON Schema export.
- `orojenesis/`: pinned two-build reproduction and provenance generation for
  the reviewed formal mapper artifact. The verifier publishes the first build
  only after the second build matches its binary and provenance byte-for-byte.

User-facing scripts remain in `scripts/`, including dataset download, dataset
execution, Docker environment setup, dataset inspection, and ROCm clock sudoers
setup.

Retired campaign entry points and their Git recovery locations are recorded in
the [retirement decisions](../../docs/internal/history-retirement-2026-11-08.md).
