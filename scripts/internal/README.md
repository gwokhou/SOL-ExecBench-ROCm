# Internal Scripts

Scripts in this directory are current repository-maintenance helpers rather than
primary user-facing entry points.

Subdirectories:

- `rdna4/`: bounded clock behavior checks, rocprofv3 overhead calibration, and
  exact RX 9060 XT/gfx1200 content-addressed validation. Local and manual
  self-hosted bundles are engineering evidence and never release authority.
- `reports/`: current compatibility-matrix JSON Schema export.
- `orojenesis/`: pinned two-build reproduction and provenance generation for
  the reviewed formal mapper artifact. The verifier publishes the first build
  only after the second build matches its binary and provenance byte-for-byte.

User-facing scripts remain in `scripts/`, including dataset download, dataset
execution, Docker environment setup, dataset inspection, and ROCm clock sudoers
setup.
