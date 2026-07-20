# Internal Scripts

Scripts in this directory are current repository-maintenance helpers rather than
primary user-facing entry points.

Subdirectories:

- `rdna4/`: bounded clock behavior checks and rocprofv3 overhead calibration.
- `reports/`: current compatibility-matrix JSON Schema export.

User-facing scripts remain in `scripts/`, including dataset download, dataset
execution, Docker environment setup, dataset inspection, and ROCm clock sudoers
setup.
