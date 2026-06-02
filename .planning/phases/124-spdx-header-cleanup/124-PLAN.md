# Phase 124: SPDX Header Cleanup - Plan

## Goal

Apply the Phase 123 provenance policy to active file headers.

## Tasks

1. Bulk update files in `nvidia_notice.allowed`.
   - Keep NVIDIA SPDX line.
   - Add project SPDX line immediately after NVIDIA.

2. Bulk update files in `nvidia_notice.cleanup_candidates`.
   - Replace NVIDIA SPDX line with project SPDX line.
   - Preserve shebang lines and Apache-2.0 license lines.

3. Update provenance policy tests.
   - Allowed files must contain NVIDIA and project attribution.
   - Cleanup candidates must contain project attribution and no NVIDIA
     file-level attribution.
   - Active files with NVIDIA attribution must be exactly the allowlist.

4. Verify.
   - Run focused provenance policy tests.
   - Run the existing ROCm migration residue audit.
   - Run Ruff on touched tests.

## Non-Goals

- Do not add release-readiness gates in this phase.
- Do not update public release wording beyond provenance policy consistency.
- Do not rewrite git history.
