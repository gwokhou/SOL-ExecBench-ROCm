---
status: complete
created_at: "2026-06-04"
slug: fix-cdna3-validation-blockers
---

# Fix CDNA3 Validation Blockers

Fix the blockers surfaced by the remote `gfx942` validation attempt without
upgrading CDNA3 validation claims locally.

## Tasks

- Avoid CUDA synchronization when `call_and_collect_outputs` is running on CPU.
- Allow legitimate Triton `triton.language.load` memory reads while preserving
  loader and file-I/O reward-hack blocks.
- Make the HIP RMSNorm example reduction portable across ROCm wavefront sizes
  by removing wave32 shuffle assumptions.
- Sync embedded RMSNorm source in `solution_hip.json`.
- Run CPU-safe focused tests and provide cloud revalidation commands.

