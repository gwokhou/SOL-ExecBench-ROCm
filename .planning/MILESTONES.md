# Milestones

## v1.0 ROCm Port (Shipped: 2026-05-21)

**Phases completed:** 6 phases, 21 plans, 18 tasks

**Key accomplishments:**

- Pinned ROCm Docker image, AMD GPU passthrough flags, and ROCm-aware entrypoint startup behavior
- PyTorch ROCm dependency declarations and lockfile resolved without CUDA/NVIDIA package residue
- ROCm runtime, HIP compiler, PyTorch ROCm, Triton ROCm, and selected library smoke tests
- ROCm-only Docker dependency pytest collection with superseded CUDA/NVIDIA smoke tests removed
- ROCm-native solution schema with hip_cpp, gfx1200, hip_cflags, and strict CUDA/NVIDIA migration errors
- ProblemPackager stages HIP/C++ solutions and injects AMD `--offload-arch` flags from explicit or local gfx targets
- HIP-aware build_ext.py discovers `.hip` and C/C++ sources, reads `hip_cflags`, and preserves the PyTorch extension loader contract
- Focused pytest audit guards Phase 2 schema/build paths against unallowlisted CUDA/NVIDIA residue

**Known gaps:**

- TEST-05 deferred: full adapted suite validation on CDNA 3 (`gfx94*`) was not recorded before v1.0 close. See `.planning/STATE.md` deferred items.

---
