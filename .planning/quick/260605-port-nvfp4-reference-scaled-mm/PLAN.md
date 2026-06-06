---
status: abandoned
created_at: "2026-06-05"
slug: port-nvfp4-reference-scaled-mm
closed_at: "2026-06-06"
resolution: superseded_by_cdna3_skip_policy
---

# Port NVFP4 Reference Scaled MM

Make Quant NVFP4 reference implementations runnable on ROCm by replacing
CUDA-only `torch._scaled_mm` dependence with a portable dequantized matmul
fallback while preserving benchmark semantics.

## Resolution

Abandoned. The project chose not to replace NVFP4/MXFP4 benchmark reference
semantics with a dequantized ROCm fallback. Current policy is to skip
NVFP4/MXFP4 Quant hardware validation on CDNA3 and defer ROCm adaptation and
hardware validation until CDNA4-class hardware is available.

## Tasks

- [x] Inspect NVFP4 reference helper shape and direct `_scaled_mm` call sites.
- [ ] Add shared unpack/dequantize/scaled-matmul helpers to inlined NVFP4 references.
- [ ] Sync `reference.py` and embedded `definition.json` reference sources.
- [ ] Run static checks and targeted validation that do not require local ROCm GPU.
