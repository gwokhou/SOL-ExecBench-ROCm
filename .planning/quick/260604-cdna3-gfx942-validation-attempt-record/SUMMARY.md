---
status: complete
completed_at: "2026-06-04"
slug: cdna3-gfx942-validation-attempt-record
---

# Summary

Recorded the remote `gfx942` CDNA3 validation attempt as an internal blocked
validation record. The attempt showed a working ROCm/PyTorch environment, a
passing CDNA3 marker, a passing PyTorch smoke, and a passing HIP `flux_rope`
focused run, but the full suite failed with 8 failures.

The blocking issues are CPU-device synchronization in `call_and_collect_outputs`,
Triton static review misclassifying `triton.language.load`, and a real HIP
RMSNorm `HSA_STATUS_ERROR_EXCEPTION` on `gfx942`.
