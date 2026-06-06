# CDNA 3 Validation Handoff

**Created:** 2026-05-21
**Milestone:** v1.1 CDNA 3 Support and Migration Closure
**Status:** Full-suite infrastructure validated with known timeout blockers

## Purpose

v1.1 adds CDNA 3 code/schema support for `gfx94*` targets. It does not record a
real CDNA 3 hardware validation pass. The next milestone should validate the
adapted suite on at least one CDNA 3 GPU before documentation claims hardware
validation.

## Required Hardware

- AMD CDNA 3 GPU reporting a `gfx94*` architecture through PyTorch ROCm, such
  as `gfx940`, `gfx941`, or `gfx942`.
- ROCm >= 7.0.
- PyTorch ROCm environment matching the project lock/dependency configuration.

## Commands

Run the full adapted suite:

```bash
uv run --no-sync pytest tests/
```

Capture environment context:

```bash
uv run python - <<'PY'
import torch
print("torch", torch.__version__)
print("hip", torch.version.hip)
print("cuda", torch.version.cuda)
print("available", torch.cuda.is_available())
if torch.cuda.is_available():
    props = torch.cuda.get_device_properties(0)
    print("name", torch.cuda.get_device_name(0))
    print("gcnArchName", getattr(props, "gcnArchName", ""))
PY
rocm-smi --showproductname --showdriverversion --showhw || true
rocminfo | grep -E "Name: *gfx94|Marketing Name" || true
```

## Evidence To Record

- Exact GPU name and `gfx94*` architecture.
- ROCm/HIP/PyTorch versions.
- Full pytest command and final pass/skip/fail counts.
- Any skipped tests and whether they are expected under CDNA 3.
- Any CDNA 3-specific deviations or follow-up issues.
- For MI300X (`gfx942`) validation, also record dataset summary, per-problem
  traces, ROCm timing evidence, clock-lock evidence, AMD-native score report,
  FP8 status, and NVFP4/MXFP4 `deferred_no_amd_path` status as defined in the
  MI300X validation handoff.

## 2026-06-06 CDNA3 Full Validation Update

Cloud validation on MI308X (`gfx942`) / CDNA3 completed against the 235-problem
adapted dataset with the following interpretation:

- 235 problems were discovered and summarized.
- 220 problems produced complete passing traces in the uploaded validation log
  bundle.
- 15 Quant NVFP4/MXFP4 problems were skipped as expected on CDNA3 because
  default NVFP4/MXFP4 hardware validation requires CDNA4-class support.
- FlashInfer safetensors blob staging and refreshed dataset availability were
  sufficient for the full validation pass; no remaining missing-blob blocker was
  observed in the uploaded logs.
- Nested `eval_driver.py` timeout logs were initially misclassified as ordinary
  CLI failures, which caused timed-out shards to be omitted from `traces.json`
  while the old `summary.json` reported the containing problem as OK. Commit
  `2984c29` fixes the timeout classification by detecting nested
  `subprocess.TimeoutExpired` records in saved CLI logs.
- Targeted verification after the fix confirmed the expected behavior for
  `FlashInfer-Bench/014_gqa_paged_prefill_causal_h32_kv4_d128_ps1`: 30 traces
  were emitted, with 29 `PASSED` and 1 `TIMEOUT`; the summary correctly reports
  `FAIL`.

Known remaining CDNA3 validation blockers:

| Problem | Remaining issue | Evidence |
| --- | --- | --- |
| `FlashInfer-Bench/014_gqa_paged_prefill_causal_h32_kv4_d128_ps1` | 1 workload shard times out. | Targeted rerun after `2984c29`: 29 passed, 1 timeout, summary fail. |
| `FlashInfer-Bench/019_mla_paged_prefill_causal_h16_ckv512_kpe64_ps1` | 3 workload shards timed out in the uploaded full-validation logs. | Same nested `eval_driver.py` timeout log shape as the verified 014 case. |
| `L2/040_altup_predict_correction_cycle_backward` | 1 workload shard timed out in the uploaded full-validation logs. | Same nested `eval_driver.py` timeout log shape as the verified 014 case. |
| `L2/055_audio_encoder_conv_positional_layer_stack` | 1 workload shard timed out in the uploaded full-validation logs. | Same nested `eval_driver.py` timeout log shape as the verified 014 case. |
| Quant NVFP4/MXFP4 problems on CDNA3 | Expected skips, not correctness failures. | `cdna3_low_precision_hardware_unsupported`; run on CDNA4-class hardware for hardware validation. |

The old full-validation summary should not be used to claim zero failures,
because it predates the nested-timeout classification fix. The corrected
interpretation is: CDNA3 validation infrastructure is operational on MI308X
(`gfx942`), Quant NVFP4/MXFP4 skips are expected, and the remaining non-skip
blockers are the recorded timeout shards above. MI308X and MI300X share the
`gfx942` code path, but MI308X evidence is not benchmark-grade MI300X hardware
validation because the hardware configurations differ.

## Acceptance Criteria

- `uv run --no-sync pytest tests/` completes successfully on a real CDNA 3 GPU.
- The recorded environment shows `gfx94*`.
- Documentation support matrix can be updated from "code/schema support;
  hardware validation deferred" to "full adapted suite passed" only after the
  timeout blockers above are resolved or explicitly accepted in the claim
  boundary.
- MI300X-specific hardware-validation claims require
  `mi300x_validation_claim_blockers()` to return no blockers.
