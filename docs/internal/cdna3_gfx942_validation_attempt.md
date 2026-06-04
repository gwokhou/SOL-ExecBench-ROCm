# CDNA3 gfx942 Validation Attempt

This document records a real `gfx942` CDNA3 validation attempt. It is not a
CDNA3 hardware-validation pass.

## Scope

- Repository HEAD: `d56fadca35b73cb116918774ab945ebcadb1d601`
- Remote workspace: `/mnt/workspace/SOL-ExecBench-ROCm`
- Local artifact archive: `/Users/guohaozhang/Downloads/cdna3-validation-artifacts.tgz`
- Hardware target: `gfx942`
- PyTorch: `2.10.0+rocm7.1`
- HIP: `7.1.25424`
- Validation date: 2026-06-04

## Environment Evidence

The remote host reported a ROCm-visible `gfx942` device through multiple
sources:

- `rocm-smi --showproductname --showdriverversion --showhw`
- `rocminfo`
- `rocm_agent_enumerator`
- `uv run sol-execbench doctor --json`

The diagnostic report status was `available`, including:

- AMD SMI probe available.
- `rocminfo` probe available.
- `rocm_agent_enumerator` probe available.
- PyTorch ROCm runtime available.
- Device memory copy succeeded.
- HIP-backed PyTorch event timing succeeded.

## Positive Results

- CDNA3 marker gate passed:
  `1 passed, 10 deselected in 3.65s`.
- PyTorch smoke evaluation passed for
  `examples/pytorch/gemma3_swiglu/solution_python.json`.
- Focused HIP/C++ `flux_rope_hip` example passed:
  `2 passed, 26 deselected in 228.08s`.
- Focused `test_torch_compile_no_reward_hack` passed:
  `1 passed in 57.41s`.

## Full Suite Result

The full adapted pytest suite did not pass:

```text
8 failed, 1390 passed, 62 skipped in 817.25s
```

The full-suite exit code was `1`.

## Blocking Failures

### CPU Device Synchronization Bug

`tests/sol_execbench/core/bench/test_utils.py` failed two CPU-device tests on a
GPU-visible host. `call_and_collect_outputs` calls
`torch.cuda.synchronize(device)` whenever `torch.cuda.is_available()` is true,
including when `device="cpu"`. PyTorch then raises:

```text
ValueError: Expected a cuda device, but got: cpu
```

### Triton Static Review Blocker

The Triton examples failed because static source review classified ordinary
Triton `tl.load` usage as `triton.language.load` under the
`unauthorized_file_or_loader` rule. The focused Triton example run ended with:

```text
3 failed, 3 passed, 22 deselected in 14.83s
```

This is a policy/static-analysis blocker, not evidence that Triton ROCm kernel
execution failed on `gfx942`.

### HIP RMSNorm CDNA3 Runtime Failure

The `rmsnorm_hip` example consistently failed on `gfx942`. The focused
serialized diagnostic run reported:

```text
RuntimeError: HIP kernel launch failed: unspecified launch failure
HSA_STATUS_ERROR_EXCEPTION: An HSAIL operation resulted in a hardware exception. code: 0x1016
Kernel Name: _Z20rmsnorm_h4096_kernelP14__hip_bfloat16PKS_S2_if
grid=[7168, 1, 1], workgroup=[1024, 1, 1]
```

The same focused run ended with:

```text
1 failed, 1 passed, 26 deselected in 223.79s
```

Because `flux_rope_hip` passed separately, this is evidence that the HIP/C++
path is not globally broken. The blocker is specific to the RMSNorm example or
its architecture assumptions on `gfx942`.

## Conclusion

CDNA3 validation was attempted on real `gfx942` hardware, but full validation
is blocked. Public and release-facing claims must continue to use deferred
language for CDNA3/MI300X hardware validation until these blockers are fixed and
a full passing evidence chain is recorded.

Do not update support matrices or prerelease claim flags to mark CDNA3,
MI300X, Triton ROCm examples, or native HIP/C++ examples as fully validated
based on this attempt.

## Follow-Up Fixes

After this attempt, quick task
`260604-fix-cdna3-validation-blockers` addressed the identified code blockers:

- `call_and_collect_outputs` now synchronizes only when the requested device is
  a CUDA/HIP device.
- Static source review allows resolved `triton.language.load` memory reads
  while preserving non-Triton `.load()` blocking.
- The HIP RMSNorm example now uses a shared-memory reduction without wave32
  shuffle assumptions.

These are remediation changes only. CDNA3/MI300X validation remains blocked
until the fixed code is re-run on real `gfx942` hardware and records a passing
full evidence chain.
