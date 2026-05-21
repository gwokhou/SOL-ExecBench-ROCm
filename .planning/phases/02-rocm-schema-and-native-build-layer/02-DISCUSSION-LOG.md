# Phase 2: ROCm Schema and Native Build Layer - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-05-21
**Phase:** 2-ROCm Schema and Native Build Layer
**Areas discussed:** Schema Language Names, AMD Hardware Targets, HIP Compile Options, CUDA Pattern Audit

---

## Schema Language Names

| Option | Description | Selected |
|--------|-------------|----------|
| Add `hip_cpp` | Introduce `hip_cpp` as the canonical ROCm-native language and migrate examples/tests toward it. | ✓ |
| Alias `cuda_cpp` | Accept `cuda_cpp` as a legacy alias for `hip_cpp` to reduce immediate schema breakage. | |
| Keep `cuda_cpp` | Keep the old value even though the port is ROCm-only. | |

**User's choice:** Add `hip_cpp`.
**Notes:** Follow-up decisions rejected `cuda_cpp` immediately, chose broad ROCm library/DSL replacement names now, and required strict explicit schema/docs errors for unsupported CUDA/NVIDIA language values.

| Option | Description | Selected |
|--------|-------------|----------|
| Reject `cuda_cpp` now | Strict ROCm-only schema; tests/examples must migrate immediately. | ✓ |
| Temporarily accept with warning | Load `cuda_cpp` as deprecated legacy input, but use `hip_cpp` for new output/docs. | |
| Auto-normalize silently | Treat `cuda_cpp` as `hip_cpp` without warning. | |

| Option | Description | Selected |
|--------|-------------|----------|
| Remove/reject now | Remove all NVIDIA-specific language values immediately. | |
| Keep until Phase 4 | Only change `cuda_cpp` now; defer library/DSL cleanup. | |
| Replace with broad ROCm names now | Add names like `hipblas`, `miopen`, `ck`, and `rocwmma`. | ✓ |

| Option | Description | Selected |
|--------|-------------|----------|
| Strict and explicit | Errors say CUDA/NVIDIA values are unsupported and name replacements where known. | ✓ |
| Migration-oriented | Explain replacement intent and leave wording/mapping discretion. | |
| Minimal | Rely mostly on enum validation. | |

---

## AMD Hardware Targets

| Option | Description | Selected |
|--------|-------------|----------|
| Architecture families + `LOCAL` | Use `RDNA4`, `CDNA3`, and `LOCAL`; map families to concrete build flags. | |
| Concrete gfx targets + `LOCAL` | Use values like `gfx1200`, `gfx942`, and `LOCAL` directly. | ✓ |
| Both families and gfx targets | Allow both architecture families and concrete gfx values. | |

**User's choice:** Concrete gfx targets plus `LOCAL`.
**Notes:** Follow-up decisions made `LOCAL` detect and inject the AMD gfx arch, limited Phase 2 to `LOCAL` plus this machine's detected gfx target, and required a documented/tested extension path for adding more targets later.

| Option | Description | Selected |
|--------|-------------|----------|
| Detect and inject gfx target | Probe AMD GPU and inject `--offload-arch=gfx...`. | ✓ |
| Compile generic/no arch flag | Let HIP defaults decide. | |
| Require explicit gfx target | Reject `LOCAL` for native HIP builds. | |

| Option | Description | Selected |
|--------|-------------|----------|
| RDNA4 + CDNA3 only | Include v1 hardware targets plus `LOCAL`. | |
| Broad ROCm 7 set | Include common supported AMD gfx targets beyond v1 hardware. | |
| Minimal detected only | Support `LOCAL` plus the gfx target on this machine. | ✓ |

| Option | Description | Selected |
|--------|-------------|----------|
| Enum-only strict | Only detected local gfx target and `LOCAL` accepted. | |
| Pattern-based | Accept any `gfx[0-9a-z]+` string. | |
| Enum + documented extension path | Strict now, with clear tests/docs for adding RDNA4/CDNA3 later. | ✓ |

---

## HIP Compile Options

| Option | Description | Selected |
|--------|-------------|----------|
| Rename to HIP fields | Replace `cuda_cflags` with `hip_cflags` and replace CUDA linker defaults. | ✓ |
| Add HIP fields, keep CUDA fields rejected | Add `hip_cflags`, keep `cuda_cflags` only to emit clear validation errors. | |
| Generic compiler fields | Use backend-neutral names like `device_cflags`. | |

**User's choice:** Rename to HIP fields.
**Notes:** Follow-up decisions chose minimal optimization defaults, continued use of `torch.utils.cpp_extension.load`, and accepted `.hip` plus C/C++ suffixes while rejecting `.cu`.

| Option | Description | Selected |
|--------|-------------|----------|
| Minimal HIP defaults | Optimization flags only; inject arch separately; avoid hard-coded ROCm links unless needed. | ✓ |
| Mirror CUDA defaults | Translate prior CUDA defaults into HIP equivalents. | |
| No defaults | Require all flags to be explicit. | |

| Option | Description | Selected |
|--------|-------------|----------|
| Use `torch.utils.cpp_extension.load` | Preserve extension-loading contract and adapt source/options. | ✓ |
| Call `hipcc` directly | Bypass PyTorch extension loading and manually build the shared object. | |
| Support both | Default to PyTorch extension loading with direct `hipcc` fallback. | |

| Option | Description | Selected |
|--------|-------------|----------|
| Accept `.hip` and C++ suffixes | Support `.hip`, C/C++ suffixes, and headers; drop `.cu`. | ✓ |
| Accept `.cu` temporarily | Allow `.cu` during migration. | |
| C++ suffixes only | HIP code must live in `.cpp`/`.cc`/`.cxx`. | |

---

## CUDA Pattern Audit

| Option | Description | Selected |
|--------|-------------|----------|
| Fail on Phase 2-owned paths | Fail schema/build paths, report/defer later-phase areas. | ✓ |
| Report only | Generate visibility but do not fail tests. | |
| Fail globally | Fail on all remaining CUDA/NVIDIA patterns anywhere. | |

**User's choice:** Fail on Phase 2-owned paths.
**Notes:** Follow-up decisions chose a focused pytest guard, limited the failing scope to schema/build core files and direct tests, and required any allowed remaining reference to have a reasoned allowlist entry.

| Option | Description | Selected |
|--------|-------------|----------|
| Pytest guard | Add focused tests that scan Phase 2-owned paths. | ✓ |
| Standalone script | Add a reporting command outside tests. | |
| Both pytest + script | Test the gate and expose a reusable command. | |

| Option | Description | Selected |
|--------|-------------|----------|
| Schema/build core only | `solution.py`, `problem_packager.py`, `build_ext.py`, and direct tests. | ✓ |
| Schema/build plus CLI wording | Include CLI compile messages/help text. | |
| Driver/data packages | All source under `src/sol_execbench/driver` and `core/data`. | |

| Option | Description | Selected |
|--------|-------------|----------|
| Allowlist with reason | Each remaining match listed with a short justification. | ✓ |
| Inline comments only | Allow references if nearby code explains why. | |
| No allowlist | Zero CUDA/NVIDIA terms in Phase 2-owned paths. | |

## the agent's Discretion

- Exact enum names for ROCm library/DSL categories beyond the locked examples.
- Exact AMD gfx detection implementation.
- Precise pytest allowlist structure.

## Deferred Ideas

- Evaluation runtime and timing/profiling migration remain Phase 3 work.
- Concrete public example migration remains Phase 4 work.
- Full hardware validation and marker semantics remain Phase 5 work.
