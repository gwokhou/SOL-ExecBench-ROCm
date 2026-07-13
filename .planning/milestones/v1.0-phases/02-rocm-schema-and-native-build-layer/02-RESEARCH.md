# Phase 2 Research - ROCm Schema and Native Build Layer

## RESEARCH COMPLETE

## Scope

Phase 2 must make solution metadata and the native build path ROCm-native without
porting the evaluator, timing stack, examples, or full test matrix. The phase is
bounded to schema/build core files and their direct tests.

## Requirements Addressed

| Requirement | Research conclusion |
|-------------|---------------------|
| SCFG-01 | Replace `cuda_cpp` with canonical `hip_cpp`; add ROCm-facing library/DSL names including `hipblas`, `miopen`, `ck`, and `rocwmma`; reject CUDA/NVIDIA language values with explicit migration errors. |
| SCFG-02 | Preserve the overall `solution.json` shape: `spec.languages`, `target_hardware`, `entry_point`, `compile_options`, `sources`, and the shared-object contract remain recognizable. |
| BUILD-01 | Keep the existing staging flow and `benchmark_kernel.so` artifact name; adapt source suffixes to `.hip` plus C/C++ suffixes. |
| BUILD-02 | Replace SM/gencode logic with AMD gfx detection and `--offload-arch=gfx...` injection. |
| BUILD-03 | Preserve compile subprocess discipline: build noise may go to stderr/stdout, but evaluation trace JSON is not touched in this phase. |
| BUILD-04 | Add a focused pytest audit gate for CUDA/NVIDIA residue in Phase 2-owned schema/build files. |

## Current Code Findings

- `src/sol_execbench/core/data/solution.py` defines `SupportedLanguages`, `SupportedHardware`, `CompileOptions`, and entry-point suffix validation. It still exposes `CUDA_CPP`, `CUTLASS`, `CUDNN`, `CUBLAS`, `CUTE_DSL`, `CUTILE`, and `CUDNN_FRONTEND`.
- `CompileOptions` currently uses `cuda_cflags` with defaults `["-O3", "--use_fast_math"]` and `ld_flags` defaulting to `["-lcuda"]`.
- `src/sol_execbench/driver/problem_packager.py` treats `CUDA_CPP`, `CUTLASS`, `CUDNN`, and `CUBLAS` as native languages, probes `nvidia-smi`, and injects CUDA `-gencode` flags.
- `src/sol_execbench/driver/templates/build_ext.py` uses `torch.utils.cpp_extension.load`, discovers `.cu` and C/C++ files, passes `extra_cuda_cflags`, and includes CUTLASS paths.
- Direct tests already exist for schema validation, problem-packager arch injection, and the build template. These can be updated without requiring real native compilation in unit tests.

## ROCm / HIP Build Facts

- ROCm HIP compilation targets concrete AMD GFX architectures such as `gfx942`, `gfx1100`, or `gfx1200`; compiler docs identify `--offload-arch=<gpu>` as the HIP offloading target selector. Source: AMD ROCm compiler reference, `--offload-arch=<gpu>` entry: https://rocm.docs.amd.com/projects/llvm-project/en/docs-7.1.1/reference/rocmcc.html
- AMD HIP compiler docs show concrete `gfx...` targets and examples such as `amdclang++ --offload-arch=gfx942 kernel.cpp -o kernel.out`. Source: HIP compilers docs: https://rocm.docs.amd.com/projects/HIP/en/latest/understand/compilers.html
- Clang HIP docs state `.hip` files are recognized as HIP programs automatically and also mention `--offload-arch=native`/architecture discovery helpers. Source: HIP support docs: https://rocm.docs.amd.com/projects/llvm-project/en/docs-6.2.0/LLVM/clang/html/HIPSupport.html
- `rocminfo` enumerates GPU agents with `Name: gfx...`; `rocm_agent_enumerator -name` is designed to emit architecture names usable by scripts. Source: ROCm agent enumerator docs: https://rocm.docs.amd.com/projects/rocminfo/en/docs-7.2.0/how-to/use-rocm-agent-enumerator.html
- Local host evidence: `rocminfo` reports AMD GPU agent `gfx1200` and HIP compiler `7.1.52802` / ROCm `7.1.1`. Phase 2 should support `LOCAL` plus `gfx1200` initially, with tests/docs making target addition straightforward.
- PyTorch’s extension utilities remain the integration point for compiling Python-loadable native extensions. The current template already uses `torch.utils.cpp_extension.load`; Phase 2 should adapt its arguments and source discovery rather than replacing the build mechanism. Source: PyTorch extension docs: https://docs.pytorch.org/docs/main/cpp_extension.html

## Implementation Guidance

### Schema

- Use `HIP_CPP = "hip_cpp"` as the native ROCm C++ enum value.
- Add broad ROCm library/DSL enum values now: `HIPBLAS = "hipblas"`, `MIOPEN = "miopen"`, `CK = "ck"`, `ROCWMMA = "rocwmma"`.
- Decide whether those broad ROCm library/DSL names are native C++ languages or mixed categories:
  - `hip_cpp`, `hipblas`, `miopen`, `ck`, and `rocwmma` should be native/C++ category values for Phase 2 planning.
  - `pytorch` and `triton` remain Python-side values.
- Do not accept `cuda_cpp`. Because Python Enum validation normally raises a generic error before model validators run, strict explicit errors may require a `field_validator("languages", mode="before")` on `BuildSpec`.
- Rename `CompileOptions.cuda_cflags` to `hip_cflags`. Keep `cflags` and `ld_flags`; change `ld_flags` default from `["-lcuda"]` to `[]` unless implementation proves a PyTorch/HIP extension requires a default link flag.
- Default `hip_cflags` should be minimal, e.g. `["-O3"]`. Architecture flags are injected by the packager.
- Update docstrings and error messages to say HIP/C++ rather than C++/CUDA in Phase 2-owned code.

### Hardware Targets

- Replace `B200` with `GFX1200 = "gfx1200"` for Phase 2. Keep `LOCAL = "LOCAL"`.
- `LOCAL` should detect the local GPU architecture and inject `--offload-arch=gfx1200`.
- Preferred detection order:
  1. `rocm_agent_enumerator -name`, filtering out `gfx000`;
  2. `rocminfo`, parsing GPU agent `Name: gfx...`;
  3. return `None` if neither command is available or no GPU target is found.
- Do not use pattern-based acceptance for arbitrary `gfx...` values yet. Add tests that make adding `gfx942` or other future values a one-line enum/test extension.

### ProblemPackager

- Replace `_get_local_sm`, `_sm_to_gencode`, `_BLACKWELL_HARDWARE`, and `_inject_gencode_flags` with ROCm equivalents:
  - `_get_local_gfx() -> str | None`
  - `_gfx_to_offload_arch(gfx: str) -> str`
  - `_inject_offload_arch_flags(sol_dict: dict) -> dict`
- Check existing `hip_cflags` for `--offload-arch`, `-offload-arch`, or `--amdgpu-target` before injecting.
- For explicit `gfx1200` target, inject `--offload-arch=gfx1200`.
- For `LOCAL`, inject detected `--offload-arch=gfx1200` when detection succeeds.
- Deduplicate flags while preserving order.

### Build Template

- Continue reading staged `solution.json` and using `torch.utils.cpp_extension.load`.
- Source discovery should include `.hip`, `.cpp`, `.cc`, `.cxx`, and `.c`; it should not include `.cu`.
- Pass HIP options through the extension loader. The exact PyTorch keyword may remain `extra_cuda_cflags` because PyTorch’s extension API historically uses CUDA naming for the device-compiler argument; if retained, allowlist this API-name occurrence in the audit with the reason that it is PyTorch’s public keyword.
- Remove CUTLASS include paths and `CUTLASS_DIR`.
- Error text should say "HIP/C++ source files" instead of "CUDA/C++ source files".
- Keep `benchmark_kernel.so` rename behavior unchanged.

### Audit Gate

- Add a direct pytest guard under `tests/sol_execbench/` for Phase 2-owned paths:
  - `src/sol_execbench/core/data/solution.py`
  - `src/sol_execbench/driver/problem_packager.py`
  - `src/sol_execbench/driver/templates/build_ext.py`
  - `tests/sol_execbench/core/data/test_solution.py`
  - `tests/sol_execbench/driver/test_build_ext.py`
  - `tests/sol_execbench/driver/test_problem_packager.py`
- Scan for terms such as `cuda_cpp`, `CUDA_CPP`, `cuda_cflags`, `-gencode`, `nvidia-smi`, `B200`, `CUTLASS_DIR`, `cutlass`, `cudnn`, and `cublas`.
- Maintain an explicit allowlist mapping `(path, pattern)` to reason. Expected likely allowlist: `extra_cuda_cflags` in `build_ext.py` and build-template tests if PyTorch requires that keyword.
- Do not fail on evaluator, examples, docs, timing, or broad tests in Phase 2.

## Validation Architecture

The phase can be validated with fast unit/static tests:

- Schema tests: `uv run pytest tests/sol_execbench/core/data/test_solution.py`
- Packager tests: `uv run pytest tests/sol_execbench/driver/test_problem_packager.py`
- Build template tests: `uv run pytest tests/sol_execbench/driver/test_build_ext.py`
- Audit guard: `uv run pytest tests/sol_execbench/test_rocm_schema_build_audit.py`
- Combined Phase 2 fast suite: `uv run pytest tests/sol_execbench/core/data/test_solution.py tests/sol_execbench/driver/test_problem_packager.py tests/sol_execbench/driver/test_build_ext.py tests/sol_execbench/test_rocm_schema_build_audit.py`

## Risks and Mitigations

- **PyTorch API naming leak:** `torch.utils.cpp_extension.load` may still require a keyword named `extra_cuda_cflags` for HIP/ROCm builds. Mitigate with a narrow audit allowlist entry explaining it is an upstream PyTorch API name, not a project schema/build concept.
- **Generic enum errors:** Pydantic enum validation may hide custom migration guidance. Mitigate with before validators for `languages` and compile option keys.
- **Local-only hardware target:** Supporting only `gfx1200` initially can block other AMD hosts. Mitigate with a documented enum/test extension path and a clear error that names accepted targets.
- **Scope bleed into evaluator/examples:** `eval_driver.py`, `docs/user/solution.md`, examples, and broader tests still contain CUDA/NVIDIA terms. Keep the audit path-scoped to Phase 2-owned files.

## Sources

- AMD ROCm compiler reference: https://rocm.docs.amd.com/projects/llvm-project/en/docs-7.1.1/reference/rocmcc.html
- AMD HIP compilers docs: https://rocm.docs.amd.com/projects/HIP/en/latest/understand/compilers.html
- AMD HIP/Clang support docs: https://rocm.docs.amd.com/projects/llvm-project/en/docs-6.2.0/LLVM/clang/html/HIPSupport.html
- ROCm agent enumerator docs: https://rocm.docs.amd.com/projects/rocminfo/en/docs-7.2.0/how-to/use-rocm-agent-enumerator.html
- PyTorch C++/CUDA extension docs: https://docs.pytorch.org/docs/main/cpp_extension.html
