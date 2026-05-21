# Phase 2: ROCm Schema and Native Build Layer - Context

**Gathered:** 2026-05-21
**Status:** Ready for planning

<domain>
## Phase Boundary

Make solution metadata and native compilation express ROCm targets and build
HIP/C++ solutions. This phase owns solution schema language/hardware values,
native HIP/C++ source staging, HIP compile options, AMD gfx target injection,
and a focused audit gate for CUDA/NVIDIA residue in schema/build core paths.

This phase does not port the evaluation runtime, timing/profiling,
reward-hack defenses, public examples, or full test-suite hardware marker
semantics except where direct schema/build tests must change.

</domain>

<decisions>
## Implementation Decisions

### Schema Language Names
- **D-01:** `hip_cpp` is the canonical native ROCm C++ language value for this port.
- **D-02:** `cuda_cpp` is rejected immediately. Do not silently normalize or keep it as a legacy alias.
- **D-03:** Phase 2 should introduce broad ROCm library/DSL language names, including `hipblas`, `miopen`, `ck`, and `rocwmma`, so NVIDIA-specific categories have ROCm-facing schema replacements. Concrete example migration remains Phase 4.
- **D-04:** Schema documentation and validation errors should be strict and explicit: CUDA/NVIDIA language values are unsupported in the ROCm port, and errors should name the ROCm replacement where one exists.

### AMD Hardware Targets
- **D-05:** Public solution metadata should use concrete AMD `gfx...` hardware targets rather than family labels such as `RDNA4` or `CDNA3`.
- **D-06:** `LOCAL` remains supported. For native HIP builds, `LOCAL` should probe the AMD GPU and inject the detected `--offload-arch=gfx...` flag.
- **D-07:** Phase 2 should initially support only `LOCAL` plus the gfx target detected on this machine.
- **D-08:** Unknown gfx targets should remain strict for now. The planner should include a documented and tested extension path for adding RDNA4/CDNA3 targets later.

### HIP Compile Options
- **D-09:** Replace CUDA-specific compile option fields with HIP-specific fields. In particular, `cuda_cflags` should become `hip_cflags`, and CUDA linker defaults should be replaced with ROCm/HIP defaults.
- **D-10:** HIP compile defaults should be minimal: optimization flags only. Target architecture injection comes from `target_hardware`; do not hard-code ROCm library links unless needed.
- **D-11:** Preserve the existing extension-loading contract by continuing to use `torch.utils.cpp_extension.load`, adapted for HIP/ROCm flags and source discovery.
- **D-12:** Native HIP solutions should accept `.hip` plus C/C++ suffixes. `.cu` should no longer be a valid native HIP entry source suffix.

### CUDA Pattern Audit
- **D-13:** The Phase 2 CUDA/NVIDIA audit should fail on Phase 2-owned schema/build paths, while leaving later-phase areas such as timing and examples out of the failing gate.
- **D-14:** Implement the audit as focused pytest guards.
- **D-15:** The failing audit scope is schema/build core only: `src/sol_execbench/core/data/solution.py`, `src/sol_execbench/driver/problem_packager.py`, `src/sol_execbench/driver/templates/build_ext.py`, and their direct tests.
- **D-16:** Any remaining CUDA/NVIDIA reference in those paths must appear in an allowlist with a short reason.

### the agent's Discretion
The planner may choose exact enum names for ROCm library/DSL categories beyond the locked examples above, the exact AMD gfx detection implementation, and the precise pytest allowlist structure, provided the decisions above are preserved.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Project Scope
- `.planning/PROJECT.md` — ROCm-only project scope, constraints, and compatibility expectations.
- `.planning/REQUIREMENTS.md` — Phase 2 requirements `SCFG-01`, `SCFG-02`, `BUILD-01`, `BUILD-02`, `BUILD-03`, and `BUILD-04`.
- `.planning/ROADMAP.md` — Phase 2 goal and success criteria.
- `.planning/phases/01-rocm-environment-baseline/01-CONTEXT.md` — prior decision that HIP/C++ staging and compiler migration are Phase 2 work.

### Codebase Maps
- `.planning/codebase/STACK.md` — current CUDA-oriented stack and native extension build dependency notes.
- `.planning/codebase/ARCHITECTURE.md` — CLI, packager, template, and schema data flow.
- `.planning/codebase/INTEGRATIONS.md` — current NVIDIA/CUDA integration points that Phase 2 replaces for schema/build.

### Phase 2 Code Surfaces
- `src/sol_execbench/core/data/solution.py` — solution language, hardware, binding, and compile-option schemas.
- `src/sol_execbench/driver/problem_packager.py` — source staging, native build detection, and target flag injection.
- `src/sol_execbench/driver/templates/build_ext.py` — native extension build template.
- `tests/sol_execbench/core/data/test_solution.py` — schema validation coverage.
- `tests/sol_execbench/driver/test_build_ext.py` — build template coverage.

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `src/sol_execbench/core/data/solution.py` already centralizes `SupportedLanguages`, `SupportedHardware`, `CompileOptions`, `BuildSpec`, and source/entry-point validation.
- `src/sol_execbench/driver/problem_packager.py` already owns native-language detection, staging, target flag injection, and compile command construction.
- `src/sol_execbench/driver/templates/build_ext.py` already validates `solution.json`, discovers native sources, calls `torch.utils.cpp_extension.load`, and normalizes the output shared object to `benchmark_kernel.so`.
- `tests/sol_execbench/core/data/test_solution.py` and `tests/sol_execbench/driver/test_build_ext.py` are the natural places to update schema/build expectations and add focused regression coverage.

### Established Patterns
- Native solutions are identified from schema language values, staged into a temporary directory, compiled before evaluation, and loaded later through the existing `benchmark_kernel.so` contract.
- The existing build path injects architecture flags during packaging rather than requiring every solution to specify all flags manually.
- Schema tests use direct Pydantic validation assertions and explicit error-message matches.
- Build template tests execute the template in a temporary directory with `torch.utils.cpp_extension` mocked, avoiding a real GPU/compiler dependency for unit coverage.

### Integration Points
- The CLI calls `ProblemPackager.compile()` only when the solution is considered native/C++ by the packager.
- `solution.json` is written to the staging directory before both compile and execute subprocesses, so any schema normalization or injected compile options must be reflected there.
- `build_ext.py` reads the staged `solution.json`; changes to `CompileOptions` must be understood by both the schema and template.
- Later Phase 3 evaluation work depends on Phase 2 preserving the shared-object contract and trace JSON discipline for compile failures.

</code_context>

<specifics>
## Specific Ideas

- Use strict ROCm-facing names now rather than compatibility aliases.
- Keep Phase 2 narrow: schema/build core paths should be clean enough to gate, while CUDA/NVIDIA residue in timing, examples, and broader tests belongs to later phases.
- The local machine's detected AMD gfx target is the only concrete target required immediately besides `LOCAL`; future RDNA4/CDNA3 targets need a clear extension path.

</specifics>

<deferred>
## Deferred Ideas

- Evaluation runtime, destination-passing behavior under ROCm, trace execution, and timing/profiling are deferred to Phase 3.
- Concrete migration of public examples to `hipblas`, `miopen`, `ck`, `rocwmma`, or other ROCm libraries is deferred to Phase 4.
- Full RDNA4/CDNA3 test-suite validation and broad hardware marker semantics are deferred to Phase 5.

</deferred>

---

*Phase: 2-ROCm Schema and Native Build Layer*
*Context gathered: 2026-05-21*
