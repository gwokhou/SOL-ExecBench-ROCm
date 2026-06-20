---
date: "2026-06-14 14:01"
promoted: false
---

# Native HIP/ROCm AST reward-hack and mutation notes

## Context

This note captures a design discussion about strengthening SOL ExecBench ROCm
reward-hack defenses for Native HIP/ROCm submissions and about adjacent benefits
from adding structured native-source analysis.

The immediate question was whether current Native HIP/ROCm reward-hack checks use
AST analysis. They do not. Current static source review uses Python AST only for
`.py` solution files. Native `.hip`, `.cpp`, `.cc`, `.cxx`, `.c`, `.h`, and
`.hpp` sources are checked with regex rules.

## Current state

Current reward-hack defenses are layered:

- Static source review before user-code import.
- Eval-driver integrity snapshots before and after user-code import.
- Runtime checks for timing monkey-patching, lazy tensor outputs, and thread
  injection.
- Multi-round correctness checks with fresh inputs.
- Timing isolation through synchronized HIP-backed PyTorch device events and
  unique tensor allocation across iterations.

Native HIP/ROCm source review currently catches obvious text patterns such as
non-default streams, explicit stream synchronization, dynamic loading, process
or network access, and some precision downgrade patterns. It does not perform
C/C++ semantic analysis, so it can miss aliases, macro indirection, wrapper
functions, function pointers, template dispatch, and conditional compilation.

## Feasibility of AST upgrade

Adding AST-level Native HIP/ROCm analysis is feasible, but it should not be
treated as a small drop-in replacement for Python AST checks.

Recommended implementation path:

1. Start with read-only native source extraction.
2. Emit a project-owned serialized IR rather than persisting full Clang ASTs.
3. Add conservative reward-hack rules on top of that IR.
4. Later add mutation planning and source rewriting.

Likely parser options:

- Clang/LLVM AST: best semantic coverage for HIP C++, but requires accurate
  compile arguments, ROCm include paths, and version coordination.
- libclang Python bindings: easier to integrate into the Python harness, but
  HIP support and parser flags may be fragile.
- Semgrep or tree-sitter: faster to adopt and better than regex, but weaker for
  type, overload, macro, and include-aware semantic analysis.

The best long-term direction is a custom `NativeSourceIR` derived from Clang or
tree-sitter output, with source ranges, calls, declarations, launch sites,
stream usage, synchronization, global state, library calls, and findings.

## Serialization model

AST results can be serialized, but the project should serialize an extracted
projection rather than raw internal Clang AST objects.

Useful sidecar candidates:

- `native_source_ir.json`
- `native_source_review.json`
- `kernel_launch_map.json`
- `mutation_candidates.json`

Example IR fields:

- Translation unit and source hash.
- Kernel declarations and host wrapper functions.
- Calls to HIP runtime, ROCm libraries, dynamic loaders, file/process/network
  APIs, and thread APIs.
- Kernel launch configuration expressions.
- Stream and synchronization usage.
- Global or static mutable state.
- Source ranges for each finding.
- Optional behavioral descriptors for optimization-search workflows.

This keeps the persisted format stable even if the parser backend changes.

## Benefits beyond reward-hack defense

Native AST/IR would help more than security review.

- Mutation and fuzzing: mutate semantically meaningful locations such as block
  size, tile size, shared memory strategy, vectorized loads, reductions,
  barriers, precision casts, and library call parameters.
- Differential validation: generate targeted mutants to test whether benchmark
  correctness checks are strong enough to kill suspicious variants.
- Provenance: record kernels, launch sites, library calls, global state, and
  synchronization points as auditable evidence.
- Source-to-profile correlation: map `rocprofv3` kernel activity back to kernel
  declarations and launch sites.
- Source classification: verify whether a declared `hip_cpp`, `hipblas`,
  `miopen`, `ck`, or `rocwmma` solution matches its actual source behavior.
- User diagnostics: emit precise findings such as `kernel.cpp:42` using
  `std::thread` in a host wrapper, instead of broad regex evidence.
- Safer automatic rewrites: insert instrumentation or normalize launch patterns
  using source ranges instead of string substitution.

## KernelFoundry connection

The paper "KernelFoundry: Hardware-aware evolutionary GPU kernel optimization"
(`https://arxiv.org/abs/2603.12440`) is aligned with this direction, though its
primary focus is optimization rather than anti-cheat.

Relevant ideas:

- GPU kernel generation needs hardware-aware structure, not just generic code
  generation plus benchmark feedback.
- KernelFoundry uses MAP-Elites quality-diversity search to preserve diverse
  optimization strategies.
- It assigns generated kernels to behavioral coordinates using static code
  analysis.
- Its behavioral dimensions are memory access pattern, algorithmic structure,
  and parallelism coordination.
- It notes that KernelBench-style validation can produce spurious speedups from
  incorrect kernels that pass tests.

Project implication: the same native-source IR can support reward-hack defense,
mutation, provenance, profiler correlation, and behavioral descriptors for
future kernel search or audit workflows.

## Recommended next steps

1. Add a read-only native source analysis spike.
2. Define a minimal `NativeSourceIR` JSON schema.
3. Implement rule parity with current regex checks before adding new behavior.
4. Add tests with macro, alias, wrapper, and direct-call variants.
5. Keep AST findings advisory at first, then promote selected rules to blocking
   once parser reliability is proven across local ROCm environments.

Open design questions:

- Whether the first parser should be Clang/libclang, tree-sitter, or Semgrep.
- Whether parser failure should be fail-open, fail-closed, or environment-gated.
- How much compile-server context should be persisted for reproducible analysis.
- Whether mutation belongs in the main package, an internal tool, or a separate
  research workflow.
