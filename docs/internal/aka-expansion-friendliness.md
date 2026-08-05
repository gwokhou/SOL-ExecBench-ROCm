# AKA × SOL-ExecBench friendliness analysis and corpus-expansion policy

> Internal design note. Answers two questions: **(1)** is SOL-ExecBench — taken
> as an *evaluation system* (agent-input paradigm, runtime model, compiler
> ecosystem) — friendly to AMD AgentKernelArena (AKA) task classes? **(2)** How
> does that friendliness verdict drive the expansion of this repo's
> AKA-derived problem set? The verdict sorts every AKA task into one of three
> handling categories that get **different** treatment.

Grounded in the SOL-ExecBench paper (arXiv 2603.19173) and this repo's code;
task counts verified against the pinned AKA clone at revision
`869228138e07e773b61dd7fc1d8cdc0435c7b405`.

---

## 1. The SOL-ExecBench capability envelope (three layers)

A task is "friendly" only if it fits *all three* layers simultaneously.

### Layer 1 — Agent input paradigm

Paper §3.3 + §4.2 + §4.5, instantiated by `src/sol_execbench/core/data/definition.py`
and `definition_reference.py`: the agent is given a **PyTorch `run()` reference +
a typed I/O tensor contract (`axes`/`inputs`/`outputs`) + a set of dynamic
workload shapes**, and must produce **one** solution bundle with **one**
`entry_point::{function}` matching that contract. The agent is *never* handed an
existing kernel to optimize and *never* handed a natural-language instruction as
the authoritative oracle. (`src/solar/` derives the SOL *bound* from its exact
semantic graph; it does not emit candidate kernels.)

### Layer 2 — Runtime model

Paper §4.4 + §4.4.1, instantiated by `driver/templates/{build_ext,eval_driver,
evaluation_orchestrator,reference_worker}.py` and `core/bench/`: a single
self-contained kernel compiled via `torch.utils.cpp_extension.load` (`hipcc`) for
HIP/C++ or Triton JIT, executed as `fn(*inputs)` / `fn(*inputs,*outputs)` on the
**default stream** under PyTorch's **eager allocator**, timed with HIP events
(10 warmup / 50 iters × 3 trials), target-derived 2×L2 clear (256 MiB fallback) + shifting-`data_ptr`
allocator before each iteration, STABLE_PEAK clocks, reference materialized in a
**trusted IPC worker**, and named per-output numeric, exact, code-distance, or
coupled top-k checks. Static + dynamic **reward-hack defenses** reject streams,
CUDAGraph, semantic caches, threading, precision downcasts, and file/loader
smuggling. There is **no** multi-stage-entry-point, custom-allocator,
non-default-stream, or repo-level-multi-file-edit path.

### Layer 3 — Compiler ecosystem

Paper §4.4, instantiated by `solution_models.py::SupportedLanguages` (closed
enum): `{pytorch, triton, hip_cpp, hipblas, miopen, ck, rocwmma}`. **Rejected:**
plain CUDA / NVIDIA runtime / PTX / cubin / ELF, FlyDSL, cuTile. `build_ext.py`
routes only `.hip/.cpp/.cc/.cxx/.c` through `hipcc`; there is no `nvcc` and no
FlyDSL compiler.

### Envelope, in one line

> *PyTorch-reference + typed-I/O-contract + dynamic-workload-shapes → single
> default-stream HIP/Triton/ROCm-library kernel, judged by tensor tolerance
> against a trusted-IPC reference, timed under HIP events with full reward-hack
> defenses.*

---

## 2. Per-suite verdict (8 AKA suites → category)

Verified transpilation mechanics (each AKA suite genuinely compiles+runs the
agent's target kernel; the differences are in *source paradigm* and *target
backend*):

| Suite (count) | Source paradigm | Target | Category | Why |
|---|---|---|---|---|
| `torch2hip` (57) | PyTorch `module_fn` | HIP C++ | **Cat1** | Isomorphic to SOL-ExecBench L1; clean C1–C4 I/O; `module_fn` cross-check available |
| `torch2flydsl` (45) | PyTorch `Model.forward` | FlyDSL | **Cat1 (~12 clean) / Cat2 (~30 FP8·MoE·arch)** | FlyDSL *target* is irrelevant (we lift the PyTorch oracle); clean elementwise/norm/matmul are Cat1; FP8/MoE/MXFP are fragile (C7/C8) |
| `instruction2triton` (31) | NL instruction + inline `y_torch` | Triton | **Cat2** | Loses instruction-following purpose + no `module_fn` cross-check; hosts the only AKA backward pass (`rmsnorm_bwd`) |
| `hip2hip` (32) | existing HIP kernel | HIP | **Cat3** | kernel-to-kernel; no liftable oracle |
| `triton2triton` (165) | existing Triton kernel | Triton | **Cat3** | kernel-to-kernel |
| `triton2flydsl` (51) | existing Triton kernel | FlyDSL (prompt-only) | **Cat3** | kernel-to-kernel; harness even runs the original Triton entry, so FlyDSL is unenforced |
| `flydsl2flydsl` (7) | existing FlyDSL kernel | FlyDSL | **Cat3** | kernel-to-kernel + FlyDSL |
| `repository` (9) | whole upstream repo | same-language | **Cat3** | multi-file repo-level; runtime has no single-bundle repo-edit concept |

**Totals:** 133 convertible (torch2hip 57 + torch2flydsl 45 + instruction2triton
31); 264 non-convertible by paradigm/ecosystem (hip2hip 32 + triton2triton 165 +
triton2flydsl 51 + flydsl2flydsl 7 + repository 9).

---

## 3. I/O representability

Paradigm fit and on-disk representability are independent. The current workload
contract supports more than the original random-tensor/single-output envelope:

- bounded integer generators for indices and masks;
- seed-sensitive custom inputs mixed with ordinary generated inputs;
- positive and simplex-valued structured inputs;
- scalar tensors, multi-output results, and mixed output dtypes;
- per-output numeric or exact checks;
- value/raw-bit code-distance checks for quantized outputs;
- coupled top-k ID and weight checks;
- `uint8`, `int8`, FP8, BF16, FP16, and FP32 schema dtypes where the target
  catalog permits them.

These capabilities admit the previously fragile CrossEntropy, BatchNorm,
KDLoss, FP8/MXFP, integer-quantization, and MoE routing families without
weakening their semantic checks. Each admitted capability has a manifest
coverage floor and focused contract tests.

Two boundaries remain deliberate:

- variable-rank cases must be split into rank-pinned Definitions because a
  tensor contract has a fixed rank;
- nondeterministic RNG kernels without a stable counter-based oracle cannot be
  given reproducible benchmark semantics.

## 4. Three-category handling policy

| Category | Verdict | Current handling |
|---|---|---|
| **Cat1** legal + structural advantage | preserve all harness strengths | Scored conversion with source cross-checking. |
| **Cat2** legal after explicit modeling | encode the structural requirement | Scored when the manifest declares and satisfies the required input/check capability; use a sentinel only when target compatibility, rather than semantic equivalence, is the intended claim. |
| **Cat3** outside the benchmark paradigm | do not weaken the benchmark to admit it | Reject; no manifest entry may reference a Cat3 suite. |

Manifest schema v7 gives every entry exactly three typed, content-addressed
`aka_artifacts`: `config`, `semantic_reference`, and `correctness_runner`.
`audit_aka_provenance` resolves and verifies all three roles at the pinned AKA
revision for torch2hip, instruction2triton, and torch2flydsl. The latter two
suites bind their actual test-file/model oracle rather than inventing a
`pytorch_code_functional/` path.

The manifest also binds `tolerance-calibration.json`. Its 163 scored workload
records pin the semantic Definition/Workload contract, formal `gfx1200` device
identity, repeated-run observations, output dtypes, sample count, and final
named output checks. Loading the corpus fails if coverage, hashes, exclusions,
or authored checks drift.

## 5. Cat3 reject log

These AKA suites or task classes remain outside the benchmark's contract:

| Suite / task class | Reason | Boundary |
|---|---|---|
| `hip2hip` | Existing kernel-to-kernel optimization has no independent PyTorch oracle. | Agent input paradigm |
| `triton2triton` | Existing kernel-to-kernel optimization has no independent PyTorch oracle. | Agent input paradigm |
| `triton2flydsl` | Kernel translation plus an unsupported FlyDSL target. | Agent input + compiler ecosystem |
| `flydsl2flydsl` | Kernel translation plus an unsupported FlyDSL target. | Agent input + compiler ecosystem |
| `repository` | Whole-repository edits cannot be represented as one solution bundle and entry point. | Agent input + runtime |
| Unstable RNG tasks | No deterministic semantic oracle is available. | Runtime/correctness |

These are scope decisions, not pending implementation promises. A proposal to
support them must define a new benchmark contract instead of adding a
compatibility alias to the current one.

## 6. Realized corpus

The current expansion contains 45 authored problems from the pinned AKA
revision: 43 scored problems covering 163 workloads, one compatibility
sentinel, and one target-incompatible sentinel. The scored entries span 37
torch2hip, six torch2flydsl, and two instruction2triton sources before role and
target filtering.

`scripts/internal/aka_author_seed.py` and the manifest's
`formal_coverage_requirements` define the realized selection and capability
floors. `tests/sol_execbench/core/dataset/test_aka_corpus.py` verifies the
denominator, roles, provenance bindings, capability coverage, and absence of
Cat3 suites.
