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
the authoritative oracle. (`src/solar/` derives the SOL *bound* via graph→einsum
conversion; it does not emit candidate kernels — `solar_bridge/learn_worker.py`
extends the einsum-handler lookup table, not a solution generator.)

### Layer 2 — Runtime model

Paper §4.4 + §4.4.1, instantiated by `driver/templates/{build_ext,eval_driver,
evaluation_orchestrator,reference_worker}.py` and `core/bench/`: a single
self-contained kernel compiled via `torch.utils.cpp_extension.load` (`hipcc`) for
HIP/C++ or Triton JIT, executed as `fn(*inputs)` / `fn(*inputs,*outputs)` on the
**default stream** under PyTorch's **eager allocator**, timed with HIP events
(10 warmup / 50 iters × 3 trials), target-derived 2×L2 clear (256 MiB fallback) + shifting-`data_ptr`
allocator before each iteration, STABLE_PEAK clocks, reference materialized in a
**trusted IPC worker**, correctness = `(atol, rtol, matched_ratio,
max_error_cap)`. Static + dynamic **reward-hack defenses** reject streams,
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

## 3. I/O representability (the second, independent axis)

Even when the paradigm/ecosystem fit (Cat1/Cat2 candidates), a task's
`(Input Representation, Output Representation)` must round-trip the on-disk
schema with structural advantage. Classification observed across the convertible
suites:

- **Cat1-friendly — C1–C4:** single random tensor; +scalars; +weight/bias
  tensors; multiple random tensors → single tensor. All harness strengths
  preserved: symbolic shape generalization, dtype-aware tolerance
  (`aka_tolerance.py`), deterministic `module_fn` oracle cross-check
  (`aka_equivalence_check.py`), harness-controlled adversarial workload sampling.
- **Cat2-fragile — C5–C8, C11, C13:**
  - C5 index/mask tensors → `RandomInput` cannot express index range;
    adversarial sampling broken (would need `CustomInput`).
  - C6 structured int32 offsets / paged-KV → needs `CustomInput` (and the
    all-or-nothing custom-mixing rule forces every input custom).
  - C7 multi-output tuples → single `ToleranceSpec` mis-calibrates mixed dtypes.
  - C8 FP8/int8 quant → paper's specialized quant-evaluator not instantiated.
  - C11 scalar/0-d output → degenerate-output guard can trip.
  - C13 pytest-parametrized (instruction2triton) → liftable, but no `module_fn`
    cross-check; correctness rests on the lifted reference.
- **Cat3-illegal — C9/C10, C12:**
  - C9/C10 variable-rank within one Definition → `_validate_tensor_axis_references`
    pins rank; must **split into rank-pinned problems** (each becomes Cat1).
  - C12 `uint8` → not in the `DType` enum (`definition_models.py`).

---

## 4. Three-category handling policy

| Category | Verdict | Handling in this repo |
|---|---|---|
| **Cat1** legal + structural advantage | keep every harness strength | `role: scored`, full conversion. Primary expansion surface |
| **Cat2** legal + structural disadvantage | convert, mark the disadvantage, no schema/runtime change | mechanical inclusion: rank-split (C9/C10 → Cat1), FP8 as `compatibility_sentinel` (C8), backward pass via instruction2triton with provenance-weakened marker (C13), clean torch2flydsl elementwise (bf16). The disadvantage is structurally visible: `suite: instruction2triton` ⇒ the equivalence cross-check skips it; FP8 ⇒ loose tolerance + sentinel role; rank-split ⇒ a separate rank-pinned problem |
| **Cat3** illegal | cannot represent | **rejected**, recorded below with a `reason_code`. No manifest entry references a Cat3 suite |

### Provenance binding note (Cat2)

Every entry's `aka_config_sha256` pins its AKA `config.yaml` at the pinned
commit (`aka_corpus.py::audit_aka_provenance`). For `torch2hip` entries the
`aka_source_sha256`/`aka_runner_sha256` additionally bind the `module_fn` oracle
and `correctness_check.py`. For `instruction2triton`/`torch2flydsl` entries those
two are empty (no `pytorch_code_functional/` tree) and the auditor skips them —
binding is config-only, which is the intended Cat2 "provenance-weakened" state.

---

## 5. Cat3 reject log

These AKA tasks/suites are **deliberately excluded** from the corpus. Each is
illegal under the envelope; admitting it would require inventing a different
benchmark (kernel-to-kernel optimization, FlyDSL compilation, repo-level edits,
or dtype-enum extension).

| Suite / task class | `reason_code` | Layer violated | Note |
|---|---|---|---|
| `hip2hip` (32) | `kernel_to_kernel_paradigm` | Layer 1 | agent is given an existing HIP kernel to optimize; no liftable PyTorch oracle |
| `triton2triton` (165) | `kernel_to_kernel_paradigm` | Layer 1 | existing Triton kernel → optimized Triton |
| `triton2flydsl` (51) | `kernel_to_kernel_paradigm` + `unsupported_backend_flydsl` | Layer 1 + 3 | existing Triton kernel; FlyDSL target is prompt-only (harness runs the original Triton entry) |
| `flydsl2flydsl` (7) | `kernel_to_kernel_paradigm` + `unsupported_backend_flydsl` | Layer 1 + 3 | existing FlyDSL kernel → optimized FlyDSL |
| `repository` (9: aiter×5, rocprim×4) | `repository_level_multi_file` | Layer 1 + 2 | whole-upstream-repo edits; runtime is single-bundle / single-entry-point |
| `torch2flydsl` FP8/MoE/MXFP (~30) | `quant_or_structured_io_fragile` (deferred, not rejected) | Layer 2 (evaluator) | C7/C8; legal but the specialized quant/structured evaluator is not instantiated — deferred to a future schema/runtime round, not a Cat3 reject |
| C12 `uint8` tasks (e.g. packed MXFP codes) | `unsupported_dtype_uint8` | schema | `DType` enum has no `uint8` |
| RNG kernels (`instruction2triton/test_randn`, `test_random_int`) | `non_deterministic_rng` | Layer 2 | no deterministic oracle (Philox counter sequences) |
| `CrossEntropy` with integer target (`l1n95`) | `index_tensor_adversarial_sampling` (deferred) | C5 | target is an index tensor; `RandomInput` miscalibrates it. Deferred to a future round with `CustomInput` author support, not a hard reject |
| `BatchNorm`-with-running-stats (`l2n52`) | `structured_input_positive_variance` (deferred) | C5/C8 | eval-mode `F.batch_norm` takes `running_var` as an input, but `RandomInput` yields `randn` (negative values) → `sqrt(negative)` → NaN. Deferred until `CustomInput` can supply a positive-variance tensor |
| `KDLoss` probability target (`14007`) | `structured_input_probability` (deferred) | C5/C8 | `F.kl_div` requires the `target` to be a probability distribution; random `target` → `log(negative)` → NaN, and the summed form returns a scalar (breaks the sanity check). Deferred with `CustomInput` + a tensor-output reduction |

The hard rejects (first 5 rows + uint8 + RNG) total **264 + a handful** tasks and
are out of scope for this benchmark's design. The *deferred* items
(torch2flydsl FP8/MoE, CrossEntropy) are legal-but-fragile Cat2 candidates held
for a later round that adds per-output `ToleranceSpec`, relaxes the
custom-input mixing rule, or extends the author to emit `CustomInput` workloads.

---

## 6. How this drove the expansion (commit reference)

The expansion grows the corpus from 15 → ~40 problems by drawing **only** from
the Cat1 and mechanical-Cat2 buckets above. See `scripts/internal/aka_author_seed.py`
`SPECS` and the manifest's `formal_coverage_requirements.combinations` for the
realized selection; the floor constraints there encode this policy (attention,
loss, ≥2 norm, a backward pass, an FP8 sentinel, a fused depth). A test in
`tests/sol_execbench/core/dataset/test_aka_corpus.py` asserts that no entry
references a Cat3 suite.
