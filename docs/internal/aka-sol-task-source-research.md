# AKA and SOL-ExecBench Task-Source Research

> Internal research note comparing only the **origin, construction, and
> modeling assumptions of benchmark tasks** in AMD AgentKernelArena (AKA) and
> NVIDIA SOL-ExecBench. Evaluation metrics, agent implementations, and hardware
> performance are out of scope except where they determine which source tasks
> can enter a benchmark. Research snapshot: 2026-07-20.

## 1. Executive Summary

SOL-ExecBench and AKA are both useful GPU-kernel benchmarks, but they begin from
different source objects:

```text
SOL-ExecBench: model architecture -> computational subgraph -> normalized problem
AKA:           existing kernel/test/repository -> executable optimization task
```

SOL-ExecBench is **model-first and schema-first**. It selects production and
emerging AI models, extracts computational subgraphs, converts them into
standalone PyTorch semantic references, and curates a balanced benchmark. This
requires each admitted computation to support a relatively strong mathematical
specification: explicit inputs and outputs, tensor shapes and dtypes, a stable
subgraph boundary, and independent execution.

AKA is **ecosystem-first and harness-first**. It packages existing HIP, Triton,
PyTorch, FlyDSL, and repository-level work from projects such as vLLM,
ROCmBench, GPU Mode, KernelBench, AITER, SGLang, and rocPRIM. An AKA task can be
defined operationally by the source tree, build command, correctness runner,
and performance runner without first being normalized into one mathematical
schema.

Consequently:

- AKA tasks can be used directly to evaluate AMD kernel-optimization agents.
- AKA tasks cannot be treated unchanged as a model-grounded AMD equivalent of
  SOL-ExecBench.
- Building an AMD SOL-like corpus from AKA requires **semantic recovery,
  boundary selection, parameter explicitization, model/workload provenance,
  deduplication, and stratified sampling**. This is more than translating
  Triton or HIP code.
- Some AKA repository and runtime tasks should remain ecosystem tasks because
  converting them into standalone tensor functions would remove the behavior
  being optimized.

## 2. Scope and Terminology

This note uses the following distinctions:

- **Source object**: the artifact from which a benchmark task is derived, such
  as a model architecture, a kernel file, a unit test, or an entire repository.
- **Semantic contract**: the mathematical computation a candidate must
  preserve.
- **Operational contract**: the commands, environment, existing source, tests,
  and modification boundaries a candidate must preserve.
- **Model-grounded task**: a task traceable to a concrete model computation and
  realistic model configurations or traces.
- **Ecosystem task**: a task traceable to a library, benchmark, competition, or
  repository, but not necessarily to one model invocation.

“Coarse” or “rough” source tasks below does not mean low quality. It means the
task retains engineering context that has not been normalized into a canonical
mathematical problem representation.

## 3. Source Inventories

### 3.1 SOL-ExecBench

The SOL-ExecBench paper describes a model-grounded construction pipeline:

1. Select 124 production and emerging models from Hugging Face, Artificial
   Analysis, and arXiv across language, diffusion, vision, audio/speech, video,
   multimodal, and hybrid architectures.
2. Load each architecture definition, source code, and configuration constants.
3. Use a frontier LLM to extract 7,400 forward and backward computational
   subgraphs as standalone PyTorch implementations.
4. Characterize candidates across 11 dimensions, including operation type,
   model domain, precision, compute intensity, and forward/backward split.
5. Use stratified sampling to balance coverage, single-operation versus fused
   problems, and quantized operations.
6. Generate benchmark drivers, then apply expert review, LLM review,
   execution-based validation, and adversarial optimization-agent validation.
7. Retain 245 validated problems, publish 235, and reserve 10 for a competition.

The published set contains:

| Category | Count | Source character |
| --- | ---: | --- |
| L1 | 94 | Single-operation kernels extracted from real models |
| L2 | 82 | Multi-operation fused model subgraphs |
| Quant | 33 | Low-precision computations from quantized models |
| FIB | 26 | FlashInfer-Bench production inference primitives |
| **Total** | **235** | Model subgraphs plus traced production primitives |

The important source property is not merely that PyTorch appears in the task.
The PyTorch program is a **standalone semantic reference** reconstructed from a
model computation. It defines what an optimized implementation must compute;
it is not the required solution language.

Primary references:

- [SOL-ExecBench paper, Sections 3 and 4](https://arxiv.org/abs/2603.19173)
- [SOL-ExecBench dataset card](https://huggingface.co/datasets/nvidia/SOL-ExecBench)
- [NVIDIA benchmark introduction](https://research.nvidia.com/benchmarks/sol-execbench/blog/introducing-sol-execbench)

### 3.2 AgentKernelArena

AKA packages executable optimization work already present in the AMD and
open-source GPU ecosystem. A typical task directory contains:

```text
task/
  source/             # baseline implementation or repository target
  scripts/task_runner.py
  config.yaml         # target symbol, commands, prompt, and constraints
```

The runner and existing source jointly define the task contract. The source can
be a HIP or Triton kernel, a PyTorch-to-kernel conversion problem, a FlyDSL
translation, or a target inside a full upstream repository.

At the research snapshot, the AKA `main` branch contained 397 task
configurations across eight task suites:

| Suite | Count | Source form |
| --- | ---: | --- |
| `triton2triton` | 165 | Existing Triton kernels |
| `torch2hip` | 57 | PyTorch references to replace with HIP |
| `triton2flydsl` | 51 | Existing Triton implementations to translate |
| `torch2flydsl` | 45 | PyTorch references to replace with FlyDSL |
| `hip2hip` | 32 | Existing HIP kernels |
| `instruction2triton` | 31 | Written specification plus harness |
| `repository` | 9 | Full AITER or rocPRIM repository targets |
| `flydsl2flydsl` | 7 | Existing FlyDSL kernels |
| **Total** | **397** | Heterogeneous executable tasks |

The largest path-identified upstream sources were vLLM (118), ROCmBench (61),
GPU Mode (46), KernelBench (33), AITER (31), SGLang (21), and GEAK evaluation
(17). These counts describe the evolving repository tree, not a frozen release
contract. AMD's initial July 2026 introduction documented an earlier set of 214
configurations across four categories.

Primary references:

- [AgentKernelArena repository](https://github.com/AMD-AGI/AgentKernelArena)
- [AMD AgentKernelArena introduction](https://rocm.blogs.amd.com/software-tools-optimization/agent-kernel-arena/README.html)
- [AMD GEAK benchmark introduction](https://rocm.blogs.amd.com/software-tools-optimization/triton-kernel-ai/README.html)
- [KernelBench repository](https://github.com/ScalingIntelligence/KernelBench)

## 4. Different Admission Contracts

The source difference creates different admission criteria.

| Constraint | SOL-ExecBench | AKA |
| --- | --- | --- |
| Concrete source model | Normally required; FIB uses production inference primitives | Not required |
| Standalone tensor semantics | Required | Optional; runner/tests may be the contract |
| Explicit inputs and outputs | Required by the problem schema | May be distributed across wrappers and source |
| Symbolic axes and shape constraints | Explicit and normalized | May be embedded in tests or launch code |
| Independent PyTorch reference | Required | Present only for relevant task types |
| Subgraph independence | Required | Not required |
| Unified corpus-level sampling | Explicit stratified curation | Inherits the mix of imported sources |
| Existing API and file boundaries | Secondary | Often central |
| Build and repository context | Intentionally reduced | Can be part of the task |
| Host/runtime behavior | Usually fixed or excluded | Can be an optimization target |

SOL-ExecBench therefore assumes that an admitted source computation can be
modeled approximately as:

```text
outputs = f(inputs, parameters, symbolic_axes)
```

This implies mathematical closure: hidden state must be eliminated or made
explicit, the selected boundary must have stable tensor semantics, and the
computation must run independently of its original model.

AKA can instead use an operational contract:

```text
modify the designated implementation in this environment;
preserve the existing API and tests; improve measured performance
```

An AKA task may consequently retain implicit workspace conventions, stream
state, launch sequencing, compilation macros, dispatch logic, multi-file
templates, or repository dependencies. These are valid engineering concerns,
but they are not automatically a normalized mathematical specification.

## 5. PyTorch Is a Semantic Reference, Not a Kernel DSL Requirement

The presence of a standalone PyTorch reference does **not** restrict
SOL-ExecBench to PyTorch implementations. It gives the benchmark a portable
correctness oracle. Candidate implementations can use lower-level GPU
languages and libraries.

The structural advantage applies most strongly to computations that are
naturally expressible as independent tensor programs:

- PyTorch operator to HIP, CUDA, Triton, or another kernel language;
- operator fusion;
- attention, MoE, normalization, GEMM, and convolution subgraphs;
- forward and backward tensor computations with explicit shape and dtype
  contracts.

It applies less well to:

- optimization of an already specialized HIP or Triton implementation where
  low-level structure is part of the task;
- repository-scale AITER or rocPRIM work;
- communication collectives and multi-GPU coordination;
- memory-pool, stream, launch, and host/device scheduling changes;
- persistent kernels and cross-kernel pipelines;
- stateful or data-dependent behavior that cannot be represented faithfully by
  a standalone PyTorch tensor function.

This is a benchmark-boundary choice: SOL-ExecBench emphasizes mathematical
correctness and local kernel performance, whereas AKA can retain broader GPU
software-engineering behavior.

## 6. LLM and Deterministic Tool Responsibilities

The SOL-ExecBench paper explicitly uses a hybrid construction process rather
than a pure-LLM or pure-static-analysis pipeline.

| Stage | Appropriate mechanism in the paper |
| --- | --- |
| Model loading and configuration collection | Deterministic tooling |
| Important subgraph identification | Frontier LLM |
| Standalone PyTorch reference construction | LLM-assisted |
| Candidate characterization | Structured/deterministic analysis |
| Stratified sampling | Deterministic selection from characterized candidates |
| Benchmark-driver generation | LLM-based driver generator |
| Numerical and workload validation | Deterministic execution |
| Semantic and quality review | Human experts plus LLM review |
| Specification-gaming detection | Optimization-agent execution plus checks |

For an AMD adaptation, LLM output should be treated as a proposal, never as the
sole correctness authority. In particular:

- source and model provenance should be collected deterministically;
- observed shapes and dtypes should come from model configuration or runtime
  traces rather than LLM invention;
- an LLM can propose subgraph boundaries and standalone reference code;
- reference code must be checked against the original computation on multiple
  workloads;
- deduplication, characterization, and sampling should be reproducible;
- final admission should combine execution checks with expert or independent
  semantic review.

## 7. Can AKA Tasks Be Used Directly?

The answer depends on the benchmark claim.

### Direct use is valid for an AMD optimization-agent benchmark

AKA already provides baseline source, compilation, correctness, and performance
contracts. No model-to-subgraph transformation is necessary when the intended
claim is:

> The agent can optimize these AMD ecosystem tasks under their existing
> engineering contracts.

### Direct use is not valid for an AMD SOL-like model-grounded corpus

Unmodified AKA tasks do not collectively support the stronger claim:

> These tasks are a representative, stratified sample of computations from
> current production and emerging AI models.

That claim requires model provenance and a controlled construction process.
Library provenance alone is insufficient: a vLLM or AITER kernel may serve real
models, but its AKA task does not necessarily record which model invocation,
subgraph boundary, production shapes, and dtype distribution justify its
inclusion.

### Expected conversion feasibility by AKA task type

| AKA task type | SOL-style conversion outlook |
| --- | --- |
| `torch2hip` / `torch2flydsl` | Best fit if model and workload provenance can be restored |
| `instruction2triton` | Possible after adding an independently validated tensor reference |
| `hip2hip` / `triton2triton` / `flydsl2flydsl` | Extract mathematical core when useful; low-level context may be lost |
| `triton2flydsl` | Translation benchmark by default; model grounding must be established separately |
| `repository` | Generally keep as an ecosystem task; standalone conversion is often lossy |

## 8. Recommended Corpus Design for This Repository

Do not collapse model-grounded and ecosystem-grounded tasks into one
unqualified benchmark population. Preserve two explicit provenance classes:

```text
model_grounded
  source model -> subgraph -> reference -> observed workloads -> stratified set

ecosystem_grounded
  source repository/benchmark -> existing runner -> executable optimization task
```

An AMD SOL-like ingestion pipeline should use AKA as a candidate and evidence
source, not as the sole corpus definition:

1. **Deterministic intake**: record upstream repository, revision, license,
   source path, original test, target architecture, and task type.
2. **Model linkage**: identify concrete consuming models or production traces;
   retain `null` rather than inventing provenance when no linkage is available.
3. **Semantic recovery**: use static inspection and LLM assistance to propose a
   stable subgraph boundary and explicit inputs, outputs, axes, dtypes, and side
   effects.
4. **Reference construction**: create an independent PyTorch semantic reference
   only when it faithfully represents the source computation.
5. **Workload derivation**: derive shapes and distributions from model configs
   or traces, not only the existing kernel microbenchmark.
6. **Equivalence validation**: compare the original source behavior and the
   normalized reference across representative and adversarial workloads.
7. **Characterization and deduplication**: classify model domain, operation,
   precision, forward/backward role, fusion depth, compute intensity, and source
   family.
8. **Stratified selection**: build a reproducible sample with declared coverage
   targets.
9. **Separate retention**: keep non-convertible repository/runtime tasks in the
   ecosystem suite rather than weakening the model-grounded claim.

Minimum provenance fields should distinguish the three layers that AKA and
SOL-ExecBench otherwise conflate:

```text
source_model        # optional; concrete model identity and revision
source_subgraph     # semantic location or extracted computation identity
source_repository   # upstream implementation/test and revision
provenance_class    # model_grounded | ecosystem_grounded
```

## 9. Final Conclusions

1. SOL-ExecBench imposes more **computation-modeling constraints** than AKA: a
   source task must become an independent, parameterized tensor computation
   with a reliable semantic reference.
2. AKA can impose more **engineering constraints** than SOL-ExecBench: existing
   APIs, files, launch code, build systems, and repository behavior may remain
   part of the task.
3. SOL-ExecBench's stronger mathematical specification improves uniformity,
   comparability, workload analysis, and model-distribution claims, but excludes
   or simplifies some real GPU systems work.
4. AKA's rougher operational sources broaden engineering coverage, but upstream
   tests and repository names do not substitute for model provenance or
   stratified corpus construction.
5. Converting AKA into a SOL-like source is primarily a **semantic modeling and
   provenance problem**, not a kernel-language translation problem.

