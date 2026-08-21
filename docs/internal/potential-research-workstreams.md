# Potential Research Workstreams Beyond SOL-ExecBench

> Internal research agenda, not an implementation roadmap or a statement of
> current benchmark capability. It separates limitations acknowledged by the
> SOL-ExecBench and SOLAR papers, methodological boundaries inferred from their
> evaluation designs, and open intersections left by adjacent GPU-kernel
> benchmarks. Research snapshot: 2026-08-21.

## 1. Executive Summary

The defensible research opportunity is not simply "run SOL-ExecBench on more
GPUs." Cross-platform kernel generation, source translation, unseen-shape
evaluation, production-derived workloads, hidden correctness tests,
profiler-guided optimization, energy measurement, and reinforcement-learning
agents all have adjacent prior work.

The narrower unresolved intersection is:

> Evaluate semantically matched GPU-kernel optimization across hardware using
> target-calibrated and uncertainty-aware SOL bounds, while factorizing the
> programming and evaluation path, auditing workload realization drift, and
> measuring how much target-side evidence an Agent needs to transfer an
> optimization mechanism.

This agenda decomposes that intersection into six workstreams:

| ID | Workstream | Primary object of study | Recommended role |
| --- | --- | --- | --- |
| W1 | CrossSOL-Eval | Evaluator and calibrated hardware bound | Immediate core |
| W2 | Value-aware and verified SOLAR | Semantic IR and bound validity | Parallel high-risk branch |
| W3 | Workload realization and representativeness | Benchmark distribution | Immediate core |
| W4 | Optimization-mechanism transfer | Transferable representation | Second-stage core |
| W5 | Target adaptation efficiency | Agent sample and resource efficiency | Second/third stage |
| W6 | Multi-objective deployment validity | External validity | Cross-cutting validation |

The recommended publication sequence is:

```text
measurement validity
  W1 CrossSOL-Eval + W3 workload realization              -> Paper A
                              |
                              v
mechanism identifiability
  W4 mechanism transfer + W5 target adaptation efficiency -> Paper B
                              |
                              v
policy learning
  hierarchical hardware-conditioned Agent/RL              -> Paper C

parallel branch: W2 value-aware/verified SOLAR
cross-cutting check: W6 deployment validity
```

Paper C is conditional on the earlier evaluator and mechanism hypotheses. It
should not be bundled into Paper A as one benchmark-plus-language-plus-RL
project.

## 2. Evidence Categories and Claim Discipline

This document uses three evidence labels:

- **Author-acknowledged limitation**: stated by the paper itself.
- **Methodological boundary**: an inference from the paper's experimental
  coverage or protocol; it must not be attributed to the authors as an
  explicit admission.
- **Adjacent-work boundary**: a neighboring paper already studies the generic
  direction, so novelty must be claimed at a narrower intersection.

The workstreams below are proposals. None is validated merely because this
repository contains a compatible schema, command, sidecar, hardware profile,
or experiment protocol.

## 3. Starting Point: Limitations and Evaluation Boundaries

### 3.1 SOL-ExecBench

The SOL-ExecBench paper explicitly identifies two limitations of the SOL
bound:

1. SOLAR derives bounds from tensor shapes rather than values, so it does not
   model value-dependent opportunities such as compression, constant
   propagation, or exploitable structure and repetition.
2. Power capping and thermal throttling can make a nominal hardware bound less
   tight under runtime variability.

Other evaluation boundaries follow from the published protocol:

- the reported hardware evaluation is centered on one NVIDIA B200 software
  stack rather than a factorized cross-hardware study;
- the scoring baseline in the paper version is internal and may evolve;
- non-default streams are disallowed, which excludes otherwise valid
  stream-based optimization and can prevent reproduction of a streamed
  baseline;
- PyTorch eager allocation can disadvantage non-PyTorch solutions at high
  memory pressure;
- best-of-k and search-budget scaling are possible extensions but are not the
  primary reported comparison object.

These boundaries motivate evaluator and deployment work, but they do not by
themselves prove that another vendor or device will produce a different
scientific conclusion.

### 3.2 SOLAR

The SOLAR paper makes the bound pipeline more explicit but retains several
important boundaries:

- translation from a PyTorch or JAX program into the analysis IR is checked by
  numerical execution on sampled inputs and has no general formal guarantee;
- the core affine representation assumes static shape constants, affine
  indexing, and no data-dependent branching, with irregular operations handled
  through modeled built-ins;
- the analysis remains shape-based rather than value-distribution-aware;
- nominal power and thermal variability can affect bound tightness;
- results projected to platforms without physical access are analytical
  explorations, not empirical evidence of performance portability.

"No observed SOL violations" is therefore an empirical result over the tested
set, not a universal proof that every translation and bound is valid.

### 3.3 What Adjacent Benchmarks Already Cover

The following generic novelty claims are not defensible as standalone claims:

| Generic claim | Adjacent coverage | Remaining narrower question |
| --- | --- | --- |
| Multi-hardware kernel benchmark | KernelBench-X, MultiKernelBench, KernelGenBench | Can hardware effects be isolated from programming abstraction, compiler, evaluator, and workload drift? |
| Cross-vendor translation | CASS and multi-backend benchmarks | Does optimization intent transfer, and how much target-native realization is required? |
| Generalization to unseen configurations | AgentKernelArena and KernelGenBench | What target-side cost is needed to reach a calibrated target-native threshold? |
| Production-representative workloads | Atrex-Bench and FastKernels | How do weighting, common support, and workload realization change conclusions across hardware? |
| Hidden correctness and memory metrics | KernelBench-Verified | Can correctness, memory, energy, and SOL evidence be integrated without collapsing incompatible quantities? |
| SOL-guided search budgeting | SOL-guided muCUTLASS work | Does calibrated cross-hardware SOL improve adaptation efficiency rather than only stop search early? |
| Profiler- or energy-guided optimization | KernelPro | Do structured mechanisms predict transfer better than device identity or latency-only history? |
| RL for kernel optimization | CUDA Agent and related systems | Does reward actually update a transferable policy, and does it reduce held-out-hardware adaptation cost? |

The intended contribution is therefore an unresolved intersection of adjacent
work, not a claim that cross-platform kernel optimization is an empty field.

## 4. Common Experimental Factorization

Every cross-hardware claim should identify the target as:

```text
H = hardware and visible device configuration
P = programming abstraction and solution language
C = compiler, backend, libraries, and toolchain
E = runtime and evaluation state
W = realized workload, including shape, dtype, layout, values, and weighting
```

A GPU model name is not a complete `H`, and a portable source program is not
evidence that performance or Agent behavior is portable. The experimental
matrix should distinguish at least:

1. **H-only shift**: hold `P/C/E/W` fixed where technically meaningful.
2. **H+C shift**: use the same high-level abstraction with target-specific
   compilation.
3. **Backend transfer**: hold task semantics fixed while changing target
   lowering or backend.
4. **Language transfer**: compare target-native CUDA/HIP/Triton or library
   realizations without calling the result hardware-only generalization.
5. **Full ecosystem transfer**: allow all target-native tools and report it as
   an upper-bound practical comparison, not a causal hardware ablation.

For every `(operator, workload, hardware)` cell, retain separate measurements:

```text
T = absolute candidate latency
R = T_baseline / T_candidate                 target-baseline speedup
E = T_SOL / T_candidate                      roofline/SOL achievement
S = 1 / (1 + (T_candidate - T_SOL)
             / (T_baseline - T_SOL))         SOL headroom coordinate
```

`R`, `E`, and `S` answer different questions and must not be pooled into one
unnamed "normalized performance" metric. Report bound uncertainty and reject
or separately classify cells where `T_baseline` is too close to `T_SOL` for a
stable headroom denominator.

## 5. W1: CrossSOL-Eval

### Research Question

Can a target-calibrated, hierarchical SOL evaluator produce comparable and
well-calibrated headroom evidence across heterogeneous GPUs?

### Falsifiable Hypothesis

A bound that combines measured hardware capabilities, workload semantics,
memory hierarchy, wave/occupancy constraints, and runtime-state uncertainty
will predict bottleneck regime and attainable headroom on held-out hardware
better than a nominal peak roofline.

### Minimum Experiment

1. Select semantically matched operators with fixed `P/C/E` and an audited
   common workload support.
2. Calibrate compute modes, effective memory levels, launch overhead, clock
   state, and stable thermal/power behavior on each physical target.
3. Evaluate multiple independently optimized target-native kernels per cell,
   including strong vendor-library and expert baselines.
4. Hold out at least one hardware configuration from model fitting.
5. Compare nominal roofline, calibrated hierarchical SOL, and achieved
   target-native performance.

### Required Metrics

- empirical bound-violation rate and severity;
- bound tightness against the best known target-native kernel;
- bottleneck-regime agreement with runtime profiling;
- uncertainty coverage and calibration error;
- ranking stability under repeated clock, cache, process, and thermal controls;
- `T`, `R`, `E`, and `S` with target-native denominators.

### Dependencies and Stop Conditions

W1 depends on stable target qualification, paired timing, content-bound SOL
inputs, and sufficiently strong baselines. If calibration cannot reduce
violation or regime error beyond nominal roofline on held-out hardware, the
result supports a negative evaluator study but not a new cross-hardware score.

### Current Repository Leverage

The repository already contains hardware identity, timing controls, formal
SOLAR artifacts, profiler evidence, and cross-hardware study planning. However,
the current [hardware-generalization protocol](../user/HARDWARE-GENERALIZATION.md)
explicitly excludes SOL evidence until a content-bound SOL input is specified.
That is the boundary W1 must close rather than assume.

## 6. W2: Value-Aware and Verified SOLAR

### Research Question

How should a hardware lower bound change when admissible optimization depends
on input values or data structure rather than shapes alone?

### Falsifiable Hypothesis

Explicit value-distribution descriptors and guarded semantic specialization can
tighten SOL bounds for sparse, repetitive, masked, indexed, or constant-rich
workloads without increasing hidden-distribution correctness failures or bound
violations.

### Contract Split

Two regimes must remain distinct:

- **Universal semantic contract**: the candidate must be correct over the full
  declared input domain. Observed value statistics cannot authorize an unsafe
  specialization.
- **Distribution-conditioned contract**: a data distribution or fixed data
  identity is part of the workload. A specialized fast path must either prove
  its precondition or retain a correct guarded fallback.

Conflating the two would turn value awareness into benchmark overfitting or
reward hacking.

### Minimum Experiment

1. Add structured descriptors such as sparsity, repetition, mask topology,
   index entropy, bounded ranges, and constants.
2. Construct adversarial hidden distributions that preserve shapes while
   violating candidate value assumptions.
3. Compare sampling-only validation with symbolic constraints, interval/range
   analysis, guarded replay, or proof-carrying lowering.
4. Evaluate dense/sparse, gather/scatter, conditional, and structured
   compression cases on at least two hardware families.

### Required Metrics

- bound-tightness improvement over shape-only SOLAR;
- semantic-translation failure rate on hidden distributions;
- empirical SOL violation rate;
- proof or guard coverage;
- analysis cost and unsupported-operation rate.

### Dependencies and Stop Conditions

W2 requires a versioned value-semantic workload contract and stronger
translation validation. If value descriptors tighten the bound only by fixing
public inputs or weakening hidden correctness, the hypothesis fails. This is a
compiler/formal-analysis branch rather than a prerequisite for the first
cross-hardware evaluator paper.

## 7. W3: Workload Realization and Representativeness

### Research Question

When hardware constraints change the feasible shapes, dtypes, layouts, or
batching regimes, what is the valid cross-hardware comparison population?

### Falsifiable Hypothesis

Reporting common support, hardware-specific extensions and omissions, and
measured distribution drift will produce more stable and attributable Agent
rankings than either a uniform shape grid or silent per-device task filtering.

### Minimum Experiment

1. Freeze task definitions and workload-generation rules before target
   realization.
2. Materialize target views from measured visible resources rather than GPU
   names alone.
3. Report the full target population and the common-support population
   separately.
4. Compare equal-Definition weighting, equal-workload weighting, and one
   independently justified production weighting.
5. Perturb capacity, shape tiers, dtype support, and workload frequencies to
   measure ranking sensitivity.

### Required Metrics

- Definition and workload retention;
- extensions, omissions, and skip-reason composition;
- categorical and continuous drift measures;
- common-scale and resource-utilization shift;
- Agent-ranking stability under alternative justified aggregations;
- uncertainty from Definition-cluster bootstrap rather than treating workloads
  from one Definition as independent.

### Dependencies and Stop Conditions

W3 depends on provenance-bound generator rules and measured device
configurations. It must not claim strict same-distribution generation when
hardware feasibility changes the realized population. If conclusions are
stable across reasonable realization and weighting choices, that negative
result is still useful evidence that drift is not the dominant explanation.

### Current Repository Leverage

The existing protocol already reports target-full and common-support results,
capacity classes, missingness, and workload drift. W3 must validate whether
those quantities explain ranking changes; merely emitting the report is not a
scientific conclusion.

## 8. W4: Optimization-Mechanism Transfer

### Research Question

Does an optimization mechanism transfer across hardware more reliably than its
concrete kernel realization?

### Representation

Separate transferable intent from target realization:

```text
intent tau:
  fusion, data reuse, parallel decomposition, memory placement,
  numerical reformulation, communication avoidance

realization rho(H, P, C):
  tile sizes, thread/wave mapping, vector width, pipeline depth,
  instruction selection, launch configuration, backend-specific code
```

### Falsifiable Hypothesis

A structured mechanism representation, combined with static and dynamic
evidence, will predict cross-hardware transfer success and failure layer better
than GPU identity, source similarity, or latency-only search history.

### Minimum Experiment

1. Build paired counterfactual kernels for one semantic intent across hardware.
2. Annotate the intended mechanism separately from its concrete realization.
3. Apply controlled interventions: preserve the mechanism while changing
   realization, then preserve implementation style while removing the
   mechanism.
4. Hold out operators and at least one hardware target when training any
   transfer predictor.
5. Compare no-transfer, code-copy, high-level-mechanism transfer, and
   target-native oracle conditions.

### Required Metrics

- mechanism-label agreement and inter-reviewer reliability;
- held-out transfer-success prediction;
- target-relative headroom loss `Delta S`;
- correctness, compile, runtime, and performance failure layers;
- profiler-predicted versus observed bottleneck change;
- mechanism stability versus realization stability across targets.

### Dependencies and Stop Conditions

W4 depends on W1 for comparable target headroom and W3 for a matched comparison
population. If mechanism labels do not predict held-out transfer beyond simple
device and latency features, a hierarchical Agent based on those labels is not
justified.

## 9. W5: Target Adaptation Efficiency

### Research Question

How much target-native interaction is required for an Agent to recover a given
fraction of the target hardware's attainable performance?

### Falsifiable Hypothesis

An Agent that transfers optimization mechanisms and adapts only their concrete
realization will reach a target-native SOL/headroom threshold with fewer target
GPU seconds, compilations, profiler calls, rollouts, and tokens than black-box
latency search.

### Minimum Experiment

1. Seal model/checkpoint, Agent code, prompt, tool set, budget, candidate
   provenance, and training exposure.
2. Use true hardware-held-out and workload-held-out targets.
3. Compare black-box search, persistent-memory/search baselines,
   mechanism-transfer search, and target-native oracle development.
4. Publish learning curves rather than only best-of-budget endpoints.
5. Repeat with fixed `P/C/E` before adding language or full-ecosystem transfer.

### Required Metrics

- `S_h(n)`: target headroom after `n` target interactions;
- `N_epsilon`: interactions needed to enter an epsilon-neighborhood of a
  target-native threshold;
- GPU seconds, compilations, profiler calls, rollouts, tokens, and API cost;
- hidden correctness and reward-hack rejection rate;
- common-support retention and failure-layer attribution;
- final target-native geomean speedup and SOL-threshold coverage.

### Dependencies and Stop Conditions

W5 depends on W1, W3, and at least a minimally predictive W4 representation.
Execution feedback, best-of-N, MCTS, evolution, Reflexion, or persistent memory
must not be labeled RL unless reward updates policy or value parameters. If the
method improves final performance but not target-side adaptation cost, the
result is an optimizer improvement rather than evidence of transfer efficiency.

The current repository intentionally leaves prompting, training, tool use, and
solution generation to an external Agent harness. W5 therefore requires a
separately sealed Agent system; the evaluator alone cannot establish this
claim.

## 10. W6: Multi-Objective and Deployment Validity

### Research Question

When does a kernel-level latency or SOL ranking predict end-to-end deployment
benefit, and when do system effects reverse it?

### Falsifiable Hypothesis

Memory pressure, allocator behavior, streams, concurrency, compilation cost,
energy, and thermal state will cause systematic ranking reversals that a
latency-only kernel benchmark misses; a separately reported Pareto view will
predict deployment selection better than a scalar score.

### Minimum Experiment

1. Select kernels embedded in reproducible model or serving paths.
2. Replay them under isolated default-stream timing and realistic concurrency,
   allocator, and stream conditions.
3. Measure compile/JIT cost, peak memory, energy, clock/thermal state, and
   end-to-end throughput or latency.
4. Compare kernel ranking with deployment ranking under multiple traffic or
   batch regimes.

### Required Metrics

- kernel and end-to-end ranking correlation;
- ranking-reversal rate with confidence intervals;
- latency/memory/energy Pareto coverage;
- compilation amortization point;
- thermal and power-state sensitivity;
- fallbacks and integration failures, not only successful kernels.

### Dependencies and Stop Conditions

W6 requires controlled system replay and reliable energy/thermal collection.
It should preserve distinct metrics rather than invent an arbitrary weighted
sum. If kernel rankings remain stable, the study establishes a domain in which
microbenchmark external validity is stronger; it does not justify a universal
deployment claim.

## 11. Proof-Obligation Ladder

The program should advance only when each prior gate has usable evidence:

| Gate | Required proof | Failure consequence |
| --- | --- | --- |
| G1 Evaluator identifiability | Stable timing, strong target baseline, calibrated bound, controlled `H/P/C/E` | Stop cross-hardware score claims; publish only evaluator diagnostics or a negative study |
| G2 Workload identifiability | Frozen realization rules, common-support audit, drift and aggregation sensitivity | Do not attribute ranking changes to hardware or Agent transfer |
| G3 Mechanism identifiability | Reproducible labels/interventions predict held-out transfer | Do not build a mechanism hierarchy on an unvalidated representation |
| G4 Adaptation efficiency | Lower target-side cost at matched correctness and target threshold | Claim optimizer quality only, not transfer or sample efficiency |
| G5 Learned-policy effect | Reward updates policy/value and beats non-RL search under matched budgets | Do not call the system RL or credit improvement to policy learning |
| G6 Deployment validity | Kernel improvements survive realistic integration and Pareto constraints | Restrict conclusions to isolated kernel evaluation |

This ladder allows every stage to yield an independently falsifiable result and
prevents a downstream Agent result from masking an invalid evaluator.

## 12. Publication Packaging

### Paper A: Factorized Cross-Hardware Evaluator

**Scope:** W1 + W3.

**Claim:** a target-calibrated, workload-audited evaluator makes cross-hardware
kernel-Agent comparisons more attributable than target-baseline speedup alone.

**Required evidence:** physical target measurements, held-out hardware,
target-native baselines, bound uncertainty, common-support and drift analysis,
and aggregation sensitivity.

**Likely fit:** MLSys for the evaluator/system contribution; NeurIPS Datasets
and Benchmarks for evaluation methodology and released artifacts.

### Paper B: Mechanism Transfer and Target Sample Efficiency

**Scope:** W4 + W5, using Paper A's evaluator.

**Claim:** mechanisms are more transferable than concrete realizations and
reduce target-side adaptation cost.

**Required evidence:** counterfactual kernel pairs, held-out operators and
hardware, target adaptation curves, failure-layer attribution, and strong
black-box/search baselines.

**Likely fit:** MLSys for an end-to-end mechanism-aware system. CGO requires a
substantive IR, lowering, or backend contribution rather than labels alone. A
main ML venue requires a distinct representation, learning, or
sample-efficiency contribution.

### Paper C: Hierarchical Hardware-Conditioned Agent/RL

**Scope:** a mechanism policy chooses transferable intent; a realization policy
adapts it to the target. This is a continuation of W5, not an evaluator feature.

**Claim:** hierarchical hardware conditioning beats black-box optimization on
held-out hardware at matched target-side budgets.

**Required evidence:** genuine policy/value updates for an RL claim, sealed
training exposure, H1-H5 ablations, matched inference budgets, and independent
hidden evaluation.

### Independent Branch: Value-Aware/Verified SOLAR

**Scope:** W2.

**Claim:** value-semantic modeling or verified lowering tightens the bound
without sacrificing semantic validity.

**Likely fit:** CGO/PL or MLSys, depending on whether the principal contribution
is IR/formal reasoning or a measured systems analysis.

W6 should initially serve as an external-validity section or artifact for the
other papers. It becomes a standalone systems paper only if it reveals a
generalizable model of ranking reversal rather than a collection of deployment
case studies.

## 13. Recommended Near-Term Decision

The highest-value near-term program is Paper A:

1. close the content-bound SOL input required by the existing generalization
   protocol;
2. calibrate and validate SOL/headroom on at least two materially different
   physical hardware families;
3. pre-register `H/P/C/E`, common-support, aggregation, and uncertainty rules;
4. use target-native baselines and publish the full evidence matrix, including
   negative and unsupported cells;
5. defer Agent/RL claims until evaluator and workload identifiability pass.

W2 can proceed in parallel if there is capacity for compiler/formal work. W4
should begin with a small manually auditable mechanism corpus, but its primary
claim should wait for W1/W3. W5 and hierarchical RL are downstream, not
shortcuts around missing evaluator evidence.

## 14. Current Repository Boundary

Relevant implemented foundations include:

- the [current claim boundary](../user/CLAIMS.md);
- the [evaluator contract](../user/EVALUATOR-CONTRACT.md);
- the [SOLAR responsibility boundary](../SOLAR-BOUNDARY.md);
- the [hardware-generalization protocol](../user/HARDWARE-GENERALIZATION.md);
- [task-source research](aka-sol-task-source-research.md) and the
  [AKA corpus-expansion policy](aka-expansion-friendliness.md);
- runtime and static evidence summarized by the
  [decision-modeling research](decision-modeling-research.md).

These foundations support future experiments but currently establish neither a
causal cross-hardware Agent-generalization result nor a target-sample-efficiency
result. In particular:

- a complete generalization report is necessary but not sufficient for a
  scientific conclusion;
- Agent identity, checkpoint, prompts, tools, training exposure, budgets, and
  candidate provenance must be externally sealed;
- holdout workloads must be genuinely withheld rather than reconstructible
  from public generator rules;
- SOL evidence must be content-bound to each compared workload and target;
- a physical target result cannot be replaced by an analytical platform
  projection;
- source portability cannot be reported as performance portability or Agent
  equivalence.

## 15. Primary References

- [SOL-ExecBench](https://arxiv.org/abs/2603.19173)
- [SOLAR](https://arxiv.org/abs/2606.26383)
- [KernelBench-X](https://arxiv.org/abs/2605.04956)
- [KernelGenBench](https://arxiv.org/abs/2607.27231)
- [MultiKernelBench](https://arxiv.org/abs/2507.17773)
- [CASS](https://arxiv.org/abs/2505.16968)
- [AgentKernelArena](https://arxiv.org/abs/2605.16819)
- [FastKernels](https://arxiv.org/abs/2605.23215)
- [Atrex-Bench](https://arxiv.org/abs/2607.14541)
- [KernelBench-Verified](https://arxiv.org/abs/2607.16241)
- [Improving Efficiency of LLM-Generated CUDA Kernels via muCUTLASS and SOL](https://arxiv.org/abs/2603.29010)
- [KernelPro](https://arxiv.org/abs/2606.26453)
- [CUDA Agent](https://arxiv.org/abs/2602.24286)
