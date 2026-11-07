# Cross-hardware Agent evaluation

The hardware-generalization protocol measures a GPU Kernel Agent across target
hardware while making workload realization drift explicit. It does not claim
that concrete workloads are identically distributed across hardware.

The benchmark owns four responsibilities:

1. derive complete evaluator target views from frozen LLM Core rules;
2. derive Agent views that expose four development slots but withhold the four
   concrete holdout workloads;
3. seal existing evaluator `Trace` evidence against an immutable study plan;
4. report Agent outcomes, common support, missingness, and workload drift.

Training, prompting, tool use, solution generation, and adaptation experiments
remain outside this repository. The study plan consumes only a minimal exposure
declaration: previously seen gfx targets, capacity classes, distribution IDs,
hardware-configuration IDs, and whether that declaration was independently
verified. A gfx target identifies an ISA family, not a complete device
configuration: MI300X and MI308X both use `gfx942` but remain distinct hardware.
Likewise, `gfx1200` does not collapse RX 9060 XT 8/16 GiB, RX 9060 XT
low-profile, and RX 9060 configurations: model, optional SKU, and visible
resources are independent configuration inputs. An unobservable SKU remains
unknown rather than being inferred from the ISA or memory size.

Hardware facts use four separate layers: published nominal profiles, declared
or provisioned configurations, runtime observations, and a resolved evaluation
context. Workload feasibility consumes the resolved visible resources. Study
classification consumes the stable configuration ID. Collection time, device
index, current free memory, and target labels remain audit facts rather than
configuration identity.

Bundled declarations mirror their schema kinds in the filesystem:
`targets/isa/` contains `isa_template`, `targets/products/` contains
`product_template`, and `targets/configurations/` contains
`configuration_template`. These are declarations, not observed devices. Only
runtime resolution may produce `observed_device`, `physical_device`,
`virtual_device`, or `partition`; the resolver derives that kind from measured
visibility plus the declared virtualization and partition boundary.

## Protocol stages

First generate one measured target view on each target machine with `dataset
corpus generate`. Copy those immutable views to the planning host, then create a
study plan:

```bash
uv run sol-execbench generalization plan \
  --study-id example-study \
  --seen-hardware gfx1200:<configuration-sha256>:8589934592:<distribution-sha256> \
  --manifest problems/LLM_CORE/releases/LLM_CORE_V2/manifest.yaml \
  --target-id gfx1200-8 --target-view artifacts/gfx1200-8.yaml \
  --target-id gfx1200-16 --target-view artifacts/gfx1200-16.yaml \
  --target-id gfx942 --target-view artifacts/gfx942.yaml \
  --output artifacts/study-plan
```

By default the plan contains `target_conditioned` and `solution_portability`
cells using full target facts. `--include-anonymous` adds the optional anonymous
facts ablation for the target-conditioned track. It is not required for the
core benchmark.

An external Agent harness consumes the emitted `*.agent-view.json` files and
produces ordinary SOL ExecBench solution bundles. Run the existing evaluator on
the planned target view, then seal its Trace files on the physical target:

```bash
uv run sol-execbench generalization run-cell \
  --plan artifacts/study-plan/plan.json \
  --manifest problems/LLM_CORE/releases/LLM_CORE_V2/manifest.yaml \
  --target-view artifacts/gfx1200-8.yaml \
  --cell-id gfx1200-8--target_conditioned--full_facts \
  --solution solutions/example.json \
  --trace traces/example.jsonl \
  --device cuda:0 \
  --output cells/gfx1200-8-target-conditioned.json
```

`run-cell` verifies the resolved hardware configuration, physical gfx target,
capacity class, plan, and cohort identities, but it does not regenerate
workloads. A missing solution is an Agent failure. Missing
Trace evidence for a supplied solution or an invalid reference is evaluator
failure evidence and invalidates that cell. Candidate compile failures,
incorrect results, timeouts, runtime failures, and OOM remain Agent outcomes.

Finally aggregate the available cells:

```bash
uv run sol-execbench generalization aggregate \
  --plan artifacts/study-plan/plan.json \
  --manifest problems/LLM_CORE/releases/LLM_CORE_V2/manifest.yaml \
  --target-id gfx1200-8 --target-view artifacts/gfx1200-8.yaml \
  --target-id gfx1200-16 --target-view artifacts/gfx1200-16.yaml \
  --target-id gfx942 --target-view artifacts/gfx942.yaml \
  --cell cells/gfx1200-8-target-conditioned.json \
  --cell cells/gfx1200-16-target-conditioned.json \
  --cell cells/gfx942-target-conditioned.json \
  --cell cells/gfx1200-8-portability.json \
  --cell cells/gfx1200-16-portability.json \
  --cell cells/gfx942-portability.json \
  --output reports/generalization.json
```

The command may emit an `incomplete` report, but such a report explicitly
forbids a generalization conclusion. A complete conclusion also requires at
least one seen control and one hardware or capacity shift target.

Shift labels distinguish an exact seen configuration, the same ISA at a new
capacity, the same ISA and capacity on a new configuration, and an unseen ISA.
This prevents two devices from being treated as identical merely because their
ISA flag matches. A same-ISA training exposure with a different distribution
ID is rejected during planning instead of being mislabeled as a hardware or
capacity shift; workload-distribution transfer requires a separate protocol.

## Reporting boundary

The report has no composite score. It records target-full and common-support
results, plus role, profile, operation-family, and regime strata. Workloads are
averaged within each Definition before Definitions receive equal weight.
Correctness, compilation, `fast_p`, and correct-only geometric mean speedup use
deterministic Definition-cluster bootstrap intervals. SOL evidence is not part
of this Trace-backed protocol until a content-bound SOL input is specified.

Representativeness drift is reported separately through Definition support and
skip reasons, latent slot signatures, categorical Jensen-Shannon divergence,
axis log-ratio shifts, common-scale ratios, and resource-utilization shifts.
These fields describe observed drift and never certify equal distributions.

For `solution_portability`, evaluator target routing may change, but candidate
sources, entry point, dependencies, language set, binding, and compile options
must retain one portability digest across targets.
