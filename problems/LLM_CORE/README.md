# LLM Core problem corpus

`LLM_CORE_V2` is a frozen, forward-only text-model corpus built independently
from pinned public architecture facts. It contains 36 semantic Definitions and
36 workload-generation rules across six selectable profiles, with no concrete
Workloads frozen in the release. Every rule defines one smoke slot, four
balanced development slots, and four corresponding holdout slots. It does not
include model weights, multimodal
encoders, NVIDIA SOL-ExecBench dataset content, scores, or real-hardware
qualification.

The directory layers are:

- `registry/`: pinned model sources and normalized semantic identities;
- `releases/LLM_CORE_V2/`: immutable Definition, generation-rule, and manifest
  files;
- `targets/isa/`: ISA capability templates without product assumptions;
- `targets/products/`: product-family templates with unknown SKU and visible
  resources;
- `targets/configurations/`: exact declared SKU, capacity, partition, and
  isolation configurations. Runtime evidence must still confirm their fixed
  facts.

Rebuild or verify the committed release deterministically:

```bash
uv run python scripts/build_llm_core_corpus.py --write
uv run python scripts/build_llm_core_corpus.py --check
```

Validate the corpus or derive a target view using measured ROCm capacity:

```bash
uv run sol-execbench dataset corpus validate
uv run sol-execbench dataset corpus generate \
  --target-template isa/gfx942 \
  --device cuda:0 \
  --profile core \
  --profile moe \
  --output problems/local/LLM_CORE/gfx942-measured
```

Use `--target-template configurations/mi300x/spx-192gb` when a full, dedicated
MI300X SPX configuration must be enforced. `products/mi308x` binds the product
identity but leaves visible CU, memory, and partition facts to runtime evidence
because the same `gfx942` ISA does not determine those configuration facts.

The same distinction applies to `gfx1200`: the generic template does not imply
an RX 9060-series SKU. `products/rx9060xt` binds only the product family, while
`configurations/rx9060xt/standard-16gb` demonstrates an exact standard 16 GiB
configuration.
The canonical configuration identity binds model, optional SKU, and visible
memory separately, so 8/16 GiB and standard/low-profile variants can remain
distinct without requiring an exhaustive committed product catalog.

Generation reason codes are recorded per Definition. Hardware measurements are
normalized into integer-byte capacity classes, then one maximum feasible common
scale is applied to all nine slots without per-slot clamping. The derived view
embeds the raw capacity evidence for audit, but only semantic generation inputs
participate in the workload-view digest. Bundled target templates remain
declarations; the measurement qualifies only the derived view.

The frozen latent slot structure is hardware-independent. Different capacity
classes may choose different common scales, while slot IDs, roles, regimes,
serving phases, bindings, coefficients, input profiles, and correctness
profiles remain identical. If all nine aligned workloads cannot be distinct and
feasible together, the entire Definition is marked `insufficient_capacity`.
Runtime OOM evidence never triggers an in-place scale downgrade or partial
workload deletion.

The four `*-low` slots form the agent-visible development view. The four
`*-high` slots are evaluator-held workloads: their concrete axes, UUIDs, and
resource envelopes are withheld while a solution is produced, then disclosed
with the completed evaluation evidence. This is an exposure boundary rather
than a cryptographic secrecy claim because the generation rules are public.

See [Cross-hardware Agent evaluation](../../../docs/user/HARDWARE-GENERALIZATION.md)
for the benchmark-owned study protocol and distributed CLI workflow.
