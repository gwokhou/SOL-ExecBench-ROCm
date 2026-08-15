# LLM Core problem corpus

`LLM_CORE_V2` is a frozen, forward-only text-model corpus built independently
from pinned public architecture facts. It contains 36 semantic Definitions and
36 workload-generation rules across six selectable profiles, with no concrete
Workloads frozen in the release. Every rule defines one smoke slot and eight
balanced development slots. It does not include model weights, multimodal
encoders, NVIDIA SOL-ExecBench dataset content, scores, or real-hardware
qualification.

The directory layers are:

- `registry/`: pinned model sources and normalized semantic identities;
- `releases/LLM_CORE_V2/`: immutable Definition, generation-rule, and manifest
  files;
- `targets/`: declared capability templates without fixed memory budgets.

Rebuild or verify the committed release deterministically:

```bash
uv run python scripts/build_llm_core_corpus.py --write
uv run python scripts/build_llm_core_corpus.py --check
```

Validate the corpus or derive a target view using measured ROCm capacity:

```bash
uv run sol-execbench dataset corpus validate
uv run sol-execbench dataset corpus generate \
  --target-template gfx942 \
  --device cuda:0 \
  --profile core \
  --profile moe \
  --output problems/local/LLM_CORE/gfx942-measured
```

Generation reason codes are recorded per Definition. Hardware measurements are
normalized into integer-byte capacity classes, then one maximum feasible common
scale is applied to all nine slots without per-slot clamping. The derived view
embeds the raw capacity evidence for audit, but only semantic generation inputs
participate in the workload-view digest. Bundled target templates remain
declarations; the measurement qualifies only the derived view.

The frozen slot distribution is hardware-independent. Different capacity
classes may choose different common scales, while slot IDs, roles, regimes,
serving phases, bindings, coefficients, input profiles, and correctness
profiles remain identical. If all nine aligned workloads cannot be distinct and
feasible together, the entire Definition is marked `insufficient_capacity`.
Runtime OOM evidence never triggers an in-place scale downgrade or partial
workload deletion.
