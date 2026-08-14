# LLM Core problem corpus

`LLM_CORE_V1` is a frozen, forward-only text-model corpus built independently
from pinned public architecture facts. It contains 84 semantic Definitions and
1260 Workloads across six selectable profiles. It does not include model
weights, multimodal encoders, NVIDIA SOL-ExecBench dataset content, scores, or
real-hardware qualification.

The directory layers are:

- `registry/`: pinned model sources and normalized semantic identities;
- `candidates/`: the append-only input for the next release;
- `releases/LLM_CORE_V1/`: immutable Definition, Workload, and manifest files;
- `targets/`: declared, unqualified static target templates.

Rebuild or verify the committed release deterministically:

```bash
uv run python scripts/build_llm_core_corpus.py --write
uv run python scripts/build_llm_core_corpus.py --check
```

Validate and select without accessing ROCm hardware:

```bash
uv run sol-execbench dataset corpus validate
uv run sol-execbench dataset corpus select \
  --target-template gfx942 \
  --memory-budget 68719476736 \
  --profile core \
  --profile moe \
  --output problems/local/LLM_CORE/gfx942-64g
```

Selection reason codes are stable and recorded for every workload. Bundled
target templates have `qualification_status: declared`; a future hardware
qualification phase must produce separate evidence before that claim changes.
