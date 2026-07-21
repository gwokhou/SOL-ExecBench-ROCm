# SOL ExecBench ROCm 3.0

SOL ExecBench ROCm evaluates GPU kernel candidates and derives formal
Speed-of-Light bounds for the AMD Radeon RX 9060 XT. The distribution contains
two sibling packages with a deliberately narrow boundary:

- `sol_execbench` owns problem/workload contracts, imported-corpus selection,
  input generation, candidate correctness and timing, local score formulae,
  aggregation policy, security, and the CLI.
- `solar` owns only the three stages described in SOL-ExecBench §4.2: operator
  graph extraction, validated extended-einsum conversion, and SOL analysis.

The paper is the source of truth when imported implementations disagree:
[SOL-ExecBench](https://arxiv.org/abs/2603.19173).

## Install

```bash
uv sync --all-groups
```

Python 3.12–3.13 is supported. Linux x86-64 resolves the ROCm 7.2 PyTorch
stack declared in `pyproject.toml`.

## Public problem corpus

The reviewed selection is tracked at
`problems/RX_9060_XT/manifest.yaml`. Problems are derived from AMD
AgentKernelArena (AKA, Apache-2.0) under the SOL-ExecBench paper §3
construction methodology: each problem's PyTorch reference is the AKA task's
own correctness oracle (`module_fn`), lifted into a standalone `def run(...)`.
The manifest pins the exact AKA commit and records per-task file checksums so
the generated problems are bound to that source revision. The current seed set
covers matmul, softmax, norm, elementwise, and conv operations across FP32 /
BF16 / FP16.

Authored definitions and workloads are committed under
`problems/RX_9060_XT/<suite>/<name>/`; materialized copies remain untracked
under `problems/local/`:

```bash
bash scripts/fetch_aka_source.sh
uv run sol-execbench dataset materialize
uv run sol-execbench dataset audit problems/local/RX_9060_XT
```

`dataset audit` also verifies the local AKA clone is pinned to the manifest
revision and that every problem's per-task checksums match (the
problems-bound-to-commit binding). The public Definition contract requires a
non-empty `op_type`; the AKA importer records it per problem.

## Evaluate and analyze

Candidate evaluation remains an outer-project operation:

```bash
./scripts/run_docker.sh -- sol-execbench evaluate \
  problems/local/RX_9060_XT/torch2hip/l1n1_square_matmul \
  --solution /sol-execbench/path/to/solution.json \
  --trace-output /outputs/matmul.trace.jsonl
```

The intended formal SOLAR target is RX 9060 XT `gfx1200`. This port's formal
publication policy requires the pinned Orojenesis toolchain; that is an
explicit ROCm release constraint, not a claim that the paper mandates this
tool for every SOLAR use. This revision deliberately blocks publication
because its locked-clock architecture audit artifact and reviewed Orojenesis
mapper artifact digest have not been published. An Orojenesis executable whose
digest is only self-declared by its local provenance manifest is rejected:

```bash
uv run sol-execbench solar analyze \
  problems/local/RX_9060_XT/torch2hip/l1n36_rmsnorm \
  --workload aka-l1n36_rmsnorm-w0 \
  --orojenesis-home /path/to/pinned/timeloop \
  --output out/solar/norm_forward_bf16
```

The isolated worker publishes an atomic directory containing only:
`operator_graph.yaml`, `einsum_graph.yaml`,
`conversion-attestation.yaml`, `solar-analysis.yaml`, and `manifest.yaml`.
It never receives candidate runtimes or computes scores.

Unknown operations fail closed during formal analysis. Offline learning writes
a verified but untrusted candidate outside the formal lookup table:

```bash
OPENAI_API_KEY=... uv run sol-execbench solar learn-handler OP sample-node.yaml \
  --output out/handler-candidates/OP
```

Formal use remains forbidden until the generated source and proofs are reviewed
and committed under `src/solar/handlers/` with an approved formal-review record
and matching source SHA-256.

## Official score

The score formula is implemented without clipping, but this v3 corpus does not
yet publish the release baseline, independent rerun, trusted candidate
execution attestation, or pinned SOLAR manifest set needed for an official
claim. No official scorer command is exposed. `sol-execbench score status`
reports the immutable authority blockers without accepting measurement or
baseline inputs; caller-authored JSON cannot be promoted to an official score.

Once those release artifacts are published and pinned, correct candidates must
satisfy `T_b > T_SOL` and `T_k >= T_SOL`; workloads are averaged within each
problem and then across problems with equal weight.

## Development

```bash
uv run --with ruff ruff check .
uv run ty check
uv run pytest tests/
```

GPU tests declare their ROCm and architecture prerequisites. Build the optional
container with `./scripts/run_docker.sh --build`.

See [SOLAR boundary](docs/SOLAR-BOUNDARY.md) and
[scoring contract](docs/SCORING-V3.md) for the normative v3 architecture.

## License

Apache-2.0. Imported SOLAR files retain their original SPDX attribution.
