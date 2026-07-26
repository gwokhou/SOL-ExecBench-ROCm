# Current Claim Boundaries

SOL ExecBench 3.0 evaluates current Definition, Workload, Solution, and Trace
contracts on AMD ROCm. The canonical benchmark result is Trace JSONL. Optional
environment, profiler, static-kernel, agent-feedback, profile-summary, and
decision artifacts are diagnostic and cannot grant correctness, timing, score,
paper-parity, or leaderboard authority.

The bundled corpus is a pinned, integrity-checked selection of the upstream
dataset. Raw upstream field names are normalized only while importing that
corpus; public models accept only the current field names.

SOLAR owns graph extraction, conversion verification, and formal lower-bound
analysis. Current einsum graphs and analysis artifacts use schema 3. Formal ROCm
bounds fail closed unless the configured architecture and pinned Orojenesis
evidence satisfy the current contract.

The repository exposes a fail-closed official scorer and machine-readable
availability. The checked-in corpus publishes a content-addressed release
policy and canonical baseline identity. Local formula experiments and raw
evidence are not official scores; the scorer requires a publisher bundle that
binds every scoring input.

Hardware-specific tests and observations establish only the behavior they
directly execute. A marker, compatibility entry, or diagnostic sidecar is not a
family-wide hardware-validation claim.

The recorded RDNA4 gate is exact to AMD Radeon RX 9060 XT `gfx1200`, ROCm
7.2.0, PyTorch `2.11.0+rocm7.2`, and its recorded HIP runtime. It does not
validate every `gfx12*` product or ROCm release. Local and manual self-hosted
GitHub bundles use `sol_execbench.rdna4_validation.v2`; they are
content-addressed engineering evidence, not publisher score bundles, and are
always rejected when release eligibility is required. See
[RDNA4 Validation Scope](RDNA4-VALIDATION.md).

## ROCm Compatibility Matrix

Each Matrix Entry separates two kinds of data:

- **Target/requested values** identify the selected compatibility target,
  including the requested ROCm user-space, image, dependency policy, and GPU
  architecture.
- **Observed evidence** records runtime probes in distinct host, container, Python dependency, dependency policy, toolchain, and GPU scopes.

Target identity is required: observed values are meaningful only relative to
the selected target. Docker Matrix Entries validate **container ROCm user-space
on recorded host driver/devices**. They do not prove native host ROCm validation;
native validation requires direct evidence for the requested host stack. Do not
describe Docker Matrix Entries as native host ROCm validation.

The 2026-05-29 live checks recorded container evidence for
`sol-execbench:rocm-7.0.2-complete` and
`sol-execbench:rocm-7.2-complete` with
`./scripts/run_docker.sh --record-container-validation`. ROCm 7.0.2 remains unlocked performance evidence. These records are container checks, not native-host ROCm hardware validation.

Illegal mixed-version Targets are blocked by default. The explicit debug
override permits bounded probes or smoke diagnostics only. It cannot create `container_validated` or
`host_validated`, score authority, paper-parity authority, or leaderboard authority.

See [Evaluator Contract](EVALUATOR-CONTRACT.md), [SOLAR Boundary](../SOLAR-BOUNDARY.md),
[Scoring](../SCORING-V3.md), and [Schema Boundaries](schema-boundaries.md).
