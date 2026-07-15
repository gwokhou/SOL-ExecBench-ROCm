# AMD SOL Bound Evidence

`sol_execbench.amd_sol_bound.v5` is the sole accepted AMD SOL sidecar. It preserves per-node work estimates while partitioning
every bound-graph node into a singleton or a versioned fusion group.

For a group, compute work is the sum of member FLOPs and memory work is based
only on tensors crossing the group boundary. Internal tensors are recorded as
eliminated intermediate traffic rather than silently discarded.

```text
T_group = max(sum(FLOPs) / compute_profile, external_bytes / memory_profile)
T_SOL = sum(T_group)
```

The registry recognizes any statically modeled producer family (attention,
GEMM, linear projection, convolution, embedding, reduction, normalization,
softmax, MoE, or SSM/Mamba) followed by a single-consumer elementwise or
activation epilogue. It uses the same group calculation for every family: exact
FLOPs are the sum of supported member estimates, and bytes are the group
boundary only.

Each artifact embeds the architecture capability budget and its provenance. A
multi-node group records `required_lds_bytes`, the conservatively required
resident intermediate size. It is eligible for `supported` only when the budget
is `supported`, has enough LDS, and every member's static family estimate is
supported. A missing, provisional, mismatched, or insufficient budget leaves
the group `inexact`; connectivity alone never upgrades it. Activation,
broadcast, data-dependent, and any family member without its own exact contract
remain inexact. In particular, only static `sum` reductions are currently exact;
activation, generic reduction, normalization, generic softmax, attention, MoE,
and SSM remain conservative unless their existing subrole estimator can prove
all required static evidence.

Every build requires an external `sol_execbench.amd_hardware_model.v3` exact-profile
model plus fusion-validation evidence, its reference path, and SHA-256 digest.
The builder validates the evidence architecture and checksum before producing
an artifact. Only v5 files are accepted by the derived-score path and official
gate; older schemas are rejected rather than migrated.

The command and artifact sequence for collecting calibration and fusion inputs,
freezing a release baseline, and emitting a gated score is documented in
[Release and Official Score Workflow](RELEASE-SCORING.md). The sequence does
not imply that every calibration or bound is authoritative: each output keeps
its recorded validation and scope boundary.
