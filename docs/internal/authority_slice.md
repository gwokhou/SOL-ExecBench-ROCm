# AMD Authority Slice

`sol_execbench.authority_slice_manifest.v1` freezes a workload subset before
candidate or release-baseline measurement. It is derived from the complete suite
manifest and an AMD bound sanity report, and never reads candidate score values.

Create the audit first with `scripts/report_amd_bound_sanity.py`, then freeze the
selection:

```bash
uv run sol-execbench baseline authority freeze \
  --suite-manifest suite.json \
  --sanity-report amd-bound-sanity.json \
  --output authority-slice.json
```

The manifest contains the source suite SHA-256, the versioned selection policy,
selected workloads, every excluded workload, stable blocker codes, and its own
payload checksum. Its `workloads` list can be passed to `baseline release build`;
the original complete suite manifest remains the denominator for a full-suite
official score report.

The v1 policy selects only warning-free workloads whose AMD SOL and SOLAR
aggregates are both `scored`, profiler timing is cited, and exact validated
hardware-profile evidence is present. Missing baseline evidence is audited but
does not block freezing because the release baseline is measured after the
selection. A v3 bound may include a multi-operator workload only when its
fusion-group IR proves external traffic and intermediate reuse, and every group
is `supported`; a recorded but inexact group remains excluded.

This artifact does not establish full 235-problem authority, NVIDIA leaderboard
equivalence, upstream SOLAR parity, or paper parity. A selected workload becomes
official only after calibrated v3 hardware evidence, release-baseline publication,
an independent rerun, and the existing official-score gate all pass.
