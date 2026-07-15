# ROCm 7.2 Evidence Rebuild Priorities

Status date: 2026-07-15

This document defines the recommended order for closing the evidence gaps after
the default software stack moved to ROCm 7.2. It is an execution-priority and
acceptance guide, not a new benchmark schema or an authority claim.

The dependency lock, default Docker target, HIP toolchain, and PyTorch ROCm
wheels have moved to ROCm 7.2. Existing hardware calibration, release baseline,
V4 bound, score, and publication artifacts remain historical ROCm 7.1 evidence.
They must not be upgraded by editing version strings or reusing old latency and
counter values.

## Guiding Rule

Do not start an expensive downstream GPU stage until its upstream evidence gate
passes. In particular, two independent patched occupancy smoke collections are
the stop/go gate for full shape-aware collection and the subsequent V5
authority rebuild.

The shortest critical path is:

```text
ROCm tool discovery
  -> patched profiler integration
  -> two independent occupancy smoke collections
  -> two ROCm 7.2 hardware calibrations
  -> complete shape-aware collection
  -> baseline and independent rerun
  -> V5 bounds and scores
  -> publication and lifecycle update
```

## P0: Prevent Invalid Evidence Generation

### 1. Unify ROCm tool discovery

Environment diagnostics, profiler collection, and batch runners must use the
same ROCm-root-aware resolution for `amd-smi`, `rocprofv3`, and
`rocprofv3-avail`. A tool installed under `/opt/rocm-*` must not be reported as
unavailable merely because it is absent from the caller's `PATH`.

Acceptance criteria:

- host `environment doctor` does not incorrectly report the installed
  `amd-smi` as unavailable;
- the profiler timing batch discovers the installed profiler without a manual
  `PATH` rewrite;
- diagnostics record the resolved executable paths.

### 2. Integrate the gfx1200 patched profiler explicitly

`scripts/patches/gfx1200_sq_wave_cycles/rocprofv3-gfx1200-patched` is a local
power-state workaround. It runs the unmodified ROCm 7.2 profiler while the GPU
is in `STABLE_PEAK`; it does not change the ROCm counter definitions or replace
the profiler libraries.

The current `scripts/internal/rdna4/run_rdna4_profiler_timing_batch.py` still
selects the bare `rocprofv3` executable. Add an explicit profiler executable
input or use a shared resolver so the selected executable is visible in the
command and evidence. Do not rely on a temporary same-name symlink or an
implicit `PATH` shadow.

The batch clock lifecycle also needs one defined owner. Prefer one batch-level
`STABLE_PEAK` lease with verified reset instead of locking and resetting once
per workload. Strict isolation must not reject an `AUTO` state before the
component responsible for acquiring the lease has run.

Acceptance criteria:

- recorded commands name the patched wrapper when it is requested;
- strict-isolation preflight and patch clock acquisition have a consistent
  order;
- success, failure, timeout, and handled-signal paths attempt and verify the
  required reset;
- an external `STABLE_PEAK` owner is preserved rather than reset by the batch.

### 3. Bind patch provenance to raw evidence

Every raw counter artifact collected through the workaround must record or
checksum-bind:

- patch ID and wrapper SHA-256;
- real `rocprofv3` path, version, and executable identity;
- `amd-smi` path and version;
- ROCm, HIP, PyTorch, GPU, and driver identity;
- pre-run, during-run, and post-run performance levels;
- whether the current process acquired the clock lease;
- reset result;
- raw profiler CSV paths and SHA-256 values.

A positive final counter value without this provenance is diagnostic evidence,
not sufficient shape-aware authority evidence.

### 4. Freeze the current V4 authority boundary

The existing V4 bundles and manifests are immutable historical evidence. The
current V5 readers do not accept their AMD SOL bounds, so V4 artifacts must not
be presented as current ROCm 7.2 authority and must not be overwritten.

Resolve the conflict between the publication guide, which states that no
current V5 authority bundle exists, and the lifecycle index, which still lists
a V4 release as active `published` evidence. Use a documented lifecycle
transition or a revision-pinned legacy verification policy; do not silently
reinterpret the existing record.

## P1: Prove That the Patch Removes the Core Blocker

### 5. Run two independent patched occupancy smoke collections

Run two host-direct collections against the same signed sampling plan and the
same gfx1200/ROCm 7.2 environment. Each run must independently establish:

- positive `SQ_WAVE_CYCLES`;
- positive `SQ_BUSY_CYCLES`;
- a finite, valid derived occupancy value;
- attributable kernel dispatch and counter rows;
- checksum-valid provider, trace, and raw evidence;
- verified `STABLE_PEAK` during collection;
- verified return to `AUTO` when the collection owns the lease.

A single positive reading, an unlocked run, selector substitution, or use of
`SQ_WAVES` without wave-cycle residence time is insufficient.

If either smoke run fails, stop the V5 authority rebuild. Preserve the result as
diagnostic evidence and do not start the complete shape-aware collection.

### 6. Update the counter-blocker conclusion

After the smoke gate passes, update the current accuracy-gap wording from
"waiting for an upstream fix or equivalent observation path" to a narrower
conclusion: the local `STABLE_PEAK` workaround is validated for the recorded
gfx1200/ROCm 7.2 environment.

Do not generalize that conclusion to all RDNA4 devices, drivers, firmware, or
future ROCm releases. Revalidate the workaround after a relevant ROCm,
rocprofiler SDK, driver, firmware, or GPU change.

## P2: Rebuild ROCm 7.2 Foundation Evidence

### 7. Rebuild and validate the default ROCm 7.2 container

Build the default image from the final source revision rather than reusing an
older local image with the correct tag. Generate and retain:

- image digest and build provenance;
- dependency preflight evidence;
- runtime and environment evidence;
- a bounded container smoke trace;
- compatibility sidecar;
- clock-lock and reset evidence where timing is claimed.

Container ROCm user-space evidence remains distinct from native-host
validation.

### 8. Rebuild semantic and authority coverage

Regenerate the inputs that define the workload and profile denominator:

- canonical suite snapshot;
- semantic graph coverage;
- authority coverage;
- hardware profile requirements;
- shape-aware sampling plan;
- timeout and blocker ledger.

Historical coverage may be retained for comparison, but its checksums must not
be used as current-model inputs after source, schema, provider, or dependency
changes.

### 9. Repeat scalar hardware calibration twice

Generate a primary calibration and an independent verification calibration on
ROCm 7.2. Also regenerate profiler overhead calibration and matching fusion
validation evidence.

Acceptance criteria:

- all required exact profiles are measured or explicitly blocked;
- both runs record the current ROCm, HIP compiler, PyTorch, GPU, clock policy,
  source revision, and requirements checksum;
- the model builder verifies the independent calibration and all referenced
  payload checksums;
- no ROCm 7.1 latency, profile value, environment fingerprint, or compiler
  identity is carried forward.

## P3: Run the Expensive Complete Collections

### Current local execution record (2026-07-15)

The P3 primary shape-aware collection for source revision
`09221dde3480bb9180903ae7084070a3ddeed9ed` measured 563 of 564 planned
authority workloads.  The remaining required workload,
`L1/013_fused_residual_rms_norm_backward`
(`ea1263c4-b63d-5fa4-a641-fc4f16d3462a`), exhausts the local GPU while
TorchInductor allocates the `batch_size=64`, `seq_len=8192` case.  A clean
retry with `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` produced the
same OOM result.

This item is **unverifiable on this host**.  The generated local blocker
sidecar binds the incomplete report and sampling-plan checksums and explicitly
sets `authority_eligible` to false.  It is not a waiver for the missing raw
evidence.  A user-authorized local diagnostic run may explicitly exclude this
UUID and collect the other 563 workloads, but that report must remain
`incomplete`; do not finalize the shape-aware artifact or model, or make a
ROCm 7.2 authority claim.  Authority collection still requires suitable
hardware (or an independently justified provider change that has been
implemented and verified).

### 10. Collect complete shape-aware roofline evidence

Run two independent complete collections for the authority sampling plan. Each
planned assignment must have exactly one checksum-valid raw case in each
required collection. Missing occupancy, layout, launch, sample, or trace
evidence must fail closed.

After both collections pass, generate:

- validated `sol_execbench.shape_aware_roofline.v1` evidence;
- a ROCm 7.2 gfx1200 hardware model that references the exact evidence payload
  checksum;
- a comparison report against the retired ROCm 7.1 model without merging their
  measurements.

### 11. Repeat profiler timing closure

Regenerate:

- profiler timing sidecars;
- workload manifests;
- profiler overhead references;
- profiler coverage summary;
- partial, missing, timeout, OOM, and readiness blocker ledgers.

Recompute the complete 235-problem denominator. Historical counts such as
88 fully profiler-backed, 28 partially profiler-backed, and 73 ready but
missing profiler timing remain historical ROCm 7.1 conclusions until reproduced
or replaced by ROCm 7.2 evidence.

### 12. Repeat release-baseline and candidate measurements

Generate, in order:

1. primary baseline traces;
2. independent rerun traces;
3. compact scoring baseline;
4. release-baseline bundle;
5. release-baseline verification;
6. independent candidate traces and timing evidence;
7. slow/equal/fast discrimination evidence or the selected release-candidate
   equivalent.

ROCm, compiler, PyTorch, library, clock-policy, or timing-policy changes
invalidate latency reuse. A checksum-valid ROCm 7.1 baseline remains historical
but cannot become a ROCm 7.2 baseline.

## P4: Rebuild V5 Scoring And Publication

### 13. Regenerate the complete V5 derived chain

Using only the new ROCm 7.2 inputs, regenerate:

- V5 AMD SOL bounds;
- bound sanity report;
- held-out contradiction report;
- authority slice;
- SOLAR derivation sidecars;
- AMD-native score report;
- official-score evidence;
- complete workload blocker ledger.

Any `T_SOL_floor > T_fastest_known` contradiction blocks publication. Do not
remove contradictory workloads and publish the remainder as if the selected
scope had passed unchanged.

### 14. Complete publication and lifecycle closure

Generate and verify:

- V5 evidence publication manifest;
- exact staged upload closure;
- immutable uploaded archive;
- clean-download verification;
- prerelease artifact bundle;
- readiness report;
- `SHA256SUMS`;
- lifecycle successor, supersession, or other documented transition.

Use "locally verified authority slice" until clean-download verification
passes. Only the verified immutable bundle may be called a "published authority
slice". Neither term implies full-suite, upstream SOLAR, NVIDIA, or leaderboard
authority.

## P5: Documentation And Deferred Scope

### 15. Reconcile current internal documentation

Update or classify the following documentation drift:

- V5 serialized lower-bound wording must use the current
  `theoretical_lower_bound.t_sol_floor_ms` surface rather than presenting the
  internal `aggregate_bound` model as the serialized field;
- authority-slice guidance that still refers to V3 bound authority;
- accuracy-gap wording that does not yet include the local patched-profiler
  validation path;
- implementation plans whose checkboxes remain open after their code landed;
- CDNA3 and decision-sidecar documents with contradictory historical/current
  status statements;
- v1.25 and v1.26 documents that need an explicit historical-release label.

### 16. Keep unrelated deferred claims separate

The following remain real gaps but must not delay the narrowly scoped ROCm 7.2
gfx1200 evidence rebuild unless the release claims them:

- exact-hardware MI300X validation;
- CDNA4 validation;
- complete 235-problem profiler-backed coverage;
- native-host MIOpen, rocBLAS, rocWMMA, and Composable Kernel example coverage;
- Origami and rocMLIR/MIGraphX provider validation;
- hard multi-tenant sandboxing;
- upstream SOLAR, NVIDIA B200, or leaderboard parity.

## Completion Definition

The ROCm 7.2 migration is evidence-complete for a declared gfx1200 authority
slice only when:

- the patched occupancy path has passed two independent smoke collections;
- all hardware, profiler, baseline, candidate, bound, and score artifacts are
  generated under ROCm 7.2 and the final source revision;
- every cross-artifact reference and checksum verifies;
- the release baseline has an independent passing rerun;
- the V5 bound and held-out gates have no unresolved contradiction in the
  declared slice;
- the staged bundle passes clean-download verification;
- the lifecycle index and documentation describe the same current authority
  state.

Until then, the safe status is: the default runtime stack has migrated to ROCm
7.2, while current authority evidence remains under reconstruction.
