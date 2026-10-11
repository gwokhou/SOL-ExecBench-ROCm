# ROCm 10.1 / Python 3.14 / PyTorch 2.14 development validation

These probes implement independent parts of the staged upgrade assessment.
The paired acquisition has its own development qualification chain. It does not
modify the historical RDNA4 release gate. Production timing and candidate
stream/graph restrictions are unchanged.

The `qualification` / `acquire` / `case_worker` workflow below retains the
original AUTO protocol for historical reproduction and compatibility checks.
It is not an entry point for new performance comparisons: current comparisons
require verified STABLE_PEAK on both stacks. Do not interpret its AUTO timing
results as upgrade benefits. The separate STABLE_PEAK archives below retain
the subsequent measurements; this directory does not yet provide their unified
acquisition entry point.

The follow-up artifacts are under
`data/outputs/stack-upgrade-followup-20261011/`. The original checkpoint below
describes the first implementation run; use the follow-up receipt for subsequent
validation and acquisition status.

## Current STABLE_PEAK regression checkpoint

The causal follow-up is archived at
`data/outputs/stack-causal-20261011/README.md`. It identifies two concrete paths:
Dynamo's per-call static-configuration hook and the gfx1200 FP32 hipBLASLt
algorithm portfolio/epilogue. These are scoped diagnostic findings; no production
guard, timing, compiler, or library policy was changed.

For irregular GEMM w1, repeated host submission took 16.55/18.40 microseconds
old/new. Hoisting the same static configuration outside the invocation loop,
while retaining guards and generated code, changed this to 15.86/15.83.
Installed source and CPU profiles bind the new overhead to ContextVar writes
and dirty-state bookkeeping in `_make_closure_patcher`. Per-call device-event
measurements retain a smaller wrapper/synchronization residual; the experiment
does not isolate every Python-version or scheduling effect.

For 16x16x32 FP32, a fixed native executable crossed old/new hipBLASLt bundles
with old/new HIP/HSA runtime bundles under the same host driver. Times followed
the library (about 4.02–4.07 versus 8.84–8.89 microseconds), not the runtime.
New-library enumeration lacks the old small-tile candidates. Furthermore, a
fixed 32x64x16 tile took 4.468 microseconds in the old library and 6.593 in the
new one. A diagnostic copy of the new code object, restricted to the frozen
alpha=1/beta=0/no-bias input, bypassed generic epilogue work and used its existing
four-wide masked edge writeback: 4.508 microseconds, with all numerical checks
passing. Separate VALU measurements were 2366 old, 2874 new, and 2304 for the
restricted intervention. This isolates generic epilogue overhead and scalar
edge writeback as concrete causes of the same-tile regression. The binary
intervention is not a general kernel fix or a production recommendation.

All four composite shapes also reproduced library-level regressions in a fixed
runtime native screen; even the fastest supported new algorithm remained slower
than the old default. Algorithm ranks vary across fresh processes, so PMC joins
must bind actual version-local algorithm IDs and kernel names. The original
rank-based counter attempt is superseded by fixed-ID runs. New Torch cannot
simply preload the old hipBLASLt: that isolated attempt failed on a missing
symbol required by new rocBLAS. No host library or image was replaced.

Prioritize restoring small/no-epilogue specialization and legal vectorized edge
stores in the library, and reducing redundant Dynamo config updates while
preserving configuration scope. The per-shape binary interventions have not
been generalized or qualified for production. All clock windows restored AUTO;
the final independent check found no remaining GPU processes.

### Earlier complete STABLE_PEAK comparison

The current performance diagnosis is archived separately at
`data/outputs/stack-stable-peak-20261011/README.md`. Per the measurement policy,
AUTO timing results below are historical observations and are excluded from the
current upgrade-benefit assessment.

All 64 old/new processes (eight frozen GEMM/composite cases, four alternating
blocks) completed at verified STABLE_PEAK with original numerical checks passing.
Irregular GEMM compiled invocation ratios were 1.041–1.099 new/old, while
graph-contained compiled ratios were 1.007–1.015. Composite preallocated,
graph-contained GEMM ratios remained 1.176–2.091. All 64 generated runners
retained the expected external mm or add-plus-mm structure; 1,156 retained
artifacts passed hash verification.

Sixteen independent C++ runs reproduced the small FP32 regression without
Python/Inductor: old about 4.02 microseconds, new 7.73–8.12 microseconds, depending
on workspace-cap group. A separate four-round new-library heuristic screen
returned four candidates; the fastest took 6.60 microseconds versus default
8.90. This identifies potential shape-specific algorithm-selection benefit,
not a production tuning qualification or an exhaustive algorithm search.
The causal follow-up above resolves the dominant library versus runtime
contribution for the small FP32 reproducer. No global backend switch or
production timing change was made. Every successful clock window
restored AUTO, independently verified with no remaining GPU processes.

## Follow-up acquisition

`freeze_workloads.py` freezes 24 public development cases across elementwise,
reduction, normalization, softmax, GEMM and composite families. It preserves
the current numerical checks (all elements must match), definition argument
ordering, source provenance, candidate bytes and seeds. Five additional public
nonaligned boundaries supplement the existing cases. These are fixed
`torch.compile(fullgraph=True, dynamic=False)` candidates; conclusions cannot
be generalized to native CK/GEMM, attention or transpose implementations.
The final denominator materializes CPU inputs once as safetensors and declares
them through current workload input descriptors. The first canary correctly
rejected matching seeds with different tensor hashes across the two stacks;
regenerating inputs from a seed is not a sufficient cross-stack identity.

`docker/install_ck_headers.sh` installs unmodified CK public headers plus CMake
generated configuration from a fixed ROCm libraries commit, retaining its
license and provenance. It targets gfx1200 and does not install a precompiled
CK device-instance library. The independent CK validation image passed all six
RDNA4 backend samples.

The reconstructed old application image is
`sha256:92769f28d0cd4332cb48535abf2fac24d7e7d5b0bd0d924bc6a8da884a5cdd9d`.
It uses the `e5a0d450` source and lock, with the official ROCm 7.2 base pinned to
`sha256:86e11093b4a7ec2a79b1b6701d10e840a6994f21c7e05929b51eb9be361c683a`.
This is a reconstruction on the current host driver, not recovery of the original
historical image. The new CK validation image is
`sha256:0ec2e6cba29e673ebe5ad83b7049e5044647fc340174d1c855379efaf36409dd`.

The qualification launcher validates the complete denominator without GPU use,
selects every per-family axis extreme for canary execution, and then checks every
case on both stacks. Gates bind exact inputs, protocol, acquisition source,
images and result artifacts. Correctness failure, changed identities, missing
predecessors or modified artifacts block acquisition. Qualification results
always have `performance_authority=false`.

Run from the repository root using an environment with the locked Pydantic
dependency and Docker access; the controller does not need host PyTorch:

```bash
export PYTHONPATH="$PWD/src:$PWD"
uv run --no-project --with pydantic==2.13.4 python -m scripts.internal.upgrade.qualification \
  qualify-static --inputs INPUTS --plan PLAN --qualification-root QUALIFICATION
# Repeat the same invocation with qualify-canary, then qualify-full.
uv run --no-project --with pydantic==2.13.4 python -m scripts.internal.upgrade.acquire \
  --inputs INPUTS --plan PLAN --qualification-root QUALIFICATION
```

Use the retained `measurement-plan-frozen-tensors.json` with
`frozen-tensor-inputs/` and `qualification-frozen-tensors/` as the local protocol.
Both image IDs,
GPU UUID/BDF, source revision and output roots must be concrete before static
qualification. Never replace a failed observation or reuse a qualification gate
after changing the plan or acquisition source. Partial acquisition cannot produce
a complete comparison.

Every pair uses fresh processes and empty Inductor/Triton caches. Cold timing
includes candidate loading, compilation and the first synchronized execution;
it excludes Python startup and eager-reference priming. Hot timing is the median
of 100 device-event measurements after ten warmups. CPU-generated input hashes
must match across stacks. The worker verifies UUID/BDF and observes AUTO policy
through AMD SMI before and after timing without changing clocks. Natural clocks,
public development inputs and this trusted worker do not create production or
publication performance authority.
The retained `peak_allocated_bytes` includes live inputs/reference outputs,
compilation, correctness checking and hot measurements. It is the Torch
allocator peak for this worker, not an isolated kernel workspace or total
device-memory measurement.

`profiler_overhead.py` separately compares no profiling, runtime/kernel tracing
with HIP events explicitly disabled, and the same tracing with HIP events
enabled. Ten forward/reverse blocks retain process/application wall time,
artifact hashes and sizes, plus actual RECORD/WAIT counts. Four host wait calls
need not yield four observed GPU WAIT records. This single delayed fixture is
not a corpus-wide calibration; it cannot justify enabling tracing by default or
matching event generations causally.

## Completed follow-up checkpoint

The retained receipt is
`data/outputs/stack-upgrade-followup-20261011/validation-receipt.json`.
The full CPU suite passed 2,499 tests with 51 prerequisite-based skips. All six
RDNA4 backend samples passed, including CK. Static, canary and full development
qualification passed with 48, 34 and 48 stack/case items respectively.

P2 completed all 240 pairs (24 cases across ten blocks; 480 fresh containers)
without a correctness failure. These are equal-workload geometric means, with
paired block bootstrap intervals:

| Metric | Old/new ratio (95% interval) | New-stack elapsed-time change | Per-case screening |
| --- | --- | --- | --- |
| Hot execution | 0.8463 (0.8322–0.8623) | +18.2% | 16 regressions, 1 improvement, 7 uncertain/below threshold |
| Cold compile plus first execution | 0.9157 (0.9148–0.9167) | +9.2% | 21 regressions, 2 improvements, 1 uncertain/below threshold |

Ratios above one favor the new stack. The largest consistent hot regressions
occurred in the composite family (family geometric mean old/new 0.6232).
`aka-3267_doubled_matmul-w1` moved from a median 0.02505 ms to 0.04308 ms;
the hot softmax improvement was specific to `aka-l1n23_softmax-w1` (1.4821 ratio).
These results do not support a general performance-upgrade claim. Retain the
old image for comparisons and prioritize composite/GEMM regression isolation
before promoting performance-sensitive workloads. Keep the complete denominator;
any compiler-option or workload changes require a separately frozen experiment.

The clean P3 run retained 30 observations. Full tracing with events increased
process wall time by a factor of 1.4979 (95% interval 1.4661–1.5176) over no
profiling. Adding HIP events to existing tracing cost 1.0050
(1.0013–1.0093) in process wall time; application wall time was 1.0012
(0.9965–1.0068). All ten event-enabled traces had two RECORD and three WAIT
records despite four host wait calls. Tracing remains opt-in, and causal event
matching remains unqualified. Earlier overhead pilots are preserved but are
superseded by `profiler-overhead-final/report.json`.

P5's `counter_probe.py` passed output and in-place-update correctness in both
application and beta kernel replay modes. Kernel replay produced both manually
separated counter groups in one application invocation, versus two invocations
for application replay. However, `SQ_INSTS_VALU` was zero for every dispatch in
both modes. Positive matching `SQ_WAVES` values alone do not qualify counter
validity. `counter-replay-final/report.json` therefore makes no speedup claim and
does not authorize production adoption. There is no automatic replay fallback.

P4's earlier ShapesSpec experiment remains isolated. Neither these development
results nor the CPU suite provide a formal RDNA4 release qualification or enable
cross-stream SOLAR predictions. The upgrade's demonstrated benefits here are
compatibility and diagnostic capabilities; aggregate execution/compilation
performance regressed in the measured public `torch.compile` scope.

## Regression diagnosis: measurement, generated code and libraries

The subsequent diagnostic artifacts are under
`data/outputs/stack-regression-20261011/`; `diagnostic-summary.json` retains
per-block values, source hashes and descriptive ratios. The original P2
observations are preserved. In particular, their event interval brackets a
Python call and can include GPU idle time while the host submits work. They
measure that invocation path, not isolated device-kernel execution.

The red-capable `reproduce.py` command replayed two unchanged frozen cases and
exited 1 for the reproduced regression (GEMM +19.2%, composite +53.1%). Further
controls covered all eight original GEMM/composite cases in four alternating
blocks: 64 fresh processes, five implementation/control variants, and five
timing methods. Every variant passed the original numerical check. GPU identity,
input hashes and AUTO policy were retained. The methods are original per-call
events, initialized pooled events, deferred synchronization, a bulk event span,
and 100 calls captured inside a graph. Graphs are trusted diagnostic controls,
not a proposed replacement for production scoring or candidate admission.

The following ratios are new/old geometric means over four independent blocks;
above one means slower. They are descriptive diagnostics, not new P2 confidence
intervals or a component-cost decomposition.

| Frozen case | Original compiled invocation | Graph-contained compiled calls | Graph-contained preallocated GEMM |
| --- | ---: | ---: | ---: |
| Irregular GEMM w0 | 1.079 | 1.013 | 1.017 |
| Irregular GEMM w1 | 1.135 | 0.976 | 0.987 |
| Irregular GEMM w2 | 1.157 | 1.061 | 1.057 |
| Irregular GEMM nonaligned | 1.036 | 0.980 | 0.998 |
| Composite w0 | 1.590 | 1.336 | 1.611 |
| Composite w1 | 1.553 | 1.681 | 2.023 |
| Composite w2 | 1.434 | 1.389 | 1.490 |
| Composite nonaligned | 1.400 | 1.261 | 1.381 |

For irregular GEMM w2, direct invocation of the generated runner reduced the
median old/new per-call times from 38.57/45.06 microseconds to 27.26/28.67;
graph-contained preallocated GEMM was 20.31/21.44. This localizes much of the
extra invocation cost above the generated runner, while retaining a smaller
library/device-path difference. It does not isolate Python 3.14 from Dynamo,
guards, allocation, HIP submission or their interactions.

All sixteen block-zero generated runners preserve the same GPU work structure:
one external `mm` for GEMM, or one Triton addition followed by external `mm` for
the composite. New wrappers contain alignment checks/copy guards, but all
diagnostic inputs were aligned. There is no evidence here of lost add/matmul
fusion or an extra computational kernel introduced by the compiler.

The composite w1 is only `(16,32) @ (32,16)` in FP32. Its graph-contained
preallocated GEMM medians were 3.82/8.11 microseconds. Independent native traces
observed hipBLASLt `MT16x16x16` in the old stack and `MT64x64x16` in the new
default path. This identifies a different selected implementation; tile size
alone is not a proven root cause, and solution indices cannot be reused across
library versions.

Selecting PyTorch's `cublas` preference alone did not isolate hipBLASLt: the
rocBLAS logs still delegated to it. Adding `ROCBLAS_USE_HIPBLASLT=0` in isolated
containers changed the actual traced kernels and removed hipBLASLt log calls.
For composite w1 this brought the new preallocated GEMM to 4.94 microseconds
(old 5.28), but made irregular FP16 GEMM w2 about 79.47 microseconds versus its
default 21.44. Do not apply this switch globally. The interpretation follows
the [PyTorch backend API](https://docs.pytorch.org/docs/stable/backends.html)
and [rocBLAS backend control](https://rocmdocs.amd.com/projects/rocBLAS/en/latest/conceptual/rocblas-design-notes.html),
and is checked against actual native dispatches rather than the preference name.

A standalone C++ client then removed Python, Dynamo and Inductor entirely. It
used fixed FP32 ones/twos, checked all 256 outputs against exactly 64, and ran
four alternating old/new blocks with both zero and 32 MiB workspace limits.
`native-client-exact-v2/` binds the Torch-bundled old libraries through temporary
diagnostic SONAME links and the new SDK libraries; `ldd` verifies each run.
With the 32 MiB limit, median graph-per-call times were 4.06/8.53 microseconds,
and both algorithms actually requested zero workspace. All sixteen runs passed.
This reproduces the regression below the Python/compiler layers; it does not
separate hipBLASLt from its corresponding HIP runtime. Initial system-library
and missing-resource trials remain archived and are not the accepted exact-library
comparison. No host library or image was replaced.

The resulting decision is to preserve separate invocation and device-work
diagnostics, retain the new stack's compatibility/diagnostic functionality, and
investigate shape-specific small-FP32 GEMM algorithm selection before promoting
performance-sensitive cases. Keep the old image as a comparison/fallback.
There is no justification here for disabling hipBLASLt globally, bypassing
guards in production, enabling graphs by default, or attributing the full P2
aggregate regression to GPU machine code. No production execution policy was
changed by this diagnosis.

The independent [VALU research](../../../docs/internal/rocm101-gfx1200-valu-counter-research.md)
identifies AUTO perfmon clock gating as the leading shared mechanism with
ROCm issue #8523. A separate, authorized STABLE_STD counter experiment remains
necessary to verify current-version VALU behavior. No zero-valued PMC was used
to explain these unprofiled timing results.

## Original implementation checkpoint

Run artifacts: `data/outputs/stack-upgrade-20261011/` (uncommitted).
Source baseline: `e5a0d450`; upgraded dependency baseline: `1b431d74`.
The new application image is
`sha256:097b9b95330f5c63081adee7dd5b4145a000ba5ea40bff0df8de3279d1996ea2`.
A separate `sol-execbench:upgrade-validation-20261011` image adds exactly the
development requirements exported from `uv.lock`. It preserves the application
image and GPU dependencies.

| Phase | GPU budget | Checkpoint |
| --- | ---: | --- |
| P0: freeze baseline and 24 workloads | 0 h | New image/device identified. Exact old 7.2 application image and representative workload manifest are not recovered/frozen. |
| P1: compatibility and correctness | 1 h | CPU regression, HIP/Triton/hipBLAS/MIOpen/rocWMMA samples and native profiler checked. CK headers missing, so acceptance remains partial. |
| P2: paired benefits | 3 h | Strict analysis tool implemented; no comparative acquisition or speedup claim. Requires completed P0/P1 and applicable static/canary/full receipts. |
| P3: event diagnostics | 2 h | Bounded JSON/rocpd readers, profile summary, informational feedback and conservative schedule correction implemented. Wait-to-record causal matching remains unverified. |
| P4: ShapesSpec | 1 h | Isolated concrete replay experiment implemented and run; no production IR registration. |
| P5: counter replay and synthesis | 1 h | Counter replay acquisition deferred. No replay speedup or production adoption recommendation. |

Budgets are ceilings, not a requirement to spend eight GPU hours. This checkpoint
does not resume previous held-out or diagnostic-model batches. Never treat a
partial directory, CPU receipt, or these experiment reports as qualification.

## Bounded probes

Run inside the new validation image with the checkout on `PYTHONPATH` and a
writable output directory. Use an external process timeout; the event probe also
requires the repository's shared GPU lock directory mounted from the host and
`ROCR_VISIBLE_DEVICES=GPU-a30fa29c1cca1511` for the verified local gfx1200 device.
Revalidate the UUID/BDF before using these local settings on another host.

```bash
python scripts/internal/upgrade/extension_recovery_probe.py --output /outputs/extension-recovery.json
python scripts/internal/upgrade/shapes_spec_probe.py --output /outputs/shapes-spec.json
rocprofv3 --hip-runtime-trace --kernel-trace --output-format json rocpd \
  --output-directory /outputs/events --output-file probe \
  -- python scripts/internal/upgrade/event_probe.py
```

The extension probe interrupts only its owned compilation process group and
retries the same temporary build directory. It validates CPU `cpp_extension`
lock recovery, not HIP compilation correctness. The event probe is trusted test
code with a short GPU delay to retain actual WAIT barriers; it is not a candidate
allowed through reward-hack admission. Clock policy is never modified by these
probes. Canonical no-profile evaluation has no new tracing cost.

ShapesSpec receipts include an owned JSON constraint description, concrete
minimum/middle/maximum/nonaligned and invalid inputs, real/fake graph code,
Python heap peak, and tracing/replay time. Raw graph acceptance of invalid inputs
is recorded rather than silently treated as constraint enforcement. Symbolic
shape branches may fail to trace; successful finite replay is not a proof over
the entire domain. The upstream `to_jsonable()` assumption repr is not persisted
as a stable SOLAR contract.

## Paired benefit analysis

`paired_analysis.py --input pairs.json --output comparison.json` accepts
`workload_ids` (exactly 24 unique preregistered IDs) and `pairs` (10 blocks per ID).
Each pair requires `workload`, `block`, `candidate_sha256`, `protocol_sha256`,
`measurement_kind` (`cold_compile` or `hot_execution`), `gpu_uuid`, immutable
`old_image`/`new_image`, `old_first`, `correct`, `old_ms`, and `new_ms`. Even blocks
run old first, odd blocks new first. Freeze the measurement protocol and source
before collecting; analyze cold compilation and hot execution in separate files.
This analysis cannot validate upstream acquisition receipts by itself.

Missing/duplicate pairs, changed candidate/GPU/image identities, wrong execution
order and numerical failures reject the whole comparison. The tool emits paired
block bootstrap 95% intervals, per-workload ratios, and an equal-workload
geometric mean. Ratios above one mean the new stack is faster. A ratio exceeding
1.05 with its interval excluding one is a development screening improvement;
the reciprocal threshold identifies regressions. There is no multiple-comparison
correction or component-level causal attribution. Log acquisition failures,
memory/artifact costs and success rates separately; do not impute failed pairs.

The original checkpoint required freezing six public development families with four cases each, including
nonaligned boundaries; bind candidate hashes, dtypes, seeds, checks, UUID/BDF,
clock state and timing policy. Recover/build and identify the exact old stack,
restore CK capability, and complete the repository qualification chain where
required. Those items were outstanding at the original checkpoint; the follow-up
acquisition and receipt above supersede that status.

## Causal evidence boundary

`performance_schedule_evidence.v2` emits only verified same-lane dependencies.
Cross-lane timestamps no longer become `happens_before` dependencies. Until host
event generations and dispatch scope can be bound, multi-lane predictions return
`cross_lane_event_dependencies_unverified`. The profile event records deliberately
retain partial information rather than reconstructing a causal graph from time
coincidence. A future implementation needs exact scope and generation evidence
before enabling cross-stream model predictions or recommending synchronization
changes. Profiling remains opt-in pending a measured overhead comparison.
