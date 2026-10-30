# Resource calibration preparation and ownership

The supported resource qualification entry is
`scripts/internal/rdna4/run_qualified_rdna4_resource_peak_calibration.py`.
It keeps the existing static -> canary -> full -> run contract. No stage grants
permission to execute a later stage; local GPU scheduling and authorization still
apply. Qualifications have `performance_authority=false`.

## Prepare without GPU ownership

Run `qualify-static --qualification-root <qualification> --out <formal.json>`
in the intended fixed compiler environment **without** the GPU host guard or
GPU device mounts. This compiles every bounded tuning configuration, not only
the first configuration per probe. Static binaries and their hashes use the
existing qualification artifact manifest; there is no parallel cache database.
Repeat the same static command to verify/reuse that chain without recompiling.
Do not use a GPU-enabled job merely to run this CPU preparation.

`qualify-static --check-only` verifies a completed static artifact chain.
`qualify-full --check-only` verifies a completed full chain, including its parents,
on CPU. Its JSON explicitly says `device_readback_performed=false`; it is not a
replacement for live device identity or clock verification. Invalid/missing
artifacts fail rather than silently rerun or initialize a new chain.

A reusable gate binds the producer and wrapper bytes, compiler hash, probe
sources, profile/gfx, tuning/measurement counts, unlocked-clock option, Git
revision, predecessor gate bytes and every listed artifact. Changes to those
inputs require a new isolated qualification root. This does not make a compiler
hash an identity for an entire arbitrary container: use a pinned image and retain
its digest in the execution record. Keep historical roots unchanged; changing a
wrapper or producer does not requalify old receipts.

## GPU execution

After preparing static artifacts, invoke canary/full/run under the existing
host guard and fixed device configuration. Qualification validates the CPU chain
once, then queries live device/clock identity inside the native lifecycle before
using it. A reused GPU gate still requires that live check. Final gate publication
occurs only after successful cleanup. A parent changed during execution is
rejected, never rebound to a different unvalidated parent.

The qualified `run` path consumes the complete binary set referenced by the
static gate. It does not call the compiler, including when invoked from a
host-managed container. Missing or changed binaries fail; there is no compile
fallback in the GPU job. ISA XML/helper prerequisites must already be provisioned:
ISA inspection is offline (`allow_download=False`) and does not install anything.
The direct legacy producer still prepares its own binaries before its native
lease, but an enclosing host guard would already hold L1: use the qualified
static-artifact path to remove compilation from that outer ownership window.

Tuning candidate order, tie-breaking, held-out ordering, sample counts, numerical
assertions and nominal-limit checks remain unchanged. Actual GPU initialization,
tuning, measurement, synchronization, clock recovery and cleanup remain covered
by native ownership. CPU preparation does not loosen device or admission checks.

## Timing and scope

Each invocation emits `calibration_timing` JSON to stderr, preserving stdout's
existing gate protocol. It records CPU preparation, native GPU lifecycle
(including lock wait and cleanup), and qualification finalization, with completion
flags retained on failures/cancellation. These are wall times, not GPU event
latencies. Under a host guard, the wrapper's inner timer does not measure the
outer L1 hold time; use the host execution record for that interval.

Preparation of all tuning configurations costs more than the prior first-probe
static build, but happens once without GPU access. Static reuse avoids that build
on subsequent unchanged invocations. Only measured CPU preparation/reuse savings
may be reported before GPU validation; no performance speedup follows from these
changes alone. Generic E2E `torch.utils.cpp_extension.load()` still loads native
constructors and has not been moved outside its GPU safety boundary.

## Bounded compilation and cache reuse

Use `qualify-static --compile-jobs 2` for cross-probe compilation. The default
remains one. The actual count is capped by CPU affinity, visible cgroup v2 CPU
quota and memory limit (1 GiB per compiler with a 512 MiB coordinator reserve).
Unknown/v1 resource layouts stay serial. These are scheduling estimates, not
peak-RSS guarantees; the container's hard limits must remain enabled. Do not
combine outer parallel probe jobs with additional compiler-internal worker flags.

Compilation order may overlap but artifact inventory and later GPU measurement
order remain deterministic. Each binary is published by atomic replace only
after success. Failure or termination cancels queued work, terminates active
compiler process groups and waits for their cleanup. Incomplete builds never
publish a static gate. An existing qualification root has a separate CPU-only
writer lock; a competing writer fails before touching files rather than
recompiling concurrently. It never uses a GPU ownership lock for compilation.

Static reuse additionally binds the compiler version, resolved clang bytes and
relevant compiler environment. Jobs count is scheduling-only and does not
invalidate identical outputs. Compiler/flags/probes/architecture/producer drift
still rejects the old root; use a new isolated root, do not rewrite old evidence.
The pinned image remains necessary: this fingerprint is not a hash of every
system header or library. Cache verification is never permission to run GPU work.

For native extensions, `MAX_JOBS` crosses `run_docker.sh` and is explicitly
bounded by the same resource policy at the compilation boundary; the sanitized
evaluation environment still omits it. Unset means one, not Ninja's unbounded
default. Native cache use remains explicit via the existing immutable-image,
source/solution/toolchain-bound cache API. Concurrent publications remain atomic;
restore preflights all existing destination paths before copying any artifact.
Worker count alone does not change the native cache key. This change does not
make all native compilation cached by default, nor prove cold-start latency from
a warm run. No writable shared Triton/MIOpen cache or new mount is introduced.

## Raw throttle evidence and immutable ISA extraction

New calibration snapshots retain `throttle_status_cli` and `throttle_evidence`
(raw uint32/uint64 values, Python types, binding conversion, metrics header,
API/CLI versions and binding identity). The derived status is THROTTLED,
UNTHROTTLED, or UNKNOWN. Unsigned maximum values, malformed/missing evidence,
and conflicting observations cannot certify unthrottled operation. The producer
uses bounded management-only child queries inside the existing native lease.

The v4 envelope remains readable. Historical snapshots without raw evidence
remain usable only for their existing limited corroboration scope; they cannot
prove a new strict unthrottled qualification. No historical record is rewritten.
Raw flags are observations, not proof of physical overheating or performance loss.

ISA inspection supplies an explicit output ELF to llvm-objcopy and uses separate
per-call extraction directories. The measured input bytes and mode are preserved;
failed calls remove their own partial files, while successful extracted code
objects remain in the caller's workspace. Consumers use the returned path rather
than assuming a fixed basename directly beneath that workspace.
