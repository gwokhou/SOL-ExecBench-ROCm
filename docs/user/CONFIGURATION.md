# Configuration

The v3 CLI is the configuration authority. Generate its exact machine-readable
surface with:

```bash
uv run sol-execbench --format json contract cli
```

Root options such as `--format json` precede the command. Command-specific
options follow it.

## Benchmark config

`evaluate --config FILE` accepts this JSON shape:

```json
{
  "schema_version": "sol_execbench.benchmark_config.v2",
  "warmup_runs": 10,
  "iterations": 50,
  "trials": 3,
  "min_measurement_time_seconds": null,
  "lock_clocks": true,
  "benchmark_reference": true,
  "seed": 200
}
```

| Field | Default | Constraint |
| --- | --- | --- |
| `warmup_runs` | `10` | integer ≥ 0 |
| `iterations` | `50` | integer > 0 |
| `trials` | `3` | integer > 0 |
| `min_measurement_time_seconds` | `null` | positive number or null |
| `lock_clocks` | `true` | Boolean |
| `benchmark_reference` | `true` | Boolean |
| `seed` | `200` | integer |

The paper timing label requires the exact defaults for warmup, iterations,
trials, minimum duration and clock locking. Changing them produces a diagnostic
custom timing protocol.

## Evaluate

```bash
uv run sol-execbench evaluate PROBLEM_DIR --solution solution.json

uv run sol-execbench evaluate \
  --definition definition.json \
  --workload workload.jsonl \
  --solution solution.json
```

| Option | Default | Meaning |
| --- | --- | --- |
| `--config` | none | benchmark config JSON |
| `--compile-timeout` | `120` | native compile timeout in seconds |
| `--timeout` | `300` | whole evaluation timeout in seconds |
| `--trace-output` | none | durable canonical Trace JSONL path |
| `--lock-clocks` | off | force the clock-lock requirement |
| `--unsafe-local-execution` | off | allow diagnostic host execution |
| `--keep-staging` | off | preserve staged process assets |
| `--profile` | `none` | `none`, `rocprofv3` or `rocprofv3-counters` |
| `--static-evidence` | `none` | `none` or `auto` |
| `--decision` | `none` | `none` or `auto` |
| `--feedback-*` | none | optional derived-sidecar identities |
| `--verbose` | off | print bounded subprocess details |

### PMC clock window

`rocprofv3-counters` collects replay evidence **after canonical timing**. Its
clock policy is independent of benchmark `lock_clocks` and trace-only profiling.
The existing evaluation GPU lock and owned `ClockLockLease` teardown are reused.

| Environment option | Default | Meaning |
| --- | --- | --- |
| `SOL_EXECBENCH_PMC_CLOCK_TARGET` | `0000:05:00.0` when `ROCR_VISIBLE_DEVICES=GPU-a30fa29c1cca1511`; otherwise no automatic window | AMD-SMI BDF, UUID, index, or explicit `all` |
| `SOL_EXECBENCH_PMC_CLOCK_MODE` | `required` | `required` refuses replay if setup fails; `best-effort` permits unlocked diagnostic replay after verified rollback |

Prefer a BDF or UUID over a mutable enumeration index. `all` deliberately changes
every AMD GPU, and is never selected implicitly. The qualified local BDF is
also checked against its known AMD-SMI UUID before any write.

For the qualified gfx1200 target, set these in the **process running evaluate**:

```bash
export ROCR_VISIBLE_DEVICES=GPU-a30fa29c1cca1511
export SOL_EXECBENCH_PMC_CLOCK_TARGET=0000:05:00.0
export SOL_EXECBENCH_PMC_CLOCK_MODE=required
```

## GPU ownership and recovery

GPU execution requires `GPU_LOCK_DIR` (an absolute host directory) and
`GPU_LOCK_UUID` (the full AMD-SMI UUID). The agreed local configuration is:

```bash
export GPU_LOCK_DIR=/home/guohao/.local/state
export GPU_LOCK_UUID=a3007590-0000-1000-800f-a29c1cca1511
export ROCR_VISIBLE_DEVICES=GPU-a30fa29c1cca1511
export SOL_EXECBENCH_CLOCK_STATE_DIR=/home/guohao/.local/state/gpu-leases/sol
```

Do not activate these new lock names while another project or installed version
still uses the old index lock. Maintenance must stop admissions, confirm cleanup
and recovery of existing owners, preserve their histories, switch both projects,
and then resume admissions. This source change performs none of those operations.
`SOL_EXECBENCH_GPU_LOCK_DIR`, `SOL_EXECBENCH_GPU_LOCK_FD` and
`SOL_EXECBENCH_GPU_LOCK_MANAGED_BY_HOST` are rejected, even when empty.

The files are `gpu-<UUID>.L1-ownership.lock` and
`gpu-<UUID>.L2-execution.lock`. Native owners acquire L1 before L2 and keep L1
through native final-state publication. Workers under an existing host owner
verify its live private receipt and L1 inode, then serialize on L2. Their scope
never restores the host's clock policy. A raw flock or an environment flag alone
is not a native owner. Same-thread nested stages reuse the active scope; an
optional SOLAR private lock cannot replace L1/L2 or cause reacquisition of them.

Direct evaluation, the SOLAR CLI/release bridge, marked GPU tests, profiler
collectors and calibration entries use the native boundary. Direct SOLAR GPU
API callers, including `ConversionReadinessRequest`, must supply an
`AnalysisExecutionPolicy` with a `device_stage_context` on GPU requests. The
readiness boundary uses the same cleanup callback and lock scope as analysis;
CPU requests need neither. The ExecBench readiness bridge also enters the
native scope before querying its formal device. Low-level evaluator/profile
workers require their supervisor's active owner and held L2. CPU paths neither
require these variables nor query devices. Direct native execution currently
supports visible device 0 only: read-only AMD-SMI identity and the selected BDF's
`unique_id` must agree with explicit `ROCR_VISIBLE_DEVICES=GPU-...`;
`HIP_VISIBLE_DEVICES` must be unset or `0`. Numeric visibility indices and
implicit/all-device ownership are not accepted.

Natural execution requires initial AUTO, records an owner and verifies AUTO on
exit without issuing a setter. Timing/PMC work requesting managed clocks instead
publishes responsibility before STABLE_PEAK activation, and restores/reads back
AUTO on exit. Inner PMC windows borrow the verified policy. A failed setup,
restore or synchronization remains a failure; it cannot silently permit work or
publish a successful release. `acquire_clock_lock`/`lock_clocks` now live in
`core.bench.gpu_execution` and only borrow an active native scope; independent
all-device callers must use the explicit native entry instead.

The host guard derives its target from `GPU_LOCK_UUID`. Optional
`SOL_EXECBENCH_HOST_CLOCK_GPU_UUID` or `--gpu-uuid` must agree. Both setters use
`/usr/bin/amd-smi set -l LEVEL -g UUID` (or the explicitly configured common
executable override); no sudoers installation occurs. The guard's
`--natural-clocks` mode is used for container dependency checks. Exceptions,
timeouts, SIGINT and SIGTERM drain child groups and owned containers before
restoring clocks and publishing the final receipt, then release L1. Unconfirmed
container cleanup leaves a blocker and does not restore under possible live work.
SIGKILL cannot run cleanup; the persisted native record blocks the next owner.

Private records remain in `$SOL_EXECBENCH_CLOCK_STATE_DIR/clock-leases/`;
without that override the legacy `/tmp/sol-execbench-locks/clock-leases/` is used.
Changing the common lock directory does not relocate or erase private history.
Pending legacy records are still checked when a new private directory is selected.
Owner records, owner flock and container-ID files retain their existing format.
PMC receipts remain in `<profiler-output-directory>.pmc-clock/`; borrowed receipts
record `borrow_ended`, never owner AUTO restoration. Controlled evaluator child
processes receive the native owner token to verify supervision. This is a
cooperative single-user execution contract, not credential-based isolation.

`run_docker.sh` requires both physical lock files to exist, then binds only those
two files at `/run/lock` and passes `GPU_LOCK_DIR=/run/lock`. It separately binds
the existing private record directory read-only at the same absolute path.
No whole host state directory is mounted or replaced. The container entrypoint
requires host-managed ownership and verifies target readback; standalone
unsupervised containers are rejected. Source and entrypoint are selected from
the mounted project so a stale image package cannot silently select the old
protocol. Use the matching host UID/GID. The host owns `--cidfile`, cleanup and
restoration; do not supply another cidfile.

After manually stopping an abandoned workload and restoring its recorded GPU,
run the original owner's verification-only recovery command with the same UUID
and private history settings:

```bash
uv run python -m sol_execbench.core.bench.clock_recovery LEASE_ID
```

Recovery takes L1, checks original identity, container absence where applicable,
GPU idleness and AUTO, then preserves the prior record in
`clock-leases/recovery-history/` before publishing confirmation. It does not set
clocks or terminate work. Missing identity, a live owner, unknown state or failed
readback leaves the gate blocked. Never delete pending records to bypass it.

The optional neutral admission record is explicitly selected with
`SOL_EXECBENCH_ADMISSION_RECORD` (an absolute persistent path outside `/tmp`).
The private directory must be distinct from the common lock directory. Configured
admission requires ready under L1, publishes held before native responsibility,
and returns ready only after native cleanup/restoration is recorded. Missing,
unknown, held or blocked records do not initialize themselves. The record is
host-only; workers borrow the native owner and do not independently admit.
There is no dependency on Spur. A scheduler must call the native entry without
holding either physical lock itself. Deployment uses an explicit record; generic
native locking without that optional setting does not provide cross-project
crash-state admission. CPU fixture results are not real GPU qualification.

Clock setter diagnostics use the existing Python stderr logging stream. Each
STABLE_PEAK setup and AUTO restore/rollback command emits one warning prefixed
`clock_command_diagnostic`, followed by a JSON object. The foreground launcher
must retain stderr; no separate log directory or lease schema is introduced.
Each stdout/stderr tail is limited to 4096 characters, with truncation flags and
the existing credential redaction. Records include the original command, return
code (null for launch failure/timeout), failure kind, UTC time, PID, executable
paths/metadata, and selected UID/GID/Groups/NoNewPrivs/Seccomp fields.

Only PATH, HOME, PYTHONPATH, PYTHONHOME, VIRTUAL_ENV, ROCM_PATH and LD_LIBRARY_PATH
are read from the environment for diagnostics, each limited to 1024 characters
and redacted; no full environment is serialized. These describe the calling
process, not proof of sudo's eventual interpreter or elevated identity. Success,
command failure, zero-exit vendor error markers and timeout retain the existing
clock semantics. Diagnostics do not retry setters or release a held owner.

JSON response mode requires a trace path:

```bash
uv run sol-execbench --format json evaluate PROBLEM_DIR \
  --solution solution.json --trace-output out/run.trace.jsonl
```

## Dataset

Only the pinned public corpus operations are exposed:

```bash
uv run sol-execbench dataset materialize \
  --manifest problems/AMD_AKA/manifest.yaml \
  --device cuda:0

uv run sol-execbench dataset audit problems/local/AMD_AKA/gfx1200
```

`--device` selects the ROCm GPU used for exact gfx detection and workload
probes. `--target-arch` optionally asserts `gfx942`, `gfx1150`, or `gfx1200`;
unknown and mismatched targets fail closed. `--probe-timeout` defaults to 120
seconds per workload. Without `--output`, the destination is
`problems/local/AMD_AKA/<gfx-target>`. This revision has no dataset batch-runner
command; evaluate materialized problems individually or use external
orchestration.

## SOLAR

```bash
uv run sol-execbench solar analyze PROBLEM_DIR \
  --workload WORKLOAD_UUID \
  --output out/solar/WORKLOAD_UUID \
  --orojenesis-home /path/to/orojenesis
```

Options are `--device` (default `cuda:0`), `--timeout` (default 14400 seconds)
`--orojenesis-home`, and `--backend` (default
`torchview_extended_einsum`). The CLI always requires the formal
capacity-constrained Orojenesis bound. It rejects paper-valid
`roofline_eq1_v1` results at the worker, bridge, and CLI boundaries as an
additional port release policy. Formal analysis is constrained by the pinned
gfx1200 architecture audit and the repository-owned Orojenesis binary
allowlist. The allowlist contains the reviewed reproducible mapper digest;
locally substituted binaries still fail closed.

Qualify the complete scored corpus through extraction, strict conversion, and
replay on the formal target with the mandatory uniform chain described in
[Large batch GPU qualification](GPU-QUALIFICATION.md). After static and canary,
the final stage is:

```bash
uv run sol-execbench solar qualify-full out/release \
  --orojenesis-home /path/to/orojenesis \
  --qualification-root out/solar-qualification
```

The chain derives the denominator from the corpus manifest, records the
stable target name `gfx1200`, and emits a content-addressed `matrix.jsonl` plus
`summary.json`. Each workload has stable stage statuses and reason codes,
source-content identities, a canonical `trace_identity_sha256`, three seeds, and
random/zero/boundary verification patterns. The trace identity binds the corpus,
definition, workload, reference, architecture profile, `gfx1200` target, and
trace seed. A failed or interrupted run can be checked and continued with
`--resume`; existing identities and artifact hashes must still match.
Concurrent writers to the same audit root are rejected before GPU work starts.

SOLAR supports only the fixed `torchview_extended_einsum` and `make_fx_aten`
paths. `--backend make_fx_aten` selects the latter explicitly; extractor and IR
dialect cannot be selected independently. One audit or release build uses the
same path for every workload, with no cross-path fallback. Operations without
complete provenance, tracing, conversion, replay, and resource-model support
fail closed.

## Official score

```bash
uv run sol-execbench --format json score status \
  --manifest problems/AMD_AKA/manifest.yaml
```

The checked-in manifest declares the content-addressed publisher policy and
canonical baseline available. The status command accepts no measurement or
baseline file. The scorer accepts only a publisher release bundle that binds
the baseline, candidate, corpus, and SOLAR statements:

```bash
uv run sol-execbench --format json score official RELEASE/release-bundle.json
```

## Environment

| Variable | Use |
| --- | --- |
| `FLASHINFER_TRACE_DIR` | additional safetensors root used by the trusted reference worker |
| `SOL_EXECBENCH_CLOCKS_LOCKED` | evaluator-owned clock-lock evidence (`1` or `0`) |
| `GPU_LOCK_DIR`, `GPU_LOCK_UUID` | explicit common L1/L2 directory and full AMD-SMI UUID |
| `SOL_EXECBENCH_CLOCK_STATE_DIR` | project-private native owner/recovery directory |
| `SOL_EXECBENCH_ADMISSION_RECORD` | optional host-only neutral admission record |
| `SOL_EXECBENCH_ALLOW_CPU_TIMING` | test-only CPU timing escape hatch |
| `SOL_EXECBENCH_SANDBOXED` | marks an externally isolated execution environment |
| `SOL_EXECBENCH_UNSAFE_LOCAL_EXECUTION` | internal marker set by the CLI flag |
| `SOL_EXECBENCH_GRACEFUL_EXIT` | profiler-controlled normal interpreter teardown |
| `SOLEXECBENCH_ENV_SNAPSHOT` | write an environment sidecar when set to `1` |
| `SOLEXECBENCH_ENV_SNAPSHOT_PATH` | explicit environment sidecar path |
| `SOLAR_OROJENESIS_HOME` | default Orojenesis toolchain directory |
| `SOL_EXECBENCH_AMD_ISA_CACHE` | static ISA tool/spec cache |
| `SOL_EXECBENCH_AMD_ISA_OFFLINE` | forbid static ISA downloads when `1` |
| `HIP_VISIBLE_DEVICES`, `ROCR_VISIBLE_DEVICES` | ROCm device visibility |

`SOL_EXECBENCH_CLOCKS_MANAGED_BY_HOST` and `GPU_LOCK_HELD_BY_HOST=1`
are wrapper-to-worker signals. `GPU_LOCK_FD`, when used, must be the actual
inherited L1 descriptor for the configured inode. Neither setting substitutes
for live native owner verification, and borrowed scopes never unlock L1 or
restore the owner's clocks.

The reference pipe descriptors, token and worker PID use private
`SOL_EXECBENCH_REFERENCE_*` variables. They are created by the staged
orchestrator, removed from the candidate environment when connected and are not
user configuration.

## Docker

Use `./scripts/run_docker.sh --build` or select a target declared in
`docker/rocm-targets.json` with `--target`. The wrapper owns its image, device
and dependency variables; inspect `./scripts/run_docker.sh --help` for the
current surface.

Clock command authorization and execution share
`core/platform/clock_commands.py`: the executable resolver, policy values, and
exact argument ordering are common to the legacy all-device clock lease, the
UUID-scoped PMC controller, and `scripts/setup_rocm_clock_sudoers.py`.
The generator requires `--gpu-uuid VERIFIED_UUID` and defaults to the stable
`/usr/bin/amd-smi` invocation, independent of PATH ordering. It grants only the
two exact UUID-scoped STABLE_PEAK and AUTO setters. Link resolution validates
root control without changing the path rendered into the rule. Legacy runtime
calls without a UUID remain all-device calls and are not covered by this rule.
UUID-scoped host setup, AUTO restoration and performance readback use the same
host selector as generation and `check`. To explicitly use another executable,
set `SOL_EXECBENCH_HOST_AMD_SMI` to its absolute invocation path in both contexts;
an empty or relative value is an error. The generator's `--amd-smi` argument
overrides that variable for that invocation only. Legacy all-device and
standalone PMC tool discovery remain unchanged; the host override is not
forwarded into Docker, whose executable layout and policy are separate.
Generation does not install permissions automatically. See the project-owned
[rule and installation instructions](ROCM-CLOCK-SUDOERS.md).
Its read-only `check` mode reports
`allowed_authentication_unverified` when sudo permits a command: successful
`sudo -l` alone does not prove passwordless execution, so this result is not
reported as verified coverage. Actual controlled lease execution remains the
verification of noninteractive acquisition and AUTO restoration.
