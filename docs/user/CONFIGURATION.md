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

Without an outer owner, the PMC window verifies initial AUTO, sets and verifies
STABLE_PEAK, then restores and verifies AUTO. With a verified outer host guard or
nested PMC owner, it **borrows** that policy and never resets clocks on inner exit.
Borrowing requires a unique owner token, a current owner record covering the same
physical GPU identities, a live owner file lock, and a fresh STABLE_PEAK readback.
An environment flag or a GPU that happens to be at STABLE_PEAK is insufficient.
Unknown external policies are not overwritten. Ownership proof failures remain
fatal even in best-effort mode.

Exceptions, timeouts, SIGINT and SIGTERM unwind the existing clock lease. The
order is child/process-group cleanup, owned Docker container removal, clock
restore and readback, final owner receipt, then GPU-lock release. If container
removal cannot be confirmed, the host leaves clocks alone and records a recovery
blocker. Release is idempotent; failed restoration does not mark the lease released.
SIGKILL and host failure cannot run teardown. A stale active owner or failed
restore blocks the next GPU-lock acquisition instead of silently resuming work.

PMC receipts are stored independently in
`<profiler-output-directory>.pmc-clock/`. They distinguish `owned`, `borrowed` and
`none`, retain the owner lease ID, and record `borrow_ended` separately from
`auto_verified`. A borrowed receipt never claims AUTO restoration. The owner's
record in `$SOL_EXECBENCH_GPU_LOCK_DIR/clock-leases/<lease-id>.json` records the
final restoration outcome; its sibling lock proves live ownership and its
container-ID file survives host crashes. These are cooperative orchestration
records on a trusted shared directory, not authentication against a process with
write access to that directory. The owner token is not forwarded to untrusted
kernel subprocesses.

`run_docker.sh` forwards the owner token and PMC options through the existing
shared lock-directory mount. Thus an unprivileged container can borrow a verified
host policy without sudo or broader container permissions. The host remains
responsible for restoration after the container is confirmed stopped. The wrapper
owns `--cidfile`; do not supply another one. Host and container must both use the
updated code. No benchmark clock policy or canonical timing protocol is changed;
PMC replay still cannot replace canonical timing samples.

An independently owned PMC window needs permission for the exact
`sudo -n amd-smi set -l STABLE_PEAK -g VERIFIED_UUID` and corresponding
`AUTO` command. BDF/index selectors are resolved to verified AMD-SMI UUIDs before
mutation; explicit `all` applies the same command separately to each verified UUID. This feature does not install permissions or elevate Python.

After a crash or restoration failure, inspect the owner record, stop its abandoned
workload/container, and restore only its recorded GPU devices with AMD-SMI. Then
run the verification-only recovery command on the host, using the same lock directory:

```bash
uv run python -m sol_execbench.core.bench.clock_recovery LEASE_ID
```

Recovery does not change clocks or kill processes. It acquires GPU/owner exclusion,
checks container absence where applicable, physical identities, GPU idleness and
AUTO, preserves the original record in `clock-leases/recovery-history/`, then
records the successful recovery. Missing container identity, unreadable state,
a live owner or failed verification leaves the gate blocked. Do not delete pending
records to bypass those checks.

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
| `SOL_EXECBENCH_GPU_LOCK_DIR` | directory for per-device lock files |
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

`SOL_EXECBENCH_CLOCKS_MANAGED_BY_HOST` and
`SOL_EXECBENCH_GPU_LOCK_MANAGED_BY_HOST` are private wrapper-to-container
signals. Do not set them manually. The entrypoint independently verifies a
host-declared clock state, and the evaluator verifies that the shared GPU lock
file is held by an external process before it skips its local lock acquisition.

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
Use the generator option `--gpu-uuid VERIFIED_UUID` for device-scoped rules;
omitting it retains the legacy all-device scope. Generation does not install
permissions automatically. Its read-only `check` mode reports
`allowed_authentication_unverified` when sudo permits a command: successful
`sudo -l` alone does not prove passwordless execution, so this result is not
reported as verified coverage. Actual controlled lease execution remains the
verification of noninteractive acquisition and AUTO restoration.
