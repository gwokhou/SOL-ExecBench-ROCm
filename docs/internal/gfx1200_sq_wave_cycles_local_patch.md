# gfx1200 `SQ_WAVE_CYCLES` local patch

## Scope

ROCm 7.2.0 advertises gfx1200 `SQ_WAVE_CYCLES` as SQ selector 24. On the
local RX 9060 XT, selector 24 remains zero while the GPU performance level is
`AUTO`, but produces positive wave-cycle values while the firmware performance
level is `STABLE_PEAK`. Selector 21 is positive in both modes but behaves like
busy cycles and is not a valid replacement.

The local patch therefore wraps `rocprofv3`; it does not modify ROCm's
`counter_defs.yaml`, replace `libhsa-amd-aqlprofile64.so`, or claim a new event
mapping. The wrapper accepts the normal `rocprofv3` arguments, acquires
`STABLE_PEAK`, verifies it, runs the real profiler, and restores `AUTO` on normal
exit or a trapped signal. An already-active external `STABLE_PEAK` setting is
preserved. Other initial performance levels are rejected to avoid destroying a
manual clock policy.

## Install and run

The sudoers rule must cover the exact installed AMD SMI path:

```bash
sudo .venv/bin/python scripts/setup_rocm_clock_sudoers.py \
  --mode install --user "${USER}" --amd-smi /opt/rocm/bin/amd-smi
```

Install the user-local wrapper:

```bash
scripts/patches/gfx1200_sq_wave_cycles/install.sh
```

Run it with ordinary profiler arguments:

```bash
~/.local/bin/rocprofv3-gfx1200-patched \
  --kernel-trace --pmc SQ_WAVE_CYCLES,SQ_BUSY_CYCLES,SQ_WAVES \
  --output-format csv --output-directory out/rocprof -- <workload>
```

The wrapper serializes patched profiler sessions. `ROCPROFV3_REAL`, `AMD_SMI`,
`SOL_EXECBENCH_SUDO`, and `XDG_RUNTIME_DIR` are explicit test and deployment
seams; normal use should keep their defaults. Because `amd-smi set -l` is
invoked without `-g`, the wrapper verifies and changes every visible AMD GPU as
one lifecycle.

## One-click rollback

Either installed or repository copy performs a checksum-guarded rollback and
resets a remaining `STABLE_PEAK` state to `AUTO`:

```bash
~/.local/bin/rollback-rocprofv3-gfx1200-patch
```

The rollback refuses to remove a modified installed file or a wrapper that is
still running. It does not remove the sudoers rule because that rule is shared
by the repository's normal benchmark clock-lock support.
