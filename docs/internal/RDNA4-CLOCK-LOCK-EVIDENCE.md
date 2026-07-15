# RDNA4 Clock-Lock Evidence

This note records the Phase 167 host clock evidence boundary for RDNA4
`gfx1200` validation.

## Historical Finding

Phase 167 used the now-retired `rocm-smi` clock-control path on the local RDNA4
host:

- manual performance mode can be requested;
- SCLK and MCLK masks can be set;
- reset returns the device to `Performance Level: auto`.

However, observed clocks did not prove stable maximum-frequency lock. After
requesting SCLK level `2` (`2780Mhz`) and MCLK level `5` (`1258Mhz`), readback
showed SCLK level `1` (`0Mhz`) and MCLK level `4` (`1124Mhz`) while the device
was idle and reporting low-power state.

## Claim Policy

`rocprofv3` kernel timing is necessary for profiler-backed timing, but it is
not sufficient for benchmark-grade authoritative timing. RDNA4 timing claims
must stay below authoritative timing unless the evidence bundle includes:

- pre-run clock state;
- benchmark-window observed locked clocks;
- post-run clock state;
- reset evidence returning the device to automatic policy;
- hardware, driver, and host policy metadata.

## Operator Policy

Clock-lock commands should be run by an operator or through a tightly scoped
sudoers policy for the exact `amd-smi version`, `amd-smi set -l STABLE_PEAK`,
and `amd-smi set -l AUTO` commands. Current verification uses
`amd-smi metric -l --json`; the historical `rocm-smi` observations above remain
only as Phase 167 evidence. The release evidence bundle must record the exact
commands and observed readback; successful set commands alone are not enough
to prove lock authority.

## Current Bundle Boundary

Phase 168 may cite Phase 167 as evidence that the host clock-control and reset
paths were exercised, but it must not upgrade RDNA4 timing to authoritative
benchmark timing on the current evidence.
