# Quick Task: RDNA4 Clock-Lock Workload Evidence

## Description

Collect benchmark-grade clock-lock evidence on RDNA4 `gfx1200` host by running a
sustained GPU workload while SCLK/MCLK are locked to maximum levels. The Phase
167 evidence was collected on an idle GPU (low-power state, clocks not at
requested levels). This task repeats the test with active GPU compute to verify
whether stable maximum-frequency lock is achievable during a benchmark window.

## Steps

1. Record pre-state: `rocm-smi --showclocks --showperflevel --showpower --showtemp --showuse`
2. Set perf level to manual
3. Record intermediate state
4. Lock SCLK to level 2 (2780Mhz) and MCLK to level 5 (1258Mhz)
5. Start sustained GPU workload (PyTorch ROCm GEMM loop for ~30 seconds)
6. Every 5 seconds during workload: record clock state with `rocm-smi --showclocks --showclkfrq`
7. After workload: record post-run state
8. Reset clocks to auto
9. Record final state
10. Write evidence report
11. Update claim boundary documentation if clocks prove stable

## Success Criteria

- SCLK observed at level 2 (2780Mhz) during active workload
- MCLK observed at level 5 (1258Mhz) during active workload
- Stable clocks maintained throughout the benchmark window (no drops)
- Clock reset succeeds and returns to auto/low-power state

## Claim Impact

If stable lock is proven: RDNA4 timing can be upgraded from non-authoritative to
benchmark-grade (for this host, with clock-lock evidence).

If stable lock is NOT proven: Timing remains non-authoritative, with stronger
evidence that the current host cannot provide benchmark-grade clock control.
