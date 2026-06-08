---
phase: 147
title: RDNA4 memory and timing closure attempt
---

# Phase 147 Context

The user clarified that the final v1.31 phase must not simply accept residual
boundaries. Phase 147 must first make a concrete attempt to reduce RDNA4
validation memory pressure and investigate why the timing sidecars still show
PyTorch/device-event fallback.

Relevant current evidence:

- Phase 143 produced 121 timing sidecars, all labeled fallback event timing.
- Phase 144 and Phase 146 showed substantial RDNA4 memory pressure on the
  current host: 32 GiB system RAM and 16 GiB RDNA4 VRAM.
- Phase 146 classified 146 failed workload records, with OOM classes dominating
  the failure set.

The phase may still end with accepted residual boundaries, but only after
recording the attempted fix or root-cause diagnosis. Public RDNA4 claims remain
bounded unless the evidence actually changes.
