# Phase 162 Context

Phase 161 and the follow-up debug session showed the remaining 10
`partial_profiler_backed` RDNA4 targets are not primarily profiler lifecycle
failures. Their non-PASSED traces are HIP OOMs in `gen_inputs`, reference
`run()`, or the staged reference user-function path on the current 16GB RDNA4
device.

The project needs the 235-problem denominator to remain accounted without
misrepresenting those OOM targets as either full profiler coverage or
retryable profiler gaps.

This phase therefore promotes the blocker class into the coverage report as
`reference_oom_blocked`.
