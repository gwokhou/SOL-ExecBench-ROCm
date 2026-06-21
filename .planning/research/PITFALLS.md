# Research: Pitfalls for v1.38 Confirmed Benchmark Evidence

**Date:** 2026-06-21

## Pitfalls

- `rocprofv3` can produce different file layouts by output format and ROCm
  version. Discovering only `output_file*` in one directory can misclassify a
  successful run as `rocprof_no_registered_artifacts`.
- Profile summary can become misleading if bottleneck hints are derived from
  incomplete counters or if raw profiler fields are treated as normalized SOL
  authority.
- Official score evidence can be confused with `speedup_factor` or a derived
  AMD score report if the schema does not record source, aggregation, and
  baseline authority explicitly.
- Falling back to `reference_latency_ms` can silently create
  `placeholder_baseline` behavior unless valid-run score production rejects or
  labels that path.
- HIP cutover logic will overclaim if it treats diagnostic profile summary or
  agent feedback as cutover authority.

## Prevention

- Register artifacts recursively or through a version-aware manifest with tests
  for CSV, JSON, rocpd, and nested output layouts.
- Keep profile summary authority flags false; put confirmed score/baseline
  authority into separate evidence reports.
- Require non-null score only when measured latency, official baseline, SOL
  bound, and aggregation policy are present and valid.
- Make baseline coverage validation explicit and expose blocker reason codes.
- Add HIP fixtures for confirmed pass, missing score, missing baseline,
  placeholder baseline, profiler partial, and diagnostic-only sidecar cases.
