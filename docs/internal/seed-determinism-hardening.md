# Seed Determinism & Value-Cache Hardening — Design Review

Scope: audit residuals `seed-seed1` (P0-4) and `seed-seed2` (P0-5) from the
2026-08-02 blindspot audit. This is the design record for why they remain open
after the hardening pass and what closing them fully entails. **No canonical
timing, calibration, corpus, inference, acceptance, or agent-feedback artifact
changes as a result of this document.**

## Threat model

### seed-seed1 — deterministic-input precompute-and-lookup

`derive_custom_input_seed` (`core/bench/custom_inputs.py`) is a pure function
of `(definition.name, workload.uuid, row_index, base_seed, round_index)`. Every
input is pinned by frozen artifacts: `base_seed` defaults to 200
(`BenchmarkConfig.seed`), `round_index` is validated to `0..9`, and the timing
phase reuses `round_index=9`. The same `(workload, row, round)` tuple produces
byte-identical inputs on every host and every run.

An attacker who knows this can reproduce the round-9 inputs offline, run the
trusted reference, and ship a kernel that recognizes the byte-identical inputs
during timing and returns the precomputed reference output from a table —
correct output, near-zero latency.

### seed-seed2 — value-keyed cache collapsing latency within a trial

`ShiftingMemoryPoolAllocator` (`core/bench/memory_pool.py`) shifts `data_ptr`
by 256 B per iteration but `copy_()`s the **same source values** into each
shifted slot. A value-keyed cache (key = a stable reduction of the inputs, e.g.
`x.sum().item()` or `torch.equal`) therefore hits on every iteration after the
first within a trial. The same `inputs` list feeds all three trials of
`measure_solution_latency`, so the cache hits for the entire timing phase after
one real computation.

## Current defenses (shipped 2026-08-02, P0/P1 remediation)

The static output-cache detector (`core/bench/reward_hack/static_review.py`,
`models.py`) already blocks:

- cache **population** writes (`_cache[key] = out`) — the exact write that makes
  a value-keyed cache collapse latency;
- cache **initialization** (`_cache = {}` / `defaultdict(...)`);
- decorator caches (`@functools.cache`, `@cache`, `@lru_cache`,
  `@cached_property`);
- key-derivation primitives (`data_ptr`, `tobytes`, `hashlib.{md5,sha1,sha256}`);
- the non-Python fallback regex covers the same families in `.hip`/`.cpp`/`.cu`.

`verify_timing_function_intact()` and the sealed integrity guards catch the
`elapsed_time` swap a precompute kernel would pair with. The contiguous-slice
gate (`core/bench/correctness.py`, P0-6) blocks a related DPS skip vector.

These close the direct, naively-named forms. Two residuals remain.

## Residual gaps (why not fully closed)

1. **Lookup containers outside the cache/memo allowlist** (`_results`, `_table`,
   `_lookup`, ...). The allowlist is `{cache, _cache, cache_dict, _cache_dict,
   memo, _memo}` plus `*_cache_dict` / `*_memo` suffixes. An attacker picks a
   name off-list. This is the open-ended static-obfuscation tail the audit flags
   as "a representative sample, not a closed set" — it is not closable by
   allowlist without high false-positive risk on legitimate result buffers.
2. **Within-trial byte-identical input values.** Even with no cache container, a
   kernel that computes once and reuses the result internally (e.g. branches on
   `x.sum().item()`) collapses latency. Closing this requires varying the input
   values per iteration, not just the `data_ptr`.

## Complete fix — and why it is boundary-constrained

Both residuals require breaking input determinism or value-repetition, which
the canonical-timing boundary currently pins.

### Option A — per-run input entropy (closes seed-seed1)

Mix a per-run nonce into `derive_custom_input_seed` so offline precompute
cannot reproduce the inputs. **Impact:** `input_sha256` (recorded in
`timing-evidence.jsonl` and the candidate evidence under
`data/outputs/.../candidate/`) changes every run. This **invalidates all frozen
v6 timing evidence and AKA evidence** and requires a full re-collection and
re-freeze. Boundary conflict: the harness pins canonical Trace timing and cited
`input_sha256` evidence as immutable.

### Option B — per-iteration input variation + reference validation (closes seed-seed2)

Vary inputs each timing iteration (not only `data_ptr`) and re-validate against
a freshly computed reference each iteration. **Impact:** this changes the timing
protocol. v6 acceptance gates on
`bench_config.timing_protocol == OFFICIAL_ROCM_TIMING_PROTOCOL`
(`warmup_runs == 10`, `iterations == 50`, `trials == 3`, single round-9 input
set, `lock_clocks`). Per-iteration variation flips the protocol to
`CUSTOM_ROCM_TIMING_PROTOCOL`, so acceptance must be re-run and re-frozen.
Boundary conflict: the official protocol is a frozen acceptance precondition.

### Recommended path

- **Do not** change the canonical-timing boundary in a hardening pass.
- Schedule Options A+B together as a **coordinated corpus re-freeze**: design
  the new protocol (per-run seed nonce + per-iteration input variation +
  per-iteration reference validation), re-collect the 440 development + 220
  held-out cases, re-fit inference, and re-accept. This is a multi-day GPU
  effort, not a defensive patch, and it must be done before describing v6
  evidence as superseded.
- Until that re-freeze, rely on the static detector (closes the naive forms),
  the contiguous-slice gate (P0-6), and `verify_timing_function_intact`.

## Safe partial hardening — considered and rejected

- **Expand the cache-name allowlist** to `_results` / `_table` / `_lookup`.
  Open-ended arms race; the next attacker uses `_buf`, `_store`, or a bare
  `list`. Each addition raises false-positive risk on legitimate buffers. The
  audit explicitly treats this family as not closed by enumeration.
- **Runtime latency-floor check** (flag kernels whose latency is below a per-
  shape floor). Requires a per-kernel latency threshold, which is itself a
  timing-protocol parameter — a boundary change, not a defensive patch.
- **Byte-identical-output detection** during timing (flag candidates whose
  output matches the reference exactly with very low latency). A correct,
  deterministic kernel legitimately produces byte-identical output, so the
  signal is only meaningful paired with a latency floor — again a boundary
  change.

No partial code change closes either residual without touching the boundary.
This document is the explicit record of that conclusion.

## Non-goals

- This document does **not** weaken the canonical-timing boundary.
- It does **not** add old-schema compatibility readers or change any schema
  version.
- It does **not** alter the frozen v6 evidence hashes.
