# Project handoff and active follow-ups

Last audited: 2026-11-27 (`v0.1` MI300X vertical SOL mainline closed out at tag `v0.1` / `70b39875`; handoff consolidated to remove superseded pre-pivot P1 backlogs and stale bootstrap instructions).

This file is the single source of truth for active repository direction, deferred research-gated follow-ups, and hard invariants. Detailed historical campaign logs remain preserved in Git history (tag `v0.1`) and under `docs/internal/`.

## Current state

- **Sole active mainline (`v0.1` ROCm vertical 3-tier SOL diagnosis — closed out across Stages 1–4)**:
  - Stages 1–3B (`ffa928e0`, `9d4be7eb`, `a528f1e8`, `768944f9`) and Stage 4 (`70b39875`) are merged on `main` and tagged `v0.1`.
  - Evaluated on cloud `AMD Instinct MI300X VF` (`amddeveloperclou-vllm0271vllmomni`: Python `3.12.16`, `torch 2.13.0+rocm10.0.0`, `hip 7.15.26333`, `triton 3.8.0`, `vllm 0.34.0`, `aiter`, `rocprofv3`) across **6 core inference primitive families (8 `FlashInfer-Trace` Definitions, 10 Smoke / 34 Full Workloads)** comparing `portable` (`34/34`), `vllm` (`34/34`), and `aiter` (`30/34` across `7/8` Definitions), plus 15 marker-scoped `rocprofv3` `Amp` hardware counter captures.
  - `src/sol_execbench/data/rocprofv3_counters/gfx942_v1.yaml` counter spellings were confirmed on-device against `rocprofv3-avail -d 0 info --pmc` (479 counters) and verified `status=collected` across all 6 primitive families.
  - Verified result bundles are retained locally under gitignored `data/outputs/mi300x_stage3b_final/{portable,vllm,aiter}.tar.gz` (SHA-256 and `FlashInfer-Trace` validated via `scripts.mvp_v1.cloud_preflight verify`). All temporary cloud instances were destroyed and verified HTTP `404` (`0` active instances).
  - Stage 4 deliverables are published in [docs/v0.1_mi300x_sol_report.md](docs/v0.1_mi300x_sol_report.md) and [README.md](README.md); 43 untracked internal research drafts were archived into gitignored `data/archive/internal_drafts_2026_11/`.
- **Scoped methodology**:
  - Input/output contracts reuse existing `FlashInfer-Trace` (`definition.json` + `workload.jsonl` + `solution.json` + `Trace`) with zero new custom artifact schema families.
  - Analytical SOL bounds use transparent dual-ceiling Roofline equations (hardware Spec + live startup Empirical microbenchmarks), bypassing `Orojenesis`.
- **Hardware & Python runtime policy**:
  - Local `RX 9060 XT (gfx1200)` serves as a free development, CPU unit-test, and script-staging host (`uv sync --all-groups` on Python `3.14` / ROCm `10.1.0` wheels).
  - Datacenter SOL captures target cloud `AMD MI300X / MI308X (gfx942)` containers (typically Python `3.12` with pre-baked `torch` / `vllm` / `aiter` ABI).
  - All first-party Python sources target native Python `3.12+` syntax (`[tool.ruff] target-version = "py312"`), eliminating runtime import-hook rewriting.
- **Permanently frozen / terminated tracks (2026-11-21 Strategic Stop-Loss)**:
  - `F2` (nested `PGID_2` timeout cleanup and `mapper_timeout` classification) closed on CPU (`5153900d`); do not trigger GPU reruns.
  - `F1` (in-repo LayerNorm `centered_fp64_v1` formal contract bump) and `F3` (`make_fx_aten` `LLM_CORE` tile-aware/formal-read coverage) are cancelled.
  - Do not run `C3 --resume`, long `timeloop-mapper` / `Orojenesis` scans, `AMD_AKA` expansion, `p1-successor-start700` collection, or RDNA4 diagnostic production re-release/tagging (`gfx1200-diagnostics-v7-production-v1` remains permanently unused).
  - Do not add new custom artifact schema families.

## Active backlog

### Deferred upstream & spin-off TODOs (research-gated)

No blocking in-repo tasks remain for `v0.1`. The following three external contribution / extraction tracks stay gated on upstream code review and ROI verification before implementation:

1. **`TODO-UPSTREAM-1` — Upstream performance / dispatch Issue or PR (`vllm-project/vllm` or `ROCm/aiter`)**
   - Candidate signals from the MI300X 3-tier vertical diagnosis: skinny-`M` FP8 GEMM default dispatch gap (`vLLM` `_scaled_mm` vs. `AITER` pre-shuffled GEMM) and long-context `MLA`/`GQA`/small-batch `MoE` dispatch.
   - Gate: verify latest upstream `main`, active RFC/maintainer priorities, and end-to-end TTFT/TPOT impact before opening a minimal reproducer or patch.
2. **`TODO-SPINOFF-2` — Zero-dependency single-file anti-distortion timing micro-module**
   - Extract the `>=2x` LLC `RotatingBufferPool` / `LLCRotatingFlusher`, read-only telemetry window, and anti-DCE output checks currently coupled to `sol_execbench.core.*` in [scripts/mvp_v1/anti_distortion.py](scripts/mvp_v1/anti_distortion.py).
   - Gate: confirm real maintainer pain points and dependency boundaries in downstream repos (`vllm bench`, `ROCm/aiter`) before publishing a standalone module.
3. **`TODO-UPSTREAM-3` — Lightweight correctness / precision-boundary Issues for `NVIDIA/SOL-ExecBench`**
   - Candidate signals: `E2` LayerNorm reference precision boundary, degenerate input distributions (all-ones weights / all-zeros bias), and tolerance blind spots.
   - Gate: diff directly against upstream `NVIDIA/SOL-ExecBench` `main` to strip all repo-specific `LLM_CORE`/`AMD_AKA` derivations before filing.

<a id="solar-evolution"></a>

## Archived milestones (historical record — frozen)

All pre-pivot `gfx1200` and SOLAR formal campaigns are completed or closed terminal. Detailed receipts and run ledgers live in Git history (`v0.1`) and `docs/internal/`:

- **Official `gfx1200` formal score (`RELEASE/release-bundle.json`, GitHub Release `gfx1200-official-score-v2`)**: completed (`0.497818327` over 43 problems / 163 workloads using `rx9060xt-gfx1200-reference-v2`). See [RELEASE/README.md](RELEASE/README.md).
- **SOLAR M0–M3 / D0 & E2 LayerNorm holdout**: M0–M3 (`B0–B2`, `E0–E2`, `C0–C3`, `D0`) closed out in [docs/internal/solar-d0-decision-2026-11-20.md](docs/internal/solar-d0-decision-2026-11-20.md), [docs/internal/solar-c2-c3-results-2026-11-20.md](docs/internal/solar-c2-c3-results-2026-11-20.md), and [docs/internal/solar-e2-layernorm-holdout-results-2026-11-19.md](docs/internal/solar-e2-layernorm-holdout-results-2026-11-19.md) (`186/186` holdout sessions passed for `centered_fp64_v1`; `C3` closed bounded `INCOMPLETE` at 12h / `168/360` rows; post-D0 roadmap superseded by the 2026-11-21 pivot).
- **RDNA4 diagnostic lifecycle (`start-460`, `start-640`, `start-700`)**: `start-460` and `start-640` closed terminal `accepted=false` (cross-design distribution shift); `p1-successor-start700` stopped at `69/440` development cases and is permanently frozen. P0 conformance (`diagnostic-lifecycle-p0-conformance-v1`) remains archived as mechanism proof only.
- **Known unsupported boundaries (intentionally unimplemented under stop-loss)**:
  - Full CCOB compressed code-object manifest parsing in [amdgpu_metadata.py](src/sol_execbench/core/bench/static_kernel/amdgpu_metadata.py) remains unsupported (best-effort gzip/zlib heuristic only).
  - Forward-only Torchview backward-reference extraction in [src/solar/graph/torchview/](src/solar/graph/torchview/extraction.py) remains unsupported.
  - Formal CDNA3/MI300X `vram_policy` and formal resource-peak calibration remain out of scope for `v0.1` (`sol-execbench solar analyze` formal publication stays `gfx1200`-only).
- **Non-canonical local data directories**: historical/editorial subtrees under gitignored `data/` (`store-control-plane-v2-historical-20260809/`, `store-invalid-production-topology-1459172/`, `calibrations/`, `report-updates/`, `report-figures/`) are marked with `NON_CANONICAL.md` rather than migrated.

## Invariants

- Performance diagnostics never change canonical Trace timing, `T_SOL`, SOL Score, leaderboard values, or rewards.
- Canonical execution precedes profiler replay. Profiler duration, achieved throughput, and measured candidate runtime never become prediction features.
- Evidence identity, schema version, calibration range, artifact hashes, and lifecycle parents fail closed. Old-schema compatibility readers are forbidden.
- `L` remains unavailable without an explicitly supplied trusted frontier.
- Partial or ungoverned diagnostics cannot request kernel code changes.
- Tuning and parameter-estimation samples cannot enter held-out acceptance.
- Candidate inputs use per-run entropy and per-invocation trusted-reference validation. The candidate process receives neither nonce nor expected output.
- Publication evaluation remains networkless, capability-free, and private-IPC.
- Mutable process evidence stays under ignored `data/outputs/`; immutable diagnostic release projections stay under ignored `data/publications/`.
- Current `sol_execbench.*` schema identifiers are defined only in `src/sol_execbench/core/integrity/schema_versions.py`; current SOLAR versions are defined only in `src/solar/schema_versions.py`.

## Verification before handoff

Run the checks relevant to the changed surface, including:

```bash
uv run python scripts/check_current_docs.py
uv run python scripts/check_schema_versions.py
uv run --with ruff ruff check .
uv run --with ruff ruff format --check .
uv run ty check
uv run python scripts/check_coupling.py
uv run python scripts/check_readability.py
uv run python scripts/check_dataclass_policy.py
uv run python scripts/check_production_reachability.py
uv run python scripts/check_non_canonical_artifacts.py
uv run python scripts/check_diagnostic_store_consistency.py
uv run python scripts/check_python_reuse.py
uv run python -m scripts.mvp_v1.cloud_preflight twin --target MI300X
uv run pytest tests/
git diff --check
```

Hardware claims additionally require the precisely marked ROCm tests on the named device. Never treat a skip as passing hardware evidence.
