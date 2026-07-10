---
quick_task: decision-sidecar
status: complete
created: 2026-07-10
branch: feat/decision-ready-data-layer
---

# Decision Sidecar (`sol_execbench.decision.v1`)

## Goal

Implement the Decision sidecar that turns decision-ready data-layer facts
(`ArchIsaBudget` + `StaticResourceFootprint`) into confidence-weighted **Layer
R** optimization hints, without ever re-asserting benchmark authority. This
delivers the modeling surveyed in `docs/decision-modeling-research.md` and
fulfills the mount-point contract in `docs/decision_sidecar_contract.md`.

## Context (already done — do NOT redo)

- **Research**: `docs/decision-modeling-research.md` — AMD official taxonomy,
  RDNA/CDNA divergence, occupancy formulas, occupancy!=performance confidence
  model, third-party credibility assessments. Primary input; cite §-numbers below.
- **Data layer**: 4 commits on `feat/decision-ready-data-layer` —
  `ArchIsaBudget` (18 fields, divergence/dialect tiers), `StaticResourceFootprint`,
  roc-objdump wiring, gfx942/gfx1150 budgets corrected and extended.
- **Mount-point contract**: `docs/decision_sidecar_contract.md`.

## Hard Constraints

- **Diagnostic-only**: `diagnostic_only: Literal[True]` + six
  `*_authority: Literal[False]` (correctness/performance/timing/score/paper_parity/leaderboard),
  mirroring `evidence_models.py:157-170`.
- **Static path emits Layer R only** (resource signals); never a compute/memory
  verdict (those are Layer M, runtime-only — research §5).
- **unknown → null, never promote**; closed enums, no unknown-promotion.
- **No backward-compat burden** (no real users): direct `v1`, no legacy readers.
- Prompt-safe outputs (no `/tmp`, `Traceback`, fixed timestamps in fixtures).

## Open Items Resolved (research §11)

1. **`nW` gap** → option (a): use the `Occupancy` value roc-objdump reports
   (`footprint.occupancy_estimate_waves_per_cu`) directly; skip the closed-form
   formula's `nW` term. Precise closed-form occupancy is a deferred limitation.
2. **`vgpr_limit` semantics** → document as architected *addressing* limit;
   occupancy's `F` uses `register_file_per_cu_bytes` (research §6, §9).
3. **`sol_execbench.decision.v1` schema** → defined in Wave 1.
4. **Cross-sidecar precedence** → Wave 4.

---

## Wave 1 — Schema + Identity

- [ ] **1.1** Create `src/sol_execbench/core/bench/decision/decision_models.py`:
  `DecisionBottleneckClass` enum (Layer R closed set: `REGISTER_PRESSURE_HIGH`,
  `LDS_PRESSURE_HIGH`, `SPILL_DETECTED`, `WORKGROUP_SIZE_LIMITED`,
  `BARRIER_LIMITED`, `WAVEFRONT_MISALIGNED`, `CACHE_LINE_MISALIGNED`);
  `DecisionConfidence` enum (`inferred_high`, `inferred_medium`, `inferred_low`).
  Acceptance: module imports clean; enums reject unknown values under `strict=True`.
- [ ] **1.2** `DecisionHint` model: `bottleneck_class`, `recommendation` (str),
  `confidence` (`DecisionConfidence`), `limitations[]`, `evidence_refs[]`,
  `identity` (`source_sha256` + `artifact_id` + `generated_at`), inline governance
  flags (diagnostic_only + 6 `*_authority=Literal[False]`). Mirror
  `StaticResourceFootprint` governance pattern.
  Acceptance: round-trips via `model_dump(mode="json")` / `model_validate`.
- [ ] **1.3** `DecisionSidecar` model: `schema_version:
  Literal["sol_execbench.decision.v1"]`, `status`, `hints: list[DecisionHint]`,
  `aggregate` summary, top-level governance flags. `DECISION_SCHEMA_VERSION`
  constant. `ConfigDict(extra="forbid", frozen=True, strict=True)`.
  Acceptance: fixture loads; unknown field rejected.

## Wave 2 — Derivation

- [ ] **2.1** Create `src/sol_execbench/core/bench/decision/derivation.py`:
  `derive_decision_hints(footprints, budget) -> list[DecisionHint]`. Select formula
  tier by `budget.register_allocation_model`:
  - `static` → derive VGPR/LDS pressure, spill, alignment signals;
  - `dynamic` (RDNA4) → emit at most an `inferred_low` note + `limitations[]`
    ("dynamic register allocation; static occupancy unreliable").
  Acceptance: gfx942 footprint (high VGPR) → `REGISTER_PRESSURE_HIGH`; scratch>0 →
  `SPILL_DETECTED` at `inferred_high`; dynamic budget → no closed-form hint.
- [ ] **2.2** Implement Layer R rules (research §6, §8.1):
  - `SPILL_DETECTED`: `scratch_bytes > 0` → `inferred_high` (deterministic).
  - `REGISTER_PRESSURE_HIGH`: `vgpr_used` near/over `vgpr_limit` or crossing a
    granularity boundary that drops occupancy → `inferred_medium`/`high`;
    recommendation from HIP guidelines (research §8.2).
  - `LDS_PRESSURE_HIGH`: LDS-limited occupancy.
  - `WAVEFRONT_MISALIGNED`: block not a multiple of `wavefront_size`.
  - `CACHE_LINE_MISALIGNED`: coalescing risk vs `cache_line_bytes`.
  - Uncovered inputs → field `null`, never promoted.
  Acceptance: each rule has a CPU-safe unit test with a crafted footprint.

## Wave 3 — Wiring + Contract

- [ ] **3.1** Wire sidecar writer (`sidecar_writer.py`) + CLI
  `--decision {none,auto}` (default `none`). Reuse the existing optional-sidecar
  pattern; never writes canonical trace JSONL.
  Acceptance: `--decision auto` emits `<trace>.decision.json` beside trace;
  `none` emits nothing.
- [ ] **3.2** `src/sol_execbench/core/data/contract.py`: add
  `"decision.sidecar": "profile:diagnostic"` capability + boundary
  `{"owner":"sol","scope":"decision","authority":"diagnostic"}`. Update
  `docs/EVALUATOR-CONTRACT.md` capability table + boundaries (enforced by
  `test_current_contract_doc_matches_builder_capabilities`).
  Acceptance: `sol-execbench contract --json` lists `decision.sidecar`;
  contract test green.

## Wave 4 — Precedence, Tests, Docs

- [ ] **4.1** Cross-sidecar precedence (research §8.4): runtime
  (`profile_summary.v2`) > static (`decision.v1`) > none. On conflict, demote the
  static hint into `limitations[]`; never override runtime. Implement as a merge
  helper with a unit test.
- [ ] **4.2** Fixtures `tests/sol_execbench/fixtures/decision/`: `valid`,
  `partial`, `unavailable`, `malformed`, `missing.case.json` — deterministic,
  prompt-safe. Parametrized loader test mirroring `test_agent_feedback_fixtures.py`.
- [ ] **4.3** CPU-safe tests: `test_decision_models.py` (round-trip, governance,
  frozen/strict), `test_decision_derivation.py` (each Layer R rule, dynamic-model
  fallback, never-promote), contract + guardrail tests.
  Acceptance: `uv run pytest tests/sol_execbench/core/bench/test_decision* -v` green;
  full CPU-safe suite no new failures vs baseline (4 pre-existing doc failures
  unchanged).
- [ ] **4.4** Docs: update `docs/decision_sidecar_contract.md` (schema now exists,
  remove "not implemented"), flip `docs/decision-modeling-research.md` open items
  to resolved, add `docs/EVALUATOR-CONTRACT.md` rows.
  Acceptance: no doc-test regression.

## Verification

- `uv run pytest tests/sol_execbench/ --tb=short -q -m "not requires_rocm and not cpp"`
  — 1863 passed baseline preserved (4 pre-existing doc failures unchanged).
- `uv run --with ruff ruff check .` clean; changed files `ruff format --check` clean.
- `uv run sol-execbench contract --json | jaq '.capabilities | keys'` includes
  `decision.sidecar`.
- Pre-commit hooks (Ruff Check / Format / DCO) pass.

## Out of Scope (future milestone, not this quick task)

- Layer C (instruction-mix from disassembly statistics) and Layer M (runtime
  compute/memory/latency bound) — require disassembly stats / runtime counters.
- RDNA4 (`gfx1200`) budget entry — only the `dynamic` fallback path is exercised.
- Real-hardware smoke (gated behind `requires_rocm`).
