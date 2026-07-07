# Repeated Helper Optimization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace repeated helper implementations with standard-library APIs, Pydantic APIs, and shared internal utilities without changing public behavior.

**Architecture:** Extend `sol_execbench.core.data.json_utils` for deterministic model serialization and JSON/JSONL model loading. Add `sol_execbench.core.text_utils` for Markdown cells, tails, subprocess output text, and ordered uniqueness. Keep checksum ownership in `sol_execbench.core.dataset.checksums`.

**Tech Stack:** Python 3.12, Pydantic v2, pytest, Ruff.

---

### Task 1: Shared Helper Tests

**Files:**
- Modify: `src/sol_execbench/core/data/json_utils.py`
- Create: `src/sol_execbench/core/text_utils.py`
- Test: `tests/sol_execbench/core/data/test_json_utils.py`
- Test: `tests/sol_execbench/core/test_text_utils.py`

- [ ] Write failing tests covering deterministic model JSON, stable model checksum payloads, JSON object loading, JSONL model loading, Markdown cell escaping, text tailing, subprocess text normalization, and ordered uniqueness.
- [ ] Run focused tests and verify they fail because the helpers do not exist yet.
- [ ] Implement the minimal shared helpers.
- [ ] Run focused tests and verify they pass.

### Task 2: Low-Risk Call-Site Refactors

**Files:**
- Modify repeated report `to_json`/`with_checksum` methods in `src/sol_execbench/core/` and `src/sol_execbench/core/dataset/`.
- Modify repeated `_md_cell`, `_tail`, `subprocess_text`, `_unique`, and sha256 helpers in touched modules.

- [ ] Replace local repeated implementations with imports from shared helpers.
- [ ] Run representative report, profiler, and helper tests.
- [ ] Run Ruff on touched files.
