# Phase 53: Dataset Contract And Acquisition Metadata - Research

**Researched:** 2026-05-23
**Domain:** Dataset acquisition contract, local layout validation, sidecar manifest metadata
**Confidence:** HIGH for repo-local behavior and phase scope; MEDIUM for Hugging Face downloader semantics because official docs were checked but real network acquisition was not exercised.

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- Model the canonical dataset root as `data/SOL-ExecBench/benchmark/{L1,L2,Quant,FlashInfer-Bench}` to match the current docs and dataset-runner examples.
- Default downloader output should be `data/SOL-ExecBench/benchmark`, with `--output-root` support and downloaded contents kept out of committed source files.
- Acquisition/local-layout manifests should record Hugging Face repository ID, subset/category, revision or local provenance, root path, discovered counts, and checksum metadata.
- Missing category directories should produce structured `missing_category` diagnostics and fail unless the user explicitly selected a partial category set.
- Keep downloader changes narrow: add category selection, output-root, manifest output, and idempotent checks without rewriting the download logic.
- Existing files and directories should be reused and layout-verified by default; do not delete or overwrite unknown files.
- Dependency, Hugging Face access, or network failures should produce clear errors while preserving the local-layout manifest path when applicable.
- Category selection should use repeatable `--category L1 --category Quant` flags, with all four public categories selected by default.
- Add reusable dataset contract/manifest code under `src/sol_execbench/core/dataset/`; scripts should stay thin CLIs over that library code.
- Stabilize manifest output through typed internal models and deterministic JSON. Keep fields conservative and sidecar-only.
- Do not modify public benchmark schemas: `definition.json`, `workload.jsonl`, `solution.json`, and trace JSONL remain unchanged.
- Write manifests to an explicit artifact/output path such as `dataset_manifest.json`; do not write into dataset roots unless explicitly requested.
- Test layout/category validation, manifest/checksum generation, idempotency, and downloader CLI behavior with fixtures and mocks; do not require real network access.
- Documentation should explicitly state that acquisition/layout completion is not readiness, execution, or paper-level validation.
- Tests may pass without a real local dataset by using temporary fixture directories. Real dataset checks are user-run commands.
- Phase 53 must not produce ready subsets or readiness classifications; those belong to Phase 54.

### the agent's Discretion
The agent may choose exact helper names, manifest field ordering, checksum algorithm details, and CLI option grouping as long as the public behavior above is preserved and follows existing repository style.

### Deferred Ideas (OUT OF SCOPE)
- Ready-subset generation and ROCm readiness classification are deferred to Phase 54.
- Bounded execution closure is deferred to Phase 55.
- Parity gap reporting is deferred to Phase 56.
- Full claim-guardrail release wording and milestone closure are deferred to Phase 57.
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| DATA-01 | Verify public dataset root contains expected `L1`, `L2`, `Quant`, and `FlashInfer-Bench` directories without GPU evaluation. | Existing runner already treats these four names as categories and discovers problem directories by `definition.json` plus `workload.jsonl`; add library-level layout validation around that same category set. [VERIFIED: scripts/run_dataset.py] |
| DATA-02 | Generate acquisition or local-layout manifest with source, category set, root path, revision/local provenance, discovered counts, and manifest checksum. | Existing repo uses Pydantic v2 models and deterministic JSON/JSONL helpers; add typed manifest models under `core/dataset` and checksum the canonical manifest payload excluding its own checksum field. [VERIFIED: src/sol_execbench/core/data/json_utils.py] |
| DATA-03 | Run downloader idempotently with explicit category selection and output-root behavior while keeping data out of committed source files. | Current downloader already loops over four subsets and writes `definition.json`, `reference.py`, and `workload.jsonl`; plan should add `--category`, `--output-root`, reuse checks, and no destructive overwrite behavior. [VERIFIED: scripts/download_solexecbench.py] |
| DATA-04 | Distinguish acquisition/layout completion from readiness, execution success, and paper-level validation in machine-readable metadata. | Prior sidecar patterns already carry explicit source/scope boundary booleans; dataset manifest should include conservative claim-boundary booleans and never add readiness statuses. [VERIFIED: tests/sol_execbench/test_public_contract_guardrails.py] |
</phase_requirements>

## Summary

Phase 53 should create a sidecar-only dataset contract layer, not a new execution path. The current downloader writes the public SOL-ExecBench rows into benchmark problem directories, while `scripts/run_dataset.py` independently discovers category/problem folders for execution; the missing layer is a reusable library that verifies the public root shape and emits deterministic acquisition/local-layout metadata before any GPU work. [VERIFIED: scripts/download_solexecbench.py] [VERIFIED: scripts/run_dataset.py]

The most important existing mismatch is path shape: the phase context and roadmap specify `data/SOL-ExecBench/benchmark`, but `scripts/download_solexecbench.py` currently defaults to `data/benchmark`, and `docs/GETTING-STARTED.md` still uses `data/benchmark` while runner examples use `data/SOL-ExecBench/benchmark`. [VERIFIED: .planning/phases/53-dataset-contract-and-acquisition-metadata/53-CONTEXT.md] [VERIFIED: scripts/download_solexecbench.py] [VERIFIED: docs/GETTING-STARTED.md] [VERIFIED: scripts/run_dataset.py]

**Primary recommendation:** implement `src/sol_execbench/core/dataset/` with category constants, layout diagnostics, manifest models, checksum helpers, and a small acquisition service, then adapt `scripts/download_solexecbench.py` to call it with repeatable `--category`, `--output-root`, `--manifest`, `--revision`, and local-layout verification options. [VERIFIED: AGENTS.md] [VERIFIED: .planning/phases/53-dataset-contract-and-acquisition-metadata/53-CONTEXT.md]

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|--------------|----------------|-----------|
| Dataset category contract | Core library | Scripts | The category set is reused by downloader, layout verification, and future inventory; scripts should not own duplicated constants. [VERIFIED: scripts/run_dataset.py] |
| Local layout validation | Core library | CLI script | Validation is pure filesystem inspection and can be unit-tested without GPU or network access. [VERIFIED: .planning/REQUIREMENTS.md] |
| Acquisition manifest generation | Core library | CLI script | Manifest shape should be typed and stable for future Phase 54-56 consumers. [VERIFIED: .planning/ROADMAP.md] |
| Hugging Face row download/unpack | Script entry point | Core helper | Current behavior lives in `scripts/download_solexecbench.py`; keep changes narrow and avoid a new package CLI unless later needed. [VERIFIED: scripts/download_solexecbench.py] |
| Execution readiness or classification | Out of scope | Phase 54 | Requirements explicitly reserve readiness classification for Phase 54. [VERIFIED: .planning/REQUIREMENTS.md] |

## Project Constraints (from AGENTS.md)

- Python package code belongs under `src/sol_execbench/`; scripts remain under `scripts/`. [VERIFIED: AGENTS.md]
- Use Python 3.12+ and Ruff style; functions/modules use `snake_case`, classes and Pydantic models use `PascalCase`. [VERIFIED: AGENTS.md]
- Tests belong under `tests/sol_execbench/` for package behavior, with descriptive `test_*` names. [VERIFIED: AGENTS.md]
- Downloaded benchmark assets and generated data must not be committed. [VERIFIED: AGENTS.md]
- Preserve SOL ExecBench benchmark semantics and public schemas unless a ROCm-specific change is unavoidable. [VERIFIED: AGENTS.md]
- Do not commit proprietary kernels, credentials, Hugging Face tokens, or downloaded datasets. [VERIFIED: AGENTS.md]
- GPU evaluation may require Docker, ROCm hardware, ROCm drivers, `/dev/kfd`, and `/dev/dri`, but Phase 53 tests should not require those resources. [VERIFIED: AGENTS.md] [VERIFIED: .planning/phases/53-dataset-contract-and-acquisition-metadata/53-CONTEXT.md]
- GSD workflow enforcement requires planning artifacts and execution context to stay in sync before repo edits. [VERIFIED: AGENTS.md]

## Existing Patterns

- `scripts/run_dataset.py` defines `CATEGORIES = {"L1", "L2", "FlashInfer-Bench", "Quant"}` and discovers problem directories by checking for `definition.json` and `workload.jsonl`. [VERIFIED: scripts/run_dataset.py]
- `scripts/run_dataset.py` accepts `--category` today as one option with `nargs="+"`; Phase 53 should use repeatable `--category` flags for the downloader as locked in context, but future runner alignment can be deferred unless needed for DATA-03. [VERIFIED: scripts/run_dataset.py] [VERIFIED: .planning/phases/53-dataset-contract-and-acquisition-metadata/53-CONTEXT.md]
- `scripts/download_solexecbench.py` uses `datasets.load_dataset(REPO_ID, name=subset, split="train")`, builds a `Definition`-shaped JSON object from dataset rows, writes `reference.py`, and serializes row workloads to `workload.jsonl`. [VERIFIED: scripts/download_solexecbench.py]
- Current downloader overwrites `definition.json`, `reference.py`, and `workload.jsonl` on each run, so idempotency must be added explicitly by comparing existing content or preserving files unless forced. [VERIFIED: scripts/download_solexecbench.py]
- Pydantic v2 models are already the project-standard schema mechanism for public data contracts. [VERIFIED: src/sol_execbench/core/data/definition.py] [VERIFIED: src/sol_execbench/core/data/workload.py]
- JSON sidecars in the runner are written with `json.dumps(..., indent=2)`, and existing tests verify safe sidecar stems for untrusted benchmark identifiers. [VERIFIED: scripts/run_dataset.py] [VERIFIED: tests/sol_execbench/test_run_dataset_amd_score.py]
- Public contract guardrails already test that SOLAR/AMD sidecar boundaries do not mutate canonical public schemas. [VERIFIED: tests/sol_execbench/test_public_contract_guardrails.py]

## Proposed Module Boundaries

Recommended package structure:

```text
src/sol_execbench/core/dataset/
├── __init__.py              # Re-export public helper names for scripts.
├── categories.py            # Public category constants and selection validation.
├── layout.py                # Filesystem discovery, missing_category diagnostics, counts.
├── manifest.py              # Pydantic manifest models and deterministic serialization.
├── checksums.py             # Stable sha256 helpers for files, directories, and manifest payloads.
└── acquisition.py           # Download-row conversion and idempotent write decisions.
```

- Keep `core/data/definition.py` and `core/data/workload.py` unchanged because Phase 53 must not modify `definition.json` or `workload.jsonl` schemas. [VERIFIED: .planning/phases/53-dataset-contract-and-acquisition-metadata/53-CONTEXT.md]
- Put category constants in `core/dataset/categories.py` and have both downloader and future inventory code import from there; later plans can decide whether `scripts/run_dataset.py` imports the same constants or remains untouched. [VERIFIED: scripts/run_dataset.py]
- Keep `scripts/download_solexecbench.py` as the operational CLI and avoid adding a new console entry point in `pyproject.toml` for this phase. [VERIFIED: pyproject.toml] [VERIFIED: .planning/phases/53-dataset-contract-and-acquisition-metadata/53-CONTEXT.md]
- Use small pure functions for discovery and manifest building so tests can use temporary directories and monkeypatched dataset rows without network access. [VERIFIED: .planning/phases/53-dataset-contract-and-acquisition-metadata/53-CONTEXT.md]

## Standard Stack

### Core

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| Python stdlib `argparse`, `json`, `hashlib`, `pathlib` | Python 3.12.13 local | CLI parsing, deterministic JSON, checksums, paths | Existing scripts use stdlib CLIs and JSON; no new framework is needed. [VERIFIED: scripts/run_dataset.py] [VERIFIED: local `python3 --version`] |
| Pydantic v2 | Installed 2.13.4; project requires `>=2.12.5` | Typed internal manifest models | Existing benchmark contracts use Pydantic v2 models. [VERIFIED: pyproject.toml] [VERIFIED: pip index] |
| `datasets` | Installed 4.8.2; registry latest observed 4.8.5 | Existing SOL-ExecBench subset loading | Current downloader already uses `datasets.load_dataset`; official docs support repository loading plus `revision`. [VERIFIED: scripts/download_solexecbench.py] [CITED: https://huggingface.co/docs/datasets/loading] |
| `huggingface-hub` CLI `hf` | Local CLI 1.16.1; installed package reported 1.14.0 by pip | Existing FlashInfer trace download in shell script | Official docs support `hf download --repo-type dataset --revision ... --local-dir`; local-dir has metadata for efficient repeated pulls. [VERIFIED: scripts/download_data.sh] [CITED: https://huggingface.co/docs/huggingface_hub/en/guides/cli] |

### Supporting

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| Pytest | Project dev dependency `>=9.0.2` | Fixture-based tests for layout, manifest, downloader CLI mocks | Use for all Phase 53 verification; no GPU markers required. [VERIFIED: pyproject.toml] |
| Ruff | Via `uv run --with ruff` | Style check | Run on touched source/tests if implementation changes Python files. [VERIFIED: AGENTS.md] |

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Stdlib JSON plus Pydantic | New dataframe/database/reporting framework | Explicitly out of scope; adds dependency without solving DATA-01 to DATA-04. [VERIFIED: .planning/REQUIREMENTS.md] |
| Script-level ad hoc dicts | Typed manifest models | Ad hoc dicts increase drift risk for future Phase 54-56 consumers. [VERIFIED: .planning/ROADMAP.md] |
| Hashing every file byte on every run by default | Count-only manifest | Count-only metadata is cheaper but does not satisfy checksum metadata intent. [VERIFIED: .planning/REQUIREMENTS.md] |

**Installation:** no new package install is recommended for Phase 53. [VERIFIED: pyproject.toml]

## Package Legitimacy Audit

No new external package is recommended. Existing relevant packages were checked because the downloader already depends on them. [VERIFIED: pyproject.toml]

| Package | Registry | Version Evidence | slopcheck | Disposition |
|---------|----------|------------------|-----------|-------------|
| `datasets` | PyPI | Installed 4.8.2; registry latest observed 4.8.5 | OK | Existing dependency; approved for continued use. [VERIFIED: pip index] |
| `huggingface-hub` | PyPI | CLI reports 1.16.1; pip reports installed 1.14.0 and latest 1.16.1 | OK | Existing transitive/CLI dependency; approved for continued use with version mismatch noted. [VERIFIED: local `hf --version`] [VERIFIED: pip index] |
| `pydantic` | PyPI | Installed/latest 2.13.4; project requires `>=2.12.5` | OK | Existing dependency; approved for typed manifest models. [VERIFIED: pyproject.toml] [VERIFIED: pip index] |

**Packages removed due to slopcheck [SLOP] verdict:** none. [VERIFIED: slopcheck]
**Packages flagged as suspicious [SUS]:** none. [VERIFIED: slopcheck]

## Manifest Schema Considerations

Recommended manifest top-level fields:

```json
{
  "schema_version": "sol_execbench.dataset_manifest.v1",
  "created_at": "2026-05-23T00:00:00Z",
  "source": {
    "kind": "huggingface_dataset",
    "repo_id": "nvidia/SOL-ExecBench",
    "revision": "main",
    "local_provenance": null
  },
  "root": {
    "path": "data/SOL-ExecBench/benchmark",
    "path_kind": "relative_to_repo"
  },
  "selected_categories": ["FlashInfer-Bench", "L1", "L2", "Quant"],
  "categories": [
    {
      "name": "L1",
      "status": "present",
      "problem_count": 0,
      "workload_count": 0,
      "required_files": ["definition.json", "workload.jsonl"],
      "checksum": {"algorithm": "sha256", "value": "..."}
    }
  ],
  "diagnostics": [],
  "claim_boundary": {
    "acquisition_or_layout_complete": true,
    "rocm_readiness": false,
    "execution_success": false,
    "paper_level_validation": false,
    "hosted_leaderboard_parity": false,
    "upstream_solar_equivalence": false
  },
  "manifest_checksum": {"algorithm": "sha256", "value": "..."}
}
```

- Use stable category ordering for deterministic manifests; `FlashInfer-Bench`, `L1`, `L2`, `Quant` alphabetical ordering is easiest, while preserving the canonical category set as constants. [VERIFIED: scripts/run_dataset.py]
- Include both selected categories and per-category records so partial explicit selections remain machine-readable. [VERIFIED: .planning/phases/53-dataset-contract-and-acquisition-metadata/53-CONTEXT.md]
- Represent missing categories as structured diagnostics such as `{"code": "missing_category", "category": "L1", "path": "..."}`. [VERIFIED: .planning/phases/53-dataset-contract-and-acquisition-metadata/53-CONTEXT.md]
- Keep checksum fields conservative: manifest checksum should be computed over deterministic JSON excluding `manifest_checksum`; category checksum can be a deterministic hash of relative file paths, file sizes, and file sha256 values for canonical files under that category. [ASSUMED]
- Count discovered problems by directories containing `definition.json` and `workload.jsonl`, matching current runner discovery. [VERIFIED: scripts/run_dataset.py]
- Count workloads by non-empty JSONL lines only after successful file read; schema parsing should be deferred to Phase 54 inventory. [VERIFIED: .planning/REQUIREMENTS.md]
- Do not parse `Definition` or `Workload` for readiness in this phase; optional schema validation may only detect malformed layout if the planner explicitly scopes it, but Phase 54 owns full inventory parsing. [VERIFIED: .planning/REQUIREMENTS.md]
- Write the manifest to an explicit path supplied by `--manifest`, defaulting outside the dataset root unless the user passes a path inside it. [VERIFIED: .planning/phases/53-dataset-contract-and-acquisition-metadata/53-CONTEXT.md]

## Downloader Idempotency Strategy

- Add CLI options to `scripts/download_solexecbench.py`: repeatable `--category`, `--output-root`, `--manifest`, `--revision`, `--force`, and `--verify-only`. [VERIFIED: .planning/phases/53-dataset-contract-and-acquisition-metadata/53-CONTEXT.md]
- Default `--output-root` to `data/SOL-ExecBench/benchmark`, replacing the current hardcoded `data/benchmark` default. [VERIFIED: scripts/download_solexecbench.py] [VERIFIED: .planning/phases/53-dataset-contract-and-acquisition-metadata/53-CONTEXT.md]
- Default categories to all four public categories; when one or more `--category` flags are provided, validate only that selected set and do not fail for unselected missing categories. [VERIFIED: .planning/phases/53-dataset-contract-and-acquisition-metadata/53-CONTEXT.md]
- Before writing a problem file, compute the intended bytes and compare with existing content; skip identical files, report changed files, and require `--force` before overwriting divergent canonical files. [ASSUMED]
- Never delete extra files or unknown directories under the output root; record them only if planner chooses an `extra_entries` diagnostic, because the context says existing unknown files must not be deleted or overwritten. [VERIFIED: .planning/phases/53-dataset-contract-and-acquisition-metadata/53-CONTEXT.md]
- On a network or Hugging Face failure, emit a local-layout manifest if the root can still be inspected; mark acquisition status as failed and include a structured diagnostic without claiming layout success unless validation passes. [VERIFIED: .planning/phases/53-dataset-contract-and-acquisition-metadata/53-CONTEXT.md]
- Keep `scripts/download_data.sh` as an orchestrator; update it to call the new output root and preserve the separate FlashInfer trace `hf download` behavior if included in the plan. [VERIFIED: scripts/download_data.sh]
- Official Hugging Face docs support `load_dataset(..., revision=...)`, so the downloader can pin a branch/tag/commit when a revision is supplied. [CITED: https://huggingface.co/docs/datasets/loading]
- Official Hugging Face Hub CLI docs state `hf download --local-dir` creates local metadata that avoids re-downloading files already up to date, so the existing FlashInfer trace path can remain CLI-based and idempotent at the Hub layer. [CITED: https://huggingface.co/docs/huggingface_hub/en/guides/cli]

## Architecture Patterns

### System Architecture Diagram

```text
User command
  |
  v
scripts/download_solexecbench.py
  |
  +--> parse category/output-root/revision/manifest flags
  |
  +--> core.dataset.categories validates selected category set
  |
  +--> if download mode:
  |       Hugging Face datasets.load_dataset(repo_id, name=category, split="train", revision=...)
  |          |
  |          v
  |       core.dataset.acquisition maps rows to definition/reference/workload files
  |          |
  |          v
  |       idempotent file writes: skip identical, fail on divergent unless --force
  |
  +--> core.dataset.layout inspects root/category/problem layout
  |
  +--> core.dataset.manifest builds deterministic sidecar
  |
  v
dataset_manifest.json + structured exit status
```

### Pattern 1: Sidecar-Only Derived Metadata

**What:** Put acquisition and layout metadata in a separate manifest file, never in canonical dataset files. [VERIFIED: .planning/phases/53-dataset-contract-and-acquisition-metadata/53-CONTEXT.md]

**When to use:** Always for Phase 53 outputs; canonical `definition.json`, `workload.jsonl`, `solution.json`, and trace JSONL are protected surfaces. [VERIFIED: .planning/REQUIREMENTS.md]

### Pattern 2: Pure Filesystem Validation

**What:** Validate directory presence and required file presence without invoking GPU evaluation or the primary benchmark CLI. [VERIFIED: .planning/REQUIREMENTS.md]

**When to use:** DATA-01 and local manifest generation. [VERIFIED: .planning/REQUIREMENTS.md]

### Pattern 3: Deterministic JSON

**What:** Sort categories, problem paths, diagnostics, and checksum inputs before serialization. [ASSUMED]

**When to use:** Manifest files consumed by later phases and tests that compare repeated output. [VERIFIED: .planning/ROADMAP.md]

### Anti-Patterns to Avoid

- **Embedding readiness in acquisition metadata:** Phase 54 owns readiness classification, so Phase 53 should only record claim-boundary booleans such as `rocm_readiness: false`. [VERIFIED: .planning/REQUIREMENTS.md]
- **Mutating canonical public schemas:** The phase context explicitly forbids changes to `definition.json`, `workload.jsonl`, `solution.json`, and trace JSONL. [VERIFIED: .planning/phases/53-dataset-contract-and-acquisition-metadata/53-CONTEXT.md]
- **Deleting or cleaning dataset roots during download:** Existing unknown files must be preserved by default. [VERIFIED: .planning/phases/53-dataset-contract-and-acquisition-metadata/53-CONTEXT.md]
- **Network-required tests:** Phase context requires fixtures and mocks rather than real dataset downloads. [VERIFIED: .planning/phases/53-dataset-contract-and-acquisition-metadata/53-CONTEXT.md]

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Hugging Face dataset access | Custom HTTP or git/LFS downloader for `nvidia/SOL-ExecBench` rows | Existing `datasets.load_dataset` call with `name`, `split`, and optional `revision` | Official docs support Hub loading and revisions; current script already uses this path. [VERIFIED: scripts/download_solexecbench.py] [CITED: https://huggingface.co/docs/datasets/loading] |
| Manifest schema validation | Loose nested dicts only | Pydantic models | Existing data contracts use Pydantic v2. [VERIFIED: src/sol_execbench/core/data/definition.py] |
| Checksum primitive | Custom hash algorithm | `hashlib.sha256` | Existing code already uses `hashlib.sha256` for deterministic sidecar stems. [VERIFIED: scripts/run_dataset.py] |
| CLI testing | Real network or GPU run | Monkeypatched loader and temporary filesystem fixtures | Existing tests already import script modules and monkeypatch runner helpers. [VERIFIED: tests/sol_execbench/test_run_dataset_amd_score.py] |

**Key insight:** The hard part is not downloading rows; it is producing a stable, non-overclaiming contract artifact that later inventory/readiness/report phases can trust without reinterpreting execution results. [VERIFIED: .planning/ROADMAP.md]

## Common Pitfalls

### Path Contract Drift

**What goes wrong:** Docs, downloader, and runner point to different dataset roots. [VERIFIED: docs/GETTING-STARTED.md] [VERIFIED: scripts/download_solexecbench.py] [VERIFIED: scripts/run_dataset.py]
**Why it happens:** `download_solexecbench.py` still defaults to `data/benchmark`, while active context standardizes `data/SOL-ExecBench/benchmark`. [VERIFIED: scripts/download_solexecbench.py]
**How to avoid:** Make one canonical default in downloader constants and docs; keep legacy paths only as user-supplied `--output-root`. [VERIFIED: .planning/phases/53-dataset-contract-and-acquisition-metadata/53-CONTEXT.md]
**Warning signs:** Tests assert `data/benchmark` after the phase, or docs show old path in dataset setup. [VERIFIED: docs/GETTING-STARTED.md]

### Overwriting Divergent Local Files

**What goes wrong:** Re-running the downloader silently replaces local edited or partially curated problem files. [VERIFIED: scripts/download_solexecbench.py]
**Why it happens:** Current script writes files unconditionally. [VERIFIED: scripts/download_solexecbench.py]
**How to avoid:** Compare intended bytes with existing bytes; skip identical content and fail on divergent content unless `--force` is passed. [ASSUMED]
**Warning signs:** No test covers a second downloader run over existing files. [VERIFIED: tests/sol_execbench/test_run_dataset_amd_score.py]

### Manifest Self-Checksum Instability

**What goes wrong:** Manifest checksum changes every time because it hashes itself or includes volatile ordering. [ASSUMED]
**Why it happens:** The checksum field is included in the payload being checksummed. [ASSUMED]
**How to avoid:** Compute checksum over a canonical payload with `manifest_checksum` omitted or set to null, with sorted keys and stable lists. [ASSUMED]
**Warning signs:** Two runs over the same fixture root produce different manifest JSON. [ASSUMED]

### Claim Boundary Leakage

**What goes wrong:** `acquisition_complete` is interpreted as ROCm-ready, passed, or paper-validated. [VERIFIED: .planning/REQUIREMENTS.md]
**Why it happens:** Metadata lacks explicit false booleans for readiness/execution/paper validation. [VERIFIED: .planning/REQUIREMENTS.md]
**How to avoid:** Include `claim_boundary` fields and docs wording stating acquisition/layout completion is not execution or validation. [VERIFIED: .planning/phases/53-dataset-contract-and-acquisition-metadata/53-CONTEXT.md]
**Warning signs:** Manifest fields named `ready`, `supported`, `validated`, or `parity` appear in Phase 53. [VERIFIED: .planning/REQUIREMENTS.md]

### Schema Validation Scope Creep

**What goes wrong:** Phase 53 starts parsing every `Definition` and `Workload` and classifying blockers. [VERIFIED: .planning/REQUIREMENTS.md]
**Why it happens:** Layout discovery and inventory parsing are adjacent concerns. [VERIFIED: .planning/ROADMAP.md]
**How to avoid:** Keep Phase 53 counts shallow and leave parse failures, dtype hints, safetensors, custom inputs, and readiness reasons to Phase 54. [VERIFIED: .planning/REQUIREMENTS.md]
**Warning signs:** DATA-01 to DATA-04 tests assert readiness statuses or input dtype classifications. [VERIFIED: .planning/REQUIREMENTS.md]

## Code Examples

### Deterministic Manifest Checksum Pattern

```python
def manifest_checksum(payload: dict[str, object]) -> str:
    payload_for_hash = dict(payload)
    payload_for_hash["manifest_checksum"] = None
    encoded = json.dumps(
        payload_for_hash,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
```

Source: proposed pattern based on existing `hashlib.sha256` usage for safe sidecar stems and stdlib JSON usage in runner outputs. [VERIFIED: scripts/run_dataset.py] [ASSUMED]

### Idempotent File Write Pattern

```python
def write_if_changed(path: Path, content: str, *, force: bool = False) -> str:
    new_bytes = content.encode("utf-8")
    if path.exists():
        old_bytes = path.read_bytes()
        if old_bytes == new_bytes:
            return "unchanged"
        if not force:
            raise FileExistsError(f"{path} exists with different content")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(new_bytes)
    return "written"
```

Source: proposed pattern to satisfy idempotency and preserve unknown files; current downloader writes unconditionally. [VERIFIED: scripts/download_solexecbench.py] [ASSUMED]

### Layout Diagnostic Shape

```python
{
    "code": "missing_category",
    "severity": "error",
    "category": "L1",
    "path": "data/SOL-ExecBench/benchmark/L1",
}
```

Source: required diagnostic code from Phase 53 context; exact fields are planner discretion. [VERIFIED: .planning/phases/53-dataset-contract-and-acquisition-metadata/53-CONTEXT.md] [ASSUMED]

## Validation And Test Strategy

### Test Framework

| Property | Value |
|----------|-------|
| Framework | Pytest, project dev dependency `>=9.0.2`. [VERIFIED: pyproject.toml] |
| Config file | `pyproject.toml` contains pytest addopts and markers. [VERIFIED: pyproject.toml] |
| Quick run command | `uv run pytest tests/sol_execbench/test_dataset_contract.py -x` [ASSUMED] |
| Full relevant command | `uv run pytest tests/sol_execbench/test_dataset_contract.py tests/sol_execbench/test_run_dataset_amd_score.py` [ASSUMED] |

### Phase Requirements → Test Map

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|--------------|
| DATA-01 | All four categories present succeeds; missing unselected category in partial selection does not fail; missing selected/default category emits `missing_category` and fails. | Unit | `uv run pytest tests/sol_execbench/test_dataset_contract.py::test_layout_reports_missing_default_category -x` | No, Wave 0. [ASSUMED] |
| DATA-02 | Manifest records source, category set, root path, revision/local provenance, counts, category checksums, and stable manifest checksum. | Unit | `uv run pytest tests/sol_execbench/test_dataset_contract.py::test_manifest_is_deterministic -x` | No, Wave 0. [ASSUMED] |
| DATA-03 | Downloader CLI honors repeated categories, output root, identical-file skip, divergent-file failure, and `--force` overwrite. | Unit with monkeypatch | `uv run pytest tests/sol_execbench/test_download_solexecbench.py -x` | No, Wave 0. [ASSUMED] |
| DATA-04 | Manifest and docs use claim-boundary wording and false readiness/execution/paper validation booleans. | Unit/docs text | `uv run pytest tests/sol_execbench/test_dataset_contract.py::test_manifest_claim_boundary_does_not_claim_readiness -x` | No, Wave 0. [ASSUMED] |

### Sampling Rate

- **Per task commit:** `uv run pytest tests/sol_execbench/test_dataset_contract.py tests/sol_execbench/test_download_solexecbench.py -x` after adding the new tests. [ASSUMED]
- **Per wave merge:** `uv run pytest tests/sol_execbench/test_dataset_contract.py tests/sol_execbench/test_download_solexecbench.py tests/sol_execbench/test_run_dataset_amd_score.py`. [ASSUMED]
- **Phase gate:** `uv run pytest tests/` if time permits; otherwise targeted tests plus Ruff on touched files should be recorded as residual risk. [VERIFIED: AGENTS.md]

### Wave 0 Gaps

- [ ] `tests/sol_execbench/test_dataset_contract.py` for category validation, manifest stability, checksums, and claim boundary. [ASSUMED]
- [ ] `tests/sol_execbench/test_download_solexecbench.py` for CLI parsing and monkeypatched `load_dataset` rows. [ASSUMED]
- [ ] Fixtures that create `tmp_path / "SOL-ExecBench" / "benchmark" / category / problem` with minimal `definition.json` and `workload.jsonl`. [ASSUMED]
- [ ] No real network fixture should be added. [VERIFIED: .planning/phases/53-dataset-contract-and-acquisition-metadata/53-CONTEXT.md]

## Risks

| Risk | Impact | Mitigation |
|------|--------|------------|
| Path contract mismatch between old docs/downloader and new roadmap | Maintainers download to the wrong root and layout validation fails. | Update downloader default and dataset setup docs together. [VERIFIED: docs/GETTING-STARTED.md] |
| Divergent local files are silently overwritten | User-curated local data can be lost. | Add skip/fail/force idempotency behavior and tests. [VERIFIED: scripts/download_solexecbench.py] [ASSUMED] |
| Manifest checksum is not reproducible | Later phases cannot trust manifest references. | Add repeated-run deterministic fixture tests. [ASSUMED] |
| Manifest fields overclaim readiness or validation | Public claims exceed evidence. | Include explicit claim-boundary false booleans and wording tests. [VERIFIED: .planning/REQUIREMENTS.md] |
| Hugging Face package/CLI version mismatch | Local `hf --version` and pip installed metadata differ, which can confuse docs or support notes. | Avoid pinning new versions in Phase 53; document commands in terms of existing project dependencies and official CLI behavior. [VERIFIED: local `hf --version`] [VERIFIED: pip index] |
| Real public dataset row shape changes | Downloader row mapping can fail. | Preserve clear errors and keep tests mocking expected row shape; manual real-download command remains user-run. [VERIFIED: scripts/download_solexecbench.py] |

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|-------------|-----------|---------|----------|
| Python | Source/tests | Yes | 3.12.13 | None needed. [VERIFIED: local `python3 --version`] |
| `uv` | Test and dev commands | Yes | 0.11.15 | Use system Python only for limited inspection, but project commands should use `uv`. [VERIFIED: local `uv --version`] |
| `datasets` | SOL-ExecBench downloader | Yes | 4.8.2 in project env | Mock in tests; real acquisition remains user-run. [VERIFIED: local import] |
| `hf` CLI | Existing FlashInfer trace shell download | Yes | 1.16.1 CLI | Phase 53 can avoid changing FlashInfer trace acquisition if not needed. [VERIFIED: local `hf --version`] |
| `slopcheck` | Package audit | Yes | 0.6.1 | Not needed at runtime. [VERIFIED: local `slopcheck --version`] |
| Context7 CLI `ctx7` | Optional docs lookup | No | — | Official web docs were used. [VERIFIED: local `which ctx7`] |
| Graphify | Optional graph context | Disabled/no graph | — | Repo grep and explicit requested files were used. [VERIFIED: gsd graphify status] |

**Missing dependencies with no fallback:** none for Phase 53 planning. [VERIFIED: local environment probes]

**Missing dependencies with fallback:** Context7 is unavailable; official Hugging Face docs were checked via web instead. [VERIFIED: local `which ctx7`] [CITED: https://huggingface.co/docs/datasets/loading]

## Security Domain

| ASVS Category | Applies | Standard Control |
|---------------|---------|------------------|
| V2 Authentication | No for tests; maybe for private Hugging Face repos | Do not store tokens in manifests; rely on existing Hugging Face auth mechanisms if users need private access. [VERIFIED: AGENTS.md] |
| V3 Session Management | No | No web sessions are introduced. [VERIFIED: .planning/REQUIREMENTS.md] |
| V4 Access Control | No | Local CLI only; no new service boundary. [VERIFIED: .planning/ROADMAP.md] |
| V5 Input Validation | Yes | Validate category names against constants, constrain manifest paths, and avoid unsafe path traversal in sidecar paths. [VERIFIED: scripts/run_dataset.py] |
| V6 Cryptography | Yes for checksums only | Use stdlib `hashlib.sha256`; no custom cryptography. [VERIFIED: scripts/run_dataset.py] |

Known threat patterns:

| Pattern | STRIDE | Standard Mitigation |
|---------|--------|---------------------|
| Path traversal through category/problem names | Tampering | Use fixed category choices and sanitize or resolve manifest output paths; existing sidecar tests show path-shaped names need guarding. [VERIFIED: tests/sol_execbench/test_run_dataset_amd_score.py] |
| Credential leakage in manifest | Information Disclosure | Do not record Hugging Face tokens, env vars, or auth headers; record only repo ID, revision, and local provenance. [VERIFIED: AGENTS.md] |
| Overwrite of local dataset files | Tampering | Idempotent compare-before-write and `--force` gate. [ASSUMED] |

## State of the Art

| Old Approach | Current Phase 53 Approach | When Changed | Impact |
|--------------|---------------------------|--------------|--------|
| `data/benchmark` downloader default | `data/SOL-ExecBench/benchmark` canonical root | Phase 53 context, 2026-05-23 | Planner must include a downloader default/doc update. [VERIFIED: .planning/phases/53-dataset-contract-and-acquisition-metadata/53-CONTEXT.md] |
| Downloader-only acquisition | Acquisition plus local-layout sidecar manifest | v1.11 requirements, 2026-05-23 | Later phases can reference manifest checksums. [VERIFIED: .planning/REQUIREMENTS.md] |
| Execution runner discovers problems directly | Dedicated dataset contract layer validates layout before execution | Phase 53 | Keeps DATA-01 separate from GPU execution. [VERIFIED: .planning/ROADMAP.md] |
| Implicit claim boundary in docs | Machine-readable claim-boundary booleans | Phase 53 | Prevents acquisition from being treated as readiness or validation. [VERIFIED: .planning/REQUIREMENTS.md] |

**Deprecated/outdated:**
- `data/benchmark` as the documented default dataset root should be treated as outdated for Phase 53 implementation, though user-supplied legacy roots can still be validated through `--output-root`. [VERIFIED: docs/GETTING-STARTED.md] [VERIFIED: .planning/phases/53-dataset-contract-and-acquisition-metadata/53-CONTEXT.md]

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | Category checksum can hash relative file paths, sizes, and file sha256 values for canonical files. | Manifest Schema Considerations | If maintainers expect full-directory hash including unknown files, manifests may not capture all local drift. |
| A2 | Downloader should fail on divergent existing canonical files unless `--force` is passed. | Downloader Idempotency Strategy | If users expect automatic overwrite, CLI behavior may feel stricter than desired. |
| A3 | New tests should be split between `test_dataset_contract.py` and `test_download_solexecbench.py`. | Validation And Test Strategy | Planner may choose different filenames without changing behavior. |
| A4 | Manifest checksum should exclude or null its own checksum field. | Common Pitfalls / Code Examples | If a different canonicalization is selected, tests and consumers must use that exact algorithm. |
| A5 | Stable alphabetical category ordering is acceptable. | Manifest Schema Considerations | If consumers require semantic ordering L1, L2, Quant, FlashInfer-Bench, manifests would need a different fixed order. |

## Open Questions

1. **Should `scripts/run_dataset.py` import the new category constants in Phase 53?**
   - What we know: the runner already has the right four-category set. [VERIFIED: scripts/run_dataset.py]
   - What's unclear: changing the runner may broaden Phase 53 beyond downloader/layout metadata. [ASSUMED]
   - Recommendation: leave runner behavior unchanged unless planner includes a tiny constant-sharing task with regression tests. [ASSUMED]

2. **Should category checksums include every file or only canonical benchmark files?**
   - What we know: unknown files must not be deleted or overwritten. [VERIFIED: .planning/phases/53-dataset-contract-and-acquisition-metadata/53-CONTEXT.md]
   - What's unclear: including unknown files detects more drift but makes manifests sensitive to local sidecars. [ASSUMED]
   - Recommendation: checksum canonical files by default and optionally count unknown entries as diagnostics without hashing them. [ASSUMED]

3. **Should `download_data.sh` gain arguments or stay a simple wrapper?**
   - What we know: it currently runs the SOL-ExecBench downloader and then `hf download` for FlashInfer trace. [VERIFIED: scripts/download_data.sh]
   - What's unclear: shell argument forwarding can grow messy if every downloader option is mirrored. [ASSUMED]
   - Recommendation: update default paths and mention direct Python downloader invocation for advanced category/manifest options. [ASSUMED]

## Sources

### Primary (HIGH confidence)
- `AGENTS.md` - project structure, conventions, tests, security, GSD workflow. [VERIFIED: AGENTS.md]
- `.planning/phases/53-dataset-contract-and-acquisition-metadata/53-CONTEXT.md` - locked Phase 53 decisions and deferred scope. [VERIFIED: phase context]
- `.planning/REQUIREMENTS.md` - DATA-01 to DATA-04 and later phase boundaries. [VERIFIED: requirements]
- `.planning/ROADMAP.md` - Phase 53-57 sequencing and success criteria. [VERIFIED: roadmap]
- `scripts/download_solexecbench.py` - current SOL-ExecBench download/unpack behavior. [VERIFIED: codebase grep/read]
- `scripts/download_data.sh` - current aggregate download script. [VERIFIED: codebase grep/read]
- `scripts/run_dataset.py` - category discovery, runner CLI, sidecar and checksum patterns. [VERIFIED: codebase grep/read]
- `src/sol_execbench/core/data/definition.py` and `workload.py` - current public Pydantic schema contracts. [VERIFIED: codebase grep/read]
- `tests/sol_execbench/test_run_dataset_amd_score.py` - script import, monkeypatch, sidecar safety test patterns. [VERIFIED: codebase grep/read]
- `pyproject.toml` - dependency and test configuration. [VERIFIED: codebase grep/read]

### Secondary (MEDIUM confidence)
- Hugging Face Datasets loading docs - `load_dataset` repository loading, `revision`, and `split`. [CITED: https://huggingface.co/docs/datasets/loading]
- Hugging Face Hub CLI docs - `hf download`, `--repo-type dataset`, `--revision`, `--local-dir`, and local metadata behavior. [CITED: https://huggingface.co/docs/huggingface_hub/en/guides/cli]
- Local environment probes for `uv`, Python, `hf`, `datasets`, `slopcheck`, and missing `ctx7`. [VERIFIED: local commands]

### Tertiary (LOW confidence)
- None used as authoritative inputs; all assumptions are listed in the Assumptions Log. [VERIFIED: this research]

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - existing dependencies and local versions were verified, and no new package is recommended. [VERIFIED: pyproject.toml] [VERIFIED: local commands]
- Architecture: HIGH - module boundary follows locked phase context and existing package/script split. [VERIFIED: AGENTS.md] [VERIFIED: phase context]
- Manifest schema: MEDIUM - required fields are locked, but exact checksum scope and field ordering remain discretionary. [VERIFIED: .planning/REQUIREMENTS.md] [ASSUMED]
- Downloader idempotency: MEDIUM - strategy is straightforward but exact overwrite semantics need planner acceptance. [ASSUMED]
- Validation strategy: HIGH - pytest fixture/mocking approach matches phase context and existing script tests. [VERIFIED: tests/sol_execbench/test_run_dataset_amd_score.py]

**Research date:** 2026-05-23
**Valid until:** 2026-06-22 for repo-local planning; recheck Hugging Face docs/package versions if implementation is delayed more than 30 days.

## Plan-Shaping Recommendation

Plan this as three narrow implementation waves: first introduce `core/dataset` pure library models/helpers with unit tests; second adapt `scripts/download_solexecbench.py` for category/output-root/revision/manifest/idempotency with monkeypatched downloader tests; third update docs and claim-boundary tests so acquisition/layout completion cannot be confused with readiness, execution, or paper validation. [VERIFIED: .planning/phases/53-dataset-contract-and-acquisition-metadata/53-CONTEXT.md] [VERIFIED: .planning/REQUIREMENTS.md]

Do not schedule GPU evaluation, real network downloads in CI, readiness classifications, ready subset generation, trace mutations, or public schema migrations in this phase. [VERIFIED: .planning/REQUIREMENTS.md]

## RESEARCH COMPLETE
