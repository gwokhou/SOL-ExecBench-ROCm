# Repository Guidelines

## Layout and Ownership

Production code is in `src/sol_execbench/` (Click CLI, benchmarks, platform
evidence, drivers) and `src/solar/` (graphs, IR, verification, formal analysis).
Mirror these under `tests/`; keep sample kernels under
`tests/sol_execbench/samples/`, Docker support in `docker/`, and maintenance in
`scripts/`.
Downloaded/generated data belongs under uncommitted `data/`.

## Commands and Verification

- Install with `uv sync --all-groups`; test with `uv run pytest tests/`.
  Never use `-n auto`: the configured eight-worker cap avoids ROCm OOM.
- Run Python checks with `uv run --with ruff ruff check .`,
  `uv run --with ruff ruff format .`, and `uv run ty check`.
- Evaluate with `uv run sol-execbench evaluate <problem_dir> --solution <path>`.
- Build the ROCm environment with `./scripts/run_docker.sh --build`.

## Architecture and Reuse

- Search with `rg` before adding helpers. Put shared helpers in the relevant
  `core/` concern (`integrity`, `process`, `data.json_utils`, `text_utils`,
  `platform.runtime`, `arguments`, or `timestamps`), never `core.utils` or
  retired evidence checksum/log modules.
- Define reusable wire identifiers once and import them directly; do not repeat
  literals or relay them through compatibility re-exports.
- Prefer `cache`, `cached_property`, or `partial` to manual memoization when
  lifetimes and immutable inputs fit. Include every behavior-changing input,
  including timeouts, in cache keys.
- Use `StrEnum`/`IntEnum` for closed parser, CLI, model, wire-code, or exhaustive
  vocabularies; normalize boundary input first. Share enums only when semantics
  and ownership match. Derive Click choices from enums instead of duplicating
  constants; one-member enums may constrain public/schema fields.
- Use constants for resources, environment variables, tools, paths, versions,
  limits, timeouts, thresholds, format fragments, and immutable collections;
  reuse alone does not make an implementation parameter an enum.
- Reserve `tools/` for external integrations; domain packages must not re-export
  generic helpers. Wrappers need a documented test seam and focused coverage.
  Migrate callers and remove obsolete imports. Orchestrators use typed stage
  contracts, parsers validate raw JSON, and writers emit canonical artifacts
  atomically.

## Naming and Resources

Target Python 3.12/Ruff and keep changes local. Import canonical definitions
directly; alias only for collisions, clear roles, or conventions such as
`numpy as np`. Update callers when symbols move; do not add private/self aliases
or incidental compatibility re-exports.

Capitalize canonical acronyms in `PascalCase` (`SOLBound`, `IRBackend`,
`AKACorpusManifest`, `GPUEvidence`, `ISAInstructionRequirement`,
`AMDSMIProcess`, `JSONDict`); use lowercase acronyms in `snake_case`, uppercase
in prose, and preserve official/third-party casing such as `ROCm`.

Load large production HIP/C++ sources through `importlib.resources`; only
focused test snippets may remain inline.

## Tests and Hardware

- Put coverage near implementation: unit-test schemas/drivers and integration-
  test subprocess/GPU behavior. Register markers only in `pyproject.toml`'s
  `[tool.pytest.ini_options]`; reuse `requires_rocm`, `cpp`, `requires_rdna4`,
  and `requires_cdna3`.
- Parameterize independent scenarios with stable semantic IDs; avoid generated
  or duplicate IDs. Use fixtures for temporary files, state, logging, and capture.
- Hardware tests declare every prerequisite and skip only for the precise
  missing capability; never hide regressions behind broad skips or `xfail`.
- Pytest upgrades atomically update the dev dependency, `minversion`, `uv.lock`,
  configuration contract tests, and newly enabled strictness fixes.

## Schemas and Public Contracts

- Register only independently serialized contracts, not public classes,
  in-memory DTOs, or nested values owned by an envelope.
- Each artifact family belongs to its domain schema registry. Import that
  registry directly; `core/integrity/artifact_registry.py` is audit-only and
  must not become a production dependency or compatibility re-export.
- Variants that evolve together share one family and a validated discriminator
  such as `artifact_kind`, `role`, or `stage`.
- Readers require the exact current version before business fields. Breaking
  changes update producers, readers, artifacts, tests, docs, and CLI contracts
  atomically, then remove superseded models, migrations, aliases, and fixtures.
- Caches and checksum preimages use local format versions; protocol selectors
  use their domain protocol registry. None are artifact schema families.
- Keep raw versions in owning registries and registered artifacts. Enforce the
  boundary with `scripts/check_schema_versions.py`.

## Quality and Process Safety

- Run relevant Ruff, `ty`, coupling, readability, and Pytest checks for Python
  changes. Keep limits canonical; never raise a baseline to admit debt.
- Scope lint exclusions to root `data/` and `examples/`, never `core/data` or
  `tests/examples` by name.
- Do not add production functions over 80 lines or 10 parameters, or test files
  over 1000 lines. Split touched violations unless correctness is urgent.
- Subprocesses require timeouts, process-group cleanup, and bounded, redacted
  output.

## Security and GPU Execution

Never commit tokens, proprietary kernels, datasets, caches, or benchmark output.
Document architecture assumptions in tests or PR notes.

On sandbox, cache, filesystem, credential, or network failure, request scoped
host escalation and retry the same bounded operation before diagnosing it or
reporting benchmark/calibration failure. A sandboxed `gh auth status`, `gh api`,
or other `gh` failure does not prove token invalidity; report that only after an
escalated retry returns an explicit authentication error. Never run
`gh auth login` or `gh auth logout`, replace credentials, or otherwise mutate
GitHub authentication state without explicit user authorization.

Sandboxes may hide `/dev/kfd` or `/dev/dri/renderD*`; retry GPU enumeration,
compilation, execution, profiling, tracing, and calibration with scoped host
access before declaring hardware/tooling or calibration unsupported. Write
evidence only to approved locations; never use service-manager or sandbox-escape
wrappers instead of approval.

## Commits and Pull Requests

Use concise imperative summaries and DCO-sign commits, for example
`git commit -s -m "Fix trace parsing"`. Keep PRs focused and report tests plus
any ROCm hardware checks.
