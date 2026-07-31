# Repository Guidelines

## Layout and Ownership

Production code is in `src/sol_execbench/` (Click CLI, benchmarks, platform
evidence, drivers) and `src/solar/` (graphs, IR, verification, formal analysis).
Mirror these under `tests/`; keep workflows in `tests/examples/`, kernels in
`examples/`, Docker support in `docker/`, and maintenance in `scripts/`.
Downloaded/generated data belongs under uncommitted `data/`.

## Commands and Verification

- Install with `uv sync --all-groups`; test with `uv run pytest tests/`.
  Never use `-n auto`: the configured eight-worker cap avoids ROCm OOM.
- Run Python checks with `uv run --with ruff ruff check .`,
  `uv run --with ruff ruff format .`, and `uv run ty check`.
- Evaluate with `uv run sol-execbench evaluate <problem_dir> --solution <path>`.
- Build the ROCm environment with `./scripts/run_docker.sh --build`.

## Architecture and Reuse

- Search with `rg` before adding helpers. Shared helpers belong in the relevant
  `core/` concern (`integrity`, `process`, `data.json_utils`, `text_utils`,
  `platform.runtime`, `arguments`, or `timestamps`), never `core.utils` or
  retired evidence checksum/log modules.
- Give reusable wire identifiers one defining module and import them directly;
  do not repeat literals or relay them through compatibility re-exports.
- Prefer `functools.cache`, `cached_property`, or `partial` to manual
  memoization when lifetime and immutable inputs fit; cache keys must include
  every behavior-changing input, including timeouts.
- Use `StrEnum`/`IntEnum` for closed parser, CLI, typed-model, wire-code, or
  exhaustive-control-flow vocabularies; normalize boundary input before domain
  logic. Share an enum only when semantics and ownership match, not merely its
  values. One-member enums remain valid when they constrain a public/schema
  field. Do not duplicate enum members as constants; derive Click choices and
  similar boundary collections from the enum.
- Use constants for resources, environment variables, tools, paths, versions,
  limits, timeouts, thresholds, format fragments, and derived immutable
  collections; reuse alone does not make an implementation parameter an enum.
- Reserve `tools/` for external integrations. Domain packages must not re-export
  generic helpers; wrappers need a documented test seam and focused coverage.
  Migrate callers and remove obsolete imports. Orchestrators use typed stage
  contracts; parsers own raw JSON validation. Write canonical artifacts
  atomically.

## Naming and Resources

Target Python 3.12/Ruff and keep changes local. Import canonical definitions
directly; alias only for collisions, clear module roles, or conventions such as
`numpy as np`. No private/self aliases or incidental compatibility re-exports;
update callers when symbols move.

Capitalize canonical acronyms in `PascalCase` (`SOLBound`, `IRBackend`,
`AKACorpusManifest`, `GPUEvidence`, `ISAInstructionRequirement`,
`AMDSMIProcess`, `JSONDict`); use lowercase acronyms in `snake_case`, uppercase
in prose, and preserve official/third-party casing such as `ROCm`.

Load large production HIP/C++ sources as `importlib.resources` package
resources; only focused test snippets may remain inline.

## Tests and Hardware

- Put coverage near implementation: unit-test schemas/drivers and integration-
  test subprocess/GPU behavior. Register markers only in
  `pyproject.toml`'s `[tool.pytest.ini_options]`; reuse `requires_rocm`, `cpp`,
  `requires_rdna4`, and `requires_cdna3`.
- Parameterize independent scenarios with stable semantic IDs; avoid generated
  or duplicate IDs. Use pytest fixtures for temporary files, state, logging,
  and capture.
- Hardware tests must declare every prerequisite and skip only for the precise
  missing capability; never hide source regressions behind broad skips/`xfail`.
- Pytest upgrades atomically update the dev dependency, `minversion`, `uv.lock`,
  configuration contract tests, and newly enabled strictness fixes.

## Schemas and Public Contracts

- Each schema family has one current canonical-registry version. Every reader,
  including diagnostics/tests, must require an exact match before business
  fields. Change producers, artifacts, tests, and docs atomically; delete all
  superseded models, readers, migrations, aliases, fixtures, and prose.
- Audit string IDs, numeric versions, versioned resources/prose, and
  multi-version acceptance. Raw numeric schema versions are allowed only in
  canonical registries and registered artifacts.
- Breaking public changes update CLI contracts, docs, examples, and tests
  together and remove old paths. Errors need stable codes, actionable hints,
  and no credential leakage.

## Quality and Process Safety

- For Python changes, run relevant Ruff, `ty`, coupling, readability, and Pytest
  checks. Keep limits canonical; never raise a baseline to admit debt.
- Lint exclusions are root-scoped: exclude root `data/` and `examples/`, never
  `core/data` or `tests/examples` by name.
- Do not add production functions over 80 lines or 10 parameters, or test files
  over 1000 lines. Split touched violations unless correctness is urgent.
- Subprocesses require bounded and redacted output, timeouts, and process-group
  cleanup.

## Security and GPU Execution

Never commit tokens, proprietary kernels, datasets, caches, or benchmark output.
Document architecture assumptions in tests or PR notes.

On sandbox/cache/filesystem permission failure, request scoped escalation and
retry the same bounded operation; do not report benchmark/calibration failure.
Because sandboxes may hide `/dev/kfd` or `/dev/dri/renderD*`, retry GPU
enumeration, compilation, execution, profiling, tracing, or calibration with
scoped host access before declaring hardware/tooling unsupported. Write evidence
only to approved locations; never substitute service-manager or sandbox-escape
wrappers for approval.

## Commits and Pull Requests

Use concise imperative summaries and DCO-sign commits, for example
`git commit -s -m "Fix trace parsing"`. Keep PRs focused and report tests plus
any ROCm hardware checks.
