# Repository Guidelines

## Layout and Ownership

Production code lives in `src/sol_execbench/` and `src/solar/`.
`sol_execbench` owns the Click CLI, benchmark models and logic, platform
evidence, and execution drivers; `solar` owns graph extraction, IR,
verification, and formal analysis. Mirror package coverage under
`tests/sol_execbench/` and `tests/solar/`; keep workflow tests in
`tests/examples/`. Runnable kernels live in `examples/`, Docker support in
`docker/`, and maintenance scripts in `scripts/`. Keep downloaded data and
generated outputs under `data/` and out of commits.

## Commands and Verification

- Install all dependencies with `uv sync --all-groups`.
- Run the suite with `uv run pytest tests/`. Never use `-n auto`; the configured
  maximum of eight workers avoids ROCm memory exhaustion. A focused example is
  `uv run pytest tests/sol_execbench/test_e2e.py`.
- Run Python checks with `uv run --with ruff ruff check .`,
  `uv run --with ruff ruff format .`, and `uv run ty check`.
- Evaluate one solution with
  `uv run sol-execbench evaluate <problem_dir> --solution <path>`.
- Build the ROCm environment with `./scripts/run_docker.sh --build`.

## Architecture and Reuse

- Search with `rg` before adding a generic helper. Shared helpers belong in
  `core/` by concern: `integrity`, `process`, `data.json_utils`, `text_utils`,
  `platform.runtime`, `arguments`, or `timestamps`. Do not create `core.utils`
  or restore retired evidence checksum/log modules.
- Reserve `tools/` for external integrations. Domain packages must not re-export
  generic helpers; wrappers need a documented test seam and focused coverage.
  Migrate in-scope callers and remove obsolete imports.
- Orchestrators use typed stage inputs and results; parsers own raw JSON
  validation. Reuse core primitives and write canonical artifacts atomically.

## Naming and Resources

Target Python 3.12 and Ruff formatting. Keep changes local to the affected
subsystem. Import canonical names from their defining modules; alias only for a
real collision, a clear module role, or conventional third-party spelling such
as `numpy as np`. Do not create private/self aliases or incidental compatibility
re-exports; update callers when symbols move.

Keep canonical acronyms fully capitalized in `PascalCase`: `IRBackend`,
`AKACorpusManifest`, `GPUEvidence`, `ISAInstructionRequirement`,
`AMDSmiProcess`, and `JSONDict`. Acronyms stay lowercase in `snake_case`;
use their all-caps form in prose, and preserve official casing such as `ROCm`
and third-party namespace spelling.

Store large production HIP or C++ sources as package resources loaded through
`importlib.resources`, not as Python string literals. Focused test snippets may
remain inline.

## Tests and Hardware

- Put coverage near the implementation. Use focused unit tests for schema and
  driver logic, and integration tests for subprocess or GPU behavior.
- Register markers only in `[tool.pytest.ini_options]` in `pyproject.toml`.
  Use existing markers such as `requires_rocm`, `cpp`, `requires_rdna4`, and
  `requires_cdna3`.
- Parameterize independent scenarios and give non-trivial cases stable semantic
  IDs; avoid generated or duplicate IDs. Prefer pytest-managed fixtures for
  temporary files, state, logging, and output capture.
- Hardware tests must declare every prerequisite and skip only for the precise
  missing capability. Do not hide source regressions behind broad `xfail` or
  skip rules.
- A pytest upgrade must atomically update the development dependency,
  `minversion`, `uv.lock`, configuration contract tests, and newly enabled
  strictness fixes.

## Schemas and Public Contracts

- Each schema family has exactly one current version in its canonical registry.
  Every reader must require and exactly match it before parsing business fields;
  diagnostic and test paths may not bypass the check.
- Change producers, artifacts, tests, and docs atomically, then delete
  superseded models, readers, migrations, aliases, fixtures, and prose. Git
  history is the archive.
- Audit string IDs, numeric versions, versioned resources/prose, and
  multi-version acceptance. Raw numeric schema versions are allowed only in
  canonical registries and registered artifacts.
- Breaking public changes are allowed only when the CLI contract, docs,
  examples, and tests change together and superseded paths are removed.
  User-facing errors need a stable code, actionable hint, and no credential
  leakage.

## Quality and Process Safety

- For Python changes, run the relevant Ruff, `ty`, coupling, readability, and
  Pytest checks. Keep limits in their canonical policy source and never raise a
  baseline to admit new debt.
- Lint exclusions are root-scoped: exclude root `data/` and `examples/`, never
  `core/data` or `tests/examples` by name.
- Do not add production functions over 80 lines or 10 parameters, or test files
  over 1000 lines. Split touched violations unless correctness is urgent.
- Subprocesses require bounded and redacted output, timeouts, and process-group
  cleanup.

## Security and GPU Execution

Never commit tokens, proprietary kernels, datasets, caches, or benchmark output.
Document architecture assumptions in tests or PR notes.

If a required container operation fails only because of sandbox, cache, or
filesystem permissions, request scoped escalation and retry the same operation;
do not classify the permission failure as a benchmark or calibration failure.
Sandboxes may incompletely expose `/dev/kfd` or hide `/dev/dri/renderD*`. For
GPU enumeration, compilation, execution, profiling, tracing, or calibration,
retry the exact bounded command with scoped host access before declaring a
device, architecture, counter, driver, compiler, or runtime unsupported. Write
evidence only to approved temporary/output locations. Do not use service-manager
or sandbox-escape wrappers instead of approval.

## Commits and Pull Requests

Use concise imperative summaries and DCO-sign commits, for example
`git commit -s -m "Fix trace parsing"`. Keep PRs focused and report tests plus
any ROCm hardware checks.
