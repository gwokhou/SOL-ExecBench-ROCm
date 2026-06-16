---
generated_by: gsd-map-codebase
focus: quality
mapped_at: 2026-06-16
---

# Conventions

## Python Style

- Python target is `>=3.12,<3.14`.
- Modules, functions, and variables use `snake_case`.
- Classes, Pydantic models, and enum classes use `PascalCase`.
- Enum values are string-backed where they cross schema boundaries.
- Most source files include SPDX headers that reflect provenance policy.
- Formatting and linting are handled by Ruff.
- Type checking is handled by Ty over `src` and `tests`.

## Schema Style

- Public benchmark schemas are Pydantic v2 models under
  `src/sol_execbench/core/data/`.
- Models use validators for compatibility, path safety, source validation, and
  migration guidance.
- Schema changes should be accompanied by tests and docs because trace,
  definition, workload, and solution JSON are public contracts.
- JSON output generally uses Pydantic `model_dump(mode="json")` or stable JSON
  helpers.

## CLI Style

- Click commands live in `src/sol_execbench/cli/main.py` and
  `src/sol_execbench/cli/baseline.py`.
- GPU-free metadata commands require `--json` and raise `click.ClickException`
  otherwise.
- Rich is used for terminal tables and progress output.
- CLI subprocess logs are bounded or filtered before diagnostic sidecars are
  written.

## Error Handling

- User-facing CLI failures use `click.ClickException`.
- Generated eval-driver failures become typed `Trace` evaluation statuses.
- No-trace outcomes write diagnostic-only JSON sidecars with bounded stdout and
  stderr tails.
- Dataset and evidence reports preserve blocked/deferred/missing states instead
  of upgrading them into validation claims.
- Reward-hack detection uses `RewardHackDetected` and explicit
  `EvaluationStatus.REWARD_HACK` traces.

## Subprocess Patterns

- Staged evaluation uses explicit command lists, not shell strings.
- `ProblemPackager` writes normalized problem files and solution sources into a
  temporary staging directory.
- Generated `eval_driver.py` redirects noisy stdout to stderr and reserves
  stdout for strict trace JSONL.
- `os._exit(0)` is used at eval-driver shutdown by default to avoid cleanup
  noise after traces are emitted; `SOL_EXECBENCH_GRACEFUL_EXIT=1` enables
  graceful exit when needed.

## Security Patterns

- Solution source paths reject absolute paths and `..` traversal.
- Native compile flags reject response files, host path injection, runtime
  loader options, and unsafe linker path behavior.
- Static source review blocks suspicious subprocess/socket/thread/cache/memory
  patterns before user-code import.
- Runtime integrity checks snapshot critical evaluation functions before user
  code is imported.
- Docs consistently state that subprocess/staging guardrails are not a hardened
  multi-tenant sandbox.

## Documentation Style

- Generated canonical docs include `<!-- generated-by: gsd-doc-writer -->`.
- Claim-bearing docs use conservative evidence wording and explicit non-claims.
- Hardware validation wording distinguishes RDNA4, CDNA3, MI300X, MI308X, and
  CDNA4 scopes.
- Diagnostic sidecars are described as diagnostic evidence, not canonical trace
  replacements.

## Contribution Conventions

- Commit titles use `#<Issue Number> - <Commit Title>` for normal
  contributions, with DCO sign-off.
- `.pre-commit-config.yaml` enforces Ruff, DCO sign-off, and Ty checks.
- PRs should stay focused and list tests or GPU checks run.
