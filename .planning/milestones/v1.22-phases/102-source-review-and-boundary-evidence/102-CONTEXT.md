# Phase 102: Source Review And Boundary Evidence - Context

**Gathered:** 2026-06-01
**Status:** Ready for planning

## Phase Boundary

Maintainers and users can see stronger static-review outcomes and boundary
evidence without mistaking them for hard sandbox guarantees.

This phase covers requirements BOUNDARY-01 through BOUNDARY-04:

- Broaden source-review test coverage for process, file, import, native loader,
  stream, cache, and obfuscation bypass families.
- Move Python review beyond raw regex where strings, comments, or simple
  obfuscation make regex scanning too broad or too easy to bypass.
- Ensure blocked or flagged review outcomes are available as structured
  evidence in logs, traces, or sidecar-adjacent output.
- Document that static review plus subprocess execution is not hardened
  multi-tenant sandboxing.

## Current Code Context

- `src/sol_execbench/core/bench/reward_hack.py` owns static source review.
- `SourceReviewIssue.to_dict()` and `SourceReview.to_dict()` already expose a
  structured issue payload with `path`, `rule`, `severity`, `message`, and
  `evidence`.
- `SourceReview.format_blocking_message()` currently emits a human-readable
  summary only. It can carry structured review payload text without changing
  public trace schemas.
- Current `_STATIC_RULES` are regex based. `_strip_comments()` removes only
  whole-line `#` or `//` comments and does not understand Python strings,
  inline comments, AST calls, dynamic imports, or `getattr(__import__(...))`
  patterns robustly.
- `review_solution_sources()` applies `_STATIC_RULES` to all sources and the
  precision-downgrade rule when any output dtype contract is `torch.float32`.
- Existing source-review tests cover non-default streams, `data_ptr` caching,
  base64/ctypes loader patterns, process execution, dynamic import, `getattr(os,
  "system")`, socket/path/pickle/importlib/torch loader cases, precision
  downgrade, and allow-list cases for plain `os.environ`, `torch.compile`, HIP
  current-stream text, and native C++ `data_ptr`.

## Documentation Context

- `.planning/codebase/CONCERNS.md` already distinguishes subprocess separation
  from hardened sandboxing.
- `docs/user/CLAIMS.md` already states the project does not claim hard sandboxing or
  multi-tenant safety.
- `docs/user/RESEARCHER-GUIDE.md` mentions source review blocks external files,
  subprocesses, network calls, and embedded payload patterns.
- `docs/user/ARCHITECTURE.md` describes the subprocess boundary, but the boundary
  section should explicitly say this is not a security sandbox.
- `README.md` does not yet surface the security boundary in quick-start usage
  where new users are most likely to see it.

## Implementation Direction

Use an AST-aware Python review path for `.py` sources, while retaining regex
rules for native text and as a fallback for Python syntax failures. Keep review
behavior conservative and local to static source review; do not redesign runtime
isolation or claim hard sandboxing.

Structured evidence should preserve current `SourceReview.to_dict()` contracts
and make blocked evidence visible in the blocking log message. This avoids trace
schema churn while satisfying the evidence requirement through logs and existing
review payloads.

## Verification Targets

- Focused source-review tests:
  `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/sol_execbench/core/bench/test_reward_hack.py -q`
- Eval driver regression coverage:
  `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/sol_execbench/driver/test_eval_driver.py -q`
- Lint touched Python files:
  `UV_CACHE_DIR=/tmp/uv-cache uv run ruff check src/sol_execbench/core/bench/reward_hack.py tests/sol_execbench/core/bench/test_reward_hack.py`

