---
status: complete
completed: 2026-05-25
---

# Configure Ty Dev Dependency Summary

## Changes

- Added `ty>=0.0.39` to the `dev` dependency group.
- Added `[tool.ty.src] include = ["src", "tests"]` to match hip-playground's source-root style.
- Updated development and testing documentation with the `uv run ty check` command.

## Verification

- `UV_CACHE_DIR=/tmp/uv-cache uv lock` resolved successfully and added `ty v0.0.39`.
- `UV_CACHE_DIR=/tmp/uv-cache uv run ty check` installed Ty and ran successfully as a tool invocation, then failed the repository type gate with 713 existing diagnostics.
- `UV_CACHE_DIR=/tmp/uv-cache uv run ruff check pyproject.toml docs/user/DEVELOPMENT.md docs/user/TESTING.md docs/user/compliance.md .planning/quick/260525-configure-ty-dev-dependency/260525-configure-ty-PLAN.md .planning/quick/260525-configure-ty-dev-dependency/260525-configure-ty-SUMMARY.md` passed.
