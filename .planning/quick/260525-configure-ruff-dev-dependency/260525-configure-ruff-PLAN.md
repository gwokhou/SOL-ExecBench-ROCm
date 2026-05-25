# Quick Task Plan: Configure Ruff Dev Dependency

## Goal

Mirror `hip-playground` by making Ruff a normal development dependency instead
of requiring ad-hoc `uv run --with ruff` invocation.

## Tasks

- [x] Add Ruff to the `dev` dependency group.
- [x] Update `uv.lock`.
- [x] Update development/testing docs to use `uv run ruff ...`.
- [x] Verify direct Ruff invocation.
