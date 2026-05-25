# Quick Audit Fix Plan: Remaining Suite Reds

## Goal

Resolve the 9 non-sandbox failures left after v1.14, without changing v1.14
profiling behavior.

## Findings

- MIOpen readiness required both `libMIOpen` and lowercase `libmiopen`.
- `build_ext.py` template imported `os`, violating the template import guard.
- Historical guardrail docs missed expected boundary phrases after doc updates.
- `.planning/REQUIREMENTS.md` compatibility path was absent.
- CUDA/NVIDIA residue audit lacked classifications for current legitimate
  boundary fixtures and dataset parity metadata.

## Tasks

- [x] Fix MIOpen readiness alias handling.
- [x] Remove AST-visible `os` import from the build template.
- [x] Restore guardrail phrases and requirement compatibility index.
- [x] Extend residue classifications for legitimate current contexts.
- [x] Verify targeted tests, Ruff, and full pytest suite.
