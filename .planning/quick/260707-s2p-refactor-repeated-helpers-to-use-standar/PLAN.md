---
status: in_progress
created_at: "2026-07-07T12:12:51Z"
---

# Quick Task Plan

Refactor repeated helper functions to use Python 3.12 standard-library APIs,
Pydantic v2 model APIs, and small shared internal utilities.

Scope:
- Replace repeated chunked sha256 file hashing with the standard
  `hashlib.file_digest` path behind the existing checksum helper.
- Add tests for deterministic Pydantic model JSON, stable model checksum,
  JSON object loading, JSONL model loading, Markdown table-cell escaping, text
  tailing, subprocess text normalization, and ordered uniqueness.
- Refactor repeated call sites for report `to_json`, report checksum payloads,
  Markdown cell escaping, subprocess text/tail helpers, and ordered uniqueness.
- Keep public schemas, sidecar payload shapes, and CLI behavior unchanged.

Verification:
- Run focused pytest coverage for the new helpers and representative existing
  report/CLI tests.
- Run Ruff check on touched source files.
