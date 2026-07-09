---
quick_id: 260709-spd
slug: import-superpowers-docs
status: complete
completed_at: "2026-07-09T09:00:00Z"
---

# Summary

Converted the legacy `docs/superpowers/` planning corpus into GSD planning
artifacts without changing the active v1.38 roadmap. Imported implementation
plans now live as archived quick task artifacts, and design specs now live as
archived research notes.

## Outputs

- Converted 25 implementation plans into `.planning/quick/*-sp*/`.
- Converted 7 design specs into `.planning/research/superpowers/`.
- Added `.planning/research/superpowers/INDEX.md` mapping every source file to
  its GSD artifact.
- Replaced legacy Superpowers execution prompts in imported artifacts with a
  GSD-compatible historical execution note.

## Verification

Passed:

```bash
python - <<'PY'
from pathlib import Path
plan_sources = sorted(Path('docs/superpowers/plans').glob('*.md'))
spec_sources = sorted(Path('docs/superpowers/specs').glob('*.md'))
plan_targets = sorted(Path('.planning/quick').glob('*-sp*/*-PLAN.md'))
spec_targets = [p for p in Path('.planning/research/superpowers').glob('*.md') if p.name != 'INDEX.md']
print(f'plans: sources={len(plan_sources)} targets={len(plan_targets)}')
print(f'specs: sources={len(spec_sources)} targets={len(spec_targets)}')
PY
```

Result: `plans: sources=25 targets=25`, `specs: sources=7 targets=7`.
