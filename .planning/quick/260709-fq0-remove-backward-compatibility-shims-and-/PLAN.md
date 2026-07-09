---
status: complete
date: 2026-07-09
---

# Remove backward compatibility shims

Task: Remove backward compatibility shims and legacy CUDA/CUPTI timing aliases now
that the project has no external users.

Plan:
- Remove compatibility facade modules that only re-export relocated modules.
- Update internal imports and tests to use canonical package paths.
- Remove legacy timing API aliases and accepted legacy methodology spellings.
- Keep ROCm-required PyTorch `torch.cuda` namespace usage intact.
- Run focused tests for imports, timing, and residue audits.
