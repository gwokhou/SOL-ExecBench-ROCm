# Research: Architecture for v1.36 SOL Agent Feedback Sidecar Producer

**Date:** 2026-06-15

## Integration Shape

1. `sol-execbench contract --json` advertises optional feedback capabilities.
2. Evaluation continues to emit canonical Trace JSONL exactly as before.
3. Optional sidecar generation reads canonical trace plus already-bounded
   diagnostic artifacts.
4. Sidecar writer persists `trace.jsonl.agent-feedback.json` beside the trace
   when output path semantics allow it.
5. HIP consumes the sidecar only through its execbench adapter.

## Suggested SOL Modules

- `core/data/contract.py`: optional capability tokens and source-boundary text.
- `core/bench/agent_feedback.py`: strict Pydantic sidecar models, builders,
  validators, freshness identity, and summary helpers.
- `cli/main.py`: output-path wiring and optional sidecar persistence after the
  canonical trace/profile/static/environment sidecars have been written.
- `docs/EVALUATOR-CONTRACT.md` or existing docs: consumer-facing contract
  description and authority boundaries.

## Build Order

1. Contract and schema first so HIP Phase 141 can lock consumer assumptions.
2. Generator and CLI persistence next so HIP Phase 142 has real artifacts.
3. Freshness and artifact citations before HIP Phase 143 uses hints.
4. Guardrail fixtures last to prove all invalid states remain diagnostic-only.
