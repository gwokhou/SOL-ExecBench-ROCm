# Phase 70 Context: Capability Registry Schema

## Objective

Define a machine-verifiable schema for tool capabilities across evidence level,
artifact type, hardware generation, GPU architecture, ROCm version, status, and
source references.

## Decisions

- Use Pydantic models consistent with existing public data model style.
- Keep routing reports separate from canonical trace JSONL.
- Status and reason codes must be explicit and auditable.
