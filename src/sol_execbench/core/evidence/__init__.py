"""Cross-run environment evidence and canonical execution-artifact identity.

Evaluation-owned traces and diagnostic sidecars remain under ``core.bench``;
``runtime_evidence`` records non-authoritative host, tool, and GPU observations.
"""

CANONICAL_BENCHMARK_OUTPUT = "trace_jsonl"

__all__ = ["CANONICAL_BENCHMARK_OUTPUT"]
