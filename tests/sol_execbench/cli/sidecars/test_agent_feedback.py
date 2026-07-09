from __future__ import annotations

import json
from pathlib import Path

from sol_execbench.cli.sidecars import agent_feedback as cli_agent_feedback_sidecar
from sol_execbench.core.data.solution import (
    BuildSpec,
    Solution,
    SourceFile,
    SupportedHardware,
    SupportedLanguages,
)
from sol_execbench.core.data.trace import (
    Environment,
    Evaluation,
    EvaluationStatus,
    Trace,
)
from sol_execbench.core.data.workload import ScalarInput
from sol_execbench.core.data.workload import Workload


def _solution(source: str = "def run(x):\n    return x\n") -> Solution:
    return Solution(
        name="candidate",
        definition="toy",
        author="agent",
        spec=BuildSpec(
            languages=[SupportedLanguages.PYTORCH],
            target_hardware=[SupportedHardware.LOCAL],
            entry_point="solution.py::run",
        ),
        sources=[SourceFile(path="solution.py", content=source)],
    )


def _trace() -> Trace:
    return Trace(
        definition="toy",
        solution="candidate",
        workload=Workload(
            uuid="w0",
            axes={"n": 1},
            inputs={"n": ScalarInput(value=1)},
        ),
        evaluation=Evaluation(
            status=EvaluationStatus.COMPILE_ERROR,
            environment=Environment(hardware="AMD gfx1200", libs={"hip": "7.0"}),
            timestamp="2026-06-16T00:00:00Z",
        ),
    )


def test_agent_feedback_sidecar_tracks_trace_output(tmp_path: Path):
    output = tmp_path / "trace.jsonl"

    assert cli_agent_feedback_sidecar._agent_feedback_sidecar_path(output) == (
        tmp_path / "trace.jsonl.agent-feedback.json"
    )
    assert cli_agent_feedback_sidecar._agent_feedback_sidecar_path(None) is None


def test_agent_feedback_sidecar_records_bounded_metadata(tmp_path: Path):
    output = tmp_path / "trace.jsonl"
    output.write_text('{"definition":"toy"}\n')
    solution = _solution()
    trace = _trace()

    written = cli_agent_feedback_sidecar._write_agent_feedback_sidecar(
        output,
        [trace],
        solution=solution,
        profile_result=None,
        static_evidence=None,
        run_id="run-001",
        feedback_target_id="gemm",
        feedback_candidate_id="candidate-sha",
        feedback_source_sha256="source-sha",
        feedback_sol_version="custom-sol-tag",
    )

    assert written == tmp_path / "trace.jsonl.agent-feedback.json"
    assert written is not None
    payload = json.loads(written.read_text())
    assert payload["schema_version"] == "sol_execbench.agent_feedback.v2"
    assert payload["authority"] == "diagnostic"
    assert payload["identity"]["trace_path"] == "trace.jsonl"
    assert payload["identity"]["target_id"] == "gemm"
    assert payload["identity"]["run_id"] == "run-001"
    assert payload["identity"]["candidate_id"] == "candidate-sha"
    assert payload["identity"]["source_sha256"] == "source-sha"
    assert payload["identity"]["sol_version"] == "custom-sol-tag"
    assert "candidate_hash" not in payload["identity"]
    assert "source_hash" not in payload["identity"]
    assert "sol_contract_version" not in payload["identity"]
    assert payload["summary"]["status_counts"] == {"COMPILE_ERROR": 1}
    assert payload["items"][0]["code"] == "compile_error"
    trace_citations = [
        citation
        for citation in payload["artifact_citations"]
        if citation["kind"] == "trace"
    ]
    assert trace_citations == [
        {
            "kind": "trace",
            "label": "canonical_trace_jsonl",
            "path": "trace.jsonl",
            "sha256": trace_citations[0]["sha256"],
            "status": None,
        }
    ]
    assert trace_citations[0]["sha256"] is not None
    assert len(trace_citations[0]["sha256"]) == 64
    assert "raw" not in json.dumps(payload).lower()


def test_agent_feedback_identity_uses_solution_source_hash(tmp_path: Path):
    output = tmp_path / "trace.jsonl"
    output.write_text('{"definition":"toy"}\n')
    trace = _trace()
    first = _solution("def run(x):\n    return x\n")
    second = _solution("def run(x):\n    return x + 1\n")

    first_identity = cli_agent_feedback_sidecar._agent_feedback_identity_fields(
        output,
        [trace],
        solution=first,
    )
    second_identity = cli_agent_feedback_sidecar._agent_feedback_identity_fields(
        output,
        [trace],
        solution=second,
    )
    no_solution_identity = cli_agent_feedback_sidecar._agent_feedback_identity_fields(
        output, [trace]
    )

    assert first_identity["candidate_id"] == second_identity["candidate_id"]
    assert first_identity["source_sha256"] == first.hash()
    assert second_identity["source_sha256"] == second.hash()
    assert first_identity["source_sha256"] != second_identity["source_sha256"]
    assert no_solution_identity["source_sha256"] is None
    assert "candidate_hash" not in first_identity
    assert "source_hash" not in first_identity
    assert "candidate_hash" not in second_identity
    assert "source_hash" not in second_identity
    assert "candidate_hash" not in no_solution_identity
    assert "source_hash" not in no_solution_identity


def test_agent_feedback_identity_accepts_consumer_identity_fields(tmp_path: Path):
    output = tmp_path / "trace.jsonl"
    output.write_text('{"definition":"toy"}\n')
    trace = _trace()

    identity = cli_agent_feedback_sidecar._agent_feedback_identity_fields(
        output,
        [trace],
        solution=_solution(),
        run_id="run-001",
        target_id="gemm",
        candidate_id="candidate-sha",
        source_sha256="source-sha",
    )

    assert identity == {
        "target_id": "gemm",
        "run_id": "run-001",
        "candidate_id": "candidate-sha",
        "source_sha256": "source-sha",
    }
