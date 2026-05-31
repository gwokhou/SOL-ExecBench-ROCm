from __future__ import annotations

import json
import re
from pathlib import Path

from sol_execbench.core.dataset.execution_closure import ExecutionClosureReport
from sol_execbench.core.dataset.paper_denominator import PaperDenominatorReport
from sol_execbench.core.matrix_diff import MatrixReportDiff
from sol_execbench.core.scoring.amd_bound_sanity import AmdBoundSanityReport

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLES_DIR = REPO_ROOT / "docs/examples/v1_19_evidence"
README = EXAMPLES_DIR / "README.md"

EXPECTED_FILES = (
    "execution_closure.demo.json",
    "paper_denominator.demo.json",
    "paper_denominator.demo.md",
    "matrix_schema_export.demo.json",
    "matrix_diff.demo.json",
    "matrix_diff.demo.md",
    "amd_bound_sanity.demo.json",
    "amd_bound_sanity.demo.md",
)
EXPECTED_SCHEMA_MARKERS = {
    "execution_closure.demo.json": "sol_execbench.execution_closure.v1",
    "paper_denominator.demo.json": "sol_execbench.paper_denominator_report.v1",
    "matrix_schema_export.demo.json": "sol_execbench.rocm_compatibility_matrix.v1",
    "matrix_diff.demo.json": "sol_execbench.rocm_compatibility_matrix_diff.v1",
    "amd_bound_sanity.demo.json": "sol_execbench.amd_bound_sanity.v1",
}
NEGATIVE_BOUNDARIES = (
    "no full 235-problem paper validation",
    "no upstream SOLAR parity",
    "no score authority",
    "no leaderboard readiness",
    "no CDNA 3/MI300X/CDNA4 validation",
    "no native-host ROCm Matrix validation",
    "no new-hardware validation",
)
FORBIDDEN_TEXT = (
    "/home/",
    "/tmp/",
    "/var/",
    "hf_",
    "token",
    "secret",
    "CLOCKS_LOCKED",
    "gfx1200",
    "gfx942",
    "latency_ms",
    "tflops",
    "GB/s",
)


def _read_text(path: Path) -> str:
    return path.read_text()


def _load_json(name: str) -> dict[str, object]:
    return json.loads(_read_text(EXAMPLES_DIR / name))


def _json_values(value: object) -> list[object]:
    if isinstance(value, dict):
        values: list[object] = []
        for nested in value.values():
            values.extend(_json_values(nested))
        return values
    if isinstance(value, list):
        values = []
        for nested in value:
            values.extend(_json_values(nested))
        return values
    return [value]


def test_v1_19_example_readme_references_existing_fixture_files():
    readme = _read_text(README)

    assert "demo-only" in readme
    assert "diagnostic-only" in readme
    assert "../../v1_19_evidence_guide.md" in readme
    for name in EXPECTED_FILES:
        assert name in readme
        path = EXAMPLES_DIR / name
        assert path.exists()
        if name.endswith(".json"):
            json.loads(_read_text(path))
        else:
            assert _read_text(path).startswith("#")


def test_v1_19_example_json_fixtures_expose_expected_schema_markers():
    for name, marker in EXPECTED_SCHEMA_MARKERS.items():
        text = _read_text(EXAMPLES_DIR / name)
        payload = json.loads(text)
        assert marker in text
        if name != "matrix_schema_export.demo.json":
            assert payload["schema_version"] == marker


def test_v1_19_report_examples_validate_against_real_contract_models():
    ExecutionClosureReport.model_validate_json(
        _read_text(EXAMPLES_DIR / "execution_closure.demo.json")
    )
    PaperDenominatorReport.model_validate_json(
        _read_text(EXAMPLES_DIR / "paper_denominator.demo.json")
    )
    MatrixReportDiff.model_validate_json(
        _read_text(EXAMPLES_DIR / "matrix_diff.demo.json")
    )
    AmdBoundSanityReport.model_validate_json(
        _read_text(EXAMPLES_DIR / "amd_bound_sanity.demo.json")
    )


def test_v1_19_example_refs_are_relative_bounded_and_checksum_backed():
    checksum_pattern = re.compile(r"^sha256:[0-9a-f]{64}$")

    for name in EXPECTED_SCHEMA_MARKERS:
        payload = _load_json(name)
        values = _json_values(payload)
        text_values = [value for value in values if isinstance(value, str)]
        for value in text_values:
            if "/" in value and not value.startswith(("https://", "sha256:")):
                assert not value.startswith("/")
        if name != "matrix_schema_export.demo.json":
            assert any(checksum_pattern.match(value) for value in text_values)

    closure = _load_json("execution_closure.demo.json")
    closure_text = json.dumps(closure, sort_keys=True)
    assert "cli_log_ref" in closure_text
    assert "stdout" not in closure_text
    assert "stderr" not in closure_text


def test_v1_19_example_wording_repeats_demo_and_negative_boundaries():
    combined = "\n".join(
        [_read_text(README)]
        + [_read_text(EXAMPLES_DIR / name) for name in EXPECTED_FILES]
    )

    assert "demo-only" in combined
    assert "diagnostic-only" in combined
    for boundary in NEGATIVE_BOUNDARIES:
        assert boundary in combined
    for forbidden in FORBIDDEN_TEXT:
        assert forbidden not in combined


def test_v1_19_example_authority_flags_remain_false_or_diagnostic_only():
    for name in (
        "paper_denominator.demo.json",
        "matrix_diff.demo.json",
        "amd_bound_sanity.demo.json",
    ):
        payload = _load_json(name)
        text = json.dumps(payload, sort_keys=True)
        for authority in (
            "score_authority",
            "leaderboard_authority",
            "paper_parity_authority",
            "native_host_validation",
            "native_host_validation_authority",
            "new_hardware_validation",
        ):
            if f'"{authority}"' in text:
                assert f'"{authority}": false' in text
        assert "diagnostic" in text


def test_amd_bound_sanity_example_uses_only_actual_script_inputs():
    demo_markdown = _read_text(EXAMPLES_DIR / "amd_bound_sanity.demo.md")
    demo_payload = _load_json("amd_bound_sanity.demo.json")
    source_text = json.dumps(demo_payload["sources"], sort_keys=True)

    for text in (demo_markdown, source_text):
        assert "paper_denominator" not in text
        assert "paper_denominator.json" not in text

    for expected_ref in (
        "trace",
        "execution_closure",
        "amd_score",
        "compatibility_matrix",
    ):
        assert expected_ref in demo_markdown
