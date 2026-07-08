from __future__ import annotations

import sol_execbench.core.dataset.solutions as solutions


def _definition(reference: str = "def run(x):\n    return x\n") -> dict:
    return {
        "name": "demo",
        "outputs": {"out": {"shape": ["N"], "dtype": "float32"}},
        "reference": reference,
    }


def test_sanitize_python_source_only_rewrites_stream_identifiers():
    source = (
        "def run(stream, mainstream):\n"
        "    text = 'stream should stay in strings'\n"
        "    # stream should stay in comments\n"
        "    return stream + mainstream\n"
    )

    sanitized = solutions.sanitize_python_source_for_static_review(source)

    assert "def run(strm, mainstream):" in sanitized
    assert "return strm + mainstream" in sanitized
    assert "'stream should stay in strings'" in sanitized
    assert "# stream should stay in comments" in sanitized
    assert "mainstream" in sanitized


def test_build_reference_solution_uses_token_aware_stream_sanitizer():
    solution = solutions.build_reference_solution(
        _definition(
            "def run(stream, x):\n"
            "    note = 'stream literal'\n"
            "    # stream comment\n"
            "    return stream + x\n"
        )
    )

    content = solution["sources"][0]["content"]
    assert "def run(strm, x):" in content
    assert "return strm + x" in content
    assert "'stream literal'" in content
    assert "# stream comment" in content
    assert solution["sources"][0]["path"] == "reference.py"
    assert solution["spec"]["entry_point"] == "reference.py::run"


def test_build_custom_solution_preserves_metadata_and_detects_dps(tmp_path):
    solution_py = tmp_path / "solution.py"
    solution_py.write_text("def run(x, out):\n    stream = x\n    return out\n")

    solution = solutions.build_custom_solution(_definition(), solution_py)

    assert solution["name"] == "custom_demo"
    assert solution["sources"][0]["path"] == "solution.py"
    assert solution["spec"]["entry_point"] == "solution.py::run"
    assert solution["spec"]["destination_passing_style"] is True
    assert "strm = x" in solution["sources"][0]["content"]
