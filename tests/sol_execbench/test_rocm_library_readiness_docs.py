from __future__ import annotations

import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_rocm_library_readiness_doc_classifies_all_schema_categories():
    text = (REPO_ROOT / "docs/rocm_libraries.md").read_text()
    for category in ("`hip_cpp`", "`hipblas`", "`miopen`", "`ck`", "`rocwmma`"):
        assert category in text
    assert "Candidate" in text
    assert "Compatibility example" in text


def test_readme_links_library_readiness_before_claiming_support():
    text = (REPO_ROOT / "README.md").read_text()
    assert "ROCm library-oriented candidate categories" in text
    assert "docs/rocm_libraries.md" in text


def test_former_nvidia_library_examples_are_pytorch_compatibility_examples():
    compatibility_examples = [
        "examples/cutlass/gemm/solution_cutlass.json",
        "examples/cudnn/softmax/solution_cudnn.json",
        "examples/cute_dsl/jamba_attn_proj/solution_cute_dsl.json",
        "examples/cutile/jamba_attn_proj/solution_cutile.json",
    ]
    for relative_path in compatibility_examples:
        data = json.loads((REPO_ROOT / relative_path).read_text())
        assert data["spec"]["languages"] == ["pytorch"], relative_path
        assert "compatibility example" in data["description"], relative_path


def test_no_public_example_advertises_candidate_library_category_as_runnable():
    candidate_categories = {"hipblas", "miopen", "ck", "rocwmma"}
    offenders = []
    for path in sorted((REPO_ROOT / "examples").glob("*/*/solution*.json")):
        data = json.loads(path.read_text())
        languages = set(data["spec"]["languages"])
        if languages & candidate_categories:
            offenders.append(str(path.relative_to(REPO_ROOT)))
    assert not offenders, "candidate categories need runnable evidence:\n" + "\n".join(
        offenders
    )
