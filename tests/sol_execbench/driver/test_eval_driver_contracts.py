from __future__ import annotations

import ast
from pathlib import Path

from sol_execbench_type_helpers import make_solution

import sol_execbench.driver as _driver_pkg

_TEMPLATES_DIR = Path(_driver_pkg.__file__).parent / "templates"


def _driver_source() -> str:
    return (_TEMPLATES_DIR / "eval_driver.py").read_text()


def test_eval_driver_is_valid_python():
    ast.parse(_driver_source(), filename="eval_driver.py")


def test_all_generated_evaluation_templates_are_valid_python():
    for name in (
        "eval_driver.py",
        "reference_worker.py",
        "evaluation_orchestrator.py",
    ):
        source = (_TEMPLATES_DIR / name).read_text()
        ast.parse(source, filename=name)


def test_candidate_driver_never_loads_or_calls_reference_code():
    source = _driver_source()

    assert "load_reference_function" not in source
    assert "measure_reference_latency" not in source
    assert "dependencies.ref_fn" not in source
    assert "reference_client=" in source


def test_eval_driver_supports_profiler_graceful_exit_switch():
    source = _driver_source()

    assert 'os.environ.get("SOL_EXECBENCH_GRACEFUL_EXIT") == "1"' in source
    assert "sys.exit(0)\nos._exit(0)" in source


def test_hip_cpp_sources_accept_pytorch_rocm_stream_api_text():
    make_solution(
        name="good_hip_stream",
        definition="test_def",
        author="good_agent",
        spec={
            "languages": ["hip_cpp"],
            "target_hardware": ["LOCAL"],
            "entry_point": "main.cpp::run",
            "destination_passing_style": True,
        },
        sources=[
            {"path": "main.cpp", "content": "void run() {}"},
            {
                "path": "kernel.hip",
                "content": (
                    "auto stream = at::cuda::getCurrentCUDAStream();\n"
                    "kernel<<<grid, block, 0, stream>>>(args);\n"
                ),
            },
        ],
    )


def test_emit_uses_strict_json():
    source = _driver_source()

    assert "emit_trace_jsonl(_trace, _real_stdout)" in source
    assert "_sanitize_floats" not in source
