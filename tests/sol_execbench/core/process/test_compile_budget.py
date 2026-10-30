from pathlib import Path

import pytest

from sol_execbench.core.process import compile_budget


@pytest.mark.parametrize(
    ("cpu", "memory", "requested", "expected"),
    [
        ("200000 100000", str(4 * 1024**3), 8, 2),
        ("50000 100000", str(4 * 1024**3), 8, 1),
        ("800000 100000", str(2 * 1024**3), 8, 1),
        ("max 100000", "max", 2, 2),
        ("invalid", "max", 8, 1),
    ],
    ids=["quota", "fractional", "memory", "explicit", "unknown"],
)
def test_compile_budget_caps_visible_resources(
    monkeypatch, tmp_path: Path, cpu, memory, requested, expected
):
    (tmp_path / "cpu.max").write_text(cpu)
    (tmp_path / "memory.max").write_text(memory)
    monkeypatch.setattr(
        compile_budget, "_cgroup_directories", lambda: (tmp_path,)
    )
    monkeypatch.setattr(
        compile_budget.os, "sched_getaffinity", lambda _: set(range(8))
    )
    assert compile_budget.compile_job_limit(requested) == expected


def test_compile_budget_unknown_and_invalid(monkeypatch, tmp_path):
    monkeypatch.setattr(
        compile_budget, "_cgroup_directories", lambda: (tmp_path,)
    )
    assert compile_budget.compile_job_limit(8) == 1
    for value in (0, -1, True):
        with pytest.raises(ValueError, match="positive"):
            compile_budget.compile_job_limit(value)


def test_compile_budget_includes_parent_limits(monkeypatch, tmp_path):
    child = tmp_path / "parent" / "child"
    child.mkdir(parents=True)
    membership = tmp_path / "membership"
    membership.write_text("0::/parent/child\n")
    for directory, quota in [
        (tmp_path, "max"),
        (child.parent, "200000"),
        (child, "max"),
    ]:
        (directory / "cpu.max").write_text(f"{quota} 100000")
        (directory / "memory.max").write_text(str(4 * 1024**3))
    monkeypatch.setattr(compile_budget, "_CGROUP_ROOT", tmp_path)
    monkeypatch.setattr(compile_budget, "_CGROUP_MEMBERSHIP", membership)
    monkeypatch.setattr(
        compile_budget.os, "sched_getaffinity", lambda _: set(range(8))
    )
    assert compile_budget.compile_job_limit(8) == 2
