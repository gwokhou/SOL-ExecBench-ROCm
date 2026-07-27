from __future__ import annotations

import ast
import json
from pathlib import Path
from types import ModuleType
from typing import Any, cast

import pytest


def test_coupling_resolves_imports_cycles_and_boundaries(
    load_script,
    tmp_path: Path,
) -> None:
    coupling = load_script("scripts/check_coupling.py")
    package = tmp_path / "package"
    package.mkdir()
    module_paths = {
        "sol_execbench.alpha": package / "alpha.py",
        "sol_execbench.beta": package / "beta.py",
        "solar.gamma": package / "gamma.py",
    }
    module_paths["sol_execbench.alpha"].write_text(
        "from . import beta\n", encoding="utf-8"
    )
    module_paths["sol_execbench.beta"].write_text(
        "from . import alpha\n", encoding="utf-8"
    )
    module_paths["solar.gamma"].write_text(
        "import sol_execbench.alpha\n", encoding="utf-8"
    )

    edges = coupling.internal_import_edges(module_paths)

    assert edges == {
        ("sol_execbench.alpha", "sol_execbench.beta"),
        ("sol_execbench.beta", "sol_execbench.alpha"),
        ("solar.gamma", "sol_execbench.alpha"),
    }
    assert coupling.strongly_connected_components(module_paths, edges) == [
        ("sol_execbench.alpha", "sol_execbench.beta")
    ]
    assert coupling.cross_package_violations(edges) == [
        ("solar.gamma", "sol_execbench.alpha")
    ]
    assert coupling.layer_violations(
        {
            (
                "sol_execbench.core.bench.worker",
                "sol_execbench.core.reports.writer",
            )
        }
    ) == [
        (
            "sol_execbench.core.bench.worker",
            "sol_execbench.core.reports.writer",
        )
    ]
    assert coupling.is_under("solar.graph.extraction", "solar.graph")
    assert not coupling.is_under("solar.graphical", "solar.graph")


def test_coupling_limit_failures_include_lines_fanout_and_exact_imports(
    load_script,
) -> None:
    coupling = load_script("scripts/check_coupling.py")
    names = {
        *coupling.P0_LIMITS,
        *coupling.P1_LIMITS,
        *coupling.SOLAR_LIMITS,
        *coupling.EXACT_IMPORTS,
    }
    stats = {
        name: coupling.ModuleStats(
            path=f"{name}.py",
            line_count=10_000,
            fanout=100,
            imports=[],
        )
        for name in names
    }

    failures = coupling.check_limits(stats)

    assert any("line_count" in failure for failure in failures)
    assert any("fanout" in failure for failure in failures)
    assert any("imports [] !=" in failure for failure in failures)


def test_current_docs_link_check_ignores_external_and_reports_missing(
    load_script,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    docs = load_script("scripts/check_current_docs.py")
    docs_root = tmp_path / "docs"
    docs_root.mkdir()
    page = docs_root / "guide.md"
    (docs_root / "present.md").write_text("# Present\n", encoding="utf-8")
    monkeypatch.setattr(docs, "ROOT", tmp_path)

    failures = docs._broken_local_links(
        page,
        (
            "[anchor](#section) [web](https://example.com) "
            "[present](present.md#title) [missing](missing.md)"
        ),
    )

    assert failures == ["docs/guide.md has broken local link 'missing.md'"]


def _dataset_policy() -> dict[str, Any]:
    return {
        "sources": [
            {
                "id": "restricted",
                "name": "Restricted corpus",
                "path_globs": ["data/restricted/**"],
                "redistribution_class": "restricted",
                "repository_redistribution": False,
                "release_bundle_redistribution": True,
            },
            "ignored-non-object",
        ]
    }


def test_dataset_redistribution_distinguishes_repository_and_release(
    load_script,
    tmp_path: Path,
) -> None:
    redistribution = load_script("scripts/check_dataset_redistribution.py")
    policy = _dataset_policy()

    repository_findings = redistribution.check_paths(
        [r".\data\restricted\sample.json", "src/package.py"],
        policy,
        mode="repository",
    )
    assert [finding.path for finding in repository_findings] == [
        "data/restricted/sample.json"
    ]
    assert (
        redistribution.check_paths(
            ["data/restricted/sample.json"], policy, mode="release"
        )
        == []
    )
    assert redistribution.check_release_root(tmp_path / "missing", policy) == []

    with pytest.raises(ValueError, match="unknown redistribution mode"):
        redistribution._is_allowed(policy["sources"][0], mode="unknown")


def test_dataset_redistribution_loads_policy_and_emits_json(
    load_script,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    redistribution = load_script("scripts/check_dataset_redistribution.py")
    provenance = tmp_path / "provenance.toml"
    provenance.write_text(
        """
[dataset_policy]
[[dataset_policy.sources]]
id = "restricted"
name = "Restricted corpus"
path_globs = ["data/restricted/**"]
redistribution_class = "restricted"
repository_redistribution = false
release_bundle_redistribution = false
""",
        encoding="utf-8",
    )

    returncode = redistribution.main(
        [
            "--provenance",
            str(provenance),
            "--path",
            "data/restricted/item.json",
            "--json",
        ]
    )
    payload = json.loads(capsys.readouterr().out)

    assert returncode == 1
    assert payload["overall_status"] == "blocking"
    assert payload["findings"][0]["source_id"] == "restricted"


@pytest.mark.parametrize(
    "content",
    (
        "",
        "[dataset_policy]\n",
        "[dataset_policy]\nsources = 'invalid'\n",
    ),
)
def test_dataset_redistribution_rejects_invalid_policy(
    load_script,
    tmp_path: Path,
    content: str,
) -> None:
    redistribution = load_script("scripts/check_dataset_redistribution.py")
    path = tmp_path / "provenance.toml"
    path.write_text(content, encoding="utf-8")

    with pytest.raises(ValueError, match=r"\[dataset_policy\]"):
        redistribution.load_dataset_policy(path)


def test_readability_ast_metrics_and_nested_qualified_names(load_script) -> None:
    readability = load_script("scripts/check_readability.py")
    tree = ast.parse(
        """
class Outer:
    def method(self, first, *, second):
        async def nested(value):
            return value
        return nested
"""
    )
    visitor = readability._QualifiedFunctionVisitor()
    visitor.visit(tree)
    class_node = cast(ast.ClassDef, tree.body[0])
    method = cast(ast.FunctionDef, class_node.body[0])

    assert readability._parameter_count(method) == 3
    assert readability._line_span(method) == 4
    assert [name for name, _node in visitor.functions] == [
        "Outer.method",
        "Outer.method.nested",
    ]


def test_readability_solar_debt_detects_added_removed_and_changed_items(
    load_script,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    readability = load_script("scripts/check_readability.py")
    baseline = {
        "long_functions": {"removed": 90, "changed": 91},
        "wide_functions": {},
        "oversized_modules": {},
        "any_modules": ["removed.py"],
        "wildcard_imports": [],
    }
    baseline_path = tmp_path / "solar-debt.json"
    baseline_path.write_text(json.dumps(baseline), encoding="utf-8")
    monkeypatch.setattr(readability, "SOLAR_DEBT_PATH", baseline_path)
    current = {
        "long_functions": {"changed": 92, "added": 100},
        "wide_functions": {},
        "oversized_modules": {},
        "any_modules": ["added.py"],
        "wildcard_imports": [],
    }

    failures = readability.check_solar_debt(current)

    assert any("removed without baseline update: removed" in item for item in failures)
    assert any("added: added=100" in item for item in failures)
    assert any("changed without baseline update" in item for item in failures)
    assert any("any_modules added: added.py" in item for item in failures)
    assert any(
        "any_modules removed without baseline update" in item for item in failures
    )


def test_loaded_quality_scripts_are_real_modules(
    load_script,
) -> None:
    modules: list[ModuleType] = [
        load_script("scripts/check_coupling.py"),
        load_script("scripts/check_current_docs.py"),
        load_script("scripts/check_dataset_redistribution.py"),
        load_script("scripts/check_readability.py"),
    ]

    assert all(module.__file__ for module in modules)
