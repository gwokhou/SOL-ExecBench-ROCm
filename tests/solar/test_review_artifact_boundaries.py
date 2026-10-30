"""Review regressions at YAML and scalar-region boundaries."""

import pytest
import yaml.constructor

from solar.analysis.orojenesis.regions import _view_axis_map, _ViewMetadata
from solar.artifacts.document import parse_yaml_artifact


def test_yaml_merge_is_rejected_before_flattening(monkeypatch):
    lines = ["a0: &a0 {value: 1}"]
    for n in range(1, 13):
        lines.append(f"a{n}: &a{n} {{<<: [*a{n - 1}, *a{n - 1}]}}")
    monkeypatch.setattr(
        yaml.constructor.SafeConstructor,
        "flatten_mapping",
        lambda *_: pytest.fail("flattened merge before validation"),
    )
    with pytest.raises(ValueError, match="merge"):
        parse_yaml_artifact(
            "merge.yaml", "\n".join(lines).encode(), max_values=100, max_depth=8
        )


def test_yaml_plain_alias_and_literal_merge_spelling_remain_valid():
    result = parse_yaml_artifact(
        "alias.yaml", b'a: &a [1, 2]\nb: *a\n"<<": 3\n'
    )
    assert result["a"] == result["b"] == [1, 2]
    assert result["<<"] == 3


@pytest.mark.parametrize("target", ["view", "squeeze", "unsqueeze"])
@pytest.mark.parametrize(
    "input_shape, output_shape", [([], [1]), ([1], []), ([], [])]
)
def test_scalar_views_are_not_matrix_regions(target, input_shape, output_shape):
    metadata = _ViewMetadata(
        target=target,
        semantic={},
        input_name="x",
        output_name="y",
        input_shape=input_shape,
        output_shape=output_shape,
        dtype="fp32",
    )
    assert _view_axis_map(metadata) is None
