"""Untrusted YAML is bounded before parsing and alias expansion."""

import pytest

from solar.artifacts.document import load_yaml_artifact, parse_yaml_artifact


def test_alias_expansion_cannot_allocate_an_exponential_tree():
    lines = ["a: &a [1, 2]"]
    previous = "a"
    for index in range(10):
        current = f"n{index}"
        lines.append(
            f"{current}: &{current} [{', '.join([f'*{previous}'] * 8)}]"
        )
        previous = current
    payload = "\n".join(lines).encode()
    with pytest.raises(ValueError, match="expanded value budget"):
        parse_yaml_artifact("aliases.yaml", payload, max_values=1000)


def test_nesting_is_rejected_before_parser_recursion():
    payload = ("value: " + "[" * 10000 + "1" + "]" * 10000).encode()
    with pytest.raises(ValueError, match="nesting depth budget"):
        parse_yaml_artifact("nested.yaml", payload, max_depth=8)


def test_recursive_aliases_are_invalid_artifact_values():
    with pytest.raises(ValueError, match="nesting depth budget"):
        parse_yaml_artifact("recursive.yaml", b"value: &x [*x]", max_depth=8)


def test_small_shared_yaml_values_and_large_nominal_shape_are_inexpensive():
    payload = (
        b"shape: &shape [1000000000000000000000000000000]\nother: *shape\n"
    )
    document = parse_yaml_artifact("small.yaml", payload, max_values=10)
    assert document["shape"] == document["other"] == [10**30]


def test_file_byte_limit_is_checked_on_bytes_read(tmp_path):
    path = tmp_path / "large.yaml"
    path.write_bytes(b"value: [" + b" " * 100)
    with pytest.raises(ValueError, match="exceeds 16 bytes"):
        load_yaml_artifact(path, max_bytes=16)


def test_invalid_budget_does_not_open_a_file(tmp_path):
    with pytest.raises(ValueError, match="budget must be positive"):
        load_yaml_artifact(tmp_path / "absent.yaml", max_bytes=-1)


def test_json_exponents_are_numbers_but_quoted_strings_remain_strings(tmp_path):
    path = tmp_path / "analysis.yaml"
    path.write_text('{"value": 6e-10, "string": "6e-10", "large": 1e30}')
    result = load_yaml_artifact(path)
    assert result["value"] == 6e-10
    assert result["large"] == 1e30
    assert result["string"] == "6e-10"
