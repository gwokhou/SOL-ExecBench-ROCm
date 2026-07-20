from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

from sol_execbench.core.data.definition import Definition
from sol_execbench.core.data.workload import Workload
from sol_execbench.core.solar_bridge import input_factory


@pytest.mark.parametrize("custom_entrypoint", [None, "make_inputs"])
def test_input_factory_binds_seed_safetensors_and_optional_custom_generator(
    tmp_path: Path, monkeypatch, custom_entrypoint: str | None
) -> None:
    custom = object()
    module = SimpleNamespace(make_inputs=custom)
    definition = cast(
        Definition,
        SimpleNamespace(custom_inputs_entrypoint=custom_entrypoint),
    )
    workload = cast(Workload, object())
    observed: dict[str, object] = {}
    entered_seeds: list[int] = []

    monkeypatch.setattr(
        input_factory,
        "load_safetensors",
        lambda *args, **kwargs: {"blob": "loaded"},
    )

    @contextmanager
    def isolated(seed: int):
        entered_seeds.append(seed)
        yield

    monkeypatch.setattr(input_factory, "isolated_torch_rng", isolated)

    def generate(definition_arg, workload_arg, device, **kwargs):
        observed.update(
            definition=definition_arg,
            workload=workload_arg,
            device=device,
            **kwargs,
        )
        return ["first", "second"]

    monkeypatch.setattr(input_factory, "gen_inputs", generate)

    factory = input_factory.build_input_factory(
        definition,
        workload,
        7,
        module,
        tmp_path / "problem",
        "hip:0",
    )

    assert factory(19) == ("first", "second")
    assert entered_seeds == [19]
    assert observed == {
        "definition": definition,
        "workload": workload,
        "device": "hip:0",
        "safe_tensors": {"blob": "loaded"},
        "custom_inputs_fn": custom if custom_entrypoint else None,
        "row_index": 7,
        "seed": 19,
    }
