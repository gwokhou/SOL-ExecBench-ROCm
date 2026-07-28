"""Lazy PyTorch executor for SOLAR IR graphs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from solar.ir.contracts import IRLifecycle, normalize_ir_kind
from solar.types import DynamicValue
from solar.verification.errors import IRExecutionError


def _shapes(layer: Mapping[str, DynamicValue]) -> list[tuple[int, ...]]:
    outputs = (layer.get("tensor_shapes") or {}).get("outputs") or []
    return [tuple(int(dimension) for dimension in shape) for shape in outputs]


class IRGraphExecutor:
    """Execute the exact subset of SOLAR IR understood by the built-in verifier."""

    def __init__(
        self,
        graph: Mapping[str, DynamicValue],
        lifecycle: IRLifecycle,
        *,
        check_shapes: bool = True,
    ) -> None:
        """Validate and configure an executable semantic graph."""
        try:
            lifecycle.validate(graph)
        except ValueError as exc:
            raise IRExecutionError(str(exc)) from exc
        discriminator = graph.get("ir_kind")
        if discriminator is None:
            raise IRExecutionError(
                "IR graph has no explicit ir_kind discriminator",
            )
        self.ir_kind = normalize_ir_kind(str(discriminator))
        if self.ir_kind is not lifecycle.kind:
            raise IRExecutionError(
                f"IR lifecycle {lifecycle.kind.value!r} cannot execute "
                f"{self.ir_kind.value!r}",
            )
        self.lifecycle = lifecycle
        layers = graph.get("layers") or {}
        if not isinstance(layers, Mapping) or not layers:
            raise IRExecutionError("IR graph has no layers")
        self.layers = dict(layers)
        declared_outputs = graph.get("outputs")
        if declared_outputs is None:
            declared_outputs = (graph.get("graph_signature") or {}).get(
                "joint_outputs",
            )
        self.declared_outputs = (
            [str(name) for name in declared_outputs]
            if isinstance(declared_outputs, list)
            else None
        )
        self.check_shapes = check_shapes
        self._validate_layers()

    def _validate_layers(self) -> None:
        for layer_id, layer in self.layers.items():
            if not isinstance(layer, Mapping):
                raise IRExecutionError(f"layer {layer_id} is not a mapping")
            if str(layer.get("type", "")).lower() == "start":
                continue
            dtypes = layer.get("tensor_dtypes") or {}
            shapes = layer.get("tensor_shapes") or {}
            for side in ("inputs", "outputs"):
                if len(dtypes.get(side) or []) != len(shapes.get(side) or []):
                    raise IRExecutionError(
                        f"layer {layer_id} lacks explicit per-tensor "
                        "dtype metadata",
                    )

    def __call__(self, *inputs: DynamicValue) -> DynamicValue:
        """Execute the graph for positional inputs and return declared outputs."""
        values, start_ids, produced, input_index = self._bind_start_inputs(
            inputs,
        )
        input_index = self._bind_external_inputs(
            inputs,
            input_index,
            start_ids,
            produced,
            values,
        )
        if input_index != len(inputs):
            raise IRExecutionError(
                f"graph consumes {input_index} inputs but reference supplied "
                f"{len(inputs)}",
            )
        consumed = self._execute_pending(values, start_ids)
        return self._terminal_outputs(values, produced, consumed)

    def _bind_start_inputs(
        self,
        inputs: Sequence[DynamicValue],
    ) -> tuple[
        dict[str, DynamicValue],
        list[str],
        set[str],
        int,
    ]:
        values: dict[str, DynamicValue] = {}
        produced = {
            name
            for layer in self.layers.values()
            for name in ((layer.get("tensor_names") or {}).get("outputs") or [])
        }
        start_ids = [
            layer_id
            for layer_id, layer in self.layers.items()
            if str(layer.get("type", "")).lower() == "start"
        ]
        input_index = 0
        for layer_id in start_ids:
            names = (self.layers[layer_id].get("tensor_names") or {}).get(
                "outputs",
            ) or []
            for name in names:
                if input_index >= len(inputs):
                    raise IRExecutionError(
                        "not enough inputs for graph start tensors",
                    )
                values[str(name)] = inputs[input_index]
                input_index += 1
        return values, start_ids, produced, input_index

    def _bind_external_inputs(
        self,
        inputs: Sequence[DynamicValue],
        input_index: int,
        start_ids: Sequence[str],
        produced: set[str],
        values: dict[str, DynamicValue],
    ) -> int:
        external_names: list[str] = []
        for layer_id, layer in self.layers.items():
            if layer_id in start_ids:
                continue
            for name in (layer.get("tensor_names") or {}).get("inputs") or []:
                if (
                    name not in produced
                    and name not in values
                    and name not in external_names
                ):
                    external_names.append(str(name))
        for name in external_names:
            if input_index >= len(inputs):
                raise IRExecutionError(f"missing external tensor {name}")
            values[name] = inputs[input_index]
            input_index += 1
        return input_index

    def _execute_pending(
        self,
        values: dict[str, DynamicValue],
        start_ids: Sequence[str],
    ) -> set[str]:
        pending = {
            key: value
            for key, value in self.layers.items()
            if key not in start_ids
        }
        consumed: set[str] = set()
        while pending:
            progressed = False
            for layer_id, layer in list(pending.items()):
                names = _tensor_names(layer, "inputs")
                if not all(name in values for name in names):
                    continue
                result = self._execute_layer(
                    layer_id,
                    layer,
                    [values[name] for name in names],
                )
                self._store_outputs(layer_id, layer, result, values)
                consumed.update(names)
                del pending[layer_id]
                progressed = True
            if not progressed:
                missing = {
                    layer_id: [
                        name
                        for name in _tensor_names(layer, "inputs")
                        if name not in values
                    ]
                    for layer_id, layer in pending.items()
                }
                raise IRExecutionError(
                    f"unresolvable graph dependencies: {missing}",
                )
        return consumed

    def _store_outputs(
        self,
        layer_id: str,
        layer: Mapping[str, DynamicValue],
        result: DynamicValue,
        values: dict[str, DynamicValue],
    ) -> None:
        import torch

        output_names = _tensor_names(layer, "outputs")
        results = (
            list(result) if isinstance(result, (tuple, list)) else [result]
        )
        if len(output_names) != len(results):
            raise IRExecutionError(
                f"layer {layer_id} returned {len(results)} outputs, "
                f"expected {len(output_names)}",
            )
        expected_shapes = _shapes(layer)
        for index, (output_name, output) in enumerate(
            zip(output_names, results, strict=True),
        ):
            if not isinstance(output, torch.Tensor):
                raise IRExecutionError(
                    f"layer {layer_id} output {index} is not a tensor",
                )
            if (
                self.check_shapes
                and tuple(output.shape) != expected_shapes[index]
            ):
                raise IRExecutionError(
                    f"layer {layer_id} output {index} produced "
                    f"{tuple(output.shape)}, expected {expected_shapes[index]}",
                )
            values[output_name] = output

    def _terminal_outputs(
        self,
        values: dict[str, DynamicValue],
        produced: set[str],
        consumed: set[str],
    ) -> DynamicValue:
        if self.declared_outputs is not None:
            missing = [
                name for name in self.declared_outputs if name not in values
            ]
            if missing:
                raise IRExecutionError(
                    f"graph declares unavailable outputs: {missing}",
                )
            ordered = self.declared_outputs
        else:
            terminal = {
                name
                for name in produced
                if name not in consumed and name in values
            }
            ordered = [
                name
                for layer in self.layers.values()
                for name in _tensor_names(layer, "outputs")
                if name in terminal
            ]
        if not ordered:
            raise IRExecutionError("IR graph has no terminal output")
        outputs = tuple(values[name] for name in ordered)
        return outputs[0] if len(outputs) == 1 else outputs

    def _execute_layer(
        self,
        layer_id: str,
        layer: Mapping[str, DynamicValue],
        operands: Sequence[DynamicValue],
    ) -> DynamicValue:
        return self.lifecycle.execute(layer_id, layer, operands, _shapes(layer))


def _tensor_names(
    layer: Mapping[str, DynamicValue],
    side: str,
) -> list[str]:
    return [
        str(name)
        for name in ((layer.get("tensor_names") or {}).get(side) or [])
    ]
