"""Source-storage observations captured before the reference executes."""

from __future__ import annotations

from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict, StrictStr, model_validator

from solar.errors import SourceInputBindingError
from solar.precision import BYTES_PER_ELEMENT, normalize_dtype
from solar.storage import NonnegativeInt, injective_strides, same_tensor_storage
from solar.types import DynamicValue


class SourceTensorStorage(BaseModel):
    """One argument's layout with a local, address-free storage group."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    source_index: NonnegativeInt
    storage_group: NonnegativeInt
    shape: tuple[NonnegativeInt, ...]
    strides: tuple[NonnegativeInt, ...]
    offset: NonnegativeInt
    storage_elements: NonnegativeInt
    dtype: StrictStr
    device: StrictStr

    @model_validator(mode="after")
    def validate_layout(self) -> SourceTensorStorage:
        """Reject malformed layout evidence; overlap remains a separate status."""
        if len(self.shape) != len(self.strides):
            raise ValueError("source layout ranks differ")
        if self.dtype not in BYTES_PER_ELEMENT:
            raise ValueError("source layout dtype is not canonical")
        extent = sum(
            (n - 1) * s for n, s in zip(self.shape, self.strides, strict=True)
        )
        if (
            0 not in self.shape
            and self.offset + extent >= self.storage_elements
        ):
            raise ValueError("source layout extends beyond storage")
        if self.offset > self.storage_elements:
            raise ValueError("source storage offset is out of bounds")
        return self

    @property
    def injective(self) -> bool:
        """Return whether element uniqueness is proven for this input."""
        return injective_strides(self.shape, self.strides)


class SourceStorage(BaseModel):
    """Top-level tensor source layouts for one concrete trace input."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    inputs: tuple[SourceTensorStorage, ...]

    @model_validator(mode="after")
    def validate_indices(self) -> SourceStorage:
        """Require unique argument bindings in stable argument order."""
        indices = [item.source_index for item in self.inputs]
        if indices != sorted(set(indices)):
            raise ValueError(
                "source storage argument indices are not unique and ordered"
            )
        return self

    def require_independent(self, indices: Sequence[int]) -> None:
        """Reject missing, potentially overlapping, or shared used inputs."""
        selected = [
            item for item in self.inputs if item.source_index in indices
        ]
        if {item.source_index for item in selected} != set(indices):
            raise SourceInputBindingError(
                "source storage evidence omits a used tensor input"
            )
        if any(not item.injective for item in selected):
            raise SourceInputBindingError(
                "source tensor element uniqueness is not proven"
            )
        groups = [item.storage_group for item in selected]
        if len(groups) != len(set(groups)):
            raise SourceInputBindingError(
                "cross-input shared storage is outside the independent-input domain"
            )

    def require_layout(
        self, observed: SourceStorage, indices: Sequence[int]
    ) -> None:
        """Check independence and the layout conditions before input cloning."""
        self.require_independent(indices)
        observed.require_independent(indices)
        expected = {item.source_index: item for item in self.inputs}
        for item in observed.inputs:
            if item.source_index not in indices:
                continue
            bound = expected[item.source_index]
            fields = (
                "shape",
                "strides",
                "offset",
                "storage_elements",
                "dtype",
                "device",
            )
            if any(
                getattr(item, name) != getattr(bound, name) for name in fields
            ):
                raise SourceInputBindingError(
                    "source tensor layout differs from the traced input"
                )


def capture_source_storage(inputs: Sequence[DynamicValue]) -> SourceStorage:
    """Record tensor layouts and storage groups without persisting addresses."""
    import torch

    tensors = []
    records = []
    for index, value in enumerate(inputs):
        if not isinstance(value, torch.Tensor):
            continue
        group = next(
            (
                i
                for i, tensor in enumerate(tensors)
                if same_tensor_storage(value, tensor)
            ),
            len(tensors),
        )
        if group == len(tensors):
            tensors.append(value)
        records.append(
            SourceTensorStorage(
                source_index=index,
                storage_group=group,
                shape=tuple(value.shape),
                strides=tuple(value.stride()),
                offset=value.storage_offset(),
                storage_elements=value.untyped_storage().nbytes()
                // value.element_size(),
                dtype=normalize_dtype(value.dtype),
                device=str(value.device),
            )
        )
    return SourceStorage(inputs=tuple(records))
