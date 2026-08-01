# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Common utilities and base classes for data models."""

import json
from collections.abc import Callable, Mapping
from typing import Annotated, ClassVar, Self, cast

from pydantic import BaseModel, ConfigDict, Field
from pydantic.config import ExtraValues

NonEmptyString = Annotated[str, Field(min_length=1)]
"""Type alias for non-empty strings with minimum length of 1."""

NonNegativeInt = Annotated[int, Field(ge=0)]
"""Type alias for non-negative integers."""


class BaseModelWithDocstrings(BaseModel):
    """Base model with the attribute docstrings being extracted to the model JSON schema."""

    model_config = ConfigDict(
        extra="forbid",
        use_attribute_docstrings=True,
        validate_assignment=True,
    )


class StrictArtifactModel(BaseModelWithDocstrings):
    """Base class for stable, strictly validated JSON artifacts.

    ``strict`` here means that unknown fields are rejected. Pydantic's scalar
    coercion remains enabled unless a schema family explicitly opts into
    ``ConfigDict(strict=True)``.

    Artifact models are deliberately mutable unless an individual schema opts in
    to freezing.  This keeps the base suitable for incremental report builders
    while ensuring that parser boundaries reject misspelled or future fields.
    """


class FrozenArtifactModel(StrictArtifactModel):
    """Immutable stable artifact model with strict field ownership."""

    model_config = ConfigDict(frozen=True)


class CurrentSchemaMixin:
    """Exact-current parsing policy shared by versioned artifact bases."""

    current_schema_version: ClassVar[str | int]

    @classmethod
    def _require_current_schema(cls, value: object) -> None:
        if isinstance(value, cls):
            return
        if not isinstance(value, Mapping):
            raise ValueError(f"{cls.__name__} must be an object")
        observed = value.get("schema_version")
        if observed != cls.current_schema_version:
            raise ValueError(
                f"{cls.__name__} requires schema_version="
                f"{cls.current_schema_version!r}",
            )

    @classmethod
    def model_validate(
        cls,
        obj: object,
        *,
        strict: bool | None = None,
        extra: ExtraValues | None = None,
        from_attributes: bool | None = None,
        context: object | None = None,
        by_alias: bool | None = None,
        by_name: bool | None = None,
    ) -> Self:
        """Require the current schema before normal Pydantic validation."""
        cls._require_current_schema(obj)
        validator = cast(
            Callable[..., Self],
            getattr(  # noqa: B009 -- cooperative call hidden from mixin typing
                super(),
                "model_validate",
            ),
        )
        return validator(
            obj,
            strict=strict,
            extra=extra,
            from_attributes=from_attributes,
            context=context,
            by_alias=by_alias,
            by_name=by_name,
        )

    @classmethod
    def model_validate_json(
        cls,
        json_data: str | bytes | bytearray,
        *,
        strict: bool | None = None,
        extra: ExtraValues | None = None,
        context: object | None = None,
        by_alias: bool | None = None,
        by_name: bool | None = None,
    ) -> Self:
        """Parse JSON only after requiring the exact current version."""
        value = json.loads(json_data)
        return cls.model_validate(
            value,
            strict=strict,
            extra=extra,
            context=context,
            by_alias=by_alias,
            by_name=by_name,
        )


class CurrentSchemaModel(CurrentSchemaMixin, StrictArtifactModel):
    """Base for one current, versioned first-party wire contract.

    Direct construction is reserved for trusted producers and may use a field
    default for the current version. Parsing entry points require the version
    to be explicitly present and exactly current before Pydantic reads any
    business fields.
    """


class CurrentFrozenSchemaModel(CurrentSchemaMixin, FrozenArtifactModel):
    """Immutable current-version wire contract."""
