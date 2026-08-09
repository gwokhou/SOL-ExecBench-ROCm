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

"""Unified JSON encoding/decoding utilities for JSON and Pydantic models."""

import json
import os
import tempfile
from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


def canonical_json_bytes(value: Any, *, sort_keys: bool = True) -> bytes:
    """Return the canonical UTF-8 representation used by JSON artifacts."""
    return (
        json.dumps(
            value,
            indent=2,
            sort_keys=sort_keys,
            allow_nan=False,
        )
        + "\n"
    ).encode()


def atomic_write_json_value(
    path: str | Path,
    value: Any,
    *,
    sort_keys: bool = True,
) -> None:
    """Atomically write JSON, sorting keys unless field order is contractual."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(
                canonical_json_bytes(value, sort_keys=sort_keys).decode()
            )
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_write_jsonl_values(
    path: str | Path,
    values: list[Any],
) -> None:
    """Atomically write deterministic JSON Lines in the destination directory."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            for value in values:
                payload = (
                    value.model_dump(mode="json")
                    if isinstance(value, BaseModel)
                    else value
                )
                handle.write(
                    json.dumps(payload, sort_keys=True, allow_nan=False) + "\n",
                )
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def save_json_file(model: BaseModel, path: str | Path) -> None:
    """Save a Pydantic BaseModel object to a JSON file.

    Parameters
    ----------
    model : BaseModel
        The Pydantic BaseModel instance to be serialized and saved.
    path : Union[str, Path]
        The file path where the JSON will be saved. Parent directories
        will be created if they don't exist.

    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(model.model_dump_json(indent=2, exclude_unset=True))


def load_json_value(path: str | Path) -> Any:
    """Load any JSON value from *path*."""
    path = Path(path)
    return json.loads(path.read_text(encoding="utf-8"))


def load_json_file[T: BaseModel](model_cls: type[T], path: str | Path) -> T:
    """Load a Pydantic BaseModel object from a JSON file.

    Parameters
    ----------
    model_cls : Type[BaseModel]
        The Pydantic BaseModel class to instantiate from the JSON data.
    path : Union[str, Path]
        The file path of the JSON file to load.

    Returns:
    -------
    BaseModel
        An instance of the specified BaseModel class populated with
        data from the JSON file.

    Raises:
    ------
    FileNotFoundError
        If the specified file does not exist.
    ValidationError
        If the JSON data doesn't match the BaseModel schema.

    """
    return model_cls.model_validate_json(Path(path).read_text(encoding="utf-8"))


def save_jsonl_file(objects: list[BaseModel], path: str | Path) -> None:
    """Save Pydantic models to a JSONL file.

    Each object is serialized as a separate JSON object on its own line.

    Parameters
    ----------
    objects : list[BaseModel]
        A list of Pydantic BaseModel instances to be serialized and saved.
    path : Union[str, Path]
        The file path where the JSONL will be saved. Parent directories
        will be created if they don't exist.

    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        object_strs = [obj.model_dump_json(indent=None) for obj in objects]
        output_str = "\n".join(object_strs) + "\n"
        f.write(output_str)


def load_jsonl_file[T: BaseModel](
    model_cls: type[T], path: str | Path
) -> list[T]:
    """Load Pydantic models from a JSONL file.

    Each nonempty line must contain a JSON object that can be deserialized
    into the specified model class.

    Parameters
    ----------
    model_cls : Type[BaseModel]
        The Pydantic BaseModel class to instantiate for each JSON object.
    path : Union[str, Path]
        The file path of the JSONL file to load.

    Returns:
    -------
    list[BaseModel]
        A list of instances of the specified BaseModel class, one for
        each valid JSON line in the file.

    Raises:
    ------
    FileNotFoundError
        If the specified file does not exist.
    ValidationError
        If any JSON line doesn't match the BaseModel schema.

    """
    out: list[T] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if line:
                out.append(model_cls.model_validate_json(line))
    return out


def append_jsonl_file(objects: list[BaseModel], path: str | Path) -> None:
    """Append Pydantic models to a JSONL file.

    Each object is serialized as a separate JSON object and appended on its
    own line.

    Parameters
    ----------
    objects : list[BaseModel]
        A list of Pydantic BaseModel instances to be serialized and appended.
    path : Union[str, Path]
        The file path of the JSONL file to append to. Parent directories
        will be created if they don't exist.

    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    needs_newline_prefix = False
    if path.exists() and path.stat().st_size > 0:
        with open(path, "rb") as f:
            f.seek(-1, 2)
            last_char = f.read(1)
            needs_newline_prefix = last_char != b"\n"

    with open(path, "a", encoding="utf-8") as f:
        if needs_newline_prefix:
            f.write("\n")

        object_strs = [obj.model_dump_json(indent=None) for obj in objects]
        output_str = "\n".join(object_strs) + "\n"
        f.write(output_str)
