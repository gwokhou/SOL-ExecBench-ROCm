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

"""Subcommand dispatch helpers for the SOL-ExecBench CLI."""

from __future__ import annotations

import os
import shlex
from collections.abc import Iterator
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from typing import Any

import click
from click.shell_completion import split_arg_string


@dataclass(frozen=True)
class SubcommandDispatch:
    result: Any


@dataclass(frozen=True)
class CompletionEnvironment:
    complete_var: str
    instruction: str


def _subcommand_for(name: str) -> click.Command | None:
    if name == "baseline":
        from . import baseline as cli_baseline

        return cli_baseline._baseline_cli
    if name == "contract":
        from . import metadata as cli_metadata

        return cli_metadata._contract_cli
    if name == "dataset":
        from . import dataset as cli_dataset

        return cli_dataset._dataset_cli
    if name == "doctor":
        from . import metadata as cli_metadata

        return cli_metadata._doctor_cli
    if name == "toolchain":
        from . import metadata as cli_metadata

        return cli_metadata._toolchain_cli
    return None


def _completion_environment(
    *,
    root_command: click.Command,
    prog_name: str | None,
    complete_var: str | None,
) -> CompletionEnvironment | None:
    if complete_var is None:
        complete_name = (prog_name or root_command.name or "").replace("-", "_")
        complete_name = complete_name.replace(".", "_")
        complete_var = f"_{complete_name}_COMPLETE".upper()

    instruction = os.environ.get(complete_var)
    if not instruction:
        return None
    return CompletionEnvironment(complete_var=complete_var, instruction=instruction)


def _completion_args(instruction: str) -> list[str]:
    words_text = os.environ.get("COMP_WORDS")
    if not words_text:
        return []

    words = split_arg_string(words_text)
    if instruction.startswith("fish"):
        incomplete = os.environ.get("COMP_CWORD", "")
        args = words[1:]
        if incomplete and args and args[-1] == incomplete:
            args.pop()
        return args

    try:
        word_index = int(os.environ.get("COMP_CWORD", "0"))
    except ValueError:
        return []
    return words[1:word_index]


@contextmanager
def _shift_completion_environment(instruction: str) -> Iterator[None]:
    words_text = os.environ.get("COMP_WORDS")
    if not words_text:
        yield
        return

    original_words = os.environ.get("COMP_WORDS")
    original_word_index = os.environ.get("COMP_CWORD")
    words = split_arg_string(words_text)
    if len(words) < 2:
        yield
        return

    shifted_words = [words[0], *words[2:]]
    try:
        os.environ["COMP_WORDS"] = shlex.join(shifted_words)
        if not instruction.startswith("fish"):
            try:
                word_index = int(original_word_index or "0")
            except ValueError:
                word_index = 0
            os.environ["COMP_CWORD"] = str(max(0, word_index - 1))
        yield
    finally:
        if original_words is None:
            os.environ.pop("COMP_WORDS", None)
        else:
            os.environ["COMP_WORDS"] = original_words
        if original_word_index is None:
            os.environ.pop("COMP_CWORD", None)
        else:
            os.environ["COMP_CWORD"] = original_word_index


def dispatch_subcommand(
    args: list[str],
    *,
    root_command: click.Command,
    prog_name: str | None,
    complete_var: str | None,
    standalone_mode: bool,
    windows_expand_args: bool,
    extra: dict[str, Any],
) -> SubcommandDispatch | None:
    completion_environment = _completion_environment(
        root_command=root_command,
        prog_name=prog_name,
        complete_var=complete_var,
    )
    if not args:
        if completion_environment is None:
            return None
        args = _completion_args(completion_environment.instruction)
        if not args:
            return None

    subcommand_name = args[0]
    subcommand = _subcommand_for(subcommand_name)
    if subcommand is None:
        return None

    subcommand_prog = f"{prog_name or root_command.name} {subcommand_name}"
    subcommand_complete_var = complete_var
    completion_context = (
        _shift_completion_environment(completion_environment.instruction)
        if completion_environment is not None
        else nullcontext()
    )
    if completion_environment is not None:
        subcommand_complete_var = completion_environment.complete_var

    with completion_context:
        return SubcommandDispatch(
            subcommand.main(
                args=args[1:],
                prog_name=subcommand_prog,
                complete_var=subcommand_complete_var,
                standalone_mode=standalone_mode,
                windows_expand_args=windows_expand_args,
                **extra,
            )
        )
