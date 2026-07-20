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


"""Tests for sol_execbench.core.bench.utils."""

import torch

from sol_execbench.core.bench.utils import call_and_collect_outputs
from sol_execbench_type_helpers import make_definition


def test_call_and_collect_outputs_normalizes_return_value_outputs():
    outputs = call_and_collect_outputs(
        lambda x: x + 1,
        [torch.tensor([1.0, 2.0])],
        destination_passing_style=False,
        definition=None,
        resolved_axes={},
        device="cpu",
        output_names=["out"],
        output_dtypes={"out": torch.float32},
    )

    assert len(outputs) == 1
    assert torch.equal(outputs[0], torch.tensor([2.0, 3.0]))


def test_call_and_collect_outputs_supports_destination_passing_style():
    definition = make_definition(
        name="dps_demo",
        op_type="dps_demo",
        axes={"N": {"type": "var"}},
        inputs={"x": {"shape": ["N"], "dtype": "float32"}},
        outputs={"out": {"shape": ["N"], "dtype": "float32"}},
        reference="def run(x):\n    return x",
    )

    def run(x, out):
        out.copy_(x + 1)

    outputs = call_and_collect_outputs(
        run,
        [torch.tensor([1.0, 2.0])],
        destination_passing_style=True,
        definition=definition,
        resolved_axes={"N": 2},
        device="cpu",
        output_names=["out"],
        output_dtypes={"out": torch.float32},
    )

    assert len(outputs) == 1
    assert torch.equal(outputs[0], torch.tensor([2.0, 3.0]))
