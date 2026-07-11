# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os

import pytest

from sol_execbench.tools.amd_isa import open_isa


pytestmark = [
    pytest.mark.cpp,
    pytest.mark.requires_network,
    pytest.mark.skipif(
        os.environ.get("SOL_EXECBENCH_RUN_NETWORK_TESTS") != "1",
        reason="set SOL_EXECBENCH_RUN_NETWORK_TESTS=1 to fetch the pinned ISA XML",
    ),
]


def test_rdna4_decoder_and_explorer_use_the_same_loaded_spec() -> None:
    with open_isa("gfx1200") as isa:
        assert isa.explorer.architecture()["name"] == "AMD RDNA 4"
        assert isa.decoder.get_instruction("V_MOV_B32")["name"] == "V_MOV_B32"
        assert isa.explorer.get_instruction("V_MOV_B32")["name"] == "V_MOV_B32"
