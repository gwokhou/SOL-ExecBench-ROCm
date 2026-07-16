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


def test_rdna4_decoder_classifies_real_llvm_wmma_disassembly() -> None:
    text = (
        "v_wmma_f32_16x16x16_bf16 v[1:8], v[9:12], v[9:12], v[1:8]  "
        "// 000000001650: CC414001 1C061309\n"
    )

    with open_isa("gfx1200") as isa:
        decoded = isa.decoder.decode_disassembly(text)

    instruction = decoded[0][0]
    assert instruction["name"] == "V_WMMA_F32_16X16X16_BF16"
    assert "WMMA" in instruction["functional"]["subgroups"]
