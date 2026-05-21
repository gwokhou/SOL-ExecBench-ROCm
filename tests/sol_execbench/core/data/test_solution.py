# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for solution schema ROCm language, hardware, and entry point validation."""

import pytest
from pydantic import ValidationError

from sol_execbench.core.data.solution import (
    BuildSpec,
    CompileOptions,
    SupportedHardware,
    SupportedLanguages,
)


def _make_spec(**overrides):
    base = dict(
        languages=["triton"],
        target_hardware=["LOCAL"],
        entry_point="kernel.py::run",
    )
    base.update(overrides)
    return BuildSpec(**base)


PYTHON_LANGUAGES = ["pytorch", "triton"]
NATIVE_LANGUAGES = ["hip_cpp", "hipblas", "miopen", "ck", "rocwmma"]
CDNA3_TARGETS = ["gfx940", "gfx941", "gfx942"]
LEGACY_LANGUAGE_REPLACEMENTS = [
    ("cuda_cpp", "hip_cpp"),
    ("cutlass", "ck or rocwmma"),
    ("cudnn", "miopen"),
    ("cudnn_frontend", "miopen"),
    ("cublas", "hipblas"),
    ("cute_dsl", "Phase 4"),
    ("cutile", "Phase 4"),
]


class TestLanguageValidation:
    """BuildSpec accepts only ROCm-native language categories."""

    @pytest.mark.parametrize("langs", [["pytorch"], ["triton"], ["pytorch", "triton"]])
    def test_pure_python_languages_accepted(self, langs):
        spec = _make_spec(languages=langs)
        assert set(spec.languages) == {SupportedLanguages(lg) for lg in langs}

    @pytest.mark.parametrize(
        "langs",
        [
            ["hip_cpp"],
            ["hipblas"],
            ["miopen"],
            ["ck"],
            ["rocwmma"],
            ["hip_cpp", "hipblas"],
            ["hip_cpp", "ck", "rocwmma"],
        ],
    )
    def test_native_rocm_languages_accepted(self, langs):
        spec = _make_spec(languages=langs, entry_point="kernel.hip::run")
        assert set(spec.languages) == {SupportedLanguages(lg) for lg in langs}

    @pytest.mark.parametrize(
        "langs",
        [
            ["pytorch", "hip_cpp"],
            ["triton", "hipblas"],
            ["pytorch", "triton", "ck"],
        ],
    )
    def test_mixed_python_and_native_languages_rejected(self, langs):
        with pytest.raises(
            ValidationError, match="HIP/C\\+\\+ and Python cannot be mixed"
        ):
            _make_spec(languages=langs, entry_point="kernel.hip::run")

    @pytest.mark.parametrize(("legacy", "replacement"), LEGACY_LANGUAGE_REPLACEMENTS)
    def test_legacy_cuda_nvidia_languages_rejected_with_guidance(
        self, legacy, replacement
    ):
        with pytest.raises(ValidationError) as exc_info:
            _make_spec(languages=[legacy], entry_point="kernel.hip::run")

        message = str(exc_info.value)
        assert legacy in message
        assert replacement in message

    @pytest.mark.parametrize("lang", [lg.value for lg in SupportedLanguages])
    def test_every_remaining_language_accepted_alone(self, lang):
        ext = ".hip" if lang in NATIVE_LANGUAGES else ".py"
        spec = _make_spec(languages=[lang], entry_point=f"kernel{ext}::run")
        assert spec.languages == [SupportedLanguages(lang)]


class TestEntryPointSuffixValidation:
    """BuildSpec rejects entry points whose suffix does not match language category."""

    @pytest.mark.parametrize("lang", PYTHON_LANGUAGES)
    def test_python_language_rejects_hip_entry(self, lang):
        with pytest.raises(ValidationError, match="require a .py entry point"):
            _make_spec(languages=[lang], entry_point="kernel.hip::run")

    @pytest.mark.parametrize("lang", PYTHON_LANGUAGES)
    def test_python_language_rejects_cpp_entry(self, lang):
        with pytest.raises(ValidationError, match="require a .py entry point"):
            _make_spec(languages=[lang], entry_point="kernel.cpp::run")

    @pytest.mark.parametrize("lang", NATIVE_LANGUAGES)
    def test_native_language_rejects_py_entry(self, lang):
        with pytest.raises(
            ValidationError, match="require a .hip or C/C\\+\\+ entry point"
        ):
            _make_spec(languages=[lang], entry_point="kernel.py::run")

    @pytest.mark.parametrize("lang", NATIVE_LANGUAGES)
    def test_native_language_rejects_cu_entry(self, lang):
        with pytest.raises(ValidationError, match="\\.hip"):
            _make_spec(languages=[lang], entry_point="kernel.cu::run")

    @pytest.mark.parametrize(
        "suffix", [".hip", ".cpp", ".cc", ".cxx", ".c", ".h", ".hpp"]
    )
    def test_native_language_accepts_valid_suffixes(self, suffix):
        spec = _make_spec(languages=["hip_cpp"], entry_point=f"kernel{suffix}::run")
        assert spec.languages == [SupportedLanguages.HIP_CPP]

    @pytest.mark.parametrize("lang", PYTHON_LANGUAGES)
    def test_python_language_accepts_py_entry(self, lang):
        spec = _make_spec(languages=[lang], entry_point="kernel.py::run")
        assert spec.languages == [SupportedLanguages(lang)]


class TestHardwareAndCompileOptions:
    """ROCm hardware targets and compile options are strict and HIP-named."""

    def test_supported_hardware_contains_rdna4_cdna3_and_local(self):
        assert SupportedHardware.LOCAL.value == "LOCAL"
        assert SupportedHardware.GFX1200.value == "gfx1200"
        assert SupportedHardware.GFX940.value == "gfx940"
        assert SupportedHardware.GFX941.value == "gfx941"
        assert SupportedHardware.GFX942.value == "gfx942"
        assert not hasattr(SupportedHardware, "B200")

    @pytest.mark.parametrize("target", ["LOCAL", "gfx1200", *CDNA3_TARGETS])
    def test_rocm_hardware_targets_accepted(self, target):
        spec = _make_spec(target_hardware=[target])
        assert spec.target_hardware == [SupportedHardware(target)]

    def test_unknown_gfx_target_rejected(self):
        with pytest.raises(ValidationError, match="gfx950"):
            _make_spec(target_hardware=["gfx950"])

    @pytest.mark.parametrize("target", CDNA3_TARGETS)
    def test_cdna3_targets_are_schema_supported_not_hardware_validated(self, target):
        spec = _make_spec(target_hardware=[target])
        assert spec.target_hardware == [SupportedHardware(target)]

    def test_compile_options_defaults_are_hip_minimal(self):
        opts = CompileOptions()
        assert opts.cflags == []
        assert opts.hip_cflags == ["-O3"]
        assert opts.ld_flags == []
        assert not hasattr(opts, "cuda_cflags")

    def test_cuda_cflags_rejected_with_hip_cflags_guidance(self):
        with pytest.raises(ValidationError) as exc_info:
            _make_spec(
                languages=["hip_cpp"],
                entry_point="kernel.hip::run",
                compile_options={"cuda_cflags": ["-O3"]},
            )

        message = str(exc_info.value)
        assert "cuda_cflags" in message
        assert "hip_cflags" in message

    def test_hip_compile_options_accepted(self):
        spec = _make_spec(
            languages=["hip_cpp"],
            entry_point="kernel.hip::run",
            compile_options={"hip_cflags": ["--offload-arch=gfx1200"]},
        )
        assert spec.compile_options is not None
        assert spec.compile_options.hip_cflags == ["--offload-arch=gfx1200"]
