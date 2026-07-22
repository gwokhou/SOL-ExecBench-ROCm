# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Tests for solution schema ROCm language, hardware, and entry point validation."""

from typing import Any

import pytest
from pydantic import ValidationError

from sol_execbench.core.data.solution import (
    BuildSpec,
    CompileOptions,
    Solution,
    SourceFile,
    SupportedHardware,
    SupportedLanguages,
)
from sol_execbench_type_helpers import make_build_spec


def _make_spec(**overrides):
    base = dict(
        languages=["triton"],
        target_hardware=["LOCAL"],
        entry_point="kernel.py::run",
    )
    base.update(overrides)
    return make_build_spec(**base)


PYTHON_LANGUAGES = ["pytorch", "triton"]
NATIVE_LANGUAGES = ["hip_cpp", "hipblas", "miopen", "ck", "rocwmma"]
CDNA3_TARGETS = ["gfx940", "gfx941", "gfx942"]
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

    def test_unknown_language_is_rejected_by_closed_enum(self):
        with pytest.raises(ValidationError, match="unsupported_language"):
            _make_spec(
                languages=["unsupported_language"], entry_point="kernel.hip::run"
            )

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
        assert SupportedHardware.GFX1150.value == "gfx1150"
        assert SupportedHardware.GFX940.value == "gfx940"
        assert SupportedHardware.GFX941.value == "gfx941"
        assert SupportedHardware.GFX942.value == "gfx942"
        assert not hasattr(SupportedHardware, "B200")

    @pytest.mark.parametrize(
        "target", ["LOCAL", "gfx1150", "gfx1200", *CDNA3_TARGETS]
    )
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

    @pytest.mark.parametrize("target", CDNA3_TARGETS)
    def test_cdna3_native_offload_arch_metadata_is_accepted_without_validation_claim(
        self, target
    ):
        spec = _make_spec(
            languages=["hip_cpp"],
            target_hardware=[target],
            entry_point="kernel.hip::run",
            compile_options={"hip_cflags": ["-O3", f"--offload-arch={target}"]},
        )

        assert spec.target_hardware == [SupportedHardware(target)]
        assert spec.compile_options is not None
        assert spec.compile_options.hip_cflags == [
            "-O3",
            f"--offload-arch={target}",
        ]

    def test_compile_options_defaults_are_hip_minimal(self):
        opts = CompileOptions()
        assert opts.cflags == []
        assert opts.hip_cflags == ["-O3"]
        assert opts.ld_flags == []
    def test_unknown_compile_option_is_rejected(self):
        with pytest.raises(ValidationError, match="extra_forbidden"):
            _make_spec(
                languages=["hip_cpp"],
                entry_point="kernel.hip::run",
                compile_options={"unsupported_flags": ["-O3"]},
            )

    def test_hip_compile_options_accepted(self):
        spec = _make_spec(
            languages=["hip_cpp"],
            entry_point="kernel.hip::run",
            compile_options={"hip_cflags": ["--offload-arch=gfx1200"]},
        )
        assert spec.compile_options is not None
        assert spec.compile_options.hip_cflags == ["--offload-arch=gfx1200"]

    def test_documented_native_compile_options_remain_accepted(self):
        spec = _make_spec(
            languages=["hip_cpp"],
            entry_point="kernel.hip::run",
            compile_options={
                "cflags": ["-Wall"],
                "hip_cflags": ["-O3", "--offload-arch=gfx1200"],
                "ld_flags": ["-lrocblas"],
            },
        )

        assert spec.compile_options is not None
        assert spec.compile_options.cflags == ["-Wall"]
        assert spec.compile_options.hip_cflags == ["-O3", "--offload-arch=gfx1200"]
        assert spec.compile_options.ld_flags == ["-lrocblas"]

    @pytest.mark.parametrize(
        ("field", "flag", "message"),
        [
            ("hip_cflags", "@/tmp/flags.rsp", "response file"),
            ("cflags", "-I/usr/local/include", "host paths"),
            ("cflags", "-isystem", "external path value"),
            ("hip_cflags", "--sysroot=/opt/rocm", "host paths"),
            ("hip_cflags", "-fplugin=/tmp/plugin.so", "host paths"),
            ("ld_flags", "-L/usr/local/lib", "host paths"),
            ("ld_flags", "-Wl,-rpath,/tmp/lib", "runtime linker paths"),
            ("ld_flags", "-Wl,--dynamic-linker=/lib64/ld-linux-x86-64.so.2", "runtime linker paths"),
        ],
    )
    def test_dangerous_native_compile_options_rejected(self, field, flag, message):
        with pytest.raises(ValidationError, match=message):
            _make_spec(
                languages=["hip_cpp"],
                entry_point="kernel.hip::run",
                compile_options={field: [flag]},
            )


class TestSolutionHashCoversBehaviorFields:
    """The build cache is keyed on solution.hash() (eval_runtime stages under a
    directory named by the hash prefix), so the hash must cover every field that
    changes the compiled artifact or runtime behavior -- otherwise two different
    solutions collide on the same cached build."""

    @staticmethod
    def _solution(**spec_overrides: Any):
        kwargs: dict[str, Any] = dict(
            languages=[SupportedLanguages.HIP_CPP],
            target_hardware=[SupportedHardware.GFX1200],
            entry_point="kernel.hip::run",
            binding=None,
        )
        kwargs.update(spec_overrides)
        return Solution(
            name="sol",
            definition="toy",
            author="agent",
            spec=BuildSpec(**kwargs),
            sources=[SourceFile(path="kernel.hip", content="void run(){}")],
        )

    def test_compile_options_perturbs_hash(self):
        base = self._solution()
        with_opts = self._solution(
            compile_options=CompileOptions(
                hip_cflags=["-O3", "--offload-arch=gfx1200"]
            )
        )
        assert base.hash() != with_opts.hash()
        assert base != with_opts
        assert hash(base) != hash(with_opts)

    def test_target_hardware_perturbs_hash(self):
        base = self._solution()
        other = self._solution(target_hardware=["gfx942"])
        assert base.hash() != other.hash()
        assert base != other

    def test_destination_passing_style_perturbs_hash(self):
        base = self._solution()
        other = self._solution(destination_passing_style=False)
        assert base.hash() != other.hash()
        assert base != other

    def test_identical_solutions_still_collide(self):
        # Cache sharing must still work for genuinely identical solutions.
        assert self._solution() == self._solution()
        assert hash(self._solution()) == hash(self._solution())
