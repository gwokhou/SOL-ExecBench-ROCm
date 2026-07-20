# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Built-in ROCm toolchain capability registry."""

from __future__ import annotations

from .models import (
    ToolLifecycle,
    ToolchainArtifactType,
    ToolchainCapability,
    ToolchainEvidenceLevel,
)


def default_toolchain_registry() -> list[ToolchainCapability]:
    """Return the built-in ROCm toolchain capability registry."""

    return [
        ToolchainCapability(
            tool_id="rocprofv3",
            display_name="ROCprofiler SDK rocprofv3",
            lifecycle=ToolLifecycle.ACTIVE,
            evidence_levels=[ToolchainEvidenceLevel.PROFILING],
            artifact_types=[ToolchainArtifactType.EXECUTABLE_RUN],
            hardware_generations=["RDNA 4", "CDNA 3", "CDNA 4"],
            gpu_arch_patterns=["gfx*"],
            rocm_version_min="6.2",
            expected_binaries=["rocprofv3"],
            probe_command=["rocprofv3", "--version"],
            source_refs=[
                "https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/how-to/using-rocprofv3.html",
                "https://github.com/ROCm/rocm-systems",
            ],
            notes="Profiling evidence only; not score or correctness authority.",
        ),
        ToolchainCapability(
            tool_id="rocprofv3-avail",
            display_name="ROCprofiler SDK rocprofv3-avail",
            lifecycle=ToolLifecycle.ACTIVE,
            evidence_levels=[ToolchainEvidenceLevel.PROFILING],
            artifact_types=[ToolchainArtifactType.EXECUTABLE_RUN],
            hardware_generations=["RDNA 4", "CDNA 3", "CDNA 4"],
            gpu_arch_patterns=["gfx*"],
            rocm_version_min="6.2",
            expected_binaries=["rocprofv3-avail"],
            probe_command=["rocprofv3-avail", "--help"],
            source_refs=[
                "https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/how-to/using-rocprofv3-avail.html",
                "https://github.com/ROCm/rocm-systems",
            ],
            notes="Counter/configuration discovery companion to rocprofv3.",
        ),
        ToolchainCapability(
            tool_id="rocprofiler-systems",
            display_name="ROCm Systems Profiler legacy repository",
            lifecycle=ToolLifecycle.MIGRATED,
            replacement_tool_id="rocm-systems",
            evidence_levels=[
                ToolchainEvidenceLevel.RUNTIME,
                ToolchainEvidenceLevel.PROFILING,
            ],
            artifact_types=[ToolchainArtifactType.EXECUTABLE_RUN],
            hardware_generations=["RDNA 4", "CDNA 3", "CDNA 4"],
            gpu_arch_patterns=["gfx*"],
            source_refs=[
                "https://github.com/ROCm/rocprofiler-systems",
                "https://github.com/ROCm/rocm-systems",
            ],
            notes="Historical repository; source of truth migrated to ROCm Systems.",
        ),
        ToolchainCapability(
            tool_id="rocm-systems",
            display_name="ROCm Systems super-repo",
            lifecycle=ToolLifecycle.ACTIVE,
            evidence_levels=[
                ToolchainEvidenceLevel.RUNTIME,
                ToolchainEvidenceLevel.PROFILING,
            ],
            artifact_types=[ToolchainArtifactType.EXECUTABLE_RUN],
            hardware_generations=["RDNA 4", "CDNA 3", "CDNA 4"],
            gpu_arch_patterns=["gfx*"],
            source_refs=["https://github.com/ROCm/rocm-systems"],
            notes="Repository source-of-truth signal, not a directly executed tool.",
        ),
        ToolchainCapability(
            tool_id="rocminfo",
            display_name="rocminfo",
            lifecycle=ToolLifecycle.ACTIVE,
            evidence_levels=[ToolchainEvidenceLevel.RUNTIME],
            artifact_types=[ToolchainArtifactType.NONE],
            hardware_generations=["RDNA 4", "CDNA 3", "CDNA 4"],
            gpu_arch_patterns=["gfx*"],
            expected_binaries=["rocminfo"],
            probe_command=["rocminfo"],
            source_refs=["https://github.com/ROCm/rocm-systems"],
            notes="Runtime/device discovery, not compiler evidence.",
        ),
        ToolchainCapability(
            tool_id="rocm_agent_enumerator",
            display_name="rocm_agent_enumerator",
            lifecycle=ToolLifecycle.ACTIVE,
            evidence_levels=[ToolchainEvidenceLevel.RUNTIME],
            artifact_types=[ToolchainArtifactType.NONE],
            hardware_generations=["RDNA 4", "CDNA 3", "CDNA 4"],
            gpu_arch_patterns=["gfx*"],
            expected_binaries=["rocm_agent_enumerator"],
            probe_command=["rocm_agent_enumerator"],
            source_refs=["https://github.com/ROCm/rocm-systems"],
            notes="Architecture discovery helper.",
        ),
        ToolchainCapability(
            tool_id="readelf",
            display_name="readelf",
            lifecycle=ToolLifecycle.ACTIVE,
            evidence_levels=[ToolchainEvidenceLevel.STATIC],
            artifact_types=[
                ToolchainArtifactType.ELF_OBJECT,
                ToolchainArtifactType.ROCM_BINARY,
                ToolchainArtifactType.STATIC_FUTURE,
            ],
            hardware_generations=["RDNA 4", "CDNA 3", "CDNA 4"],
            gpu_arch_patterns=["gfx*"],
            expected_binaries=["readelf"],
            probe_command=["readelf", "--version"],
            source_refs=["https://sourceware.org/binutils/docs/binutils/readelf.html"],
            notes="Optional fallback for ELF metadata.",
        ),
        ToolchainCapability(
            tool_id="llvm-objdump",
            display_name="LLVM objdump",
            lifecycle=ToolLifecycle.ACTIVE,
            evidence_levels=[ToolchainEvidenceLevel.STATIC],
            artifact_types=[
                ToolchainArtifactType.ELF_OBJECT,
                ToolchainArtifactType.ROCM_BINARY,
                ToolchainArtifactType.STATIC_FUTURE,
            ],
            hardware_generations=["RDNA 4", "CDNA 3", "CDNA 4"],
            gpu_arch_patterns=["gfx*"],
            expected_binaries=["llvm-objdump"],
            probe_command=["llvm-objdump", "--version"],
            source_refs=["https://llvm.org/docs/CommandGuide/llvm-objdump.html"],
            notes="Optional object inspection route.",
        ),
        ToolchainCapability(
            tool_id="roc-objdump",
            display_name="roc-objdump",
            lifecycle=ToolLifecycle.CANDIDATE,
            evidence_levels=[ToolchainEvidenceLevel.STATIC],
            artifact_types=[
                ToolchainArtifactType.ELF_OBJECT,
                ToolchainArtifactType.ROCM_BINARY,
                ToolchainArtifactType.STATIC_FUTURE,
            ],
            hardware_generations=["RDNA 4", "CDNA 3", "CDNA 4"],
            gpu_arch_patterns=["gfx*"],
            expected_binaries=["roc-objdump"],
            probe_command=["roc-objdump", "--version"],
            source_refs=[
                "https://rocm.docs.amd.com/projects/HIP/en/develop/understand/compilers.html"
            ],
            notes="Distribution-dependent candidate for static evidence.",
        ),
        ToolchainCapability(
            tool_id="rga",
            display_name="Radeon GPU Analyzer",
            lifecycle=ToolLifecycle.PLANNED,
            evidence_levels=[ToolchainEvidenceLevel.STATIC],
            artifact_types=[
                ToolchainArtifactType.HIP_COMPILER_OUTPUT,
                ToolchainArtifactType.ROCM_BINARY,
                ToolchainArtifactType.STATIC_FUTURE,
            ],
            hardware_generations=["RDNA 4", "CDNA 3", "CDNA 4"],
            gpu_arch_patterns=["gfx*"],
            expected_binaries=["rga"],
            probe_command=["rga", "--version"],
            source_refs=[
                "https://github.com/GPUOpen-Tools/radeon_gpu_analyzer",
                "https://gpuopen.com/manuals/rga_manual/help_manual/",
            ],
            notes="Optional compiler-facing static evidence route.",
        ),
    ]
