# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Canonical nominal, configured, observed, and resolved GPU facts."""

from __future__ import annotations

import re
from datetime import datetime
from enum import StrEnum

from pydantic import ConfigDict, Field, field_validator, model_validator

from sol_execbench.core.data.base_model import FrozenArtifactModel
from sol_execbench.core.integrity import SHA256Digest, stable_json_checksum

HARDWARE_IDENTITY_FORMAT = "hardware_identity.v1"
HARDWARE_RESOLUTION_POLICY = "hardware_context_resolution.v1"
_PCI_BDF_PATTERN = r"[0-9a-f]{4}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]"
_PCI_BDF_RE = re.compile(f"^{_PCI_BDF_PATTERN}$")
_HARDWARE_CONFIG = ConfigDict(
    extra="forbid",
    frozen=True,
    strict=True,
    allow_inf_nan=False,
)


class HardwareConfigurationKind(StrEnum):
    """Lifecycle tier of a declared template or resolved accelerator."""

    ISA_TEMPLATE = "isa_template"
    PRODUCT_TEMPLATE = "product_template"
    CONFIGURATION_TEMPLATE = "configuration_template"
    OBSERVED_DEVICE = "observed_device"
    PHYSICAL_DEVICE = "physical_device"
    VIRTUAL_DEVICE = "virtual_device"
    PARTITION = "partition"


class HardwareVirtualizationMode(StrEnum):
    """Virtualization boundary visible to the benchmark."""

    UNKNOWN = "unknown"
    BARE_METAL = "bare_metal"
    PASSTHROUGH_VM = "passthrough_vm"
    SR_IOV_VF = "sr_iov_vf"


class HardwareIsolationClass(StrEnum):
    """Whether accelerator resources are dedicated to one evaluator."""

    UNKNOWN = "unknown"
    DEDICATED = "dedicated"
    SHARED = "shared"


class HardwareFactConfidence(StrEnum):
    """Evidence quality for nominal hardware facts."""

    DECLARED = "declared"
    SPEC_DERIVED = "spec_derived"
    MEASURED = "measured"
    CALIBRATED = "calibrated"


class PCIeLinkIdentity(FrozenArtifactModel):
    """Negotiated and maximum identity of one PCIe link endpoint."""

    model_config = _HARDWARE_CONFIG

    bdf: str = Field(pattern=f"^{_PCI_BDF_PATTERN}$")
    current_speed_gtps: float = Field(gt=0)
    max_speed_gtps: float = Field(gt=0)
    current_width: int = Field(gt=0)
    max_width: int = Field(gt=0)

    @model_validator(mode="after")
    def _negotiated_values_fit_capability(self) -> PCIeLinkIdentity:
        if self.current_speed_gtps > self.max_speed_gtps:
            raise ValueError("PCIe current speed exceeds maximum speed")
        if self.current_width > self.max_width:
            raise ValueError("PCIe current width exceeds maximum width")
        return self


class PCIeTopologyIdentity(FrozenArtifactModel):
    """Ordered CPU-root-to-GPU PCIe path and its effective bottleneck."""

    model_config = _HARDWARE_CONFIG

    links: tuple[PCIeLinkIdentity, ...] = Field(min_length=1)
    bottleneck_bdf: str = Field(pattern=f"^{_PCI_BDF_PATTERN}$")
    effective_speed_gtps: float = Field(gt=0)
    effective_width: int = Field(gt=0)

    @property
    def endpoint_bdf(self) -> str:
        """Return the final device BDF in the ordered path."""
        return self.links[-1].bdf

    @field_validator("links", mode="before")
    @classmethod
    def _json_links_are_immutable(cls, value: object) -> object:
        return tuple(value) if isinstance(value, list) else value

    @model_validator(mode="after")
    def _effective_link_is_derived(self) -> PCIeTopologyIdentity:
        bdfs = [link.bdf for link in self.links]
        if len(bdfs) != len(set(bdfs)):
            raise ValueError("PCIe topology repeats a BDF")
        bottleneck = min(
            self.links,
            key=lambda link: link.current_speed_gtps * link.current_width,
        )
        if (
            self.bottleneck_bdf != bottleneck.bdf
            or self.effective_speed_gtps != bottleneck.current_speed_gtps
            or self.effective_width != bottleneck.current_width
        ):
            raise ValueError("PCIe effective link is not canonically derived")
        return self


class HardwareNominalProfile(FrozenArtifactModel):
    """Versioned published or calibrated facts for one accelerator model."""

    vendor: str = Field(min_length=1)
    device_model: str = Field(min_length=1)
    product_sku: str | None = Field(
        default=None,
        pattern=r"^[a-z0-9][a-z0-9._-]*$",
    )
    gfx_target: str = Field(pattern=r"^gfx[0-9a-z]+$")
    isa_features: tuple[str, ...] = ()
    compute_units: int = Field(gt=0)
    memory_capacity_bytes: int = Field(gt=0)
    memory_bandwidth_bytes_per_second: float = Field(gt=0)
    l2_cache_bytes: int = Field(ge=0)
    last_level_cache_bytes: int = Field(ge=0)
    peak_ops_per_second: dict[str, float]
    profile_revision: str = Field(min_length=1)
    source: str = Field(min_length=1)
    confidence: HardwareFactConfidence

    @field_validator("isa_features")
    @classmethod
    def _canonical_features(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        expected = tuple(sorted({item.strip().lower() for item in value}))
        if value != expected:
            raise ValueError(
                "ISA features must be normalized, sorted, and unique"
            )
        return value

    @field_validator("peak_ops_per_second")
    @classmethod
    def _positive_peaks(cls, value: dict[str, float]) -> dict[str, float]:
        if not value or any(rate <= 0 for rate in value.values()):
            raise ValueError("nominal peak rates must be positive")
        return dict(sorted((key.lower(), rate) for key, rate in value.items()))

    @property
    def profile_digest(self) -> SHA256Digest:
        """Return the immutable nominal-profile identity."""
        return stable_json_checksum(
            {
                "format": HARDWARE_IDENTITY_FORMAT,
                "kind": "nominal_profile",
                **self.model_dump(mode="json"),
            }
        )


class HardwareConfiguration(FrozenArtifactModel):
    """Stable configured accelerator identity, independent of runtime state."""

    target_id: str = Field(min_length=1)
    vendor: str = Field(min_length=1)
    device_model: str | None = None
    product_sku: str | None = Field(
        default=None,
        pattern=r"^[a-z0-9][a-z0-9._-]*$",
    )
    gfx_target: str = Field(pattern=r"^gfx[0-9a-z]+$")
    isa_features: tuple[str, ...] = ()
    kind: HardwareConfigurationKind
    visible_compute_units: int | None = Field(default=None, gt=0)
    visible_memory_bytes: int | None = Field(default=None, gt=0)
    l2_cache_bytes: int | None = Field(default=None, gt=0)
    partition: str | None = None
    virtualization: HardwareVirtualizationMode = (
        HardwareVirtualizationMode.UNKNOWN
    )
    isolation: HardwareIsolationClass = HardwareIsolationClass.UNKNOWN
    gpu_id: str | None = None
    gpu_bdf: str | None = Field(default=None, pattern=f"^{_PCI_BDF_PATTERN}$")
    pcie_topology: PCIeTopologyIdentity | None = None
    nominal_profile_digest: SHA256Digest | None = None

    @field_validator("gfx_target", mode="before")
    @classmethod
    def _normalize_gfx_target(cls, value: object) -> object:
        return value.strip().lower() if isinstance(value, str) else value

    @field_validator("isa_features")
    @classmethod
    def _canonical_features(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        expected = tuple(sorted({item.strip().lower() for item in value}))
        if value != expected:
            raise ValueError(
                "ISA features must be normalized, sorted, and unique"
            )
        return value

    @field_validator("partition", mode="before")
    @classmethod
    def _normalize_partition(cls, value: object) -> object:
        if isinstance(value, str):
            normalized = value.strip().lower()
            return normalized or None
        return value

    @model_validator(mode="after")
    def _topology_terminates_at_gpu(self) -> HardwareConfiguration:
        if (
            self.pcie_topology is not None
            and self.gpu_bdf != self.pcie_topology.endpoint_bdf
        ):
            raise ValueError("PCIe topology does not terminate at gpu_bdf")
        return self

    def identity_payload(self) -> dict[str, object]:
        """Return configuration semantics without audit-only target labels."""
        payload = self.model_dump(mode="json", exclude={"target_id"})
        payload["device_model"] = (
            self.device_model.strip().lower() if self.device_model else None
        )
        payload["vendor"] = self.vendor.strip().lower()
        return {
            "format": HARDWARE_IDENTITY_FORMAT,
            "kind": "configuration",
            **payload,
        }

    @property
    def configuration_id(self) -> SHA256Digest:
        """Return the stable configured-hardware identity."""
        return stable_json_checksum(self.identity_payload())


class HardwareObservation(FrozenArtifactModel):
    """One runtime observation without conflating it with nominal facts."""

    probe_method: str = Field(min_length=1)
    probe_version: str = Field(min_length=1)
    device: str = Field(min_length=1)
    device_index: int = Field(ge=0)
    gpu_name: str = Field(min_length=1)
    product_sku: str | None = Field(
        default=None,
        pattern=r"^[a-z0-9][a-z0-9._-]*$",
    )
    gfx_target: str = Field(pattern=r"^gfx[0-9a-z]+$")
    collected_at: datetime
    torch_version: str = Field(min_length=1)
    hip_version: str = Field(min_length=1)
    visible_compute_units: int | None = Field(default=None, gt=0)
    runtime_total_bytes: int = Field(gt=0)
    runtime_free_bytes: int = Field(gt=0)
    stable_allocatable_bytes: int = Field(gt=0)
    usable_quota_bytes: int = Field(gt=0)
    l2_cache_bytes: int | None = Field(default=None, gt=0)
    gfx_clock_hz: float | None = Field(default=None, gt=0)
    memory_bandwidth_bytes_per_second: float | None = Field(default=None, gt=0)

    @field_validator("gfx_target", mode="before")
    @classmethod
    def _normalize_gfx_target(cls, value: object) -> object:
        return value.strip().lower() if isinstance(value, str) else value

    @model_validator(mode="after")
    def _memory_bounds_are_consistent(self) -> HardwareObservation:
        if self.runtime_free_bytes > self.runtime_total_bytes:
            raise ValueError("runtime free memory exceeds total memory")
        if self.stable_allocatable_bytes > self.runtime_total_bytes:
            raise ValueError("stable allocation exceeds total memory")
        if self.usable_quota_bytes > self.runtime_total_bytes:
            raise ValueError("usable quota exceeds total memory")
        return self

    @property
    def observation_digest(self) -> SHA256Digest:
        """Hash measured semantics while excluding time and device numbering."""
        return stable_json_checksum(
            {
                "format": HARDWARE_IDENTITY_FORMAT,
                "kind": "observation",
                **self.model_dump(
                    mode="json",
                    exclude={"collected_at", "device", "device_index"},
                ),
            }
        )


class ResolvedHardwareContext(FrozenArtifactModel):
    """Canonical join of configuration, observation, and execution limits."""

    configuration: HardwareConfiguration
    observation: HardwareObservation
    hardware_configuration_id: SHA256Digest
    hardware_observation_digest: SHA256Digest
    capacity_class_bytes: int = Field(ge=0)
    supported_dtypes: tuple[str, ...]
    supported_quantization: tuple[str, ...]
    capabilities: tuple[str, ...]
    max_tensor_bytes: int = Field(gt=0)
    reference_ipc_limit_bytes: int = Field(gt=0)
    resolution_policy: str = HARDWARE_RESOLUTION_POLICY
    context_digest: SHA256Digest

    @model_validator(mode="after")
    def _identities_are_canonical(self) -> ResolvedHardwareContext:
        if (
            self.hardware_configuration_id
            != self.configuration.configuration_id
        ):
            raise ValueError("hardware configuration identity does not match")
        if (
            self.hardware_observation_digest
            != self.observation.observation_digest
        ):
            raise ValueError("hardware observation identity does not match")
        if self.context_digest != resolved_hardware_context_digest(self):
            raise ValueError(
                "resolved hardware context identity does not match"
            )
        return self


class HardwareExecutionIdentity(FrozenArtifactModel):
    """Configured GPU plus software and control state for performance evidence."""

    gpu_architecture: str = Field(min_length=1)
    gpu_id: str | None = None
    gpu_bdf: str | None = None
    pcie_topology: PCIeTopologyIdentity | None = None
    rocm_version: str | None = None
    compiler_version: str | None = None
    clock_mode: str | None = None
    power_profile: str | None = None

    @model_validator(mode="after")
    def _topology_terminates_at_gpu(self) -> HardwareExecutionIdentity:
        if self.gpu_bdf is not None and not _PCI_BDF_RE.fullmatch(self.gpu_bdf):
            raise ValueError("invalid GPU BDF")
        if (
            self.pcie_topology is not None
            and self.gpu_bdf != self.pcie_topology.endpoint_bdf
        ):
            raise ValueError("PCIe topology does not terminate at gpu_bdf")
        return self

    @property
    def hardware_configuration(self) -> HardwareConfiguration:
        """Project the stable hardware portion of this execution identity."""
        return HardwareConfiguration(
            target_id=self.gpu_id or self.gpu_bdf or self.gpu_architecture,
            vendor="AMD",
            device_model=None,
            gfx_target=self.gpu_architecture.split(":", maxsplit=1)[0].lower(),
            isa_features=tuple(
                sorted(set(self.gpu_architecture.split(":")[1:]))
            ),
            kind=HardwareConfigurationKind.OBSERVED_DEVICE,
            gpu_id=self.gpu_id,
            gpu_bdf=self.gpu_bdf,
            pcie_topology=self.pcie_topology,
        )

    @property
    def execution_context_id(self) -> SHA256Digest:
        """Bind stable hardware identity to software and control state."""
        return stable_json_checksum(
            {
                "format": HARDWARE_IDENTITY_FORMAT,
                "kind": "execution_context",
                "hardware_configuration_id": (
                    self.hardware_configuration.configuration_id
                ),
                "rocm_version": self.rocm_version,
                "compiler_version": self.compiler_version,
                "clock_mode": self.clock_mode,
                "power_profile": self.power_profile,
            }
        )


def resolve_hardware_configuration(
    template: HardwareConfiguration,
    observation: HardwareObservation,
) -> HardwareConfiguration:
    """Resolve measured visible resources beneath one declared configuration."""
    if template.gfx_target != observation.gfx_target:
        raise ValueError(
            "hardware observation gfx target differs from template"
        )
    if (
        template.device_model is not None
        and template.device_model.strip().lower()
        != observation.gpu_name.strip().lower()
    ):
        raise ValueError(
            "observed device model differs from target configuration"
        )
    if (
        template.product_sku is not None
        and observation.product_sku is not None
        and template.product_sku != observation.product_sku
    ):
        raise ValueError(
            "observed product SKU differs from target configuration"
        )
    if (
        template.visible_compute_units is not None
        and observation.visible_compute_units is not None
        and template.visible_compute_units != observation.visible_compute_units
    ):
        raise ValueError(
            "observed compute units differ from target configuration"
        )
    if (
        template.visible_memory_bytes is not None
        and template.visible_memory_bytes != observation.runtime_total_bytes
    ):
        raise ValueError("observed memory differs from target configuration")
    if (
        template.l2_cache_bytes is not None
        and observation.l2_cache_bytes is not None
        and template.l2_cache_bytes != observation.l2_cache_bytes
    ):
        raise ValueError("observed L2 differs from target configuration")
    kind = _resolved_configuration_kind(template)
    return template.model_copy(
        update={
            "device_model": template.device_model or observation.gpu_name,
            "product_sku": template.product_sku or observation.product_sku,
            "kind": kind,
            "visible_compute_units": (
                observation.visible_compute_units
                or template.visible_compute_units
            ),
            "visible_memory_bytes": observation.runtime_total_bytes,
            "l2_cache_bytes": observation.l2_cache_bytes
            or template.l2_cache_bytes,
        }
    )


def _resolved_configuration_kind(
    template: HardwareConfiguration,
) -> HardwareConfigurationKind:
    templates = {
        HardwareConfigurationKind.ISA_TEMPLATE,
        HardwareConfigurationKind.PRODUCT_TEMPLATE,
        HardwareConfigurationKind.CONFIGURATION_TEMPLATE,
    }
    if template.kind not in templates:
        return template.kind
    if template.virtualization in {
        HardwareVirtualizationMode.PASSTHROUGH_VM,
        HardwareVirtualizationMode.SR_IOV_VF,
    }:
        return HardwareConfigurationKind.VIRTUAL_DEVICE
    if template.partition not in {None, "spx", "full"}:
        return HardwareConfigurationKind.PARTITION
    if (
        template.kind is HardwareConfigurationKind.CONFIGURATION_TEMPLATE
        and template.virtualization is HardwareVirtualizationMode.BARE_METAL
    ):
        return HardwareConfigurationKind.PHYSICAL_DEVICE
    return HardwareConfigurationKind.OBSERVED_DEVICE


def build_resolved_hardware_context(
    *,
    configuration: HardwareConfiguration,
    observation: HardwareObservation,
    capacity_class_bytes: int,
    supported_dtypes: tuple[str, ...],
    supported_quantization: tuple[str, ...],
    capabilities: tuple[str, ...],
    max_tensor_bytes: int,
    reference_ipc_limit_bytes: int,
) -> ResolvedHardwareContext:
    """Resolve and bind one hardware context without changing workload identity."""
    resolved = resolve_hardware_configuration(configuration, observation)
    payload: dict[str, object] = {
        "configuration": resolved,
        "observation": observation,
        "hardware_configuration_id": resolved.configuration_id,
        "hardware_observation_digest": observation.observation_digest,
        "capacity_class_bytes": capacity_class_bytes,
        "supported_dtypes": tuple(sorted(set(supported_dtypes))),
        "supported_quantization": tuple(sorted(set(supported_quantization))),
        "capabilities": tuple(sorted(set(capabilities))),
        "max_tensor_bytes": max_tensor_bytes,
        "reference_ipc_limit_bytes": reference_ipc_limit_bytes,
        "resolution_policy": HARDWARE_RESOLUTION_POLICY,
        "context_digest": "0" * 64,
    }
    provisional = ResolvedHardwareContext.model_construct(
        _fields_set=None,
        **payload,
    )
    payload["context_digest"] = resolved_hardware_context_digest(provisional)
    return ResolvedHardwareContext.model_validate(payload)


def resolved_hardware_context_digest(
    context: ResolvedHardwareContext,
) -> SHA256Digest:
    """Hash the canonical join, including observation but excluding wall time."""
    return stable_json_checksum(
        {
            "format": HARDWARE_IDENTITY_FORMAT,
            "kind": "resolved_context",
            "hardware_configuration_id": context.hardware_configuration_id,
            "hardware_observation_digest": context.hardware_observation_digest,
            "capacity_class_bytes": context.capacity_class_bytes,
            "supported_dtypes": context.supported_dtypes,
            "supported_quantization": context.supported_quantization,
            "capabilities": context.capabilities,
            "max_tensor_bytes": context.max_tensor_bytes,
            "reference_ipc_limit_bytes": context.reference_ipc_limit_bytes,
            "resolution_policy": context.resolution_policy,
        }
    )


def require_complete_execution_identity(
    identity: HardwareExecutionIdentity | None,
    *,
    context: str,
    require_pcie_topology: bool = False,
) -> None:
    """Reject partial performance-evidence identities."""
    if identity is None:
        if require_pcie_topology:
            raise ValueError(f"{context} requires a complete gpu_identity")
        return
    required = (
        "gpu_id",
        "gpu_bdf",
        "rocm_version",
        "compiler_version",
        "clock_mode",
        "power_profile",
    )
    missing = [name for name in required if getattr(identity, name) is None]
    if require_pcie_topology and identity.pcie_topology is None:
        missing.append("pcie_topology")
    if missing:
        raise ValueError(
            f"{context} bound hardware but gpu_identity is missing "
            f"required fields: {', '.join(missing)}"
        )


__all__ = [
    "HARDWARE_IDENTITY_FORMAT",
    "HARDWARE_RESOLUTION_POLICY",
    "HardwareConfiguration",
    "HardwareConfigurationKind",
    "HardwareExecutionIdentity",
    "HardwareFactConfidence",
    "HardwareIsolationClass",
    "HardwareNominalProfile",
    "HardwareObservation",
    "HardwareVirtualizationMode",
    "PCIeLinkIdentity",
    "PCIeTopologyIdentity",
    "ResolvedHardwareContext",
    "build_resolved_hardware_context",
    "require_complete_execution_identity",
    "resolve_hardware_configuration",
    "resolved_hardware_context_digest",
]
