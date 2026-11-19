from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from sol_execbench.core.dataset.corpus import load_target_descriptor
from sol_execbench.core.platform.hardware import (
    HardwareConfiguration,
    HardwareConfigurationKind,
    HardwareExecutionIdentity,
    HardwareIsolationClass,
    HardwareObservation,
    HardwareVirtualizationMode,
    resolve_hardware_configuration,
)
from sol_execbench.core.platform.runtime import RocmDeviceInfo

ROOT = Path(__file__).resolve().parents[4]
TARGETS = ROOT / "problems/LLM_CORE/targets"
GIB = 1024**3


def _template(**updates: object) -> HardwareConfiguration:
    payload: dict[str, object] = {
        "target_id": "audit-label",
        "vendor": "AMD",
        "device_model": None,
        "gfx_target": "gfx942",
        "kind": HardwareConfigurationKind.ISA_TEMPLATE,
    }
    payload.update(updates)
    return HardwareConfiguration.model_validate(payload)


def _observation(
    model: str,
    *,
    total_gib: int = 192,
    compute_units: int | None = None,
) -> HardwareObservation:
    total = total_gib * GIB
    return HardwareObservation(
        probe_method="mock",
        probe_version="mock.v1",
        device="cuda:0",
        device_index=0,
        gpu_name=model,
        gfx_target="gfx942",
        collected_at=datetime(2026, 8, 18, tzinfo=UTC),
        torch_version="test",
        hip_version="test",
        visible_compute_units=compute_units,
        runtime_total_bytes=total,
        runtime_free_bytes=total,
        stable_allocatable_bytes=total,
        usable_quota_bytes=total * 4 // 5,
    )


def test_configuration_identity_excludes_audit_label_but_binds_resources() -> (
    None
):
    base = _template(
        device_model="AMD Instinct MI300X",
        visible_compute_units=304,
        visible_memory_bytes=192 * GIB,
    )
    renamed = base.model_copy(update={"target_id": "another-label"})
    resized = base.model_copy(update={"visible_memory_bytes": 24 * GIB})

    assert renamed.configuration_id == base.configuration_id
    assert resized.configuration_id != base.configuration_id


def test_gfx942_runtime_resolution_distinguishes_mi300x_and_mi308x() -> None:
    template = _template()
    mi300x = resolve_hardware_configuration(
        template,
        _observation("AMD Instinct MI300X", compute_units=304),
    )
    mi308x = resolve_hardware_configuration(
        template,
        _observation("AMD Instinct MI308X", compute_units=304),
    )

    assert mi300x.gfx_target == mi308x.gfx_target == "gfx942"
    assert mi300x.device_model == "AMD Instinct MI300X"
    assert mi308x.device_model == "AMD Instinct MI308X"
    assert mi300x.configuration_id != mi308x.configuration_id


def test_model_specific_template_rejects_another_product_on_same_isa() -> None:
    template = _template(
        device_model="AMD Instinct MI300X",
        visible_compute_units=304,
        visible_memory_bytes=192 * GIB,
    )

    with pytest.raises(ValueError, match="device model"):
        resolve_hardware_configuration(
            template,
            _observation("AMD Instinct MI308X", compute_units=304),
        )


def test_declared_gfx942_templates_cover_isa_product_and_configuration() -> (
    None
):
    isa_template = load_target_descriptor(TARGETS / "isa/gfx942.yaml")
    product = load_target_descriptor(TARGETS / "products/mi300x.yaml")
    mi300x = load_target_descriptor(
        TARGETS / "configurations/mi300x/spx-192gb.yaml"
    )
    mi308x = load_target_descriptor(TARGETS / "products/mi308x.yaml")

    assert isa_template.hardware.device_model is None
    assert isa_template.hardware.target_id == "amd-gfx942-isa-declared"
    assert isa_template.hardware.kind is HardwareConfigurationKind.ISA_TEMPLATE
    assert product.hardware.device_model == "AMD Instinct MI300X"
    assert product.hardware.target_id == "amd-mi300x-product-declared"
    assert product.hardware.kind is HardwareConfigurationKind.PRODUCT_TEMPLATE
    assert product.hardware.visible_compute_units is None
    assert mi300x.hardware.device_model == "AMD Instinct MI300X"
    assert mi300x.hardware.target_id == (
        "amd-mi300x-spx-192gb-configuration-declared"
    )
    assert (
        mi300x.hardware.kind is HardwareConfigurationKind.CONFIGURATION_TEMPLATE
    )
    assert mi300x.hardware.product_sku == "mi300x-oam"
    assert mi300x.hardware.partition == "spx"
    assert mi300x.hardware.visible_compute_units == 304
    assert mi308x.hardware.device_model == "AMD Instinct MI308X"
    assert mi308x.hardware.visible_compute_units is None
    assert (
        len(
            {
                isa_template.hardware.configuration_id,
                product.hardware.configuration_id,
                mi300x.hardware.configuration_id,
                mi308x.hardware.configuration_id,
            }
        )
        == 4
    )


def test_product_template_resolves_to_observed_device() -> None:
    product = load_target_descriptor(TARGETS / "products/mi300x.yaml")
    resolved = resolve_hardware_configuration(
        product.hardware,
        _observation("AMD Instinct MI300X", compute_units=304),
    )

    assert resolved.kind is HardwareConfigurationKind.OBSERVED_DEVICE
    assert resolved.visible_compute_units == 304


def test_bare_metal_configuration_template_resolves_to_physical_device() -> (
    None
):
    configuration = load_target_descriptor(
        TARGETS / "configurations/mi300x/spx-192gb.yaml"
    )
    resolved = resolve_hardware_configuration(
        configuration.hardware,
        _observation("AMD Instinct MI300X", compute_units=304),
    )

    assert resolved.kind is HardwareConfigurationKind.PHYSICAL_DEVICE
    assert resolved.partition == "spx"


@pytest.mark.parametrize(
    ("updates", "expected_kind"),
    [
        (
            {"virtualization": HardwareVirtualizationMode.SR_IOV_VF},
            HardwareConfigurationKind.VIRTUAL_DEVICE,
        ),
        ({"partition": "CPX"}, HardwareConfigurationKind.PARTITION),
    ],
)
def test_configuration_template_resolves_runtime_kind(
    updates: dict[str, object],
    expected_kind: HardwareConfigurationKind,
) -> None:
    template = _template(
        kind=HardwareConfigurationKind.CONFIGURATION_TEMPLATE,
        **updates,
    )

    resolved = resolve_hardware_configuration(
        template,
        _observation("AMD Instinct MI300X", compute_units=304),
    )

    assert resolved.kind is expected_kind
    assert resolved.partition != "CPX"


def test_runtime_device_projects_to_canonical_configuration() -> None:
    device = RocmDeviceInfo(
        device="cuda:2",
        index=2,
        name="AMD Instinct MI308X",
        gfx_target="gfx942",
        visible_compute_units=20,
        total_memory_bytes=24 * GIB,
        l2_cache_bytes=None,
        torch_version="test",
        hip_version="test",
    )

    configuration = device.hardware_configuration
    assert configuration.kind is HardwareConfigurationKind.OBSERVED_DEVICE
    assert configuration.device_model == "AMD Instinct MI308X"
    assert configuration.visible_compute_units == 20
    assert configuration.visible_memory_bytes == 24 * GIB


def test_execution_context_separates_hardware_from_software_state() -> None:
    first = HardwareExecutionIdentity(
        gpu_architecture="gfx942",
        gpu_id="gpu-1",
        rocm_version="7.2",
        compiler_version="clang-a",
        clock_mode="locked",
        power_profile="peak",
    )
    second = first.model_copy(update={"compiler_version": "clang-b"})

    assert (
        first.hardware_configuration.configuration_id
        == second.hardware_configuration.configuration_id
    )
    assert first.execution_context_id != second.execution_context_id


def test_configuration_identity_binds_virtualization_and_isolation() -> None:
    bare = _template(
        virtualization=HardwareVirtualizationMode.BARE_METAL,
        isolation=HardwareIsolationClass.DEDICATED,
    )
    virtual = bare.model_copy(
        update={
            "virtualization": HardwareVirtualizationMode.SR_IOV_VF,
            "isolation": HardwareIsolationClass.SHARED,
        }
    )

    assert bare.configuration_id != virtual.configuration_id


def test_gfx1200_identity_can_distinguish_product_matrix() -> None:
    configurations = (
        _template(
            device_model="AMD Radeon RX 9060 XT",
            product_sku="rx9060xt-standard",
            gfx_target="gfx1200",
            visible_memory_bytes=16 * GIB,
        ),
        _template(
            device_model="AMD Radeon RX 9060 XT",
            product_sku="rx9060xt-standard",
            gfx_target="gfx1200",
            visible_memory_bytes=8 * GIB,
        ),
        _template(
            device_model="AMD Radeon RX 9060 XT",
            product_sku="rx9060xt-lp",
            gfx_target="gfx1200",
            visible_memory_bytes=16 * GIB,
        ),
        _template(
            device_model="AMD Radeon RX 9060",
            product_sku="rx9060-standard",
            gfx_target="gfx1200",
            visible_memory_bytes=8 * GIB,
        ),
    )

    assert {item.gfx_target for item in configurations} == {"gfx1200"}
    assert len({item.configuration_id for item in configurations}) == 4


def test_generic_runtime_observation_does_not_guess_product_sku() -> None:
    observation = _observation(
        "AMD Radeon RX 9060 XT",
        total_gib=16,
        compute_units=32,
    ).model_copy(update={"gfx_target": "gfx1200"})
    resolved = resolve_hardware_configuration(
        _template(gfx_target="gfx1200"),
        observation,
    )

    assert resolved.visible_memory_bytes == 16 * GIB
    assert resolved.product_sku is None


def test_exact_rx9060xt_target_binds_sku_and_capacity() -> None:
    target = load_target_descriptor(
        TARGETS / "configurations/rx9060xt/standard-16gb.yaml"
    )

    assert target.hardware.product_sku == "rx9060xt-standard"
    assert target.hardware.visible_memory_bytes == 16 * GIB


def test_rx9060xt_standard_16gb_resolves_rdna4_wgp_and_firmware_carveout() -> (
    None
):
    target = load_target_descriptor(
        TARGETS / "configurations/rx9060xt/standard-16gb.yaml"
    )
    observation = HardwareObservation(
        probe_method="torch_rocm_allocation_search.v1",
        probe_version="torch_rocm_allocation_search.v1",
        device="cuda:0",
        device_index=0,
        gpu_name="AMD Radeon RX 9060 XT",
        gfx_target="gfx1200",
        collected_at=datetime(2026, 11, 19, tzinfo=UTC),
        torch_version="2.14.0+rocm10.1.0",
        hip_version="7.16.26385",
        visible_compute_units=16,
        runtime_total_bytes=17_095_983_104,
        runtime_free_bytes=16_800_000_000,
        stable_allocatable_bytes=15_000_000_000,
        usable_quota_bytes=12_000_000_000,
        l2_cache_bytes=4_194_304,
    )

    resolved = resolve_hardware_configuration(target.hardware, observation)

    assert resolved.visible_compute_units == 32
    assert resolved.visible_memory_bytes == 17_095_983_104
    assert resolved.product_sku == "rx9060xt-standard"
    assert resolved.kind is HardwareConfigurationKind.OBSERVED_DEVICE


@pytest.mark.parametrize(
    ("template_path", "updates", "match"),
    [
        pytest.param(
            "configurations/rx9060xt/standard-16gb.yaml",
            {"visible_compute_units": 8},
            "compute units",
            id="wrong-rdna4-wgp",
        ),
        pytest.param(
            "configurations/mi300x/spx-192gb.yaml",
            {"visible_compute_units": 152},
            "compute units",
            id="cdna-half-cu",
        ),
        pytest.param(
            "configurations/rx9060xt/standard-16gb.yaml",
            {"runtime_total_bytes": 8 * GIB, "runtime_free_bytes": 8 * GIB},
            "observed memory",
            id="8gib-sku",
        ),
        pytest.param(
            "configurations/rx9060xt/standard-16gb.yaml",
            {"runtime_total_bytes": 16 * GIB - (512 * 1024**2 + 1)},
            "observed memory",
            id="excess-carveout",
        ),
        pytest.param(
            "configurations/rx9060xt/standard-16gb.yaml",
            {"runtime_total_bytes": 16 * GIB + 4096},
            "observed memory",
            id="above-nominal",
        ),
    ],
)
def test_configuration_template_rejects_mismatched_wgp_or_memory(
    template_path: str,
    updates: dict[str, int],
    match: str,
) -> None:
    target = load_target_descriptor(TARGETS / template_path)
    is_cdna = "mi300x" in template_path
    total = 192 * GIB if is_cdna else 17_095_983_104
    base_obs = HardwareObservation(
        probe_method="test",
        probe_version="test.v1",
        device="cuda:0",
        device_index=0,
        gpu_name="AMD Instinct MI300X" if is_cdna else "AMD Radeon RX 9060 XT",
        gfx_target="gfx942" if is_cdna else "gfx1200",
        collected_at=datetime(2026, 11, 19, tzinfo=UTC),
        torch_version="test",
        hip_version="test",
        visible_compute_units=304 if is_cdna else 16,
        runtime_total_bytes=total,
        runtime_free_bytes=min(
            total, int(updates.get("runtime_total_bytes", total))
        ),
        stable_allocatable_bytes=8 * GIB,
        usable_quota_bytes=4 * GIB,
        l2_cache_bytes=4_194_304 if not is_cdna else None,
    )
    obs = base_obs.model_copy(update=updates)

    with pytest.raises(ValueError, match=match):
        resolve_hardware_configuration(target.hardware, obs)
