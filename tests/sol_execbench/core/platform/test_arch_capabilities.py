# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""CPU-safe tests for the arch capability budget catalog and derivation."""

from __future__ import annotations

from sol_execbench.core.platform.arch_capabilities import (
    ArchCapabilityBudgetStatus,
    ArchIsaBudget,
    default_arch_capability_budgets,
    derive_arch_capability_budget,
    load_packaged_arch_capability_budget,
)
from sol_execbench.core.platform.environment_models import GpuEnvironmentSummary
from sol_execbench.core.platform.environment_snapshot import (
    derive_capability_budgets,
)


def test_default_catalog_covers_validated_archs():
    catalog = default_arch_capability_budgets()
    assert set(catalog) == {"gfx942", "gfx1150"}
    for budget in catalog.values():
        assert isinstance(budget, ArchIsaBudget)


def test_gfx942_budget_values():
    budget = load_packaged_arch_capability_budget("gfx942")

    assert budget.architecture == "gfx942"
    assert budget.wavefront_size == 64
    assert budget.vgpr_limit == 256
    assert budget.sgpr_limit == 104
    assert budget.lds_per_workgroup_bytes == 65536
    assert budget.register_file_per_cu_bytes == 524288
    assert budget.waves_per_cu_max == 40
    assert budget.confidence.value == "inexact"
    assert "mfma" in budget.mfma_variants
    assert budget.source


def test_gfx1150_budget_values():
    budget = load_packaged_arch_capability_budget("gfx1150")

    assert budget.wavefront_size == 32
    assert budget.vgpr_limit == 256
    assert budget.register_file_per_cu_bytes is None
    assert budget.mfma_variants == []
    assert budget.confidence.value == "inexact"
    assert "RDNA 3.5" in budget.source


def test_derive_known_arch():
    budget = derive_arch_capability_budget("gfx942")
    assert budget is not None
    assert budget.architecture == "gfx942"


def test_derive_normalizes_gcn_suffix():
    budget = derive_arch_capability_budget("gfx942:sramecc+:xnack-")
    assert budget is not None
    assert budget.architecture == "gfx942"


def test_derive_unknown_returns_none():
    assert derive_arch_capability_budget("gfx9999") is None


def test_derive_none_target_returns_none():
    assert derive_arch_capability_budget(None) is None


def test_derive_capability_budgets_available_and_unsupported():
    budgets = derive_capability_budgets(
        [
            GpuEnvironmentSummary(source="rocminfo", index=0, gfx_target="gfx942"),
            GpuEnvironmentSummary(source="rocminfo", index=1, gfx_target="gfx1100"),
        ]
    )

    assert [(budget.status, budget.architecture) for budget in budgets] == [
        (ArchCapabilityBudgetStatus.AVAILABLE, "gfx942"),
        (ArchCapabilityBudgetStatus.UNSUPPORTED, "gfx1100"),
    ]
    assert budgets[0].budget is not None
    assert budgets[0].reason_code is None
    assert budgets[1].budget is None
    assert budgets[1].reason_code == "unsupported_architecture"


def test_derive_capability_budgets_dedupes_repeated_arch():
    budgets = derive_capability_budgets(
        [
            GpuEnvironmentSummary(source="rocminfo", index=0, gfx_target="gfx942"),
            GpuEnvironmentSummary(source="pytorch", index=0, gfx_target="gfx942"),
        ]
    )

    assert len(budgets) == 1
