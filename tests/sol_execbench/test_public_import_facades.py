"""Public import compatibility checks for refactored facade modules."""

from __future__ import annotations


def test_data_definition_facade_exports_public_schema_types() -> None:
    from sol_execbench.core.data.definition import (
        AxisConst,
        AxisExpr,
        AxisSpec,
        AxisVar,
        DType,
        Definition,
        TensorSpec,
    )

    assert Definition.__name__ == "Definition"
    assert AxisConst.__name__ == "AxisConst"
    assert AxisVar.__name__ == "AxisVar"
    assert AxisExpr.__name__ == "AxisExpr"
    assert TensorSpec.__name__ == "TensorSpec"
    assert DType.FLOAT32.value == "float32"
    assert AxisSpec is not None


def test_amd_score_facade_exports_public_scoring_api() -> None:
    from sol_execbench.core.scoring.amd_score import (
        AmdNativeScore,
        AmdNativeSuiteReport,
        build_amd_native_suite_report,
        build_amd_native_suite_report_from_traces,
        score_amd_native_trace_workload,
        score_amd_native_workload,
    )

    assert AmdNativeScore.__name__ == "AmdNativeScore"
    assert AmdNativeSuiteReport.__name__ == "AmdNativeSuiteReport"
    assert callable(build_amd_native_suite_report)
    assert callable(build_amd_native_suite_report_from_traces)
    assert callable(score_amd_native_trace_workload)
    assert callable(score_amd_native_workload)


def test_runtime_evidence_facade_exports_public_api() -> None:
    from sol_execbench.core.runtime_evidence import (
        RuntimeFailureEvidence,
        build_aggregate_report,
        build_dependency_observation,
        build_host_evidence,
        build_runtime_matrix_entry,
        collect_gpu_evidence,
        collect_visible_device_environment,
        load_matrix_entry,
        write_aggregate_report,
        write_json_payload,
        write_matrix_entry,
    )

    assert RuntimeFailureEvidence.__name__ == "RuntimeFailureEvidence"
    assert callable(build_aggregate_report)
    assert callable(build_dependency_observation)
    assert callable(build_host_evidence)
    assert callable(build_runtime_matrix_entry)
    assert callable(collect_gpu_evidence)
    assert callable(collect_visible_device_environment)
    assert callable(load_matrix_entry)
    assert callable(write_aggregate_report)
    assert callable(write_json_payload)
    assert callable(write_matrix_entry)


def test_dataset_inventory_facade_exports_public_api() -> None:
    from sol_execbench.core.dataset.inventory import (
        CategoryInventoryRecord,
        DatasetInventory,
        InventoryDenominators,
        InventoryDiagnostic,
        ProblemDefinitionInventory,
        ProblemInventoryRecord,
        WorkloadInventoryRecord,
        build_dataset_inventory,
        write_dataset_inventory,
    )

    assert CategoryInventoryRecord.__name__ == "CategoryInventoryRecord"
    assert DatasetInventory.__name__ == "DatasetInventory"
    assert InventoryDenominators.__name__ == "InventoryDenominators"
    assert InventoryDiagnostic.__name__ == "InventoryDiagnostic"
    assert ProblemDefinitionInventory.__name__ == "ProblemDefinitionInventory"
    assert ProblemInventoryRecord.__name__ == "ProblemInventoryRecord"
    assert WorkloadInventoryRecord.__name__ == "WorkloadInventoryRecord"
    assert callable(build_dataset_inventory)
    assert callable(write_dataset_inventory)
