from __future__ import annotations

from sol_execbench.core.bench.timing_policy import (
    TimingActivityDomain,
    TimingBackend,
    TimingSourceType,
    classify_timing_source,
    select_timing_policy,
    timing_policy_for_languages,
    timing_policy_table,
)
from sol_execbench.core.data.solution import SupportedLanguages


def test_supported_languages_map_to_timing_source_types():
    assert classify_timing_source([SupportedLanguages.PYTORCH]) == TimingSourceType.PYTORCH
    assert classify_timing_source([SupportedLanguages.TRITON]) == TimingSourceType.TRITON

    for language in (
        SupportedLanguages.HIP_CPP,
        SupportedLanguages.HIPBLAS,
        SupportedLanguages.MIOPEN,
        SupportedLanguages.CK,
        SupportedLanguages.ROCWMMA,
    ):
        assert classify_timing_source([language]) == TimingSourceType.HIP_NATIVE


def test_classifier_handles_strings_empty_and_mixed_inputs():
    assert classify_timing_source(["pytorch"]) == TimingSourceType.PYTORCH
    assert classify_timing_source(["triton", "pytorch"]) == TimingSourceType.TRITON
    assert classify_timing_source([]) == TimingSourceType.UNKNOWN
    assert classify_timing_source(["not-a-language"]) == TimingSourceType.UNKNOWN
    assert classify_timing_source(["pytorch", "hip_cpp"]) == TimingSourceType.MIXED


def test_policy_table_has_distinct_source_specific_interpretations():
    policies = {policy.source_type: policy for policy in timing_policy_table()}

    assert set(policies) == set(TimingSourceType)
    assert policies[TimingSourceType.PYTORCH].backend == TimingBackend.PYTORCH_PROFILER
    assert (
        policies[TimingSourceType.PYTORCH].activity_domain
        == TimingActivityDomain.PYTORCH_OPERATOR_ATTRIBUTION
    )
    assert policies[TimingSourceType.TRITON].backend == TimingBackend.ROCPROFV3
    assert (
        policies[TimingSourceType.TRITON].activity_domain
        == TimingActivityDomain.KERNEL_ACTIVITY
    )
    assert policies[TimingSourceType.HIP_NATIVE].backend == TimingBackend.ROCPROFV3
    assert (
        policies[TimingSourceType.HIP_NATIVE].activity_domain
        == TimingActivityDomain.KERNEL_ACTIVITY
    )
    assert len(
        {
            policies[TimingSourceType.PYTORCH].interpretation,
            policies[TimingSourceType.TRITON].interpretation,
            policies[TimingSourceType.HIP_NATIVE].interpretation,
        }
    ) == 3


def test_every_policy_exposes_auditable_metadata():
    for policy in timing_policy_table():
        payload = policy.to_dict()

        assert payload["source_type"]
        assert payload["backend"]
        assert payload["activity_domain"]
        assert payload["aggregation_rule"]
        assert payload["interpretation"]
        assert payload["reason"]
        assert isinstance(payload["fallback_applied"], bool)


def test_event_timing_fallback_is_labeled_and_not_profiler_backed():
    policy = select_timing_policy(TimingSourceType.HIP_NATIVE, profiler_available=False)

    assert policy.backend == TimingBackend.DEVICE_EVENTS
    assert policy.activity_domain == TimingActivityDomain.FALLBACK_EVENT_TIMING
    assert policy.fallback_applied is True
    assert "profiler-backed timing is unavailable" == policy.reason
    assert "not profiler-backed kernel activity timing" in policy.interpretation


def test_language_policy_helper_combines_classification_and_selection():
    policy = timing_policy_for_languages(["hip_cpp"])

    assert policy.source_type == TimingSourceType.HIP_NATIVE
    assert policy.backend == TimingBackend.ROCPROFV3
