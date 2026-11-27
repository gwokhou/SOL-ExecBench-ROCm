# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Tier 2 vLLM and Tier 3 AITER adapters for BF16/FP8 Fused MoE."""

from __future__ import annotations

import contextlib
import importlib
from collections.abc import Callable

import torch

import sol_execbench.vertical.adapters.common as _common
from sol_execbench.vertical.adapters.common import (
    _FP8_TORCH_DTYPES,
    DynamicCallable,
    PreparedRunner,
    _as_callable,
    _as_int,
    _as_tensor,
    _self_validate,
    _try_aiter_preshuffle_helper,
    _unavailable,
)


def _pack_moe_weights_once(
    gate_w: torch.Tensor,
    up_w: torch.Tensor,
    down_w: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    w1_packed = torch.cat([gate_w, up_w], dim=-1).transpose(1, 2).contiguous()
    w2_packed = down_w.transpose(1, 2).contiguous()
    return w1_packed, w2_packed


def _resolve_vllm_moe_callables() -> (
    tuple[DynamicCallable | None, DynamicCallable | None] | None
):
    try:
        moe_mod = importlib.import_module(
            "vllm.model_executor.layers.fused_moe"
        )
    except Exception:  # noqa: BLE001 -- vllm fused_moe optional
        return None
    fused_moe_fn = getattr(moe_mod, "fused_moe", None)
    if fused_moe_fn is not None and not callable(fused_moe_fn):
        fused_moe_fn = getattr(fused_moe_fn, "fused_moe", None)
    fused_experts_fn = getattr(moe_mod, "fused_experts", None)
    if fused_moe_fn is None and fused_experts_fn is None:
        sub_mod = getattr(moe_mod, "fused_moe", None)
        if sub_mod is not None:
            fused_moe_fn = getattr(sub_mod, "fused_moe", None)
            fused_experts_fn = getattr(sub_mod, "fused_experts", None)
    if fused_experts_fn is None:
        with contextlib.suppress(ImportError, RuntimeError, AttributeError):
            sub_mod = importlib.import_module(
                "vllm.model_executor.layers.fused_moe.fused_moe"
            )
            fused_experts_fn = getattr(sub_mod, "fused_experts", None)
    return fused_moe_fn, fused_experts_fn


def _probe_vllm_bf16_experts(
    fused_experts_fn: Callable[..., object],
    cand_args: list[object],
    prototype_args: list[object],
    *,
    ref_fn: Callable[..., torch.Tensor],
    max_atol: float,
) -> PreparedRunner | None:
    for use_inplace in (False, True):

        def _make_experts_runner(
            *, pass_inplace: bool
        ) -> Callable[..., object]:
            def _run_experts(
                t: torch.Tensor,
                w1: torch.Tensor,
                w2: torch.Tensor,
                lg: torch.Tensor,
                k: int,
            ) -> object:
                probs = torch.softmax(lg.float(), dim=-1)
                weights, ids = torch.topk(probs, k=int(k), dim=-1)
                w_norm = (weights / weights.sum(dim=-1, keepdim=True)).to(
                    t.dtype
                )
                if pass_inplace:
                    return fused_experts_fn(
                        t,
                        w1,
                        w2,
                        w_norm,
                        ids.to(torch.int32),
                        inplace=False,
                    )
                return fused_experts_fn(t, w1, w2, w_norm, ids.to(torch.int32))

            return _run_experts

        experts_runner = _make_experts_runner(pass_inplace=use_inplace)
        if _self_validate(
            experts_runner, cand_args, ref_fn, prototype_args, max_atol
        ):
            return PreparedRunner(
                fn=experts_runner,
                implementation_name="vllm_fused_moe",
                candidate_args=cand_args,
            )
    return None


def try_make_vllm_moe_runner(
    prototype_args: list[object],
    *,
    ref_fn: Callable[..., torch.Tensor],
    max_atol: float,
) -> PreparedRunner:
    """Build a setup-packed Tier-2 vLLM Fused MoE runner (BF16 or FP8)."""
    if _as_tensor(prototype_args[0]).dtype in _FP8_TORCH_DTYPES:
        return try_make_vllm_fp8_moe_runner(
            prototype_args, ref_fn=ref_fn, max_atol=max_atol
        )
    if not _common._is_rocm_torch():
        return _unavailable(prototype_args, "rocm_torch_required")
    resolved = _resolve_vllm_moe_callables()
    if resolved is None:
        return _unavailable(prototype_args, "vllm_moe_not_installed")
    fused_moe_fn, fused_experts_fn = resolved
    tokens = _as_tensor(prototype_args[0])
    gate_w = _as_tensor(prototype_args[1])
    up_w = _as_tensor(prototype_args[2])
    down_w = _as_tensor(prototype_args[3])
    logits = _as_tensor(prototype_args[4])
    top_k = _as_int(prototype_args[5])
    w1_packed, w2_packed = _pack_moe_weights_once(gate_w, up_w, down_w)
    cand_args: list[object] = [tokens, w1_packed, w2_packed, logits, int(top_k)]

    if callable(fused_moe_fn):
        for runner in (
            lambda t, w1, w2, lg, k: fused_moe_fn(
                t, w1, w2, lg, k, renormalize=True
            ),
            lambda t, w1, w2, lg, k: fused_moe_fn(t, w1, w2, lg, k),
        ):
            if _self_validate(
                runner, cand_args, ref_fn, prototype_args, max_atol
            ):
                return PreparedRunner(
                    fn=runner,
                    implementation_name="vllm_fused_moe",
                    candidate_args=cand_args,
                )

    if callable(fused_experts_fn):
        matched = _probe_vllm_bf16_experts(
            fused_experts_fn,
            cand_args,
            prototype_args,
            ref_fn=ref_fn,
            max_atol=max_atol,
        )
        if matched is not None:
            return matched
    return _unavailable(prototype_args, "vllm_moe_unsupported")


def _build_aiter_moe_weight_pairs(
    gate_w: torch.Tensor,
    up_w: torch.Tensor,
    down_w: torch.Tensor,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    w1_packed, w2_packed = _pack_moe_weights_once(gate_w, up_w, down_w)
    pairs: list[tuple[torch.Tensor, torch.Tensor]] = []
    shuffle_helper = _try_aiter_preshuffle_helper()
    if shuffle_helper is not None:
        with contextlib.suppress(
            RuntimeError, TypeError, ValueError, AttributeError
        ):
            pairs.append(
                (
                    shuffle_helper(w1_packed, layout=(16, 16)),
                    shuffle_helper(w2_packed, layout=(16, 16)),
                )
            )
    pairs.append((w1_packed, w2_packed))
    return pairs


def try_make_aiter_moe_runner(
    prototype_args: list[object],
    *,
    ref_fn: Callable[..., torch.Tensor],
    max_atol: float,
) -> PreparedRunner:
    """Build a setup-packed Tier-3 AITER Fused MoE runner (BF16 or FP8)."""
    if _as_tensor(prototype_args[0]).dtype in _FP8_TORCH_DTYPES:
        return try_make_aiter_fp8_moe_runner(
            prototype_args, ref_fn=ref_fn, max_atol=max_atol
        )
    if not _common._is_rocm_torch():
        return _unavailable(prototype_args, "rocm_torch_required")
    try:
        aiter_mod = importlib.import_module("aiter")
    except Exception:  # noqa: BLE001 -- aiter optional on ROCm hosts
        return _unavailable(prototype_args, "aiter_not_installed")

    tokens = _as_tensor(prototype_args[0])
    gate_w = _as_tensor(prototype_args[1])
    up_w = _as_tensor(prototype_args[2])
    down_w = _as_tensor(prototype_args[3])
    logits = _as_tensor(prototype_args[4])
    top_k = _as_int(prototype_args[5])
    weight_pairs = _build_aiter_moe_weight_pairs(gate_w, up_w, down_w)
    candidates: list[tuple[str, Callable[..., object]]] = []
    with contextlib.suppress(ImportError, RuntimeError, AttributeError):
        fm_mod = importlib.import_module("aiter.fused_moe")
        fm_fn = getattr(fm_mod, "fused_moe", None)
        if callable(fm_fn):
            candidates.append(("aiter_fused_moe", fm_fn))
    for attr, impl_name in (
        ("fused_moe", "aiter_fused_moe"),
        ("ck_moe", "aiter_ck_moe"),
    ):
        target = getattr(aiter_mod, attr, None)
        if target is not None and not callable(target):
            target = getattr(target, attr, None)
        if callable(target):
            candidates.append((impl_name, target))
    if not candidates:
        return _unavailable(prototype_args, "aiter_moe_unavailable")

    for w1_cand, w2_cand in weight_pairs:
        cand_args: list[object] = [tokens, w1_cand, w2_cand, logits, int(top_k)]
        for impl_name, moe_fn in candidates:

            def _make_runner(
                fn: Callable[..., object], wt_dtype: torch.dtype | None
            ) -> Callable[..., object]:
                def _run(
                    t: torch.Tensor,
                    w1: torch.Tensor,
                    w2: torch.Tensor,
                    lg: torch.Tensor,
                    k: int,
                ) -> object:
                    probs = torch.softmax(lg.float(), dim=-1)
                    weights, ids = torch.topk(probs, k=int(k), dim=-1)
                    w_norm = weights / weights.sum(dim=-1, keepdim=True)
                    if wt_dtype is not None:
                        w_norm = w_norm.to(wt_dtype)
                    return fn(t, w1, w2, w_norm, ids.to(torch.int32))

                return _run

            for wt_dtype in (torch.float32, tokens.dtype):
                runner = _make_runner(moe_fn, wt_dtype)
                if _self_validate(
                    runner, cand_args, ref_fn, prototype_args, max_atol
                ):
                    return PreparedRunner(
                        fn=runner,
                        implementation_name=impl_name,
                        candidate_args=cand_args,
                    )
    return _unavailable(prototype_args, "aiter_moe_unsupported")


def _build_vllm_fp8_quant_configs(
    w1_dtype: torch.dtype,
    scales: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> list[object]:
    w1_s, w2_s, a_s = scales
    configs: list[object] = []
    mods: list[object] = []
    for mod_name in (
        "vllm.model_executor.layers.fused_moe.config",
        "vllm.model_executor.layers.fused_moe",
    ):
        with contextlib.suppress(ImportError, RuntimeError, AttributeError):
            mods.append(importlib.import_module(mod_name))
    scale_pairs = (
        (w1_s, w2_s),
        (w1_s.squeeze(-1).contiguous(), w2_s.squeeze(-1).contiguous()),
    )
    for mod in mods:
        helper = getattr(mod, "fp8_w8a8_moe_quant_config", None)
        if callable(helper):
            for ws1, ws2 in scale_pairs:
                for act_s in (a_s, None):
                    with contextlib.suppress(Exception):
                        configs.append(
                            helper(w1_scale=ws1, w2_scale=ws2, a1_scale=act_s)
                        )
        cfg_cls = getattr(mod, "FusedMoEQuantConfig", None)
        make_fn = getattr(cfg_cls, "make", None) if cfg_cls else None
        if callable(make_fn):
            for ws1, ws2 in scale_pairs:
                for act_s in (a_s, None):
                    with contextlib.suppress(Exception):
                        configs.append(
                            make_fn(
                                w1_dtype,
                                w1_scale=ws1,
                                w2_scale=ws2,
                                a1_scale=act_s,
                            )
                        )
    return configs


def _make_vllm_experts_runner(
    fused_experts_fn: Callable[..., object],
    *,
    quant_config: object = None,
    scales: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    weight_dtype: torch.dtype = torch.float32,
) -> Callable[..., object]:
    def _run(
        t: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        lg: torch.Tensor,
        k: int,
    ) -> object:
        probs = torch.softmax(lg.float(), dim=-1)
        weights, ids = torch.topk(probs, k=int(k), dim=-1)
        w_norm = (weights / weights.sum(dim=-1, keepdim=True)).to(weight_dtype)
        if quant_config is not None:
            return fused_experts_fn(
                t,
                w1,
                w2,
                w_norm,
                ids.to(torch.int32),
                quant_config=quant_config,
            )
        if scales is None:
            raise ValueError("scales required when quant_config is None")
        w1_scale, w2_scale, a_scale = scales
        return fused_experts_fn(
            t,
            w1,
            w2,
            w_norm,
            ids.to(torch.int32),
            inplace=False,
            use_fp8_w8a8=True,
            w1_scale=w1_scale,
            w2_scale=w2_scale,
            a1_scale=a_scale,
        )

    return _run


def _probe_vllm_fp8_moe_candidates(
    fused_moe_fn: object,
    fused_experts_fn: object,
    cand_args: list[object],
    prototype_args: list[object],
    scales: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    *,
    ref_fn: Callable[..., object],
    max_atol: float,
) -> PreparedRunner | None:
    w1_scale, w2_scale, a_scale = scales
    quant_configs = _build_vllm_fp8_quant_configs(
        _as_tensor(cand_args[1]).dtype, scales
    )
    moe_call = _as_callable(fused_moe_fn)
    if moe_call is not None:
        moe_runners: list[Callable[..., object]] = [
            lambda t, w1, w2, lg, k, _qc=qc: moe_call(
                t, w1, w2, lg, k, renormalize=True, quant_config=_qc
            )
            for qc in quant_configs
        ]
        moe_runners.extend(
            (
                lambda t, w1, w2, lg, k: moe_call(
                    t,
                    w1,
                    w2,
                    lg,
                    k,
                    renormalize=True,
                    use_fp8_w8a8=True,
                    w1_scale=w1_scale,
                    w2_scale=w2_scale,
                    a1_scale=a_scale,
                ),
                lambda t, w1, w2, lg, k: moe_call(
                    t, w1, w2, lg, k, renormalize=True, use_fp8_w8a8=True
                ),
            )
        )
        for runner in moe_runners:
            if _self_validate(
                runner, cand_args, ref_fn, prototype_args, max_atol
            ):
                return PreparedRunner(
                    fn=runner,
                    implementation_name="vllm_fp8_fused_moe",
                    candidate_args=cand_args,
                )
    experts_call = _as_callable(fused_experts_fn)
    if experts_call is not None:
        expert_runners = [
            _make_vllm_experts_runner(
                experts_call, quant_config=qc, weight_dtype=w_dt
            )
            for qc in quant_configs
            for w_dt in (torch.float32, torch.bfloat16)
        ]
        expert_runners.append(
            _make_vllm_experts_runner(experts_call, scales=scales)
        )
        for runner in expert_runners:
            if _self_validate(
                runner, cand_args, ref_fn, prototype_args, max_atol
            ):
                return PreparedRunner(
                    fn=runner,
                    implementation_name="vllm_fp8_fused_moe",
                    candidate_args=cand_args,
                )
    return None


def try_make_vllm_fp8_moe_runner(
    prototype_args: list[object],
    *,
    ref_fn: Callable[..., torch.Tensor],
    max_atol: float,
) -> PreparedRunner:
    """Build a setup-packed Tier-2 vLLM FP8 Fused MoE runner."""
    if not _common._is_rocm_torch():
        return _unavailable(prototype_args, "rocm_torch_required")
    resolved = _resolve_vllm_moe_callables()
    if resolved is None:
        return _unavailable(prototype_args, "vllm_moe_not_installed")
    fused_moe_fn, fused_experts_fn = resolved
    tokens = _as_tensor(prototype_args[0])
    gate_w = _as_tensor(prototype_args[1])
    up_w = _as_tensor(prototype_args[2])
    down_w = _as_tensor(prototype_args[3])
    logits = _as_tensor(prototype_args[4])
    top_k = _as_int(prototype_args[5])
    w1_packed, w2_packed = _pack_moe_weights_once(gate_w, up_w, down_w)
    num_e = int(gate_w.shape[0])
    scales = (
        torch.ones((num_e, 1), dtype=torch.float32, device=tokens.device),
        torch.ones((num_e, 1), dtype=torch.float32, device=tokens.device),
        torch.ones((1,), dtype=torch.float32, device=tokens.device),
    )
    t_bf16 = tokens.to(torch.bfloat16)
    for t_cand in (t_bf16, tokens):
        cand_args: list[object] = [
            t_cand,
            w1_packed,
            w2_packed,
            logits,
            int(top_k),
        ]
        matched = _probe_vllm_fp8_moe_candidates(
            fused_moe_fn,
            fused_experts_fn,
            cand_args,
            prototype_args,
            scales,
            ref_fn=ref_fn,
            max_atol=max_atol,
        )
        if matched is not None:
            return matched
    return _unavailable(prototype_args, "vllm_fp8_moe_unsupported")


def _probe_aiter_fp8_moe_fn(
    moe_fn: Callable[..., object],
    cand_args: list[object],
    prototype_args: list[object],
    scales: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    *,
    ref_fn: Callable[..., object],
    max_atol: float,
) -> PreparedRunner | None:
    w1_scale, w2_scale, a_scale = scales
    calls: tuple[Callable[..., object], ...] = (
        lambda t, w1, w2, wn, ids: moe_fn(
            t,
            w1,
            w2,
            wn,
            ids,
            w1_scale=w1_scale,
            w2_scale=w2_scale,
            a1_scale=a_scale,
        ),
        lambda t, w1, w2, wn, ids: moe_fn(
            t, w1, w2, wn, ids, w1_scale, w2_scale
        ),
        lambda t, w1, w2, wn, ids: moe_fn(t, w1, w2, wn, ids),
    )
    for call_fn in calls:

        def _make_runner(
            fn: Callable[..., object],
        ) -> Callable[..., object]:
            def _run(
                t: torch.Tensor,
                w1: torch.Tensor,
                w2: torch.Tensor,
                lg: torch.Tensor,
                k: int,
            ) -> object:
                probs = torch.softmax(lg.float(), dim=-1)
                weights, ids = torch.topk(probs, k=int(k), dim=-1)
                w_norm = (weights / weights.sum(dim=-1, keepdim=True)).to(
                    torch.float32
                )
                return fn(t, w1, w2, w_norm, ids.to(torch.int32))

            return _run

        runner = _make_runner(call_fn)
        if _self_validate(runner, cand_args, ref_fn, prototype_args, max_atol):
            return PreparedRunner(
                fn=runner,
                implementation_name="aiter_fp8_fused_moe",
                candidate_args=cand_args,
            )
    return None


def try_make_aiter_fp8_moe_runner(
    prototype_args: list[object],
    *,
    ref_fn: Callable[..., torch.Tensor],
    max_atol: float,
) -> PreparedRunner:
    """Build a setup-packed Tier-3 AITER FP8 Fused MoE runner."""
    if not _common._is_rocm_torch():
        return _unavailable(prototype_args, "rocm_torch_required")
    try:
        aiter_mod = importlib.import_module("aiter")
    except Exception:  # noqa: BLE001 -- aiter optional on ROCm hosts
        return _unavailable(prototype_args, "aiter_not_installed")
    tokens = _as_tensor(prototype_args[0])
    gate_w = _as_tensor(prototype_args[1])
    up_w = _as_tensor(prototype_args[2])
    down_w = _as_tensor(prototype_args[3])
    logits = _as_tensor(prototype_args[4])
    top_k = _as_int(prototype_args[5])
    weight_pairs = _build_aiter_moe_weight_pairs(gate_w, up_w, down_w)
    candidates: list[Callable[..., object]] = []
    quant_type: object = None
    with contextlib.suppress(ImportError, RuntimeError, AttributeError):
        fm_mod = importlib.import_module("aiter.fused_moe")
        fm_fn = getattr(fm_mod, "fused_moe", None)
        if callable(fm_fn):
            candidates.append(fm_fn)
        qt_enum = getattr(fm_mod, "QuantType", None)
        quant_type = getattr(qt_enum, "per_Tensor", None) if qt_enum else None
    for attr in ("fused_moe", "fmoe_fp8_blockscale_g1u1", "ck_moe"):
        target = getattr(aiter_mod, attr, None)
        if target is not None and not callable(target):
            target = getattr(target, attr, None)
        if callable(target):
            candidates.append(target)
    if not candidates:
        return _unavailable(prototype_args, "aiter_fp8_moe_unavailable")
    num_e = int(gate_w.shape[0])
    scales = (
        torch.ones((num_e, 1), dtype=torch.float32, device=tokens.device),
        torch.ones((num_e, 1), dtype=torch.float32, device=tokens.device),
        torch.ones((1,), dtype=torch.float32, device=tokens.device),
    )
    t_bf16 = tokens.to(torch.bfloat16)
    for t_cand in (t_bf16, tokens):
        for w1_cand, w2_cand in weight_pairs:
            cand_args: list[object] = [
                t_cand,
                w1_cand,
                w2_cand,
                logits,
                int(top_k),
            ]
            for moe_fn in candidates:
                bound_fn = (
                    (
                        lambda *args, _fn=moe_fn, _qt=quant_type, **kw: _fn(
                            *args, quant_type=_qt, **kw
                        )
                    )
                    if quant_type is not None
                    else moe_fn
                )
                matched = _probe_aiter_fp8_moe_fn(
                    bound_fn,
                    cand_args,
                    prototype_args,
                    scales,
                    ref_fn=ref_fn,
                    max_atol=max_atol,
                )
                if matched is not None:
                    return matched
    return _unavailable(prototype_args, "aiter_fp8_moe_unsupported")
