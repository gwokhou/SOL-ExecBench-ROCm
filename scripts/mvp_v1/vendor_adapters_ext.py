# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Setup-time pre-packing Tier 2 vLLM / Tier 3 AITER adapters for Stage-2 ops."""

from __future__ import annotations

import importlib
import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch

from scripts.mvp_v1.vendor_adapters import (
    PreparedRunner,
    _build_aiter_moe_weight_pairs,
    _import_vllm_custom_ops,
    _is_rocm_torch,
    _pack_moe_weights_once,
    _resolve_vllm_moe_callables,
    _self_validate,
    _unavailable,
    build_paged_tables_and_seqlens,
)


def _probe_vllm_fp8_moe_candidates(
    fused_moe_fn: Any,
    fused_experts_fn: Any,
    cand_args: list[Any],
    prototype_args: list[Any],
    scales: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    *,
    ref_fn: Callable[..., Any],
    max_atol: float,
) -> PreparedRunner | None:
    w1_scale, w2_scale, a_scale = scales
    if callable(fused_moe_fn):
        for runner in (
            lambda t, w1, w2, lg, k: fused_moe_fn(
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
            lambda t, w1, w2, lg, k: fused_moe_fn(
                t, w1, w2, lg, k, renormalize=True, use_fp8_w8a8=True
            ),
        ):
            if _self_validate(
                runner, cand_args, ref_fn, prototype_args, max_atol
            ):
                return PreparedRunner(
                    fn=runner,
                    implementation_name="vllm_fp8_fused_moe",
                    candidate_args=cand_args,
                )
    if callable(fused_experts_fn):

        def _run_experts(
            t: torch.Tensor,
            w1: torch.Tensor,
            w2: torch.Tensor,
            lg: torch.Tensor,
            k: int,
        ) -> torch.Tensor:
            probs = torch.softmax(lg.float(), dim=-1)
            weights, ids = torch.topk(probs, k=int(k), dim=-1)
            w_norm = (weights / weights.sum(dim=-1, keepdim=True)).to(
                torch.float32
            )
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

        if _self_validate(
            _run_experts, cand_args, ref_fn, prototype_args, max_atol
        ):
            return PreparedRunner(
                fn=_run_experts,
                implementation_name="vllm_fp8_fused_moe",
                candidate_args=cand_args,
            )
    return None


def try_make_vllm_fp8_moe_runner(
    prototype_args: list[Any],
    *,
    ref_fn: Callable[..., torch.Tensor],
    max_atol: float,
) -> PreparedRunner:
    """Build a setup-packed Tier-2 vLLM FP8 Fused MoE runner."""
    if not _is_rocm_torch():
        return _unavailable(prototype_args, "rocm_torch_required")
    resolved = _resolve_vllm_moe_callables()
    if resolved is None:
        return _unavailable(prototype_args, "vllm_moe_not_installed")
    fused_moe_fn, fused_experts_fn = resolved
    tokens, gate_w, up_w, down_w, logits, top_k = prototype_args
    w1_packed, w2_packed = _pack_moe_weights_once(gate_w, up_w, down_w)
    num_e = int(gate_w.shape[0])
    scales = (
        torch.ones((num_e, 1), dtype=torch.float32, device=tokens.device),
        torch.ones((num_e, 1), dtype=torch.float32, device=tokens.device),
        torch.ones((1,), dtype=torch.float32, device=tokens.device),
    )
    t_bf16 = tokens.to(torch.bfloat16)
    for t_cand in (t_bf16, tokens):
        cand_args: list[Any] = [
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
    moe_fn: Callable[..., torch.Tensor],
    cand_args: list[Any],
    prototype_args: list[Any],
    scales: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    *,
    ref_fn: Callable[..., Any],
    max_atol: float,
) -> PreparedRunner | None:
    w1_scale, w2_scale, a_scale = scales
    calls: tuple[
        Callable[
            [
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
            ],
            torch.Tensor,
        ],
        ...,
    ] = (
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
            fn: Callable[..., torch.Tensor],
        ) -> Callable[..., torch.Tensor]:
            def _run(
                t: torch.Tensor,
                w1: torch.Tensor,
                w2: torch.Tensor,
                lg: torch.Tensor,
                k: int,
            ) -> torch.Tensor:
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
    prototype_args: list[Any],
    *,
    ref_fn: Callable[..., torch.Tensor],
    max_atol: float,
) -> PreparedRunner:
    """Build a setup-packed Tier-3 AITER FP8 Fused MoE runner."""
    if not _is_rocm_torch():
        return _unavailable(prototype_args, "rocm_torch_required")
    try:
        aiter_mod = importlib.import_module("aiter")
    except Exception:  # noqa: BLE001 -- aiter optional on ROCm hosts
        return _unavailable(prototype_args, "aiter_not_installed")
    tokens, gate_w, up_w, down_w, logits, top_k = prototype_args
    weight_pairs = _build_aiter_moe_weight_pairs(gate_w, up_w, down_w)
    candidates: list[Callable[..., Any]] = []
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
            cand_args: list[Any] = [
                t_cand,
                w1_cand,
                w2_cand,
                logits,
                int(top_k),
            ]
            for moe_fn in candidates:
                matched = _probe_aiter_fp8_moe_fn(
                    moe_fn,
                    cand_args,
                    prototype_args,
                    scales,
                    ref_fn=ref_fn,
                    max_atol=max_atol,
                )
                if matched is not None:
                    return matched
    return _unavailable(prototype_args, "aiter_fp8_moe_unsupported")


@dataclass(frozen=True, slots=True, kw_only=True)
class _PackedMLA:
    cand_args: list[Any]
    qo_indptr: torch.Tensor
    d_ckv: int
    sm_scale: float


def _pack_mla_paged_once(prototype_args: list[Any]) -> _PackedMLA:
    q, kv_cache, kv_indptr, kv_indices, kv_last_len, d_ckv = prototype_args
    d_ckv_i = int(d_ckv)
    kv_4d = kv_cache.unsqueeze(2).contiguous()
    batch = int(q.shape[0])
    qo_indptr = torch.arange(0, batch + 1, dtype=torch.int32, device=q.device)
    block_tables, seq_lens, _ = build_paged_tables_and_seqlens(
        kv_indptr,
        kv_indices,
        kv_last_len,
        page_size=int(kv_cache.shape[1]),
        device=q.device,
    )
    cand_args: list[Any] = [
        q,
        kv_4d,
        kv_indptr,
        kv_indices,
        kv_last_len,
        block_tables,
        seq_lens,
    ]
    return _PackedMLA(
        cand_args=cand_args,
        qo_indptr=qo_indptr,
        d_ckv=d_ckv_i,
        sm_scale=1.0 / math.sqrt(int(q.shape[2])),
    )


def _build_mla_candidate_runners(
    mla_fn: Callable[..., Any],
    packed: _PackedMLA,
    *,
    allow_return: bool,
) -> list[Callable[..., torch.Tensor]]:
    qo_indptr = packed.qo_indptr
    d_ckv = packed.d_ckv
    sm_scale = packed.sm_scale

    def _run_varlen(
        q_in: torch.Tensor,
        kv_in: torch.Tensor,
        indptr_in: torch.Tensor,
        indices_in: torch.Tensor,
        last_in: torch.Tensor,
        _bt: torch.Tensor,
        _sl: torch.Tensor,
    ) -> torch.Tensor:
        out = torch.empty(
            (q_in.shape[0], q_in.shape[1], d_ckv),
            dtype=q_in.dtype,
            device=q_in.device,
        )
        mla_fn(
            q_in,
            kv_in,
            out,
            qo_indptr,
            indptr_in,
            indices_in,
            last_in,
            1,
            sm_scale=sm_scale,
        )
        return out

    def _run_paged(
        q_in: torch.Tensor,
        kv_in: torch.Tensor,
        _indptr: torch.Tensor,
        _indices: torch.Tensor,
        _last: torch.Tensor,
        bt_in: torch.Tensor,
        sl_in: torch.Tensor,
    ) -> torch.Tensor:
        out = torch.empty(
            (q_in.shape[0], q_in.shape[1], d_ckv),
            dtype=q_in.dtype,
            device=q_in.device,
        )
        mla_fn(out, q_in, kv_in, bt_in, sl_in, sm_scale)
        return out

    candidates: list[Callable[..., torch.Tensor]] = [_run_varlen, _run_paged]
    if allow_return:
        candidates.append(
            lambda q_in, kv_in, indptr_in, indices_in, last_in, _bt, _sl: (
                mla_fn(
                    q_in,
                    kv_in,
                    qo_indptr,
                    indptr_in,
                    indices_in,
                    last_in,
                    d_ckv,
                    sm_scale,
                )
            )
        )
    return candidates


def _probe_mla_decode_fn(
    mla_fn: Callable[..., Any],
    packed: _PackedMLA,
    prototype_args: list[Any],
    *,
    impl_name: str,
    allow_return: bool,
    ref_fn: Callable[..., Any],
    max_atol: float,
) -> PreparedRunner | None:
    candidates = _build_mla_candidate_runners(
        mla_fn, packed, allow_return=allow_return
    )
    for runner in candidates:
        if _self_validate(
            runner, packed.cand_args, ref_fn, prototype_args, max_atol
        ):
            return PreparedRunner(
                fn=runner,
                implementation_name=impl_name,
                candidate_args=packed.cand_args,
            )
    return None


def try_make_vllm_mla_runner(
    prototype_args: list[Any],
    *,
    ref_fn: Callable[..., torch.Tensor],
    max_atol: float,
) -> PreparedRunner:
    """Build a setup-packed Tier-2 vLLM MLA Paged Decode runner."""
    if not _is_rocm_torch():
        return _unavailable(prototype_args, "rocm_torch_required")
    loaded = _import_vllm_custom_ops()
    if loaded is None or loaded[1] is None:
        return _unavailable(prototype_args, "vllm_ops_not_installed")
    _vllm_mod, ops_mod = loaded
    mla_fn = getattr(ops_mod, "mla_decode_fwd", None)
    if not callable(mla_fn):
        return _unavailable(prototype_args, "vllm_mla_not_installed")
    packed = _pack_mla_paged_once(prototype_args)
    matched = _probe_mla_decode_fn(
        mla_fn,
        packed,
        prototype_args,
        impl_name="vllm_mla_decode_fwd",
        allow_return=False,
        ref_fn=ref_fn,
        max_atol=max_atol,
    )
    if matched is not None:
        return matched
    return _unavailable(prototype_args, "vllm_mla_unsupported")


def try_make_aiter_mla_runner(
    prototype_args: list[Any],
    *,
    ref_fn: Callable[..., torch.Tensor],
    max_atol: float,
) -> PreparedRunner:
    """Build a setup-packed Tier-3 AITER MLA Paged Decode runner."""
    if not _is_rocm_torch():
        return _unavailable(prototype_args, "rocm_torch_required")
    try:
        aiter_mod = importlib.import_module("aiter")
    except Exception:  # noqa: BLE001 -- aiter optional on ROCm hosts
        return _unavailable(prototype_args, "aiter_not_installed")
    mla_fn = getattr(aiter_mod, "mla_decode_fwd", None)
    if not callable(mla_fn):
        mla_sub = getattr(aiter_mod, "mla", None)
        mla_fn = getattr(mla_sub, "mla_decode_fwd", None) if mla_sub else None
    if not callable(mla_fn):
        try:
            mla_mod = importlib.import_module("aiter.mla")
            mla_fn = getattr(mla_mod, "mla_decode_fwd", None)
        except Exception:  # noqa: BLE001 -- aiter.mla optional
            mla_fn = None
    if not callable(mla_fn):
        return _unavailable(prototype_args, "aiter_mla_not_installed")
    packed = _pack_mla_paged_once(prototype_args)
    matched = _probe_mla_decode_fn(
        mla_fn,
        packed,
        prototype_args,
        impl_name="aiter_mla_decode_fwd",
        allow_return=True,
        ref_fn=ref_fn,
        max_atol=max_atol,
    )
    if matched is not None:
        return matched
    return _unavailable(prototype_args, "aiter_mla_unsupported")


def _cos_sin_candidates(
    x: torch.Tensor,
    cos_sin: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    cos_sin_cast = cos_sin.to(dtype=x.dtype).contiguous()
    if cos_sin.dtype == x.dtype:
        return (cos_sin_cast,)
    return (cos_sin_cast, cos_sin.contiguous())


def _probe_vllm_rmsnorm_fused(
    fused_fn: Callable[..., Any],
    cand_args_list: list[list[Any]],
    prototype_args: list[Any],
    *,
    ref_fn: Callable[..., Any],
    max_atol: float,
) -> PreparedRunner | None:
    def _run_fused(
        x_in: torch.Tensor,
        res_in: torch.Tensor,
        w_in: torch.Tensor,
        cs_in: torch.Tensor,
        pos_in: torch.Tensor,
    ) -> torch.Tensor:
        x_clone = x_in.clone()
        res_clone = res_in.clone()
        out = fused_fn(pos_in, x_clone, res_clone, w_in, cs_in, True)
        return x_clone if out is None else out

    for cand_args in cand_args_list:
        if _self_validate(
            _run_fused, cand_args, ref_fn, prototype_args, max_atol
        ):
            return PreparedRunner(
                fn=_run_fused,
                implementation_name="vllm_fused_qk_norm_rope",
                candidate_args=cand_args,
            )
    return None


def try_make_vllm_rmsnorm_rope_runner(
    prototype_args: list[Any],
    *,
    ref_fn: Callable[..., torch.Tensor],
    max_atol: float,
) -> PreparedRunner:
    """Build a setup-packed Tier-2 vLLM Add-RMSNorm + RoPE runner."""
    if not _is_rocm_torch():
        return _unavailable(prototype_args, "rocm_torch_required")
    loaded = _import_vllm_custom_ops()
    if loaded is None or loaded[1] is None:
        return _unavailable(prototype_args, "vllm_ops_not_installed")
    _vllm_mod, ops_mod = loaded
    x, residual, weight, cos_sin = prototype_args
    m, heads, head_dim = x.shape
    positions = torch.arange(m, dtype=torch.int64, device=x.device)
    cand_args_list: list[list[Any]] = [
        [x, residual, weight, cs_cand, positions]
        for cs_cand in _cos_sin_candidates(x, cos_sin)
    ]

    fused_fn = getattr(ops_mod, "fused_qk_norm_rope", None)
    if callable(fused_fn):
        matched = _probe_vllm_rmsnorm_fused(
            fused_fn,
            cand_args_list,
            prototype_args,
            ref_fn=ref_fn,
            max_atol=max_atol,
        )
        if matched is not None:
            return matched

    norm_fn = getattr(ops_mod, "fused_add_rms_norm", None)
    rope_fn = getattr(ops_mod, "rotary_embedding", None)
    if callable(norm_fn) and callable(rope_fn):

        def _run_two_step(
            x_in: torch.Tensor,
            res_in: torch.Tensor,
            w_in: torch.Tensor,
            cs_in: torch.Tensor,
            pos_in: torch.Tensor,
        ) -> torch.Tensor:
            h = x_in.clone()
            res_work = res_in.clone()
            norm_fn(h, res_work, w_in, 1e-6)
            h_flat = h.view(m, heads * head_dim)
            rope_fn(pos_in, h_flat, None, head_dim, cs_in, True)
            return h_flat.view(m, heads, head_dim)

        for cand_args in cand_args_list:
            if _self_validate(
                _run_two_step, cand_args, ref_fn, prototype_args, max_atol
            ):
                return PreparedRunner(
                    fn=_run_two_step,
                    implementation_name="vllm_rmsnorm_rope",
                    candidate_args=cand_args,
                )
    return _unavailable(prototype_args, "vllm_rmsnorm_rope_unsupported")


def _probe_aiter_rmsnorm_two_step(
    norm_fn: Callable[..., Any],
    rope_fn: Callable[..., Any],
    cand_args: list[Any],
    prototype_args: list[Any],
    *,
    with_add: bool,
    ref_fn: Callable[..., Any],
    max_atol: float,
) -> PreparedRunner | None:
    def _run_with_add(
        x_in: torch.Tensor,
        res_in: torch.Tensor,
        w_in: torch.Tensor,
        cs_in: torch.Tensor,
        pos_in: torch.Tensor,
    ) -> torch.Tensor:
        out = torch.empty_like(x_in)
        res_out = torch.empty_like(res_in)
        norm_fn(out, x_in, res_in, res_out, w_in, 1e-6)
        rope_out = rope_fn(out, cs_in, pos_in)
        return out if rope_out is None else rope_out

    def _run_separate(
        x_in: torch.Tensor,
        res_in: torch.Tensor,
        w_in: torch.Tensor,
        cs_in: torch.Tensor,
        pos_in: torch.Tensor,
    ) -> torch.Tensor:
        out = norm_fn(x_in + res_in, w_in, 1e-6)
        rope_out = rope_fn(out, cs_in, pos_in)
        return out if rope_out is None else rope_out

    runner = _run_with_add if with_add else _run_separate
    if _self_validate(runner, cand_args, ref_fn, prototype_args, max_atol):
        return PreparedRunner(
            fn=runner,
            implementation_name="aiter_rmsnorm_rope",
            candidate_args=cand_args,
        )
    return None


def try_make_aiter_rmsnorm_rope_runner(
    prototype_args: list[Any],
    *,
    ref_fn: Callable[..., torch.Tensor],
    max_atol: float,
) -> PreparedRunner:
    """Build a setup-packed Tier-3 AITER Add-RMSNorm + RoPE runner."""
    if not _is_rocm_torch():
        return _unavailable(prototype_args, "rocm_torch_required")
    try:
        aiter_mod = importlib.import_module("aiter")
    except Exception:  # noqa: BLE001 -- aiter optional on ROCm hosts
        return _unavailable(prototype_args, "aiter_not_installed")
    x, residual, weight, cos_sin = prototype_args
    positions = torch.arange(x.shape[0], dtype=torch.int64, device=x.device)
    cand_args_list: list[list[Any]] = [
        [x, residual, weight, cs_cand, positions]
        for cs_cand in _cos_sin_candidates(x, cos_sin)
    ]

    fused_fn = getattr(aiter_mod, "fused_qk_norm_rope", None)
    if callable(fused_fn):

        def _run_fused(
            x_in: torch.Tensor,
            res_in: torch.Tensor,
            w_in: torch.Tensor,
            cs_in: torch.Tensor,
            pos_in: torch.Tensor,
        ) -> torch.Tensor:
            x_clone = x_in.clone()
            res_clone = res_in.clone()
            out = fused_fn(pos_in, x_clone, res_clone, w_in, cs_in)
            return x_clone if out is None else out

        for cand_args in cand_args_list:
            if _self_validate(
                _run_fused, cand_args, ref_fn, prototype_args, max_atol
            ):
                return PreparedRunner(
                    fn=_run_fused,
                    implementation_name="aiter_fused_qk_norm_rope",
                    candidate_args=cand_args,
                )

    rope_fn = getattr(aiter_mod, "rope_cached_positions_2c_fwd", None) or (
        getattr(aiter_mod, "rope_fwd", None)
    )
    add_norm = getattr(aiter_mod, "rmsnorm2d_fwd_with_add", None)
    rms_norm = getattr(aiter_mod, "rms_norm", None)
    if callable(rope_fn):
        for norm_fn, with_add in ((add_norm, True), (rms_norm, False)):
            if not callable(norm_fn):
                continue
            for cand_args in cand_args_list:
                matched = _probe_aiter_rmsnorm_two_step(
                    norm_fn,
                    rope_fn,
                    cand_args,
                    prototype_args,
                    with_add=with_add,
                    ref_fn=ref_fn,
                    max_atol=max_atol,
                )
                if matched is not None:
                    return matched
    return _unavailable(prototype_args, "aiter_rmsnorm_rope_unsupported")


def _probe_swiglu_quant_pair(
    fused_fn: Any,
    silu_fn: Any,
    quant_fn: Any,
    cand_args: list[Any],
    *,
    fused_name: str,
    split_name: str,
    fp8_dtype: torch.dtype,
    ref_fn: Callable[..., Any],
    max_atol: float,
) -> PreparedRunner | None:
    gate_up = cand_args[0]
    m, i2 = gate_up.shape
    inter = i2 // 2
    if callable(fused_fn):

        def _run_fused_out(
            gu_in: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            y_fp8 = torch.empty(
                (m, inter), dtype=fp8_dtype, device=gu_in.device
            )
            y_scale = torch.empty(
                (m, 1), dtype=torch.float32, device=gu_in.device
            )
            fused_fn(y_fp8, y_scale, gu_in)
            return y_fp8, y_scale

        def _run_fused_ret(
            gu_in: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            return fused_fn(gu_in)

        for runner in (_run_fused_ret, _run_fused_out):
            if _self_validate(runner, cand_args, ref_fn, cand_args, max_atol):
                return PreparedRunner(
                    fn=runner,
                    implementation_name=fused_name,
                    candidate_args=cand_args,
                )

    if callable(silu_fn) and callable(quant_fn):

        def _run_two_step(
            gu_in: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            act_out = torch.empty(
                (m, inter), dtype=gu_in.dtype, device=gu_in.device
            )
            silu_fn(act_out, gu_in)
            y_fp8 = torch.empty(
                (m, inter), dtype=fp8_dtype, device=gu_in.device
            )
            y_scale = torch.empty(
                (m, 1), dtype=torch.float32, device=gu_in.device
            )
            quant_fn(y_fp8, act_out, y_scale)
            return y_fp8, y_scale

        if _self_validate(
            _run_two_step, cand_args, ref_fn, cand_args, max_atol
        ):
            return PreparedRunner(
                fn=_run_two_step,
                implementation_name=split_name,
                candidate_args=cand_args,
            )
    return None


def try_make_vllm_swiglu_quant_runner(
    prototype_args: list[Any],
    *,
    fp8_dtype: torch.dtype,
    ref_fn: Callable[..., torch.Tensor],
    max_atol: float,
) -> PreparedRunner:
    """Build a Tier-2 vLLM SwiGLU + Dynamic Per-Token FP8 Quant runner."""
    if not _is_rocm_torch():
        return _unavailable(prototype_args, "rocm_torch_required")
    loaded = _import_vllm_custom_ops()
    if loaded is None or loaded[1] is None:
        return _unavailable(prototype_args, "vllm_ops_not_installed")
    _vllm_mod, ops_mod = loaded
    matched = _probe_swiglu_quant_pair(
        getattr(ops_mod, "silu_and_mul_quant", None),
        getattr(ops_mod, "silu_and_mul", None),
        getattr(ops_mod, "dynamic_per_token_scaled_fp8_quant", None),
        prototype_args,
        fused_name="vllm_silu_and_mul_fp8_quant",
        split_name="vllm_silu_mul_then_fp8_quant",
        fp8_dtype=fp8_dtype,
        ref_fn=ref_fn,
        max_atol=max_atol,
    )
    if matched is not None:
        return matched
    return _unavailable(prototype_args, "vllm_swiglu_quant_unsupported")


def try_make_aiter_swiglu_quant_runner(
    prototype_args: list[Any],
    *,
    fp8_dtype: torch.dtype,
    ref_fn: Callable[..., torch.Tensor],
    max_atol: float,
) -> PreparedRunner:
    """Build a Tier-3 AITER SwiGLU + Dynamic Per-Token FP8 Quant runner."""
    if not _is_rocm_torch():
        return _unavailable(prototype_args, "rocm_torch_required")
    try:
        aiter_mod = importlib.import_module("aiter")
    except Exception:  # noqa: BLE001 -- aiter optional on ROCm hosts
        return _unavailable(prototype_args, "aiter_not_installed")
    matched = _probe_swiglu_quant_pair(
        getattr(aiter_mod, "silu_and_mul_scaled_quant", None),
        getattr(aiter_mod, "silu_and_mul", None),
        getattr(aiter_mod, "dynamic_per_token_scaled_quant", None),
        prototype_args,
        fused_name="aiter_silu_and_mul_fp8_quant",
        split_name="aiter_silu_mul_then_fp8_quant",
        fp8_dtype=fp8_dtype,
        ref_fn=ref_fn,
        max_atol=max_atol,
    )
    if matched is not None:
        return matched
    return _unavailable(prototype_args, "aiter_swiglu_quant_unsupported")
