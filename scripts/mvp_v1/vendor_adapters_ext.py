# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Setup-time pre-packing Tier 2 vLLM / Tier 3 AITER adapters for Stage-2 ops."""

from __future__ import annotations

import contextlib
import importlib
import inspect
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


def _build_vllm_fp8_quant_configs(
    w1_dtype: torch.dtype,
    scales: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> list[Any]:
    w1_s, w2_s, a_s = scales
    configs: list[Any] = []
    mods: list[Any] = []
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
    fused_experts_fn: Callable[..., torch.Tensor],
    *,
    quant_config: Any = None,
    scales: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    weight_dtype: torch.dtype = torch.float32,
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
    quant_configs = _build_vllm_fp8_quant_configs(cand_args[1].dtype, scales)
    if callable(fused_moe_fn):
        moe_runners: list[Callable[..., Any]] = [
            lambda t, w1, w2, lg, k, _qc=qc: fused_moe_fn(
                t, w1, w2, lg, k, renormalize=True, quant_config=_qc
            )
            for qc in quant_configs
        ]
        moe_runners.extend(
            (
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
    if callable(fused_experts_fn):
        expert_runners = [
            _make_vllm_experts_runner(
                fused_experts_fn, quant_config=qc, weight_dtype=w_dt
            )
            for qc in quant_configs
            for w_dt in (torch.float32, torch.bfloat16)
        ]
        expert_runners.append(
            _make_vllm_experts_runner(fused_experts_fn, scales=scales)
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
    calls: tuple[Callable[..., torch.Tensor], ...] = (
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
    quant_type: Any = None
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
            cand_args: list[Any] = [
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


def _expand_csr_pages_to_tokens(
    kv_indptr: torch.Tensor,
    kv_indices: torch.Tensor,
    kv_last_len: torch.Tensor,
    *,
    page_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    device = kv_indices.device
    offsets = torch.arange(page_size, dtype=torch.int32, device=device)
    indptr_cpu = kv_indptr.detach().cpu().tolist()
    last_cpu = kv_last_len.detach().cpu().tolist()
    tok_chunks: list[torch.Tensor] = []
    tok_offsets = [0]
    for b_idx, tail in enumerate(last_cpu):
        s_pg = int(indptr_cpu[b_idx])
        e_pg = int(indptr_cpu[b_idx + 1])
        pages = kv_indices[s_pg:e_pg].to(torch.int32)
        n_pg = int(pages.numel())
        slen = max(0, (n_pg - 1) * page_size + int(tail))
        expanded = (pages[:, None] * page_size + offsets[None, :]).reshape(-1)
        tok_chunks.append(expanded[:slen])
        tok_offsets.append(tok_offsets[-1] + slen)
    tok_indices = (
        torch.cat(tok_chunks, dim=0).contiguous()
        if tok_chunks
        else torch.zeros((0,), dtype=torch.int32, device=device)
    )
    tok_indptr = torch.tensor(tok_offsets, dtype=torch.int32, device=device)
    tok_last = torch.ones_like(kv_last_len, dtype=torch.int32)
    return tok_indptr, tok_indices, tok_last


@dataclass(frozen=True, slots=True, kw_only=True)
class _PackedMLA:
    cand_args: list[Any]
    qo_indptr: torch.Tensor
    tok_indptr: torch.Tensor
    tok_indices: torch.Tensor
    tok_last_len: torch.Tensor
    page_size: int
    d_ckv: int
    d_total: int
    sm_scale: float


def _pack_mla_paged_once(prototype_args: list[Any]) -> _PackedMLA:
    q, kv_cache, kv_indptr, kv_indices, kv_last_len, d_ckv = prototype_args
    d_ckv_i = int(d_ckv)
    page_size = int(kv_cache.shape[1])
    d_total = int(kv_cache.shape[2])
    kv_4d = kv_cache.unsqueeze(2).contiguous()
    batch = int(q.shape[0])
    qo_indptr = torch.arange(0, batch + 1, dtype=torch.int32, device=q.device)
    block_tables, seq_lens, _ = build_paged_tables_and_seqlens(
        kv_indptr,
        kv_indices,
        kv_last_len,
        page_size=page_size,
        device=q.device,
    )
    tok_indptr, tok_indices, tok_last_len = _expand_csr_pages_to_tokens(
        kv_indptr, kv_indices, kv_last_len, page_size=page_size
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
        tok_indptr=tok_indptr,
        tok_indices=tok_indices,
        tok_last_len=tok_last_len,
        page_size=page_size,
        d_ckv=d_ckv_i,
        d_total=d_total,
        sm_scale=1.0 / math.sqrt(int(q.shape[2])),
    )


def _make_mla_varlen_runner(
    mla_fn: Callable[..., Any],
    packed: _PackedMLA,
    *,
    use_token_pages: bool,
    squeeze_kv: bool = False,
    extra_kwargs: dict[str, Any] | None = None,
) -> Callable[..., torch.Tensor]:
    qo_indptr = packed.qo_indptr
    d_ckv = packed.d_ckv
    d_total = packed.d_total
    ps = 1 if use_token_pages else packed.page_size
    kw = dict(extra_kwargs or {})

    def _run(
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
        if use_token_pages:
            kv_buf = (
                kv_in.view(-1, 1, d_total)
                if squeeze_kv
                else kv_in.view(-1, 1, 1, d_total)
            )
            ind_arg = packed.tok_indptr
            idx_arg = packed.tok_indices
            last_arg = packed.tok_last_len
        else:
            kv_buf = kv_in.squeeze(2) if squeeze_kv else kv_in
            ind_arg, idx_arg, last_arg = indptr_in, indices_in, last_in
        mla_fn(
            q_in,
            kv_buf,
            out,
            qo_indptr,
            ind_arg,
            idx_arg,
            last_arg,
            1,
            page_size=ps,
            nhead_kv=1,
            sm_scale=packed.sm_scale,
            **kw,
        )
        return out

    return _run


def _build_legacy_mla_runners(
    mla_fn: Callable[..., Any],
    packed: _PackedMLA,
    *,
    allow_return: bool,
) -> list[Callable[..., torch.Tensor]]:
    qo_indptr = packed.qo_indptr
    d_ckv = packed.d_ckv
    sm_scale = packed.sm_scale

    def _run_legacy_varlen(
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

    runners: list[Callable[..., torch.Tensor]] = [
        _run_legacy_varlen,
        _run_paged,
    ]
    if allow_return:
        runners.append(
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
    return runners


def _build_mla_candidate_runners(
    mla_fn: Callable[..., Any],
    packed: _PackedMLA,
    *,
    allow_return: bool,
) -> list[Callable[..., torch.Tensor]]:
    candidates: list[Callable[..., torch.Tensor]] = [
        _make_mla_varlen_runner(mla_fn, packed, use_token_pages=True),
    ]
    if not packed.cand_args[0].is_cuda:
        candidates.append(
            _make_mla_varlen_runner(mla_fn, packed, use_token_pages=False)
        )
        candidates.extend(
            _build_legacy_mla_runners(mla_fn, packed, allow_return=allow_return)
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


def _resolve_vllm_triton_decode_fn() -> Callable[..., Any] | None:
    for mod_name in (
        "vllm.v1.attention.ops.triton_decode_attention",
        "vllm.attention.ops.triton_decode_attention",
        "vllm.v1.attention.backends.mla.triton_mla",
    ):
        with contextlib.suppress(ImportError, RuntimeError, AttributeError):
            mod = importlib.import_module(mod_name)
            dec_fn = getattr(mod, "decode_attention_fwd", None)
            if callable(dec_fn):
                return dec_fn
    return None


def _make_legacy_triton_mla_runner(
    dec_fn: Callable[..., Any],
    packed: _PackedMLA,
    attn_logits: torch.Tensor,
    num_splits: int,
) -> Callable[..., torch.Tensor]:
    q = packed.cand_args[0]
    bt = packed.cand_args[5]
    batch, q_heads = int(q.shape[0]), int(q.shape[1])
    b_req_idx = torch.arange(batch, dtype=torch.int32, device=q.device)
    offsets = torch.arange(packed.page_size, dtype=torch.int32, device=q.device)
    req_tok = (
        bt[:, :, None].to(torch.int32) * packed.page_size
        + offsets[None, None, :]
    ).reshape(batch, -1)

    def _run_legacy(
        q_in: torch.Tensor,
        kv_in: torch.Tensor,
        _ind: torch.Tensor,
        _idx: torch.Tensor,
        _last: torch.Tensor,
        _bt: torch.Tensor,
        sl_in: torch.Tensor,
    ) -> torch.Tensor:
        out = torch.empty(
            (batch, q_heads, packed.d_ckv),
            dtype=q_in.dtype,
            device=q_in.device,
        )
        k_buf = kv_in.view(-1, 1, packed.d_total)
        dec_fn(
            q_in,
            k_buf,
            k_buf[..., : packed.d_ckv],
            out,
            req_tok,
            b_req_idx,
            sl_in,
            attn_logits,
            num_splits,
            packed.sm_scale,
            1,
        )
        return out

    return _run_legacy


def _try_vllm_triton_mla_runner(
    packed: _PackedMLA,
    prototype_args: list[Any],
    *,
    ref_fn: Callable[..., Any],
    max_atol: float,
) -> PreparedRunner | None:
    dec_fn = _resolve_vllm_triton_decode_fn()
    if dec_fn is None:
        return None
    q = packed.cand_args[0]
    bt = packed.cand_args[5]
    batch, q_heads = int(q.shape[0]), int(q.shape[1])
    num_splits = 4 if int(bt.shape[1]) <= 16 else 16
    lse = torch.empty((batch, q_heads), dtype=torch.float32, device=q.device)
    attn_logits = torch.empty(
        (batch, q_heads, num_splits, packed.d_ckv + 1),
        dtype=torch.float32,
        device=q.device,
    )

    def _run_v1(
        q_in: torch.Tensor,
        kv_in: torch.Tensor,
        _ind: torch.Tensor,
        _idx: torch.Tensor,
        _last: torch.Tensor,
        bt_in: torch.Tensor,
        sl_in: torch.Tensor,
    ) -> torch.Tensor:
        out = torch.empty(
            (batch, q_heads, packed.d_ckv),
            dtype=q_in.dtype,
            device=q_in.device,
        )
        dec_fn(
            q_in,
            kv_in,
            kv_in[..., : packed.d_ckv],
            out,
            lse,
            bt_in,
            sl_in,
            attn_logits,
            num_splits,
            packed.sm_scale,
            packed.page_size,
            is_mla=True,
        )
        return out

    runners: list[Callable[..., torch.Tensor]] = [_run_v1]
    if not q.is_cuda:
        runners.append(
            _make_legacy_triton_mla_runner(
                dec_fn, packed, attn_logits, num_splits
            )
        )
    for runner in runners:
        if _self_validate(
            runner, packed.cand_args, ref_fn, prototype_args, max_atol
        ):
            return PreparedRunner(
                fn=runner,
                implementation_name="vllm_triton_mla_decode",
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
    if loaded is None:
        return _unavailable(prototype_args, "vllm_ops_not_installed")
    _vllm_mod, ops_mod = loaded
    packed = _pack_mla_paged_once(prototype_args)
    mla_fn = getattr(ops_mod, "mla_decode_fwd", None) if ops_mod else None
    if callable(mla_fn):
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
    triton_matched = _try_vllm_triton_mla_runner(
        packed, prototype_args, ref_fn=ref_fn, max_atol=max_atol
    )
    if triton_matched is not None:
        return triton_matched
    with contextlib.suppress(ImportError, RuntimeError, AttributeError):
        aiter_ops_mod = importlib.import_module("vllm._aiter_ops")
        rocm_ops = getattr(aiter_ops_mod, "rocm_aiter_ops", None)
        rocm_mla = getattr(rocm_ops, "mla_decode_fwd", None)
        if callable(rocm_mla):
            matched = _probe_mla_decode_fn(
                rocm_mla,
                packed,
                prototype_args,
                impl_name="vllm_rocm_aiter_mla",
                allow_return=False,
                ref_fn=ref_fn,
                max_atol=max_atol,
            )
            if matched is not None:
                return matched
    if ops_mod is None or not callable(mla_fn):
        return _unavailable(prototype_args, "vllm_mla_not_installed")
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
    mla_mod = getattr(aiter_mod, "mla", None)
    mla_fn = getattr(aiter_mod, "mla_decode_fwd", None)
    if not callable(mla_fn) and mla_mod is not None:
        mla_fn = getattr(mla_mod, "mla_decode_fwd", None)
    if not callable(mla_fn):
        with contextlib.suppress(Exception):
            mla_mod = importlib.import_module("aiter.mla")
            mla_fn = getattr(mla_mod, "mla_decode_fwd", None)
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
    x0 = prototype_args[0]
    m, heads, head_dim = int(x0.shape[0]), int(x0.shape[1]), int(x0.shape[2])

    def _run_qkv_packed(
        x_in: torch.Tensor,
        res_in: torch.Tensor,
        w_in: torch.Tensor,
        cs_in: torch.Tensor,
        pos_in: torch.Tensor,
    ) -> torch.Tensor:
        h_flat = (x_in + res_in).reshape(m, heads * head_dim)
        fused_fn(
            h_flat,
            heads,
            0,
            0,
            head_dim,
            1e-6,
            w_in,
            w_in,
            cs_in,
            True,
            pos_in,
        )
        return h_flat.view(m, heads, head_dim)

    def _run_legacy_fused(
        x_in: torch.Tensor,
        res_in: torch.Tensor,
        w_in: torch.Tensor,
        cs_in: torch.Tensor,
        pos_in: torch.Tensor,
    ) -> torch.Tensor:
        out = fused_fn(pos_in, x_in, res_in, w_in, cs_in, True)
        return x_in if out is None else out

    for runner in (_run_qkv_packed, _run_legacy_fused):
        for cand_args in cand_args_list:
            if _self_validate(
                runner, cand_args, ref_fn, prototype_args, max_atol
            ):
                return PreparedRunner(
                    fn=runner,
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
    m, heads, head_dim = int(x.shape[0]), int(x.shape[1]), int(x.shape[2])
    positions = torch.arange(m, dtype=torch.int64, device=x.device)
    cand_args_list: list[list[Any]] = [
        [
            x.detach().clone(),
            residual.detach().clone(),
            weight,
            cs_cand,
            positions,
        ]
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
            norm_fn(x_in, res_in, w_in, 1e-6)
            x_flat = x_in.view(m, heads * head_dim)
            rope_fn(pos_in, x_flat, None, head_dim, cs_in, True)
            return x_in

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


def _apply_aiter_rope_candidates(
    rope_fn: Callable[..., Any],
) -> tuple[
    Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor], ...
]:
    n_params: int | None = None
    with contextlib.suppress(TypeError, ValueError):
        n_params = len(inspect.signature(rope_fn).parameters)

    def _call_pos_enc(
        h_3d: torch.Tensor, cs_in: torch.Tensor, pos_in: torch.Tensor
    ) -> torch.Tensor:
        m, heads, head_dim = h_3d.shape
        h_flat = h_3d.view(m, heads * head_dim)
        rope_fn(pos_in, h_flat, None, head_dim, cs_in, True)
        return h_3d

    def _call_cached_2c(
        h_3d: torch.Tensor, cs_in: torch.Tensor, pos_in: torch.Tensor
    ) -> torch.Tensor:
        d2 = h_3d.shape[-1] // 2
        cos = cs_in[:, :d2].contiguous()
        sin = cs_in[:, d2:].contiguous()
        dummy_y = torch.empty_like(h_3d)
        res = rope_fn(h_3d, dummy_y, cos, sin, pos_in, 0, False, False)
        return (
            res[0] if isinstance(res, tuple) else (h_3d if res is None else res)
        )

    def _call_simple(
        h_3d: torch.Tensor, cs_in: torch.Tensor, pos_in: torch.Tensor
    ) -> torch.Tensor:
        res = rope_fn(h_3d, cs_in, pos_in)
        return h_3d if res is None else res

    if n_params is not None and n_params >= 8:
        return (_call_cached_2c, _call_pos_enc, _call_simple)
    if n_params is not None and n_params >= 6:
        return (_call_pos_enc, _call_simple)
    return (_call_pos_enc, _call_cached_2c, _call_simple)


def _probe_aiter_rmsnorm_two_step(
    norm_fn: Callable[..., Any],
    rope_fn: Callable[..., Any],
    cand_args: list[Any],
    prototype_args: list[Any],
    *,
    mode: str,
    ref_fn: Callable[..., Any],
    max_atol: float,
) -> PreparedRunner | None:
    for rope_call in _apply_aiter_rope_candidates(rope_fn):

        def _make_runner(
            rc: Callable[
                [torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor
            ],
        ) -> Callable[..., torch.Tensor]:
            def _run(
                x_in: torch.Tensor,
                res_in: torch.Tensor,
                w_in: torch.Tensor,
                cs_in: torch.Tensor,
                pos_in: torch.Tensor,
            ) -> torch.Tensor:
                m, heads, head_dim = x_in.shape
                if mode == "add_2d":
                    out = torch.empty_like(x_in)
                    res_out = torch.empty_like(res_in)
                    norm_fn(
                        out.view(m * heads, head_dim),
                        x_in.view(m * heads, head_dim),
                        res_in.view(m * heads, head_dim),
                        res_out.view(m * heads, head_dim),
                        w_in,
                        1e-6,
                    )
                    return rc(out, cs_in, pos_in)
                if mode == "add_3d":
                    out = torch.empty_like(x_in)
                    res_out = torch.empty_like(res_in)
                    norm_fn(out, x_in, res_in, res_out, w_in, 1e-6)
                    return rc(out, cs_in, pos_in)
                if mode == "inplace_add":
                    norm_fn(x_in, res_in, w_in, 1e-6)
                    return rc(x_in, cs_in, pos_in)
                h = (x_in + res_in).view(m * heads, head_dim)
                out_2d = norm_fn(h, w_in, 1e-6)
                return rc(out_2d.view(m, heads, head_dim), cs_in, pos_in)

            return _run

        runner = _make_runner(rope_call)
        if _self_validate(runner, cand_args, ref_fn, prototype_args, max_atol):
            return PreparedRunner(
                fn=runner,
                implementation_name="aiter_rmsnorm_rope",
                candidate_args=cand_args,
            )
    return None


def _resolve_aiter_norm_and_rope_ops(
    aiter_mod: Any,
    *,
    allow_pos_enc_submod: bool,
) -> tuple[list[tuple[Callable[..., Any], str]], list[Callable[..., Any]]]:
    rmsnorm_mod: Any = None
    pos_mod: Any = None
    with contextlib.suppress(ImportError, RuntimeError, AttributeError):
        rmsnorm_mod = importlib.import_module("aiter.ops.rmsnorm")
    if allow_pos_enc_submod:
        with contextlib.suppress(ImportError, RuntimeError, AttributeError):
            pos_mod = importlib.import_module("aiter.ops.pos_encoding")

    norms: list[tuple[Callable[..., Any], str]] = []
    add_2d = getattr(aiter_mod, "rmsnorm2d_fwd_with_add", None) or (
        getattr(rmsnorm_mod, "rmsnorm2d_fwd_with_add", None)
        if rmsnorm_mod
        else None
    )
    if callable(add_2d):
        norms.extend(((add_2d, "add_2d"), (add_2d, "add_3d")))
    inplace_add = (
        getattr(rmsnorm_mod, "fused_add_rms_norm_cu", None)
        if rmsnorm_mod
        else None
    )
    if callable(inplace_add):
        norms.append((inplace_add, "inplace_add"))
    rms_norm = getattr(aiter_mod, "rms_norm", None) or (
        getattr(rmsnorm_mod, "rms_norm", None) if rmsnorm_mod else None
    )
    if callable(rms_norm):
        norms.append((rms_norm, "separate"))

    ropes: list[Callable[..., Any]] = []
    for cand in (
        getattr(pos_mod, "rotary_embedding_fwd", None) if pos_mod else None,
        getattr(aiter_mod, "rope_cached_positions_2c_fwd", None),
        getattr(aiter_mod, "rope_fwd", None),
    ):
        if callable(cand):
            ropes.append(cand)
    return norms, ropes


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
        [
            x.detach().clone(),
            residual.detach().clone(),
            weight,
            cs_cand,
            positions,
        ]
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
            out = fused_fn(pos_in, x_in, res_in, w_in, cs_in)
            return x_in if out is None else out

        for cand_args in cand_args_list:
            if _self_validate(
                _run_fused, cand_args, ref_fn, prototype_args, max_atol
            ):
                return PreparedRunner(
                    fn=_run_fused,
                    implementation_name="aiter_fused_qk_norm_rope",
                    candidate_args=cand_args,
                )

    norms, ropes = _resolve_aiter_norm_and_rope_ops(
        aiter_mod, allow_pos_enc_submod=not x.is_cuda
    )
    for rope_fn in ropes:
        for norm_fn, mode in norms:
            for cand_args in cand_args_list:
                matched = _probe_aiter_rmsnorm_two_step(
                    norm_fn,
                    rope_fn,
                    cand_args,
                    prototype_args,
                    mode=mode,
                    ref_fn=ref_fn,
                    max_atol=max_atol,
                )
                if matched is not None:
                    return matched
    return _unavailable(prototype_args, "aiter_rmsnorm_rope_unsupported")


def _build_swiglu_quant_runners(
    fused_fn: Any,
    silu_fn: Any,
    quant_fn: Any,
    *,
    scaled_quant_fn: Any,
    m: int,
    inter: int,
    fp8_dtype: torch.dtype,
    fused_name: str,
    split_name: str,
) -> list[
    tuple[str, Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]]]
]:
    pairs: list[
        tuple[str, Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]]]
    ] = []
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

        pairs.extend(((fused_name, fused_fn), (fused_name, _run_fused_out)))

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

        pairs.append((split_name, _run_two_step))

    if callable(silu_fn) and callable(scaled_quant_fn):

        def _run_scaled_two_step(
            gu_in: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            act_out = torch.empty(
                (m, inter), dtype=gu_in.dtype, device=gu_in.device
            )
            silu_fn(act_out, gu_in)
            return scaled_quant_fn(act_out, use_per_token_if_dynamic=True)

        pairs.append((split_name, _run_scaled_two_step))
    return pairs


def _probe_swiglu_quant_pair(
    fused_fn: Any,
    silu_fn: Any,
    quant_fn: Any,
    cand_args: list[Any],
    *,
    scaled_quant_fn: Any = None,
    fused_name: str,
    split_name: str,
    fp8_dtype: torch.dtype,
    ref_fn: Callable[..., Any],
    max_atol: float,
) -> PreparedRunner | None:
    gate_up = cand_args[0]
    m, i2 = int(gate_up.shape[0]), int(gate_up.shape[1])
    for impl_name, runner in _build_swiglu_quant_runners(
        fused_fn,
        silu_fn,
        quant_fn,
        scaled_quant_fn=scaled_quant_fn,
        m=m,
        inter=i2 // 2,
        fp8_dtype=fp8_dtype,
        fused_name=fused_name,
        split_name=split_name,
    ):
        if _self_validate(runner, cand_args, ref_fn, cand_args, max_atol):
            return PreparedRunner(
                fn=runner,
                implementation_name=impl_name,
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
    silu_fn = getattr(ops_mod, "silu_and_mul", None)
    if not callable(silu_fn):
        c_ns = getattr(torch.ops, "_C", None)
        silu_fn = getattr(c_ns, "silu_and_mul", None) if c_ns else None
    matched = _probe_swiglu_quant_pair(
        getattr(ops_mod, "silu_and_mul_quant", None),
        silu_fn,
        getattr(ops_mod, "dynamic_per_token_scaled_fp8_quant", None),
        prototype_args,
        scaled_quant_fn=getattr(ops_mod, "scaled_fp8_quant", None),
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
    fused_fn = getattr(aiter_mod, "silu_and_mul_scaled_quant", None)
    if fused_fn is None:
        with contextlib.suppress(ImportError, RuntimeError, AttributeError):
            act_mod = importlib.import_module("aiter.ops.activation")
            fused_fn = getattr(act_mod, "silu_and_mul_quant", None)
    matched = _probe_swiglu_quant_pair(
        fused_fn,
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
