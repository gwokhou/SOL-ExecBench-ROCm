# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Tier 2 vLLM and Tier 3 AITER adapters for MLA paged decode."""

from __future__ import annotations

import contextlib
import importlib
import math
from collections.abc import Callable
from dataclasses import dataclass

import torch

import sol_execbench.vertical.adapters.common as _common
from sol_execbench.vertical.adapters.common import (
    PreparedRunner,
    _as_int,
    _as_tensor,
    _self_validate,
    _unavailable,
)
from sol_execbench.vertical.adapters.gqa import build_paged_tables_and_seqlens


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
    cand_args: list[object]
    qo_indptr: torch.Tensor
    tok_indptr: torch.Tensor
    tok_indices: torch.Tensor
    tok_last_len: torch.Tensor
    page_size: int
    d_ckv: int
    d_total: int
    sm_scale: float


def _pack_mla_paged_once(prototype_args: list[object]) -> _PackedMLA:
    q = _as_tensor(prototype_args[0])
    kv_cache = _as_tensor(prototype_args[1])
    kv_indptr = _as_tensor(prototype_args[2])
    kv_indices = _as_tensor(prototype_args[3])
    kv_last_len = _as_tensor(prototype_args[4])
    d_ckv_i = _as_int(prototype_args[5])
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
    cand_args: list[object] = [
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
    mla_fn: Callable[..., object],
    packed: _PackedMLA,
    *,
    use_token_pages: bool,
    squeeze_kv: bool = False,
    extra_kwargs: dict[str, object] | None = None,
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
    mla_fn: Callable[..., object],
    packed: _PackedMLA,
    *,
    allow_return: bool,
) -> list[Callable[..., object]]:
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

    runners: list[Callable[..., object]] = [
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
    mla_fn: Callable[..., object],
    packed: _PackedMLA,
    *,
    allow_return: bool,
) -> list[Callable[..., object]]:
    candidates: list[Callable[..., object]] = [
        _make_mla_varlen_runner(mla_fn, packed, use_token_pages=True),
    ]
    if not _as_tensor(packed.cand_args[0]).is_cuda:
        candidates.append(
            _make_mla_varlen_runner(mla_fn, packed, use_token_pages=False)
        )
        candidates.extend(
            _build_legacy_mla_runners(mla_fn, packed, allow_return=allow_return)
        )
    return candidates


def _probe_mla_decode_fn(
    mla_fn: Callable[..., object],
    packed: _PackedMLA,
    prototype_args: list[object],
    *,
    impl_name: str,
    allow_return: bool,
    ref_fn: Callable[..., object],
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


def _resolve_vllm_triton_decode_fn() -> Callable[..., object] | None:
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
    dec_fn: Callable[..., object],
    packed: _PackedMLA,
    attn_logits: torch.Tensor,
    num_splits: int,
) -> Callable[..., torch.Tensor]:
    q = _as_tensor(packed.cand_args[0])
    bt = _as_tensor(packed.cand_args[5])
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
    prototype_args: list[object],
    *,
    ref_fn: Callable[..., object],
    max_atol: float,
) -> PreparedRunner | None:
    dec_fn = _resolve_vllm_triton_decode_fn()
    if dec_fn is None:
        return None
    q = _as_tensor(packed.cand_args[0])
    bt = _as_tensor(packed.cand_args[5])
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
    prototype_args: list[object],
    *,
    ref_fn: Callable[..., torch.Tensor],
    max_atol: float,
) -> PreparedRunner:
    """Build a setup-packed Tier-2 vLLM MLA Paged Decode runner."""
    if not _common._is_rocm_torch():
        return _unavailable(prototype_args, "rocm_torch_required")
    loaded = _common._import_vllm_custom_ops()
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
    prototype_args: list[object],
    *,
    ref_fn: Callable[..., torch.Tensor],
    max_atol: float,
) -> PreparedRunner:
    """Build a setup-packed Tier-3 AITER MLA Paged Decode runner."""
    if not _common._is_rocm_torch():
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
