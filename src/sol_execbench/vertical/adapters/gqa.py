# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Tier 2 vLLM and Tier 3 AITER adapters for GQA paged decode."""

from __future__ import annotations

import contextlib
import importlib
import inspect
import math
from collections.abc import Callable
from dataclasses import dataclass

import torch

import sol_execbench.vertical.adapters.common as _common
from sol_execbench.vertical.adapters.common import (
    PreparedRunner,
    _as_tensor,
    _self_validate,
    _unavailable,
)


@dataclass(frozen=True, slots=True, kw_only=True)
class _PackedROCmKV:
    k_paged: torch.Tensor
    v_linear: torch.Tensor
    v_shuffled: torch.Tensor
    block_tables: torch.Tensor
    seq_lens: torch.Tensor
    max_seq_len: int
    page_size: int
    q_heads: int
    kv_heads: int
    head_dim: int


def build_paged_tables_and_seqlens(
    kv_indptr: torch.Tensor,
    kv_indices: torch.Tensor,
    kv_last_len: torch.Tensor,
    *,
    page_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Build dense ``block_tables`` and ``seq_lens`` once from ragged CSR KV."""
    starts = [int(v) for v in kv_indptr.tolist()]
    batch = max(0, len(starts) - 1)
    counts = [starts[i + 1] - starts[i] for i in range(batch)]
    max_blocks = max(counts, default=1)
    block_tables = torch.zeros(
        (batch, max_blocks), dtype=torch.int32, device=device
    )
    seq_lens_list: list[int] = []
    last_lens = [int(v) for v in kv_last_len.tolist()]
    indices_i32 = kv_indices.to(dtype=torch.int32)
    for b_idx in range(batch):
        p0, p1 = starts[b_idx], starts[b_idx + 1]
        cnt = p1 - p0
        if cnt > 0:
            block_tables[b_idx, :cnt] = indices_i32[p0:p1]
        slen = (
            max(0, (cnt - 1) * page_size + last_lens[b_idx]) if cnt > 0 else 0
        )
        seq_lens_list.append(slen)
    seq_lens = torch.tensor(seq_lens_list, dtype=torch.int32, device=device)
    return block_tables, seq_lens, max(seq_lens_list, default=page_size)


def _pack_rocm_paged_kv_once(
    prototype_args: list[object],
) -> _PackedROCmKV | None:
    q = _as_tensor(prototype_args[0])
    k_cache = _as_tensor(prototype_args[1])
    v_cache = _as_tensor(prototype_args[2])
    kv_indptr = _as_tensor(prototype_args[3])
    kv_indices = _as_tensor(prototype_args[4])
    kv_last_len = _as_tensor(prototype_args[5])
    n_pages, ps, hk, d = k_cache.shape
    _, q_heads, _ = q.shape
    x_pack = 16 // max(int(k_cache.element_size()), 1)
    if x_pack <= 0 or d % x_pack != 0 or ps % x_pack != 0:
        return None
    k_paged = (
        k_cache.view(n_pages, ps, hk, d // x_pack, x_pack)
        .permute(0, 2, 3, 1, 4)
        .contiguous()
    )
    v_linear = v_cache.permute(0, 2, 3, 1).contiguous()
    v_shuffled = (
        v_cache.view(n_pages, ps // x_pack, x_pack, hk, d)
        .permute(0, 3, 1, 4, 2)
        .contiguous()
    )
    block_tables, seq_lens, max_seq_len = build_paged_tables_and_seqlens(
        kv_indptr,
        kv_indices,
        kv_last_len,
        page_size=ps,
        device=q.device,
    )
    return _PackedROCmKV(
        k_paged=k_paged,
        v_linear=v_linear,
        v_shuffled=v_shuffled,
        block_tables=block_tables,
        seq_lens=seq_lens,
        max_seq_len=max_seq_len,
        page_size=ps,
        q_heads=q_heads,
        kv_heads=hk,
        head_dim=d,
    )


def _make_vllm_pa_v2_runner(
    pa_fn: Callable[..., object],
    packed: _PackedROCmKV,
    q: torch.Tensor,
    scale: float,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    partition_size: int = 256,
    *,
    with_query_start_loc: bool = False,
) -> Callable[..., torch.Tensor]:
    num_parts = max(1, math.ceil(packed.max_seq_len / partition_size))
    scratch_shape = (q.shape[0], packed.q_heads, num_parts)
    exp_sums = torch.empty(scratch_shape, dtype=torch.float32, device=q.device)
    max_logits = torch.empty(
        scratch_shape, dtype=torch.float32, device=q.device
    )
    tmp_out = torch.empty(
        (*scratch_shape, packed.head_dim), dtype=q.dtype, device=q.device
    )

    def _run_rocm_v2(
        q_in: torch.Tensor,
        k_in: torch.Tensor,
        v_in: torch.Tensor,
        bt_in: torch.Tensor,
        sl_in: torch.Tensor,
    ) -> torch.Tensor:
        out = torch.empty_like(q_in)
        if with_query_start_loc:
            pa_fn(
                out,
                exp_sums,
                max_logits,
                tmp_out,
                q_in,
                k_in,
                v_in,
                packed.kv_heads,
                scale,
                bt_in,
                sl_in,
                None,
                packed.page_size,
                packed.max_seq_len,
                None,
                "auto",
                k_scale,
                v_scale,
                None,
            )
            return out
        pa_fn(
            out,
            exp_sums,
            max_logits,
            tmp_out,
            q_in,
            k_in,
            v_in,
            packed.kv_heads,
            scale,
            bt_in,
            sl_in,
            packed.page_size,
            packed.max_seq_len,
            None,
            "auto",
            k_scale,
            v_scale,
            None,
        )
        return out

    return _run_rocm_v2


def _make_vllm_pa_v1_runner(
    pa_fn: Callable[..., object],
    packed: _PackedROCmKV,
    scale: float,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
) -> Callable[..., torch.Tensor]:
    def _run_v1(
        q_in: torch.Tensor,
        k_in: torch.Tensor,
        v_in: torch.Tensor,
        bt_in: torch.Tensor,
        sl_in: torch.Tensor,
    ) -> torch.Tensor:
        out = torch.empty_like(q_in)
        pa_fn(
            out,
            q_in,
            k_in,
            v_in,
            packed.kv_heads,
            scale,
            bt_in,
            sl_in,
            packed.page_size,
            packed.max_seq_len,
            None,
            "auto",
            k_scale,
            v_scale,
            0,
            0,
            0,
            1,
            0,
        )
        return out

    return _run_v1


def _probe_vllm_pa_op(
    pa_fn: Callable[..., object],
    packed: _PackedROCmKV,
    cand_args: list[object],
    prototype_args: list[object],
    *,
    impl_name: str,
    ref_fn: Callable[..., torch.Tensor],
    max_atol: float,
) -> PreparedRunner | None:
    q = _as_tensor(cand_args[0])
    scale = 1.0 / math.sqrt(packed.head_dim)
    k_scale = torch.ones(1, dtype=torch.float32, device=q.device)
    v_scale = torch.ones(1, dtype=torch.float32, device=q.device)
    has_qsl = False
    with contextlib.suppress(TypeError, ValueError):
        has_qsl = "query_start_loc" in inspect.signature(pa_fn).parameters
    qsl_order = (True, False) if has_qsl else (False, True)
    candidates = [
        *(
            _make_vllm_pa_v2_runner(
                pa_fn,
                packed,
                q,
                scale,
                k_scale,
                v_scale,
                partition_size=part_sz,
                with_query_start_loc=use_qsl,
            )
            for use_qsl in qsl_order
            for part_sz in (256, 512)
        ),
        _make_vllm_pa_v1_runner(pa_fn, packed, scale, k_scale, v_scale),
    ]

    for runner in candidates:
        if _self_validate(runner, cand_args, ref_fn, prototype_args, max_atol):
            return PreparedRunner(
                fn=runner,
                implementation_name=impl_name,
                candidate_args=cand_args,
            )
    return None


def try_make_vllm_gqa_runner(
    prototype_args: list[object],
    *,
    ref_fn: Callable[..., torch.Tensor],
    max_atol: float,
) -> PreparedRunner:
    """Build a setup-packed Tier-2 vLLM PagedAttention GQA runner."""
    if not _common._is_rocm_torch():
        return _unavailable(prototype_args, "rocm_torch_required")
    loaded = _common._import_vllm_custom_ops()
    if loaded is None or loaded[1] is None:
        return _unavailable(prototype_args, "vllm_ops_not_installed")
    _vllm_mod, ops_mod = loaded
    packed = _pack_rocm_paged_kv_once(prototype_args)
    if packed is None:
        return _unavailable(prototype_args, "rocm_kv_shape_not_divisible")

    q = prototype_args[0]
    cand_args: list[object] = [
        q,
        packed.k_paged,
        packed.v_linear,
        packed.block_tables,
        packed.seq_lens,
    ]
    for attr, impl_name in (
        ("paged_attention_rocm", "vllm_paged_attention_rocm"),
        ("paged_attention_v1", "vllm_paged_attention_v1"),
    ):
        pa_fn = getattr(ops_mod, attr, None)
        if pa_fn is None:
            continue
        matched = _probe_vllm_pa_op(
            pa_fn,
            packed,
            cand_args,
            prototype_args,
            impl_name=impl_name,
            ref_fn=ref_fn,
            max_atol=max_atol,
        )
        if matched is not None:
            return matched
    return _unavailable(prototype_args, "vllm_gqa_unsupported")


def try_make_aiter_gqa_runner(
    prototype_args: list[object],
    *,
    ref_fn: Callable[..., torch.Tensor],
    max_atol: float,
) -> PreparedRunner:
    """Build a setup-packed Tier-3 AITER PagedAttention GQA runner."""
    if not _common._is_rocm_torch():
        return _unavailable(prototype_args, "rocm_torch_required")
    try:
        aiter_mod = importlib.import_module("aiter")
    except Exception:  # noqa: BLE001 -- aiter optional on ROCm hosts
        return _unavailable(prototype_args, "aiter_not_installed")
    packed = _pack_rocm_paged_kv_once(prototype_args)
    if packed is None:
        return _unavailable(prototype_args, "rocm_kv_shape_not_divisible")

    q = prototype_args[0]
    max_blocks = int(packed.block_tables.shape[1])
    pa_asm_fn = getattr(aiter_mod, "pa_fwd_asm", None)
    if callable(pa_asm_fn) and packed.head_dim == 128:
        for v_cand in (packed.v_shuffled, packed.v_linear):
            cand_args: list[object] = [
                q,
                packed.k_paged,
                v_cand,
                packed.block_tables,
                packed.seq_lens,
            ]

            def _run_asm(
                q_in: torch.Tensor,
                k_in: torch.Tensor,
                v_in: torch.Tensor,
                bt_in: torch.Tensor,
                sl_in: torch.Tensor,
            ) -> torch.Tensor:
                return pa_asm_fn(q_in, k_in, v_in, bt_in, sl_in, max_blocks)

            if _self_validate(
                _run_asm, cand_args, ref_fn, prototype_args, max_atol
            ):
                return PreparedRunner(
                    fn=_run_asm,
                    implementation_name="aiter_pa_fwd_asm",
                    candidate_args=cand_args,
                )

    pa_rocm_fn = getattr(aiter_mod, "paged_attention_rocm", None)
    if callable(pa_rocm_fn):
        cand_args_lin: list[object] = [
            q,
            packed.k_paged,
            packed.v_linear,
            packed.block_tables,
            packed.seq_lens,
        ]
        matched = _probe_vllm_pa_op(
            pa_rocm_fn,
            packed,
            cand_args_lin,
            prototype_args,
            impl_name="aiter_paged_attention_rocm",
            ref_fn=ref_fn,
            max_atol=max_atol,
        )
        if matched is not None:
            return matched
    return _unavailable(prototype_args, "aiter_gqa_unsupported")
