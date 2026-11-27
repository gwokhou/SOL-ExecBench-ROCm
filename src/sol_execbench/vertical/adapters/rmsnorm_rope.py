# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Tier 2 vLLM and Tier 3 AITER adapters for Fused RMSNorm + RoPE."""

from __future__ import annotations

import contextlib
import importlib
import inspect
from collections.abc import Callable

import torch

import sol_execbench.vertical.adapters.common as _common
from sol_execbench.vertical.adapters.common import (
    PreparedRunner,
    _as_tensor,
    _self_validate,
    _unavailable,
)


def _cos_sin_candidates(
    x: torch.Tensor,
    cos_sin: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    cos_sin_cast = cos_sin.to(dtype=x.dtype).contiguous()
    if cos_sin.dtype == x.dtype:
        return (cos_sin_cast,)
    return (cos_sin_cast, cos_sin.contiguous())


def _probe_vllm_rmsnorm_fused(
    fused_fn: Callable[..., object],
    cand_args_list: list[list[object]],
    prototype_args: list[object],
    *,
    ref_fn: Callable[..., object],
    max_atol: float,
) -> PreparedRunner | None:
    x0 = _as_tensor(prototype_args[0])
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
    ) -> object:
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
    prototype_args: list[object],
    *,
    ref_fn: Callable[..., torch.Tensor],
    max_atol: float,
) -> PreparedRunner:
    """Build a setup-packed Tier-2 vLLM Add-RMSNorm + RoPE runner."""
    if not _common._is_rocm_torch():
        return _unavailable(prototype_args, "rocm_torch_required")
    loaded = _common._import_vllm_custom_ops()
    if loaded is None or loaded[1] is None:
        return _unavailable(prototype_args, "vllm_ops_not_installed")
    _vllm_mod, ops_mod = loaded
    x = _as_tensor(prototype_args[0])
    residual = _as_tensor(prototype_args[1])
    weight = _as_tensor(prototype_args[2])
    cos_sin = _as_tensor(prototype_args[3])
    m, heads, head_dim = int(x.shape[0]), int(x.shape[1]), int(x.shape[2])
    positions = torch.arange(m, dtype=torch.int64, device=x.device)
    cand_args_list: list[list[object]] = [
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
    rope_fn: Callable[..., object],
) -> tuple[Callable[[torch.Tensor, torch.Tensor, torch.Tensor], object], ...]:
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
    ) -> object:
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
    ) -> object:
        res = rope_fn(h_3d, cs_in, pos_in)
        return h_3d if res is None else res

    if n_params is not None and n_params >= 8:
        return (_call_cached_2c, _call_pos_enc, _call_simple)
    if n_params is not None and n_params >= 6:
        return (_call_pos_enc, _call_simple)
    return (_call_pos_enc, _call_cached_2c, _call_simple)


def _probe_aiter_rmsnorm_two_step(
    norm_fn: Callable[..., object],
    rope_fn: Callable[..., object],
    cand_args: list[object],
    prototype_args: list[object],
    *,
    mode: str,
    ref_fn: Callable[..., object],
    max_atol: float,
) -> PreparedRunner | None:
    for rope_call in _apply_aiter_rope_candidates(rope_fn):

        def _make_runner(
            rc: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], object],
        ) -> Callable[..., object]:
            def _run(
                x_in: torch.Tensor,
                res_in: torch.Tensor,
                w_in: torch.Tensor,
                cs_in: torch.Tensor,
                pos_in: torch.Tensor,
            ) -> object:
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
                out_2d = _as_tensor(norm_fn(h, w_in, 1e-6))
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
    aiter_mod: object,
    *,
    allow_pos_enc_submod: bool,
) -> tuple[
    list[tuple[Callable[..., object], str]], list[Callable[..., object]]
]:
    rmsnorm_mod: object = None
    pos_mod: object = None
    with contextlib.suppress(ImportError, RuntimeError, AttributeError):
        rmsnorm_mod = importlib.import_module("aiter.ops.rmsnorm")
    if allow_pos_enc_submod:
        with contextlib.suppress(ImportError, RuntimeError, AttributeError):
            pos_mod = importlib.import_module("aiter.ops.pos_encoding")

    norms: list[tuple[Callable[..., object], str]] = []
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

    ropes: list[Callable[..., object]] = []
    for cand in (
        getattr(pos_mod, "rotary_embedding_fwd", None) if pos_mod else None,
        getattr(aiter_mod, "rope_cached_positions_2c_fwd", None),
        getattr(aiter_mod, "rope_fwd", None),
    ):
        if callable(cand):
            ropes.append(cand)
    return norms, ropes


def try_make_aiter_rmsnorm_rope_runner(
    prototype_args: list[object],
    *,
    ref_fn: Callable[..., torch.Tensor],
    max_atol: float,
) -> PreparedRunner:
    """Build a setup-packed Tier-3 AITER Add-RMSNorm + RoPE runner."""
    if not _common._is_rocm_torch():
        return _unavailable(prototype_args, "rocm_torch_required")
    try:
        aiter_mod = importlib.import_module("aiter")
    except Exception:  # noqa: BLE001 -- aiter optional on ROCm hosts
        return _unavailable(prototype_args, "aiter_not_installed")
    x = _as_tensor(prototype_args[0])
    residual = _as_tensor(prototype_args[1])
    weight = _as_tensor(prototype_args[2])
    cos_sin = _as_tensor(prototype_args[3])
    positions = torch.arange(x.shape[0], dtype=torch.int64, device=x.device)
    cand_args_list: list[list[object]] = [
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
