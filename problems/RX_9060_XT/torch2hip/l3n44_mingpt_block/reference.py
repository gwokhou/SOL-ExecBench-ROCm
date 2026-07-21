"""Standalone PyTorch reference for l3n44_mingpt_block (debug mirror)."""

import math
import torch
import torch.nn.functional as F


def _new_gelu(z):
    return (
        0.5
        * z
        * (
            1.0
            + torch.tanh(math.sqrt(2.0 / math.pi) * (z + 0.044715 * torch.pow(z, 3.0)))
        )
    )


def run(
    x,
    ln1_w,
    ln1_b,
    attn_c_attn_w,
    attn_c_attn_b,
    attn_c_proj_w,
    attn_c_proj_b,
    attn_bias,
    n_head,
    ln2_w,
    ln2_b,
    mlp_cfc_w,
    mlp_cfc_b,
    mlp_cproj_w,
    mlp_cproj_b,
    n_embd,
):
    a = F.layer_norm(x, (int(n_embd),), ln1_w, ln1_b)
    B, T, C = a.size()
    nh = int(n_head)
    qkv = F.linear(a, attn_c_attn_w, attn_c_attn_b)
    q, k, v = qkv.split(C, dim=2)
    k = k.view(B, T, nh, C // nh).transpose(1, 2)
    q = q.view(B, T, nh, C // nh).transpose(1, 2)
    v = v.view(B, T, nh, C // nh).transpose(1, 2)
    att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
    att = att.masked_fill(attn_bias[:, :, :T, :T] == 0, float("-inf"))
    att = F.softmax(att, dim=-1)
    y = att @ v
    y = y.transpose(1, 2).contiguous().view(B, T, C)
    x = x + F.linear(y, attn_c_proj_w, attn_c_proj_b)
    m = F.layer_norm(x, (int(n_embd),), ln2_w, ln2_b)
    h = F.linear(m, mlp_cfc_w, mlp_cfc_b)
    h = _new_gelu(h)
    h = F.linear(h, mlp_cproj_w, mlp_cproj_b)
    return x + h
