"""Standalone PyTorch reference for gpumode_positionwise_ffn (debug mirror)."""

import torch.nn.functional as F


def run(
    x,
    W_1_weight,
    W_1_bias,
    W_2_weight,
    W_2_bias,
    layer_norm_weight,
    layer_norm_bias,
    dropout_p,
    training,
):
    out = F.linear(x, W_1_weight, W_1_bias)
    out = F.relu(out)
    out = F.linear(out, W_2_weight, W_2_bias)
    out = F.dropout(out, p=dropout_p, training=bool(training))
    out = out + x
    return F.layer_norm(out, out.shape[-1:], layer_norm_weight, layer_norm_bias)
