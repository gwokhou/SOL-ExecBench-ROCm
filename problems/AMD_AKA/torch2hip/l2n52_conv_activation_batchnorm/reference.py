"""Standalone PyTorch reference for l2n52_conv_activation_batchnorm (debug mirror)."""

import torch
import torch.nn.functional as F


def run(x, conv_weight, conv_bias, bn_weight, bn_bias, bn_mean, bn_var, bn_eps):
    value = F.conv2d(x, conv_weight, conv_bias)
    value = torch.multiply(torch.tanh(F.softplus(value)), value)
    return F.batch_norm(
        value, bn_mean, bn_var, bn_weight, bn_bias, training=False, eps=bn_eps
    )
