"""Standalone PyTorch reference for gpumode_transpose (debug mirror)."""


def run(input, dim1, dim2):
    return input.transpose(int(dim1), int(dim2)).contiguous()
