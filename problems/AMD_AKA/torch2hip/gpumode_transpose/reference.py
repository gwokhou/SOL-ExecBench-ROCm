"""Standalone PyTorch reference for gpumode_transpose (debug mirror)."""


def run(input, dim1, dim2):  # noqa: A002 - benchmark ABI
    return input.transpose(int(dim1), int(dim2)).contiguous()
