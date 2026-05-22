import torch


@torch.no_grad()
def run(A, B):
    return torch.mm(A.float(), B.float())
