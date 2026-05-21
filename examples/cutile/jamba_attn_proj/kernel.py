import torch


@torch.no_grad()
def run(attn_output, residual, o_proj_weight):
    shape = attn_output.shape
    hidden_size = shape[-1]
    projected = torch.matmul(
        attn_output.contiguous().view(-1, hidden_size),
        o_proj_weight.t(),
    )
    output = projected + residual.contiguous().view(-1, hidden_size)
    return output.view(shape)
