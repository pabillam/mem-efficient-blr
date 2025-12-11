import torch
import numpy as np

def torch_monarch_baseline(
    x: torch.Tensor, 
    w1_bfly_t: torch.Tensor, 
    w2_bfly_t: torch.Tensor,
    best_config=None) -> (torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor):

    """
    Performs full Monarch matrix multiplication as in [ICML22] https://arxiv.org/abs/2204.00595.
    Assumes innermost dimension of w1_bfly is contiguous along the block dimension.

    Parameters:
    ----------
    x : torch.Tensor
        Input tensor of shape (num_batches, num_seq, in_f)
    w1_bfly_t : torch.Tensor
        Tensor of shape (b1, b2 * (r // b2), in_f // b1)
    w2_bfly_t : torch.Tensor
        Tensor of shape (b2, out_f // b2, b1 * (r // b1))

    Returns:
    -------
    out: torch.Tensor
        Output tensor of shape (num_batches, num_seq, out_f), resulting from the Monarch matrix multiplication
    """

    batch_shape, n = x.shape[:-1], x.shape[-1]
    k, q, p = w1_bfly_t.shape
    l, s, r = w2_bfly_t.shape
    assert k * p == n
    assert l * r == k * q

    x_shape = x.shape
    x = x.flatten(0, -2)
    x = x.view(-1, k, x_shape[-1] // k).transpose(0, 1)

    pre_out1 = torch.empty(k, x_shape[0] * x_shape[1], q, device=x.device, dtype=x.dtype)
    pre_out1 = torch.bmm(x, w1_bfly_t.transpose(-1, -2), out=pre_out1)
    out1 = pre_out1.transpose(0, 1).reshape(x_shape[0] * x_shape[1], r, l).transpose(-1, -2).contiguous().transpose(0, 1)
    pre_out2 = torch.empty(l, x_shape[0] * x_shape[1], s, device=x.device, dtype=x.dtype)
    pre_out2 = torch.bmm(out1, w2_bfly_t.transpose(-1, -2), out=pre_out2)
    out2 = pre_out2.permute(1, 2, 0).reshape(*batch_shape, s * l)
    return out2, pre_out2, out1, pre_out1

