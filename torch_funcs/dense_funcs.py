import torch
import torch.nn as nn

def torch_dense_baseline(
    x: torch.Tensor,
    Wt: torch.Tensor,
    best_config=None) -> torch.Tensor:

    """
    Performs dense matrix multiplication 

    Parameters:
    ----------
    x : torch.Tensor
        Input tensor of shape (num_batches, num_seq, in_f)
    Wt : torch.Tensor
        Tensor of shape (out_f, in_f)

    Returns:
    -------
    out: torch.Tensor
        Output tensor of shape (num_batches, num_seq, out_f), resulting from the dense matrix multiplication
    """

    return nn.functional.linear(x, Wt)