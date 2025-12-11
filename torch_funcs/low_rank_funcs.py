import torch
import torch.nn as nn

def torch_low_rank_baseline(
    x: torch.Tensor,  
    Vt: torch.Tensor, 
    Ut: torch.Tensor,
    best_config=None) -> torch.Tensor:

    """
    Performs low-rank matrix multiplication 

    Parameters:
    ----------
    x : torch.Tensor
        Input tensor of shape (num_batches, num_seq, in_f)
    Vt : torch.Tensor
        Tensor of shape (rank, in_f)
    Ut : torch.Tensor
        Tensor of shape (out_f, rank)

    Returns:
    -------
    out: torch.Tensor
        Output tensor of shape (num_batches, num_seq, out_f), resulting from the low-rank matrix multiplication
    """

    return nn.functional.linear(nn.functional.linear(x, Vt), Ut)