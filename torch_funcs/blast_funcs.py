import torch

def torch_blast_baseline(
    x: torch.Tensor, 
    U: torch.Tensor, 
    Vt: torch.Tensor, 
    S: torch.Tensor,
    best_config=None) -> torch.Tensor:

    """
    Performs full BLAST matrix multiplication as in [NeurIPS24] https://arxiv.org/pdf/2410.21262v1

    Parameters:
    ----------
    x : torch.Tensor
        Input tensor of shape (num_batches, num_seq, in_f)
    U : torch.Tensor
        Tensor of shape (b2, out_f // b2, rank)
    Vt : torch.Tensor
        Tensor of shape (b1, rank, in_f // b1)
    S : torch.Tensor
        Scaling tensor of shape (b2, b1, rank)

    Returns:
    -------
    out: torch.Tensor
        Output tensor of shape (num_batches, num_seq, out_f), resulting from the BLAST matrix multiplication
    """

    b1, r, p = Vt.shape
    b2, q, _ = U.shape

    assert b1 == b2
    assert b1 != 1 and b2 != 1

    x_shape = x.shape
    x = x.flatten(0, -2)
    x = x.view(-1, b1, x_shape[-1] // b1).transpose(0, 1)

    y = torch.bmm(x, Vt.transpose(1, 2))
    z = y.unsqueeze(0) * S.unsqueeze(2)
    z = z.sum(1)

    out = torch.bmm(z, U.transpose(1, 2))
    out = out.transpose(0, 1).reshape(*(x_shape[:-1] + (b2 * q,)))
    
    return out, z, y

def torch_blast_bmm(
    x: torch.Tensor, 
    U: torch.Tensor, 
    Vt: torch.Tensor, 
    S: torch.Tensor) -> torch.Tensor:

    """
    Performs BLAST Matrix Multiplication using 3 BMMs

    Parameters:
    ----------
    x : torch.Tensor
        Input tensor of shape (num_batches, num_seq, in_f)
    U : torch.Tensor
        Tensor of shape (b2, out_f // b2, rank)
    Vt : torch.Tensor
        Tensor of shape (b1, rank, in_f // b1)
    S : torch.Tensor
        Scaling tensor of shape (rank, b2, b1)

    Returns:
    -------
    out: torch.Tensor
        Output tensor of shape (num_batches, num_seq, out_f), resulting from the BLAST matrix multiplication
    """

    b1, r, p = Vt.shape
    b2, q, _ = U.shape

    assert b1 == b2
    assert b1 != 1 and b2 != 1

    x_shape = x.shape
    x = x.flatten(0, -2)
    x = x.view(-1, b1, x_shape[-1] // b1).transpose(0, 1)

    y = torch.bmm(x, Vt.transpose(1, 2))
    z =  torch.bmm(y.transpose(0, 2).contiguous(), S.transpose(1, 2))

    out = torch.bmm(z.transpose(0, 2).contiguous(), U.transpose(1, 2))
    out = out.transpose(0, 1).reshape(*(x_shape[:-1] + (q * b2,)))
    
    return out, z, y