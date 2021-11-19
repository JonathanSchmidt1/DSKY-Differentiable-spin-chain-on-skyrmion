import torch
import numpy as np

def partial_trace(vec, dims, keep):
    """
    Calculates the reduced density matrix of 'vec' with the dimensions len('keep')^2.
    
    Parameters
    ----------
    vec   : torch.tensor with dimensionality 1 and size of 2^L
    dims  : list of spin dimensions with length L
    keep  : list or tuple of site indices to keep
    
    Returns
    -------
    red_dens_mat : torch.tensor
    """
    if isinstance(keep, int):
        keep = (keep,)
    vec  = torch.reshape(vec, dims)
    lose = tuple(i for i in range(len(dims)) if i not in keep)
    vec = torch.tensordot(vec, torch.conj(vec), dims = (lose, lose))
    d = int(torch.numel(vec)**0.5)
    red_dens_mat = torch.reshape(vec, (d, d))
    return red_dens_mat

if __name__ == "__main__":
    a = torch.tensor([0 , 1 / np.sqrt(2), 1 / np.sqrt(2), 0])
    print(partial_trace(a, [2, 2], keep = 0))





