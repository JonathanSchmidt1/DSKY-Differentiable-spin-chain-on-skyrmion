
import torch
from partial_trace import partial_trace

def calculate_entropy(state, L, dims, sys_a):
    """
    Calculates the entanglement of 'state' with the dimensions 'dims' between 'sys_a' and the rest.
    
    Parameters
    ----------
    state : torch.tensor with dimensionality 1 and size of 2^L
    L     : int length of the spin chain
    dims  : list of spin dimensions with length L
    sys_a : list of sites in system A

    Returns
    -------
    entanglement : float
    """

    assert(len(dims) == L)
    if len(sys_a) > L // 2:
        sys = range(L)
        sys_a = [k for k in sys if k not in sys_a]

    reduced_densmat = partial_trace(state, dims, keep = sys_a)
    eigvals = torch.linalg.eigh(reduced_densmat)[0]
    eigvals = eigvals[eigvals > 1e-6]
    entanglement = torch.sum(- eigvals * torch.log(eigvals))
    return entanglement

def calculate_entropies(state, L, dims, device = 'cuda'):
    """
    Calculates the entanglements of 'state' with the dimensions 'dims' for all 'two-cuts'.

    Parameters
    ----------
    state : torch.tensor with dimensionality 1 and size of 2^L
    L     : int length of the spin chain
    dims  : list of spin dimensions with length L

    Returns
    -------
    entanglements : torch.tensor of floats
    """

    entanglements = torch.empty((L - 1,), dtype = torch.float64, device = device)
    for bond in range(L - 1):
        entanglements[bond] = calculate_entropy(state, L, dims, range(bond + 1))
    return entanglements

    


if __name__ == "__main__":
    import numpy as np
    a = torch.tensor([0 , 1 / np.sqrt(2), 1 / np.sqrt(2), 0, 0, 0, 0, 0], dtype = torch.complex128, device = 'cuda').requires_grad_()
    print("Entropy divided by ln(2):", calculate_entropy(a, 3, [2, 2, 2], [0]) / np.log(2))
    print(calculate_entropies(a, 3, [2, 2, 2]))
    loss = torch.sum(calculate_entropies(a, 3, [2, 2, 2]))
    print("check if derivative can be calculated and is non-zero:", (torch.sum(torch.abs(torch.autograd.grad(loss, a)[0]))>0).item())





