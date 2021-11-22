from torch import nn
import torch
from hamiltonian import ham_total
from xitorch import linalg
from test_sparse_eigen import CsrLinOp


class HamModule(nn.Module):
    def __init__(self, L, J1,  B_0, B_ext, phi_i, device='cuda',dtype=torch.double):
        """
        Parameters
        ----------
        L: int Length of spin chain
        B_0: float initial B_0
        B_ext: float initial B_ext
        phi_i: float array of angle differences
        Returns
        ----------
        """
        super(HamModule, self).__init__()
        self.J1 = J1
        self.L = nn.Parameter(torch.tensor(L), requires_grad=False)
        self.B_0 = nn.Parameter(torch.tensor(B_0, device=device, dtype=dtype), requires_grad=True)
        self.B_ext = nn.Parameter(torch.tensor(B_ext, device=device, dtype=dtype), requires_grad=True)
        self.phi_i = nn.Parameter(torch.tensor(phi_i, device=device, dtype=dtype), requires_grad=True)


    def forward(self,n_eigs):
        """
        Parameters
        ----------
        n_eigs: number of eigenvalues/vectors that are calculated
        Returns
        ----------
        eigvals (torch.tensor) and eigvectors (torch.tensor)
        """
        H = ham_total(self.L.item(), self.J1 , self.B_0.item(), self.B_ext.item(), self.phi_i.data, prec=64)
        #H_linop = CsrLinOp(torch.stack([H.storage._row, H.storage._col], dim=0), H.storage._value * self.B_0, H.size(0))
        H_linop = CsrLinOp(torch.stack([H.storage._row, H.storage._col], dim=0), H.storage._value, H.size(0))
        eigvals, eigvecs = linalg.symeig(H_linop, neig=n_eigs, method="davidson", max_niter=1000, nguess=None,
                                         v_init="randn",
                                         max_addition=None, min_eps=1e-07, verbose=False,
                                         bck_options={'method': 'bicgstab', 'rtol': 1e-05, 'atol': 1e-06, 'eps': 1e-8,
                                                      'verbose': False, 'max_niter': 10000})
        return eigvals, eigvecs

if __name__ == "__main__":
    from bipartite_entropy import calculate_entropies
    from hamiltonian import Sky_phi
    import numpy as np

    L=12
    #weight list for loss function
    weight_list = torch.tensor([L//2 - i for i in range(1, L//2)] + [-L + 2] + [i for i in range(1, L//2)]).cuda()
    print(weight_list)

    J1 = -1.0
    B_0 = 0.4
    B_ext = -0.0
    scalfac = 1.0
    delta = 0.5
    center = L/2 - 0.5

    phi_i = Sky_phi(L, center, delta, scalfac)
    n_eigs = 3
    H = HamModule(L, J1, B_0, B_ext, phi_i, device='cpu')
    optimizer = torch.optim.Adam(H.parameters(),
                           lr= 0.0001)
    optimizer.zero_grad()

    for i in range(10):
        eigvals, eigvecs = H.forward(n_eigs)
        loss = torch.tensor([0.]).requires_grad_().cuda()
        for i_eig in range(1):
            loss += torch.sum(weight_list * calculate_entropies(eigvecs[:, i_eig], L, [2] * L))
        #TODO: if you want to test backward comment back in line 37 to get a dependence of the Hamiltonian on the parameters
        print(loss)
        loss.backward()
        optimizer.step()
