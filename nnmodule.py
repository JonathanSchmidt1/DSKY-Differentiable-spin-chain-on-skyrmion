from torch import nn
import torch
from hamiltonian import ham_total
from xitorch import linalg
from test_sparse_eigen import CsrLinOp
import numpy as np


class HamModule(nn.Module):

    pi = np.pi

    def __init__(self, L, J_1,  B_0, B_ext, phi_i, device='cuda',dtype=torch.double):
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
        self.J_1 = J_1
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

        self.phi_i2 = torch.square(self.phi_i)
        self.phi_i2 = torch.cumsum(self.phi_i2, 0)
        self.phi_i2 = self.phi_i2 * self.pi / self.phi_i2[-1]
        self.phi_i2 = self.phi_i2 - self.phi_i2[0]
        self.phi_i2 = torch.cat((self.phi_i2, torch.flip(2 * self.pi - self.phi_i2, (0,))))
        
        H = ham_total(self.L.item(), self.J_1 , self.B_0, self.B_ext, self.phi_i2, prec=64)
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

    L = 16
    #weight list for loss function
    #weight_list = torch.tensor([L//2 - i for i in range(1, L//2)] + [L - 2] + [i for i in range(1, L//2)]).cuda()
    weight_list = torch.tensor([1 for i in range(1, L//2)] + [L - 2] + [1 for i in range(1, L//2)]).cuda()
    print(weight_list)

    J1 = -1.0
    B_0 = -0.4
    B_ext = -0.08
    scalfac = 1.0
    delta = 0.5
    center = L/2 - 0.5

    phis = np.array(Sky_phi(L, center, delta, scalfac))[:L//2 + 1] + np.pi
    phi_i = np.sqrt(np.diff(phis))
    n_eigs = 3
    H = HamModule(L, J1, B_0, B_ext, phi_i, device='cpu')
    optimizer = torch.optim.Adam(H.parameters(),
                           lr = 0.0001)

    ent_list = []
    ideal_ent = torch.zeros(L - 1, dtype = torch.double).cuda()
    ideal_ent[L // 2 - 1] = np.log(2)

    for i in range(100):
        eigvals, eigvecs = H.forward(n_eigs)
        #print(eigvals)
        loss = torch.tensor([0.]).requires_grad_().cuda()
        for i_eig in range(1):
            ent = calculate_entropies(eigvecs[:, i_eig], L, [2] * L)
            loss += torch.sum(torch.square(weight_list * (ent - ideal_ent)))
            ent_list.append(ent)
        
        print(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(ent)

    for para in H.parameters():

        print(para)
    
