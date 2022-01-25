from torch import nn
import torch
from hamiltonian import ham_total
from xitorch import linalg
from test_sparse_eigen import CsrLinOp
import numpy as np
from hamiltonian import Sky_phi


class HamModule(nn.Module):

    def __init__(self, L, J_1, B_0, B_ext, device='cuda',dtype=torch.double):
        """
        Parameters
        ----------
        L: int Length of spin chain
        B_0: float initial B_0
        B_ext: float initial B_ext
        ----------
        """

        super(HamModule, self).__init__()

        self.J_1 = J_1
        # named tensors are not supported for our usage
        self.L = torch.tensor([L], device=device)#nn.Parameter(torch.tensor([L]), requires_grad=False)
        self.B_0 = nn.Parameter(torch.tensor([B_0], device=device, dtype=dtype), requires_grad=True)
        self.B_ext = nn.Parameter(torch.tensor([B_ext], device=device, dtype=dtype), requires_grad=True)
        self.phi_i = torch.zeros((L,), dtype = torch.float64)

    def output_parameters(self):
        
        return [(param, getattr(self, param).detach().cpu().numpy()) for param in ["B_0", "B_ext", "phi_i"]]

    def MakeHamAndSolve(self, n_eigs):

        if self.B_0.dtype==torch.double:
            H = ham_total(self.L.item(), self.J_1 , self.B_0, self.B_ext, self.phi_i, prec=64)
        else:
            H = ham_total(self.L.item(), self.J_1 , self.B_0, self.B_ext, self.phi_i, prec=32)

        H_linop = CsrLinOp(torch.stack([H.storage._row, H.storage._col], dim=0), H.storage._value, H.size(0))
        eigvals, eigvecs = linalg.symeig(H_linop, neig=n_eigs, method="davidson", max_niter=1000, nguess=None,
                                         v_init="randn",
                                         max_addition=None, min_eps=1e-07, verbose=False,
                                         bck_options={'method': 'bicgstab', 'rtol': 1e-05, 'atol': 1e-06, 'eps': 1e-8,
                                                      'verbose': False, 'max_niter': 10000})
        
        return eigvals, eigvecs
    
    def forward(self,n_eigs):

        pass



class HamModule_param(HamModule):

    def __init__(self, L, J_1,  B_0, B_ext, scalfac, delta, device='cuda', dtype=torch.double):
        """
        Parameters
        ----------
        L: int Length of spin chain
        B_0: float initial B_0
        B_ext: float initial B_ext
        scalfac: float initial scaling factor
        delta: float initial wall width parameter
        Returns
        ----------
        """

        super().__init__(L, J_1, B_0, B_ext, device = device, dtype = dtype)
        self.scalfac = nn.Parameter(torch.tensor([scalfac], device=device, dtype=dtype), requires_grad=True)
        self.delta = nn.Parameter(torch.tensor([delta], device=device, dtype=dtype), requires_grad=True)
        self.phi_i = torch.zeros((L,), device = device, dtype = dtype)
    
    def output_parameters(self):
        
        return [(param, getattr(self, param).detach().cpu().numpy()) for param in ["B_0", "B_ext", "scalfac", "delta", "phi_i"]]
    
    def set_phi_i(self):

        self.phi_i = torch.tensor(Sky_phi(self.L.item(), self.L.item()/2 - 0.5, self.delta.item(), self.scalfac.item()), device = self.phi_i.device, dtype = self.phi_i.dtype) + torch.pi

        return
    
    def forward(self, n_eigs):
        """
        Parameters
        ----------
        n_eigs: number of eigenvalues/vectors that are calculated
        Returns
        ----------
        eigvals (torch.tensor) and eigvectors (torch.tensor)
        """

        self.set_phi_i()
        return super().MakeHamAndSolve(n_eigs)


class HamModule_phi(HamModule):

    def __init__(self, L, J_1,  B_0, B_ext, phi_diff, device='cuda',dtype=torch.double):
        """
        Parameters
        ----------
        L: int Length of spin chain
        B_0: float initial B_0
        B_ext: float initial B_ext
        phi_diff: float array of angle differences
        Returns
        ----------
        """

        super().__init__(L, J_1, B_0, B_ext, device = device, dtype = dtype)
        self.phi_diff = nn.Parameter(torch.tensor(phi_diff, device=device, dtype=dtype), requires_grad=True)

    def set_phi_i(self, which):

        self.phi_i = torch.square(self.phi_diff)
        self.phi_i = torch.cumsum(self.phi_i, 0)
        self.phi_i = self.phi_i * self.pi / self.phi_i[-1]
        self.phi_i = self.phi_i - self.phi_i[0]

        if which == "skyrmion":

            self.phi_i = torch.cat((self.phi_i, torch.flip(-self.phi_i, (0,))))

            return
        
        elif which == "domainwall":

            self.phi_i = torch.cat((self.phi_i, torch.flip(self.phi_i, (0,))))

            return
        
        else:

            raise ValueError("which must be either skyrmion or domainwall")

    def forward(self, n_eigs):
        """
        Parameters
        ----------
        n_eigs: number of eigenvalues/vectors that are calculated
        Returns
        ----------
        eigvals (torch.tensor) and eigvectors (torch.tensor)
        """

        self.set_phi_i("skyrmion")
        return super().MakeHamAndSolve(n_eigs)