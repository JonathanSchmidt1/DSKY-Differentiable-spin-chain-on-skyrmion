from torch import nn
import torch
from hamiltonian import ham_total
from xitorch import linalg
from test_sparse_eigen import CsrLinOp
import numpy as np


class HamModule(nn.Module):

    pi = np.pi

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

        super(HamModule, self).__init__()
        self.J_1 = J_1
        # named tensors are not supported for our usage
        self.L = torch.tensor([L], device=device)#nn.Parameter(torch.tensor([L]), requires_grad=False)
        self.B_0 = nn.Parameter(torch.tensor([B_0], device=device, dtype=dtype), requires_grad=True)
        self.B_ext = nn.Parameter(torch.tensor([B_ext], device=device, dtype=dtype), requires_grad=True)
        self.phi_diff = nn.Parameter(torch.tensor(phi_diff, device=device, dtype=dtype), requires_grad=True)
        self.phi_i = torch.zeros((L,), dtype = torch.float64)

    def output_parameters(self):
        
        return [(param, getattr(self, param).detach().cpu().numpy()) for param in ["B_0", "B_ext", "phi_i"]]

    def forward(self,n_eigs):
        """
        Parameters
        ----------
        n_eigs: number of eigenvalues/vectors that are calculated
        Returns
        ----------
        eigvals (torch.tensor) and eigvectors (torch.tensor)
        """

        self.phi_i = torch.square(self.phi_diff)
        self.phi_i = torch.cumsum(self.phi_i, 0)
        self.phi_i = self.phi_i * self.pi / self.phi_i[-1]
        self.phi_i = self.phi_i - self.phi_i[0]
        #self.phi_i = torch.cat((self.phi_i, torch.flip(2 * self.pi - self.phi_i, (0,))))
        self.phi_i = torch.cat((self.phi_i, torch.flip(self.phi_i, (0,))))
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

if __name__ == "__main__":
    from bipartite_entropy import calculate_entropies
    from hamiltonian import Sky_phi
    from pathlib import Path
    import h5py
    import time
    Path("output_dwall2").mkdir(parents = True, exist_ok = True)

    L = 12
    nsteps = 2000
    #weight list for loss function
    #weight_list = torch.tensor([L//2 - i for i in range(1, L//2)] + [L - 2] + [i for i in range(1, L//2)]).cuda()
    #weight_list = torch.tensor([L//2 - i for i in range(1, L//2)] + [L - 2] + [i for i in range(1, L//2)]).cuda()
    weight_list = torch.full((L-1,),1.0).cuda()
    print("The weight list for the entropy:", weight_list.tolist())

    para_names = ["B_0", "B_ext", "phi_diff"]
    J1 = -1.0
    B_0 = -0.4
    B_ext = -0.08
    scalfac = 1.0
    delta = 0.5
    center = L / 2 - 0.5

    phis = np.array(Sky_phi(L, center, delta, scalfac))[:L//2 + 1] + np.pi
    phi_diff = np.sqrt(np.diff(phis))
    n_eigs = 3
    H = HamModule(L, J1, B_0, B_ext, phi_diff, device='cuda')
    optimizer = torch.optim.Adam(H.parameters(),
                           lr = 0.001)

    ideal_ent = torch.zeros(L - 1, dtype = torch.double).cuda()
    ideal_ent[L // 2 - 1] = np.log(2)

    out_file = h5py.File('output_dwall2/test_output.h5', 'w', libver = 'latest')
    fixedset = out_file.create_dataset("fixed values", (2,), data = [L, J1])

    entset = out_file.create_dataset("entropy", (nsteps,L - 1))
    lossset = out_file.create_dataset("loss", (nsteps,))

    paramsset = []
    for i_para, para in enumerate(H.output_parameters()):
        paramsset.append(out_file.create_dataset(para[0], (nsteps,) + para[1].shape))
    out_file.swmr_mode = True

    #out_file = open("output/entropy_loss.txt", "w")
    start = time.time()
    for i in range(nsteps):
        eigvals, eigvecs = H.forward(n_eigs)
        #print(eigvals)
        loss = torch.tensor([0.]).requires_grad_().cuda()
        for i_eig in range(1):
            ent = calculate_entropies(eigvecs[:, i_eig], L, [2] * L)
            loss += torch.sum(torch.square(weight_list * (ent - ideal_ent)))
        
        entlist = ent.tolist()
        entset[i] = entlist
        entset.flush()
        lossset[i] = loss.item()
        lossset.flush()

        for i_para, para in enumerate(H.output_parameters()):
            paramsset[i_para][i] = para[1]
            paramsset[i_para].flush()
        print('loss[{}] ={}'.format(i + 1, loss.item()))
        #for i in range(L - 1):
        #    out_file.write(str(entlist[i]) + "\t")
        #out_file.write(str(loss.item()) + "\n")

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print((time.time()-start)/(i+1))
    out_file.close()
    print("Entropy after optimization:", ent.tolist())

    for para in H.parameters():

        print(para)
   
