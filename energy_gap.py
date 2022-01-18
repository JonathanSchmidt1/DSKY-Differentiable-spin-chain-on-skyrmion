from logging import root
import torch
import numpy as np
from hamiltonian import Sky_phi
from nnmodule import HamModule
from pathlib import Path
from xitorch.optimize import minimize, rootfinder
from hamiltonian import ham_total
from xitorch import linalg
from test_sparse_eigen import CsrLinOp
Path("output").mkdir(parents = True, exist_ok = True)

L = 12
nsteps = 2000
device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
dtype = torch.float64

para_names = ["B_0", "B_ext", "phi_i"]
J_1 = -1.0
B_0 = -0.4
B_ext = -0.08
scalfac = 1.0
delta = 0.5
center = L / 2 - 0.5
phis = np.array(Sky_phi(L, center, delta, scalfac))[:L//2 + 1] + np.pi
phi_i = np.sqrt(np.diff(phis))
n_eigs = 3
#H = HamModule(L, J1, B_0, B_ext, phi_i, device = 'cuda')
param0 = torch.tensor([B_0, B_ext] + list(phi_i), device = device, dtype = dtype)
print(param0)

def energy_gap(param_values):
    B_0    = param_values[0]
    B_ext  = param_values[1]
    phi_i  = param_values[2:]
    phi_i2 = torch.square(phi_i)
    phi_i2 = torch.cumsum(phi_i2, 0)
    phi_i2 = phi_i2 * torch.pi / phi_i2[-1]
    phi_i2 = phi_i2 - phi_i2[0]
    phi_i2 = torch.cat((phi_i2, torch.flip(2 * torch.pi - phi_i2, (0,))))
    if B_0.dtype == torch.float64:
        H = ham_total(L, J_1 , B_0, B_ext, phi_i2, prec = 64)
    else:
        H = ham_total(L, J_1 , B_0, B_ext, phi_i2, prec = 32)
    H_linop = CsrLinOp(torch.stack([H.storage._row, H.storage._col], dim=0), H.storage._value, H.size(0))
    eigvals = linalg.symeig(H_linop, neig=n_eigs, method="davidson", max_niter=1000, nguess=None,
                            v_init="randn",
                            max_addition=None, min_eps=1e-07, verbose=False,
                            bck_options={'method': 'bicgstab', 'rtol': 1e-05, 'atol': 1e-06, 'eps': 1e-8,
                                         'verbose': False, 'max_niter': 10000})[0]
    return eigvals[1] - eigvals[0]



ymin = minimize(energy_gap, param0, verbose = True)
#ymin = rootfinder(energy_gap, param0, verbose = True)
print(ymin)


