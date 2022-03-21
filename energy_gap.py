from torch import tensor, stack, float64, float32, profiler
from torch import device
import numpy as np
from hamiltonian import Sky_phi
from hamiltonian import ham_total
from xitorch import linalg
from test_sparse_eigen import CsrLinOp
#import memory_profiler


dev    = device("cuda")

#@memory_profiler.profile
def energy_gap(param_values, J_1 = -1.0, L = 16, n_eigs = 3, dtype = float64):#, scalfac = 1.0, delta = 0.5):
    with profiler.profile(activities=[ profiler.ProfilerActivity.CPU, profiler.ProfilerActivity.CUDA], with_stack = True, profile_memory = True) as prof:
        with profiler.record_function("INITIALIZATION"):
            center  = L / 2 - 0.5
            delta   = param_values[2]
            scalfac = param_values[3]
            B_0     = tensor([param_values[0]], dtype = dtype, device = dev)
            B_ext   = tensor([param_values[1]], dtype = dtype, device = dev)
            phi_i   = tensor(Sky_phi(L, center, delta, scalfac), dtype = dtype, device = dev)
        with profiler.record_function("MATRIXBUILDER"):
            if B_0.dtype == float64:
                H = ham_total(L, J_1 , B_0, B_ext, phi_i, prec = 64)
            else:
                H = ham_total(L, J_1 , B_0, B_ext, phi_i, prec = 32)
            H_linop = CsrLinOp(stack([H.storage._row, H.storage._col], dim = 0), H.storage._value, H.size(0))
        with profiler.record_function("EIGENSOLVER"):
            eigvals = linalg.symeig(H_linop, neig = n_eigs, method = "davidson", max_niter = 1000, nguess = None,
                                    v_init = "randn", max_addition = None, min_eps = 1e-07, verbose = False,
                                    bck_options={'method': 'bicgstab', 'rtol': 1e-05, 'atol': 1e-06, 'eps': 1e-8,
                                                 'verbose': False, 'max_niter': 100})[0]
    print(prof.key_averages().table(sort_by = "self_cuda_memory_usage", row_limit = 40))
    prof.export_chrome_trace("trace.json")
    return np.log((eigvals[1] - eigvals[0]).cpu().detach().numpy() + 1e-6)


def energy_gap_2param_test(param_values, J_1 = -1.0, L = 16, n_eigs = 3, dtype = float64):#, scalfac = 1.0, delta = 0.5):
    center  = L / 2 - 0.5
    delta   = 0.5
    scalfac = 1
    B_0     = tensor([param_values[0]], dtype = dtype, device = dev)
    B_ext   = tensor([param_values[1]], dtype = dtype, device = dev)
    phi_i   = tensor(Sky_phi(L, center, delta, scalfac), dtype = dtype, device = dev)

    if B_0.dtype == float64:
        H = ham_total(L, J_1 , B_0, B_ext, phi_i, prec = 64)
    else:
        H = ham_total(L, J_1 , B_0, B_ext, phi_i, prec = 32)
    H_linop = CsrLinOp(stack([H.storage._row, H.storage._col], dim = 0), H.storage._value, H.size(0))
        
    eigvals = linalg.symeig(H_linop, neig = n_eigs, method = "davidson", max_niter = 1000, nguess = None,
                            v_init = "randn", max_addition = None, min_eps = 1e-07, verbose = False,
                            bck_options={'method': 'bicgstab', 'rtol': 1e-05, 'atol': 1e-06, 'eps': 1e-8,
                                         'verbose': False, 'max_niter': 10000})[0]

    return (eigvals[1] - eigvals[0]).cpu().detach().numpy()#np.log((eigvals[1] - eigvals[0]).cpu().detach().numpy() + 1e-6)

