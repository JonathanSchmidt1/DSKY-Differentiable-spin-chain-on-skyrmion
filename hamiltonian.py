import numpy as np
import scipy
import torch
import torch_sparse
from torch_sparse import coalesce

pauli32 = [
    torch_sparse.SparseTensor(row=torch.tensor([0, 1]), col=torch.tensor([1, 0]),
                              value=torch.tensor([1, 1], dtype=torch.float32), sparse_sizes=(2, 2)),
    torch_sparse.SparseTensor(row=torch.tensor([0, 1]), col=torch.tensor([1, 0]),
                              value=torch.tensor([-1j, 1j], dtype=torch.complex64), sparse_sizes=(2, 2)),
    torch_sparse.SparseTensor(row=torch.tensor([0, 1]), col=torch.tensor([0, 1]),
                              value=torch.tensor([1, -1], dtype=torch.float32), sparse_sizes=(2, 2))
]

pauli64 = [
    torch_sparse.SparseTensor(row=torch.tensor([0, 1]), col=torch.tensor([1, 0]),
                              value=torch.tensor([1, 1], dtype=torch.float64), sparse_sizes=(2, 2)),
    torch_sparse.SparseTensor(row=torch.tensor([0, 1]), col=torch.tensor([1, 0]),
                              value=torch.tensor([-1, 1], dtype=torch.float64), sparse_sizes=(2, 2)),
    torch_sparse.SparseTensor(row=torch.tensor([0, 1]), col=torch.tensor([0, 1]),
                              value=torch.tensor([1, -1], dtype=torch.float64), sparse_sizes=(2, 2))
]

ladder32 = [
    torch_sparse.SparseTensor(row=torch.tensor([0]), col=torch.tensor([1]),
                              value=torch.tensor([1], dtype=torch.float32), sparse_sizes=(2, 2)),
    torch_sparse.SparseTensor(row=torch.tensor([1]), col=torch.tensor([0]),
                              value=torch.tensor([1], dtype=torch.float32), sparse_sizes=(2, 2)),
]

ladder64 = [
    torch_sparse.SparseTensor(row=torch.tensor([0]), col=torch.tensor([1]),
                              value=torch.tensor([1], dtype=torch.float64), sparse_sizes=(2, 2)),
    torch_sparse.SparseTensor(row=torch.tensor([1]), col=torch.tensor([0]),
                              value=torch.tensor([1], dtype=torch.float64), sparse_sizes=(2, 2)),
]

eye32 = torch_sparse.SparseTensor(row=torch.tensor([0, 1]), col=torch.tensor([0, 1]),
                                  value=torch.tensor([1, 1], dtype=torch.float32), sparse_sizes=(2, 2))
eye64 = torch_sparse.SparseTensor(row=torch.tensor([0, 1]), col=torch.tensor([0, 1]),
                                  value=torch.tensor([1, 1], dtype=torch.float64), sparse_sizes=(2, 2))


def add_tensor(a, b):
    """Matrix addition of two sparse matrices.
    Args:
        indexA (:class:`torch_sparse.SparseTenor`): first sparse matrix.
        valueA (:class:`torch_sparse.SparseTenor`): second sparse matrix.
    Returns:
        (:class:`torch_sparse.SparseTenor`) sum of the two sparse matrices
    """
    indexA = torch.stack((a.storage._row, a.storage._col))
    indexB = torch.stack((b.storage._row, b.storage._col))
    valueA = a.storage._value
    valueB = b.storage._value
    m, n = a.size(0), a.size(1)
    index_value_tuple = spadd(indexA, valueA, indexB, valueB, m, n)
    return torch_sparse.SparseTensor(row = index_value_tuple[0][0] , col = index_value_tuple[0][1],
                                     value = index_value_tuple[1],  sparse_sizes =(m,n))



def spadd(indexA, valueA, indexB, valueB, m, n):
    """Matrix addition of two sparse matrices.
    Args:
        indexA (:class:`LongTensor`): The index tensor of first sparse matrix.
        valueA (:class:`Tensor`): The value tensor of first sparse matrix.
        indexB (:class:`LongTensor`): The index tensor of second sparse matrix.
        valueB (:class:`Tensor`): The value tensor of second sparse matrix.
        m (int): The first dimension of the sparse matrices.
        n (int): The second dimension of the sparse matrices.
    """
    index = torch.cat([indexA, indexB], dim=-1)
    value = torch.cat([valueA, valueB], dim=0)
    #print(index, value)
    return coalesce(index=index, value=value, m=m, n=n, op='add')


# kronecker product of two torch_sparse.SparseTensor objects
def sparse_kron(A, B):
    if A.nnz == 0 or B.nnz == 0:
        return torch_sparse.SparseTensor.from_dense(torch.zeros(A.sparse_sizes[0] * B.sparse_sizes[0], A.sparse_sizes[1] * B.sparse_sizes[1]))

    row = A.storage.row().repeat_interleave(B.nnz())
    col = A.storage.col().repeat_interleave(B.nnz())
    data = A.storage.value().repeat_interleave(B.nnz())

    row *= B.sparse_sizes()[0]
    col *= B.sparse_sizes()[1]

    row, col = row.reshape((-1, B.nnz())), col.reshape((-1, B.nnz()))
    row += B.storage.row()
    col += B.storage.col()
    row, col = row.reshape((-1)), col.reshape((-1))

    data = data.reshape((-1, B.nnz())) * B.storage.value()
    data = data.reshape((-1))
    return torch_sparse.SparseTensor(row=row, col=col, value=data, sparse_sizes = (A.sparse_sizes()[0] * B.sparse_sizes()[0], A.sparse_sizes()[1] * B.sparse_sizes()[1]))


# tensoring A of shape(2n,2n) sitting at sites=[i, ..., i+n] into larger hilbertspace by padding with 2x2-identities from left and right
def sparse_ikron(A, L, sites, prec=64):
    if prec == 64:
        eye = eye64

    elif prec == 32:
        eye = eye32

    else:
        raise ValueError("prec must be either 32 or 64")

    # check for square matrix with dimension of power 2
    assert (A.sparse_sizes()[0] == A.sparse_sizes()[1] and (A.sparse_sizes()[0] & (A.sparse_sizes()[0] - 1) == 0) and
            A.sparse_sizes()[0] != 0)

    if sites[0] > 0:
        res = eye
    else:
        res = A

        for i in range(L - sites[-1] - 1):
            res = sparse_kron(res, eye)

        return res

    for k in range(sites[0] - 1):
        res = sparse_kron(res, eye)

    res = sparse_kron(res, A)

    for k in range(L - sites[-1] - 1):
        res = sparse_kron(res, eye)

    return res


def tensor_sum(it):
    for i, tensor in enumerate(it):
        if i == 0:
            res = tensor.copy()

        else:
            res = add_tensor(res, tensor)

    return res


# product with scalar
def tensor_elem_mul(A, c, inplace = False):

    if not inplace:
        A_cp = A.copy()
        A_cp.storage._value = torch.mul(A_cp.storage._value, c)

        return A_cp
    
    else:
        A.storage._value = torch.mul(A.storage._value, c)
        return



def set_prec(prec):

    if prec ==64:
        pauli = pauli64
        ladder = ladder64

    elif prec == 32:
        pauli = pauli32
        ladder = ladder32

    else:
        raise ValueError("prec must be either 32 or 64")
    
    return pauli,ladder




def ham_j1(L, J1 = 1.0, prec = 64):
    """
    Constructs the isotropic J1-Hamiltonian 
    
    Parameters
    ----------
    L     : int length of the spin chain
    J1    : float coupling strength of interaction, optional
    prec  : either 32 or 64, sets float precision to be used

    Returns
    -------
    ham : torch_sparse.SparseTensor
    """

    pauli,ladder = set_prec(prec)

    summands = [sparse_kron(ladder[0], ladder[1]), sparse_kron(ladder[1], ladder[0]), sparse_kron(pauli[2], pauli[2])]

    for S in summands[:2]: tensor_elem_mul(S, 2, inplace = True)

    def j1_terms():

        for i in range(L - 1):
            coo = [i, i + 1]

            yield sparse_ikron(tensor_sum(summands), L, coo)

    ham = tensor_elem_mul(tensor_sum(j1_terms()), J1 / 4)

    return ham


def ham_mag(L, B0, B_ext, theta, prec=64):
    """
    Constructs the magnetic interaction Hamiltonian produced by a magnetic Skyrmion of amplitude B0 and a global magnetic field in z-direction with amplitude B_ext
    
    Parameters
    ----------
    L     : int length of the spin chain
    B0    : float amplitude of Skyrmion
    B_ext : float amplitude of global magnetic field in z-direction
    theta   : list of float of length L contains the angles in xz-plane between magnetic field and z-axis

    Returns
    -------
    ham : torch_sparse.SparseTensor
    """

    pauli,ladder = set_prec(prec)

    def ext_terms():

        for i in range(L):
            yield tensor_elem_mul(sparse_ikron(pauli[2], L, [i], prec), -B_ext / 2)
    

    def sky_x_terms():

        for i,cphi in enumerate(theta):
            yield sparse_ikron(tensor_elem_mul(pauli[0], - B0 * torch.sin(cphi) / 2), L, [i], prec = prec)


    def sky_z_terms():

        for i,cphi in enumerate(theta):
            yield sparse_ikron(tensor_elem_mul(pauli[2], - B0 * torch.cos(cphi) / 2), L, [i], prec = prec)
    
    ham = tensor_sum([tensor_sum(ext_terms()), tensor_sum(sky_x_terms()), tensor_sum(sky_z_terms())])

    return ham


def ham_total(L, J1, B0, B_ext, theta, prec=64):

    return add_tensor(ham_j1(L, J1 = J1), ham_mag(L, B0, B_ext, theta))


def Sky_phi(L, q, delta, scalfac):

    def theta(x, q, delta, scalfac):

        if np.abs(x-q) < 1.0e-10:
            return 0


        return(np.sign(x-q)*2*np.arctan(np.exp((np.abs((x-q)/scalfac)-1/np.abs((x-q)/scalfac))/delta)))

    return [theta(i, q, delta, scalfac) for i in range(L)]



if __name__ == '__main__':
    L = 10

    J1 = -1.0
    B0 = -0.4
    B_ext = -0.08

    scalfac = 1.0
    delta = 0.5
    center = L/2 - 0.5
    

    theta_list = torch.tensor(Sky_phi(L,center,delta,scalfac)) + np.pi
    phis = np.array(Sky_phi(L, center, delta, scalfac))[:L//2 + 1] + np.pi
    phi_i = torch.tensor(np.sqrt(np.diff(phis)))
    phi_i2 = torch.square(phi_i)
    phi_i2 = torch.cumsum(phi_i2, 0)
    phi_i2 = phi_i2 * np.pi / phi_i2[-1]
    phi_i2 = phi_i2 - phi_i2[0]
    theta_list = torch.cat((phi_i2, torch.flip(2 * np.pi - phi_i2, (0,))))

    torch.set_printoptions(precision = 8)
    print("Theta: ", theta_list / np.pi)

    
    H_j1 = ham_j1(L, J1 = J1)
    H_mag = ham_mag(L, B0, B_ext, theta_list)

    H = add_tensor(H_j1, H_mag)

    #print(H.to_dense())

    neigs = 3

    from xitorch import linalg
    from test_sparse_eigen import CsrLinOp

    H_linop = CsrLinOp(torch.stack([H.storage._row, H.storage._col], dim=0), H.storage._value, H.size(0))

    eigvals, eigvecs = linalg.symeig(H_linop, neig=neigs, method="davidson", max_niter=1000, nguess=None,
                                     v_init="randn",
                                     max_addition=None, min_eps=1e-07, verbose=False,
                                     bck_options={'method': 'bicgstab', 'rtol': 1e-05, 'atol': 1e-06, 'eps': 1e-8,
                                                  'verbose': False, 'max_niter': 10000})
    from bipartite_entropy import calculate_entropies

    import time

    start_time = time.time()
    print("Energies: ", eigvals)
    for i_eig in range(neigs):
        print("Entanglement for state " + str(i_eig) +" is:", calculate_entropies(eigvecs[:, i_eig], L, [2]*L))
        print("Maximum entanglement for state " + str(i_eig) +" is:", calculate_entropies(eigvecs[:, i_eig], L, [2]*L)[L//2 - 1])
    print("Time for entanglement calculation:", time.time() - start_time)
