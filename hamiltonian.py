import numpy
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
        return torch_sparse.SparseTensor.from_dense(
            torch.zeros(A.sparse_sizes[0] * B.sparse_sizes[0], A.sparse_sizes[1] * B.sparse_sizes[1]))

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
    return torch_sparse.SparseTensor(row=row, col=col, value=data, sparse_sizes = (A.sparse_sizes()[0]*B.sparse_sizes()[0], A.sparse_sizes()[1]*B.sparse_sizes()[1]))


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

            res = tensor

        else:

            res = add_tensor(res, tensor)
    return res


# product with scalar
def tensor_elem_mul(A, c, ret=False):
    A.storage._value = torch.mul(A.storage._value, c)

    if ret:
        return A

    return


def ham_j1(L, J1=1.0, prec=64):
    if prec == 64:

        pauli = pauli64
        ladder = ladder64

    elif prec == 32:

        pauli = pauli32
        ladder = ladder32

    else:
        raise ValueError("prec must be either 32 or 64")

    summands = [sparse_kron(ladder[0], ladder[1]), sparse_kron(ladder[1], ladder[0]), sparse_kron(pauli[2], pauli[2])]

    for S in summands[:2]: tensor_elem_mul(S, 2)

    def j1_terms():

        for i in range(L - 1):
            coo = [i, i + 1]

            yield sparse_ikron(tensor_sum(summands), L, coo)

    ham = tensor_elem_mul(tensor_sum(j1_terms()), J1 / 4, ret=True)

    return ham


if __name__ == '__main__':
    L = 16
    neigs = 3
    H = ham_j1(L)

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
    for i_eig in range(neigs):
        print("Entanglement for state " + str(i_eig) + " is:", calculate_entropies(eigvecs[:, i_eig], L, [2] * L))
    print("Time for entanglement calculation:", time.time() - start_time)