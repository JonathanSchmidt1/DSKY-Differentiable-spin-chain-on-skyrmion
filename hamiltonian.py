import sys
sys.path.append("/nfs/home/mamelz/PythonModules")

import numpy
import scipy
import torch
import torch_sparse

pauli32 = [
           torch_sparse.SparseTensor(row = torch.tensor([0,1]), col = torch.tensor([1,0]), value = torch.tensor([1,1], dtype = torch.float32), sparse_sizes=(2,2)),
           torch_sparse.SparseTensor(row = torch.tensor([0,1]), col = torch.tensor([1,0]), value = torch.tensor([-1j,1j], dtype = torch.complex64), sparse_sizes=(2,2)),
           torch_sparse.SparseTensor(row = torch.tensor([0,1]), col = torch.tensor([0,1]), value = torch.tensor([1,-1], dtype = torch.float32), sparse_sizes=(2,2))
           ]

pauli64 = [
           torch_sparse.SparseTensor(row = torch.tensor([0,1]), col = torch.tensor([1,0]), value = torch.tensor([1,1], dtype = torch.float64), sparse_sizes=(2,2)),
           torch_sparse.SparseTensor(row = torch.tensor([0,1]), col = torch.tensor([1,0]), value = torch.tensor([-1,1], dtype = torch.float64), sparse_sizes=(2,2)),
           torch_sparse.SparseTensor(row = torch.tensor([0,1]), col = torch.tensor([0,1]), value = torch.tensor([1,-1], dtype = torch.float64), sparse_sizes=(2,2))
           ]

ladder32 = [
            torch_sparse.SparseTensor(row = torch.tensor([0]), col = torch.tensor([1]), value = torch.tensor([1], dtype = torch.float32), sparse_sizes=(2,2)),
            torch_sparse.SparseTensor(row = torch.tensor([1]), col = torch.tensor([0]), value = torch.tensor([1], dtype = torch.float32), sparse_sizes=(2,2)),
            ]

ladder64 = [
            torch_sparse.SparseTensor(row = torch.tensor([0]), col = torch.tensor([1]), value = torch.tensor([1], dtype = torch.float64), sparse_sizes=(2,2)),
            torch_sparse.SparseTensor(row = torch.tensor([1]), col = torch.tensor([0]), value = torch.tensor([1], dtype = torch.float64), sparse_sizes=(2,2)),
            ]

eye32 = torch_sparse.SparseTensor(row = torch.tensor([0,1]), col = torch.tensor([0,1]), value = torch.tensor([1,1], dtype = torch.float32), sparse_sizes = (2,2))
eye64 = torch_sparse.SparseTensor(row = torch.tensor([0,1]), col = torch.tensor([0,1]), value = torch.tensor([1,1], dtype = torch.float64), sparse_sizes = (2,2))

#kronecker product of two torch_sparse.SparseTensor objects
def sparse_kron(A,B):

    if A.nnz == 0 or B.nnz == 0:

        return torch_sparse.SparseTensor.from_dense(torch.zeros(A.sparse_sizes[0] * B.sparse_sizes[0], A.sparse_sizes[1] * B.sparse_sizes[1]))

    row = A.storage.row().repeat_interleave(B.nnz())
    col = A.storage.col().repeat_interleave(B.nnz())
    data = A.storage.value().repeat_interleave(B.nnz())

    row *= B.sparse_sizes()[0]
    col *= B.sparse_sizes()[1]

    row,col = row.reshape((-1,B.nnz())),col.reshape((-1,B.nnz()))
    row += B.storage.row()
    col += B.storage.col()
    row,col = row.reshape((-1)),col.reshape((-1))

    data = data.reshape((-1,B.nnz())) * B.storage.value()
    data = data.reshape((-1))

    return torch_sparse.SparseTensor(row = row, col = col, value = data)


#tensoring A of shape(2n,2n) sitting at sites=[i, ..., i+n] into larger hilbertspace by padding with 2x2-identities from left and right
def sparse_ikron(A, L, sites, prec = 64):

    if prec == 64:
        eye = eye64
    
    elif prec == 32:
        eye = eye32
    
    else:
        raise ValueError("prec must be either 32 or 64")

    #check for square matrix with dimension of power 2
    assert (A.sparse_sizes()[0] == A.sparse_sizes()[1] and (A.sparse_sizes()[0] & (A.sparse_sizes()[0]-1) == 0) and A.sparse_sizes()[0] != 0)

    if sites[0] > 0:
        res = eye
    else:
        res = A

        for i in range(L-sites[-1]-1):

            res = sparse_kron(res, eye)
        
        return res

    for k in range(sites[0]-1):
        res = sparse_kron(res, eye)
    
    res = sparse_kron(res,A)

    for k in range(L-sites[-1]-1):
        res = sparse_kron(res, eye)

    return res


def tensor_sum(it):

    for i,tensor in enumerate(it):

        if i == 0:

            res = tensor
        
        else:

            res = torch_sparse.add(res,tensor)
    
    return res


#product with scalar
def tensor_elem_mul(A, c, ret = False):

    A.storage._value = torch.mul(A.storage._value, c)

    if ret:

        return A

    return


def ham_j1(L, J1 = 1.0, prec = 64):

    if prec ==64:

        pauli = pauli64
        ladder = ladder64
    
    elif prec == 32:

        pauli = pauli32
        ladder = ladder32
    
    else:
        raise ValueError("prec must be either 32 or 64")

    summands = [sparse_kron(ladder[0],ladder[1]), sparse_kron(ladder[1],ladder[0]), sparse_kron(pauli[2],pauli[2])]

    for S in summands[:2]: tensor_elem_mul(S,2)

    def j1_terms():

        for i in range(L-1):
            coo = [i,i+1]

            yield sparse_ikron(tensor_sum(summands), L, coo)

    ham = tensor_elem_mul(tensor_sum(j1_terms()), J1/4, ret = True)

    return ham


H = ham_j1(16).storage._value.requires_grad_()

print(H.coo)
print(torch.linalg.eigh(H.to_dense())[0])

            


