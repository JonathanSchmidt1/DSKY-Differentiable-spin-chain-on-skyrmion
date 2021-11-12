import torch, xitorch, numpy as np, torch_sparse as ts
import scipy
mat = np.load('matrix.npy', allow_pickle=True)[()]
import torch
from torch_sparse import coalesce


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
    return coalesce(index=index, value=value, m=m, n=n, op='add')


rowptr = torch.from_numpy(mat.indptr).to(torch.long).cuda()
mat = mat.tocoo()

row = torch.from_numpy(mat.row).to(torch.long).cuda()
col = torch.from_numpy(mat.col).to(torch.long).cuda()
index = torch.stack([row, col],dim=0)
value = torch.from_numpy(mat.data).cuda().requires_grad_().double()
sparse_sizes = mat.shape[:2]
storage = ts.SparseStorage(row=row, rowptr=rowptr, col=col, value=value,
                            sparse_sizes=sparse_sizes)
sparset = ts.SparseTensor.from_storage(storage)


class CsrLinOp(xitorch.LinearOperator):
    def __init__(self,index, value ,size):
        super().__init__(shape=(size,size), is_hermitian=True, device=value.device, dtype=value.dtype)
        self.v = value
        self.index = index
        self.size = size
    def _mv(self, x):
        if len(x.shape)==1:
            return ts.spmm(self.index,self.v, self.size, self.size , x.reshape(-1,1)).squeeze()
        else:
            return ts.spmm(self.index,self.v, self.size, self.size , x)
    def _mm(self, x):
        return ts.spmm(self.index,self.v, self.size, self.size , x)
    def _getparamnames(self, prefix=""):
        return [prefix+"v"]

linop = CsrLinOp(index, value, mat.shape[0])
from time import time
from xitorch import linalg


new_value = value.clone()
m,n = mat.shape[0], mat.shape[0]
for i in range(100):
    linop = CsrLinOp(index, new_value, mat.shape[0])
    eigval, eigvec = linalg.symeig(linop,neig=3, method="davidson", max_niter=1000, nguess=None, v_init="randn",
                                    max_addition=None, min_eps=1e-05, verbose=True, bck_options={'method':
                                    'bicgstab','rtol':1e-05, 'atol':1e-06, 'eps':1e-8,'verbose':False, 'max_niter':10000})
    #careful with the convergence criteria
    test = torch.sum(torch.matmul(eigvec[:,0], torch.arange(mat.shape[0]).double().cuda())/65000.0)
    #random loss function
    print("result", test)
    new_value =new_value - 0.2*torch.autograd.grad(test, new_value)[0]
    #basic gradient update
    index2, value2 = ts.transpose(index, new_value, m, n)
    index, new_value = spadd(index, new_value, index2, value2, m,n)
    new_value=new_value/2
    #make it hermitic again
    index2, value2 = ts.transpose(index, new_value, m, n)
    print('checking hermiticity', torch.sum(spadd(index, new_value, index2, -value2, m,n)[1]))
    
