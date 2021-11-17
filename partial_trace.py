import torch
import numpy as np

def partial_trace(vec, dims, keep):
    if isinstance(keep, int):
        keep = (keep,)
    vec  = torch.reshape(vec, dims)
    lose = tuple(i for i in range(len(dims)) if i not in keep)
    vec = torch.tensordot(vec, torch.conj(vec), dims = (lose, lose))
    d = int(torch.numel(vec)**0.5)
    return torch.reshape(vec, (d, d))

if __name__ == "__main__":
    a = torch.tensor([0 , 1 / np.sqrt(2), 1 / np.sqrt(2), 0])
    print(partial_trace(a, [2, 2], keep = 0))





