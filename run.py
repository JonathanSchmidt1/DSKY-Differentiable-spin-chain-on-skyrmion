from bipartite_entropy import calculate_entropies
from hamiltonian import Sky_phi
from pathlib import Path
from nnmodule import HamModule_param, HamModule_phi
import h5py
import time
import torch
import numpy as np


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
#H = HamModule_phi(L, J1, B_0, B_ext, phi_diff, device='cuda')

H = HamModule_param(L, J1, B_0, B_ext, scalfac, delta)

n_eigs = 3
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