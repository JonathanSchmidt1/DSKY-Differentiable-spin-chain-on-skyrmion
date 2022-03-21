import h5py
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def read(filename, folder = "output/", plot = True, min_params = ["B_0", "B_ext", "entropy"]):
    if not folder.endswith('/'):
        folder = folder + '/'
    path = Path(folder + filename)
    if path.is_file():
        file = h5py.File(path, "r", libver = 'latest', swmr = True)
    else:
        raise ValueError("There is no file called {} in the folder {}.".format(filename, folder))

    loss = np.array(file["loss"][()])
    min_ind = np.argmin(loss[loss != 0])

    for name in min_params:
        try:
            print("Minimum of " + name + ":", file[name][()][min_ind])
        except:
            raise ValueError("There is no parameter called {}.".format(name))

    phi_i = file["phi_i"][()][min_ind]

    if plot:

        for param in min_params:

            plt.plot(file[param][()])
            plt.title(param)
            plt.savefig(folder + param + ".png")
        
        plt.plot(file["loss"][()])
        plt.savefig(folder + "loss.png")
        plt.close()

        plt.plot(np.cos(phi_i))
        plt.plot(np.sin(phi_i))
        plt.title("Polar angle for lowest loss function")
        plt.savefig(folder + "phi_i.png")
        plt.close()

read("test_output.h5", min_params = ["B_0", "B_ext", "scalfac", "delta", "entropy"])#, folder="output_dwall2/")
#read("test_output.h5", folder="output_skyrmion2/")