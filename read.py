import h5py
import matplotlib.pyplot as plt
import numpy as np

folder = "output/"
file = h5py.File(folder + "test_output.h5", "r", libver = 'latest', swmr = True)

loss = np.array(file["loss"][()])
min_ind = np.argmin(loss[loss != 0])
min_params = ["B_0", "B_ext", "entropy"]

for name in min_params:
    print("Minimum of " + name + ":", file[name][()][min_ind])

plt.plot(file["B_0"][()])
plt.savefig(folder + "B_0.png")
plt.close()

plt.plot(file["B_ext"][()])
plt.savefig(folder + "B_ext.png")
plt.close()

phi_i = file["phi_i"][()][min_ind]
phi_i2 = np.square(phi_i)
phi_i2 = np.cumsum(phi_i2, 0)
phi_i2 = phi_i2 * np.pi / phi_i2[-1]
phi_i2 = phi_i2 - phi_i2[0]
phi_i2 = np.concatenate((phi_i2, np.flip(2 * np.pi - phi_i2, (0,))))

plt.plot(np.cos(phi_i2))
plt.plot(np.sin(phi_i2))
plt.savefig(folder + "phi_i.png")
plt.close()

plt.plot(file["entropy"][()])
plt.savefig(folder + "entropy.png")
plt.close()

plt.plot(file["loss"][()])
plt.savefig(folder + "loss.png")
plt.close()
