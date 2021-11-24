import h5py
import matplotlib.pyplot as plt

folder = "output/"
file = h5py.File(folder + "test_output.h5", "r", libver = 'latest', swmr = True)

plt.plot(file["B_0"][()])
plt.savefig(folder + "B_0.png")
plt.close()

plt.plot(file["B_ext"][()])
plt.savefig(folder + "B_ext.png")
plt.close()

plt.plot(file["phi_i"][()])
plt.savefig(folder + "phi_i.png")
plt.close()

plt.plot(file["entropy"][()])
plt.savefig(folder + "entropy.png")
plt.close()

plt.plot(file["loss"][()])
plt.savefig(folder + "loss.png")
plt.close()
