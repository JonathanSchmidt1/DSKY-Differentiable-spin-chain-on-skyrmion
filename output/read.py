import h5py
import matplotlib.pyplot as plt

file = h5py.File("test_output.h5", "r")

plt.plot(file["entropy"][()])
plt.show()
plt.close()
