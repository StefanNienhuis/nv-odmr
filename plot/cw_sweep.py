import numpy as np
import matplotlib.pyplot as plt

counts = np.load('/home/dl-lab-pc3/QB2025/nv-odmr/persist/cw_sweep/2026-05-06T12:13:59.362988.npy')

# counts = counts.sum(axis=1)

print(counts[0:5])
counts = counts / np.max(counts)


plt.plot(counts)
plt.show()