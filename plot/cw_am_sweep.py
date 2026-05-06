import numpy as np
import matplotlib.pyplot as plt

counts = np.load('/home/dl-lab-pc3/QB2025/nv-odmr/persist/cw_am_sweep/2026-05-06T12:22:05.931585.npy')

active_counts = counts[:,:,0]
inactive_counts = counts[:,:,1]

mean_active_counts = np.mean(active_counts, axis=1)
mean_inactive_counts = np.mean(inactive_counts, axis=1)

am_counts = (mean_inactive_counts - mean_active_counts) / mean_inactive_counts

plt.plot(mean_active_counts, label='on')
plt.plot(mean_inactive_counts, label='off')
plt.legend()
plt.show()


plt.plot(am_counts, label='on')
plt.show()