import numpy as np
import matplotlib.pyplot as plt

start_freq      = 2.84e9   # Sweep start frequency (Hz)
n_sweep         = 401      # Number of sweep steps
stop_freq       = 2.90e9   # Sweep stop frequency (Hz)

freq = np.linspace(start_freq, stop_freq, n_sweep)

counts = np.load('/home/dl-lab-pc3/QB2025/nv-odmr/persist/cw_fm_sweep/2026-05-06T12:51:18.508017.npy')

low_counts = counts[:,:,0]
high_counts = counts[:,:,1]

mean_low_counts = np.mean(low_counts, axis=1)
mean_high_counts = np.mean(high_counts, axis=1)

fm_counts = (mean_high_counts - mean_low_counts) / (mean_high_counts + mean_low_counts)

plt.plot(freq, mean_low_counts)
plt.plot(freq, mean_high_counts)
plt.show()

plt.plot(freq, fm_counts)
plt.show()