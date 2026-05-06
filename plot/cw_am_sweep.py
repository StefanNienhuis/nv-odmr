import numpy as np
import matplotlib.pyplot as plt
from util.double_lorentzian_fit import curve, fit_curve

start_freq      = 2.84e9
stop_freq       = 2.90e9
n_sweep         = 401

freqs = np.linspace(start_freq, stop_freq, n_sweep)
counts = np.load('../persist/cw_am_sweep/2026-05-06T12.22.05.931585.npy')

active_counts = counts[:,:,0]
inactive_counts = counts[:,:,1]

mean_active_counts = np.mean(active_counts, axis=1)
mean_inactive_counts = np.mean(inactive_counts, axis=1)

am_counts = (mean_inactive_counts - mean_active_counts) / mean_inactive_counts

params = fit_curve(freqs, am_counts, dip=False)
fit = curve(freqs, *params)

slope = np.gradient(fit, freqs)
max_slope = np.max(np.abs(slope))
max_slope_freq = freqs[np.argmax(np.abs(slope))]

# Plot individual components
# plt.plot(mean_active_counts, label='on')
# plt.plot(mean_inactive_counts, label='off')
# plt.legend()
# plt.show()

print(f"Max slope at {max_slope_freq/1e9} GHz")

plt.plot(freqs, am_counts, 'x', color='gray', label='Data')
plt.plot(freqs, fit, color='red', label='Fit')
plt.axvline(x=max_slope_freq, linestyle='--', label='Max Slope')
plt.legend()
plt.show()