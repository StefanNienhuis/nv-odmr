import numpy as np
import matplotlib.pyplot as plt
from util.double_lorentzian_fit import fit_curve

center_freq = 2.87e9

start_freq      = 2.84e9
stop_freq       = 2.90e9
n_sweep         = 201

freqs = np.linspace(start_freq, stop_freq, n_sweep)
detuning = freqs - center_freq

results = np.load('../data/cw_am_sweep/2026-05-12T11.06.53.528274.npz')
counts = results['data']

active_counts = counts[:,:,0]
inactive_counts = counts[:,:,1]

# mean_active_counts = np.mean(active_counts, axis=1)
# mean_inactive_counts = np.mean(inactive_counts, axis=1)

# am_counts = (mean_inactive_counts - mean_active_counts) / mean_inactive_counts

am_counts = np.mean((inactive_counts - active_counts) / inactive_counts, axis=1)

curve, params = fit_curve(detuning, am_counts, dip=False)
fit = curve(detuning, *params)

max_peak = np.max(fit)
max_peak_detuning = detuning[np.argmax(np.abs(fit))]

slope = np.gradient(fit, detuning)
max_slope = np.max(np.abs(slope))
max_slope_detuning = detuning[np.argmax(np.abs(slope))]

# Plot individual components
# plt.plot(mean_active_counts, label='on')
# plt.plot(mean_inactive_counts, label='off')
# plt.legend()
# plt.show()

print(f"Max peak at {max_peak_detuning/1e6} MHz ({(center_freq + max_peak_detuning)/1e9} GHz)")
print(f"Max slope at {max_slope_detuning/1e6} MHz ({(center_freq + max_slope_detuning)/1e9} GHz)")

plt.plot(detuning / 1e6, am_counts, 'x', color='gray', label='Data')
plt.plot(detuning / 1e6, fit, color='red', label='Fit')
plt.axvline(x=max_peak_detuning / 1e6, linestyle='--', label='Max Peak')
plt.axvline(x=max_slope_detuning / 1e6, linestyle='--', label='Max Slope')
plt.xlabel('Detuning [MHz]')
plt.legend()
plt.show()