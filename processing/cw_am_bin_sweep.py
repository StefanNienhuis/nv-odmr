import numpy as np
import matplotlib.pyplot as plt
from util.double_lorentzian_fit import fit_curve

center_freq = 2.87e9

results = np.load('../data/cw_am_bin_sweep/2026-06-01T15.26.49.236401.npz', allow_pickle=True)
counts = results['data']
params = results['params'].item()

start_freq      = params['start_freq']
stop_freq       = params['stop_freq']
n_sweep         = params['n_sweep']

freqs = np.linspace(start_freq, stop_freq, n_sweep)
detuning = freqs - center_freq

curve, params = fit_curve(detuning, counts, dip=False)
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

plt.plot(detuning / 1e6, counts, 'x', color='gray', label='Data')
plt.plot(detuning / 1e6, fit, color='red', label='Fit')
plt.axvline(x=max_peak_detuning / 1e6, linestyle='--', label='Max Peak')
plt.axvline(x=max_slope_detuning / 1e6, linestyle='--', label='Max Slope')
plt.xlabel('Detuning [MHz]')
plt.legend()
plt.show()