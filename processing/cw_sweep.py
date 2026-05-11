import numpy as np
import matplotlib.pyplot as plt
from util.double_lorentzian_fit import fit_curve

center_freq = 2.87e9

start_freq      = 2.84e9
stop_freq       = 2.90e9
n_sweep         = 101

freqs = np.linspace(start_freq, stop_freq, n_sweep)
detuning = freqs - center_freq
results = np.load('../data/cw_sweep/2026-05-11T12.01.16.725109.npy.npz')

counts = results['data']
counts = counts / np.max(counts)

curve, params = fit_curve(detuning, counts)
fit = curve(detuning, *params)

max_dip = np.min(fit)
max_dip_detuning = detuning[np.argmin(np.abs(fit))]

slope = np.gradient(fit, detuning)
max_slope = np.max(np.abs(slope))
max_slope_detuning = detuning[np.argmax(np.abs(slope))]

print(f"Max dip at {max_dip_detuning/1e9} MHz")
print(f"Max slope at {max_slope_detuning/1e9} MHz")

plt.plot(detuning / 1e6, counts, 'x', color='gray', label='Data')
plt.plot(detuning / 1e6, fit, color='red', label='Fit')
plt.axvline(x=max_dip_detuning / 1e6, linestyle='--', label='Max Dip')
plt.axvline(x=max_slope_detuning / 1e6, color='orange', linestyle='--', label='Max Slope')
plt.xlabel('Detuning [MHz]')
plt.legend()
plt.show()