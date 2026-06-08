import numpy as np
import matplotlib.pyplot as plt
from util.double_lorentzian_diff_fit import fit_curve

center_freq = 2.87e9

results = np.load('../data/cw_fm_bin_sweep/2026-06-05T10.59.42.656244.npz', allow_pickle=True)
counts = results['data']
params = results['params'].item()

start_freq      = params['start_freq']
stop_freq       = params['stop_freq']
n_sweep         = params['n_sweep']
mod_depth       = params['mod_depth']

freq = np.linspace(start_freq, stop_freq, n_sweep)
detuning = freq - center_freq

curve, params = fit_curve(detuning, counts, mod_depth)

fit = curve(detuning, *params)

slope = np.gradient(fit, detuning)
max_slope = np.max(np.abs(slope))
max_slope_detuning = detuning[np.argmax(np.abs(slope))]

print(f"Max slope at {max_slope_detuning/1e6} MHz")

plt.plot(detuning / 1e6, counts, 'x', color='Gray', label='Data')
plt.plot(detuning / 1e6, fit, color='red', label='Fit')
plt.axvline(x=max_slope_detuning / 1e6, linestyle='--', label='Max Slope')
plt.xlabel('Detuning [MHz]')
plt.xlim((start_freq - center_freq) / 1e6, (stop_freq - center_freq) / 1e6)
plt.legend()
plt.show()