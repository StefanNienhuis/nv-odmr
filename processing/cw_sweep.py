import numpy as np
import matplotlib.pyplot as plt
from util.double_lorentzian_fit import fit_curve

center_freq = 2.87e9

results = np.load('../data/cw_sweep/2026-06-03T15.56.51.126233.npz', allow_pickle=True)

counts = results['data']
normalizer = np.max(counts)
counts = counts / np.max(counts)

params = results['params'].item()

start_freq      = params['start_freq']
stop_freq       = params['stop_freq']
n_sweep         = params['n_sweep']

freqs = np.linspace(start_freq, stop_freq, n_sweep)
detuning = freqs - center_freq

curve, params = fit_curve(detuning, counts)
fit = curve(detuning, *params)

[b0, b1, A1, A2, x1, x2, gamma1, gamma2] = params
print("A:", A1, A2)
print("gamma: ", gamma1, gamma2)

max_dip = np.min(fit)
max_dip_detuning = detuning[np.argmin(np.abs(fit))]

slope = np.gradient(fit, detuning)
max_slope = np.max(np.abs(slope))
max_slope_detuning = detuning[np.argmax(np.abs(slope))]

print(f"Max dip at {max_dip_detuning/1e9} MHz")
print(f"Max slope at {max_slope_detuning/1e9} MHz")

print()
print(f"drive_freq = {center_freq + max_slope_detuning}")
print(f"slope = {max_slope}")
print(f"normalizer = {normalizer}")

plt.plot(freqs, counts)
plt.show()

plt.plot(detuning / 1e6, counts, 'x', color='gray', label='Data')
plt.plot(detuning / 1e6, fit, color='red', label='Fit')
plt.axvline(x=max_dip_detuning / 1e6, linestyle='--', label='Max Dip')
plt.axvline(x=max_slope_detuning / 1e6, color='orange', linestyle='--', label='Max Slope')
plt.xlabel('Detuning [MHz]')
plt.legend()
plt.show()