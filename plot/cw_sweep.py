import numpy as np
import matplotlib.pyplot as plt
from util.double_lorentzian_fit import fit_curve

center_freq = 2.87e9

start_freq      = 2.84e9
stop_freq       = 2.90e9
n_sweep         = 401

freqs = np.linspace(start_freq, stop_freq, n_sweep)
detuning = freqs - center_freq
counts = np.load('../persist/cw_sweep/2026-05-06T12.13.59.362988.npy')

counts = counts / np.max(counts)

curve, params = fit_curve(detuning, counts)
fit = curve(detuning, *params)

slope = np.gradient(fit, detuning)
max_slope = np.max(np.abs(slope))
max_slope_detuning = detuning[np.argmax(np.abs(slope))]

print(f"Max slope at {max_slope_detuning/1e9} MHz")

plt.plot(detuning / 1e6, counts, 'x', color='gray', label='Data')
plt.plot(detuning / 1e6, fit, color='red', label='Fit')
plt.axvline(x=max_slope_detuning / 1e6, linestyle='--', label='Max Slope')
plt.xlabel('Detuning [MHz]')
plt.legend()
plt.show()