import numpy as np
import matplotlib.pyplot as plt
from util.double_lorentzian_fit import fit_curve

center_freq = 2.87e9

start_freq      = 2.84e9   # Sweep start frequency (Hz)
n_sweep         = 401      # Number of sweep steps
stop_freq       = 2.90e9   # Sweep stop frequency (Hz)
freq_dev = 1.5e6

freqs = np.linspace(start_freq, stop_freq, n_sweep)
detuning = freqs - center_freq

counts = np.load('../persist/cw_fm_sweep/2026-05-06T12.51.18.508017.npy')

# TODO: current data very slanted in negative detuning - remove range limit with proper data
detuning = detuning[120:-120]
counts = counts[120:-120]

low_counts = counts[:,:,0]
high_counts = counts[:,:,1]

mean_low_counts = np.mean(low_counts, axis=1)
mean_high_counts = np.mean(high_counts, axis=1)

fm_counts = (mean_high_counts - mean_low_counts) / (mean_high_counts + mean_low_counts)

low_detuning = detuning - freq_dev
high_detuning = detuning + freq_dev

fit_detuning = np.concatenate((low_detuning, high_detuning))
fit_counts = np.concatenate((mean_low_counts, mean_high_counts))

curve, params = fit_curve(fit_detuning, fit_counts)

low_fit = curve(low_detuning, *params)
high_fit = curve(high_detuning, *params)

fit = (high_fit - low_fit) / (high_fit + low_fit)

slope = np.gradient(fit, detuning)
max_slope = np.max(np.abs(slope))
max_slope_detuning = detuning[np.argmax(np.abs(slope))]

print(f"Max slope at {max_slope_detuning/1e6} MHz")

plt.plot(detuning / 1e6, fm_counts, 'x', color='Gray', label='Data')
plt.plot(detuning / 1e6, fit, color='red', label='Fit')
plt.axvline(x=max_slope_detuning / 1e6, linestyle='--', label='Max Slope')
plt.xlabel('Detuning [MHz]')
plt.xlim((start_freq - center_freq) / 1e6, (stop_freq - center_freq) / 1e6)
plt.legend()
plt.show()