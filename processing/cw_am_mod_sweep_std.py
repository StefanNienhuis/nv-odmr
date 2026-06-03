import numpy as np
import matplotlib.pyplot as plt
from util.sensitivity import calculate_sensitivity

results = np.load('../data/cw_am_mod_sweep_std/2026-06-03T14.49.18.962079.npz', allow_pickle=True)
counts_per_modulation_freq = results['data']

params = results['params'].item()

modulation_freqs = params['modulation_freqs']
slopes_per_freq = params['slopes']
meas_time = params['meas_time']

gamma_nv = 28.0e9

sensitivities_per_modulation_freq = []

for freq, slope, counts in zip(modulation_freqs, slopes_per_freq, counts_per_modulation_freq):
    sensitivity = calculate_sensitivity(counts, slope, meas_time)
    print(f"{freq}:\t\t{sensitivity*1e6} uT/sqrt(Hz)")
    sensitivities_per_modulation_freq.append(sensitivity)

print(np.std(counts_per_modulation_freq[3]))
print(slopes_per_freq[3])
plt.plot(counts_per_modulation_freq[3] - np.mean(counts_per_modulation_freq[3]))
plt.show()

plt.semilogx(modulation_freqs, np.array(sensitivities_per_modulation_freq) * 1e6)
plt.xlabel('Modulation frequency (Hz)')
plt.ylabel('Sensitivity ($\\mu T/\\sqrt{Hz}$)')
plt.legend()
plt.show()
