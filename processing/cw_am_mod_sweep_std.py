import numpy as np
import matplotlib.pyplot as plt

results = np.load('../persist/cw_am_mod_sweep_std/2026-05-12T15.34.26.334173.npz', allow_pickle=True)
counts_per_modulation_freq = results['data']

params = results['params'].item()

modulation_freqs = params['modulation_freqs']
slopes_per_freq = params['slopes']
meas_time = params['meas_time']

gamma_nv = 28.0e9

sensitivities_per_modulation_freq = []

for freq, slope, counts in zip(modulation_freqs, slopes_per_freq, counts_per_modulation_freq):
    S = counts_per_modulation_freq
    
    std_S = np.std(S, ddof=1)
    
    std_f = std_S / slope
    std_B = std_f / gamma_nv

    sensitivity = std_B * np.sqrt(meas_time)
    print(f"{freq}:\t\t{sensitivity*1e6} uT/sqrt(Hz)")
    sensitivities_per_modulation_freq.append(sensitivity)

plt.semilogx(modulation_freqs[:-5], np.array(sensitivities_per_modulation_freq)[:-5] *1e6)
plt.xlabel('Modulation frequency (Hz)')
plt.ylabel('Sensitivity ($\\mu T/\\sqrt{Hz}$)')
plt.legend()
plt.show()
