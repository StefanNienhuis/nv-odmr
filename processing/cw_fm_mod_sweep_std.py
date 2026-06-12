import numpy as np
import matplotlib.pyplot as plt
from util.sensitivity import calculate_sensitivity
import util.plot_style

files = {
    '../persist/cw_fm_mod_sweep_std/2026-06-03T18.36.15.131925.npz': 'Simplified lock-in',
    '../persist/cw_fm_bin_mod_sweep_std/2026-06-08T14.52.55.391741.npz': 'Phase-sensitive'
}

for file, label in files.items():
    results = np.load(file, allow_pickle=True)
    counts_per_modulation_freq = results['data']

    params = results['params'].item()

    modulation_freqs = params['modulation_freqs']
    slopes_per_freq = params['slopes']
    meas_time = params['meas_time']

    gamma_nv = 28.0e9

    sensitivities_per_modulation_freq = []

    for freq, slope, counts in zip(modulation_freqs, slopes_per_freq, counts_per_modulation_freq):
        sensitivity = calculate_sensitivity(np.abs(counts), slope, meas_time)
        print(f"{freq:5.0f}:\t{sensitivity*1e6} uT/sqrt(Hz)")
        sensitivities_per_modulation_freq.append(sensitivity)

    plt.semilogx(modulation_freqs, np.array(sensitivities_per_modulation_freq) * 1e6, label=label)

plt.suptitle('Sensitivity vs FM modulation frequency')
plt.xlabel('Modulation frequency (Hz)')
plt.ylabel('Sensitivity ($\\mu\\text{T}/\\sqrt{\\text{Hz}}$)')
plt.axhline(7.613385554967387, color='C3', linestyle='--', label='CW sensitivity')

plt.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=3,
           handlelength=1.6, columnspacing=1.1, borderaxespad=0.0)
plt.show()
