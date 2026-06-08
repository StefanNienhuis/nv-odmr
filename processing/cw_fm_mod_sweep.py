import numpy as np
import matplotlib.pyplot as plt

from util.double_lorentzian_diff_fit import fit_curve

center_freq = 2.8e9

results = np.load('../persist/cw_fm_bin_mod_sweep/2026-06-08T13.49.04.621379.npz', allow_pickle=True)
fm_counts_per_modulation_freq = results['data']

params = results['params'].item()

modulation_freqs = params['modulation_freqs']
mod_depth = params['mod_depth']

freqs = np.linspace(params['start_freq'], params['stop_freq'], params['n_sweep'])
detuning = freqs - center_freq

max_slopes = []
max_slope_freqs = []

for i, (modulation_freq, fm_counts) in enumerate(list(zip(modulation_freqs, fm_counts_per_modulation_freq))):
    curve, params = fit_curve(detuning, fm_counts, mod_depth, dip=False)
    
    fit = curve(detuning, *params)
    max_peak = np.max(fit)
    max_peak_detuning = detuning[np.argmax(fit)]

    slope = np.gradient(fit, detuning)
    max_slope = np.max(np.abs(slope))
    max_slope_detuning = detuning[np.argmax(np.abs(slope))]

    max_slopes.append(max_slope)
    max_slope_freqs.append(center_freq + max_slope_detuning)

    print(f"Max peak at {max_peak_detuning/1e6} MHz ({(center_freq + max_peak_detuning)/1e9} GHz)")
    print(f"Max slope at {max_slope_detuning/1e6} MHz ({(center_freq + max_slope_detuning)/1e9} GHz)")

    plt.subplot(2, 5, i+1)
    plt.title(f'{modulation_freq} Hz modulation')
    plt.plot(detuning / 1e6, fm_counts, 'x', color='gray', label='Data')
    plt.plot(detuning / 1e6, fit, '--', color='red', label='Fit')
    # plt.axvline(x=max_peak_detuning / 1e6, linestyle='--', label='Max Peak')
    # plt.axvline(x=max_slope_detuning / 1e6, linestyle='--', label='Max Slope')
    plt.tick_params('x', labelbottom=(i >= 15))
    # plt.ylim(-0.04, 0.14)
    # plt.xlabel('Detuning [MHz]')

print()
print(f"modulation_freqs = {list(modulation_freqs)}")
print(f"drive_freqs = {max_slope_freqs}")
print(f"slopes = {max_slopes}")

plt.show()