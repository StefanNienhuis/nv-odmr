import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

from util.double_lorentzian_diff_fit import fit_curve

center_freq = 2.8e9

results = np.load('../data/cw_fm_mod_sweep/2026-06-03T17.19.41.042999.npz', allow_pickle=True)
fm_counts_per_modulation_freq = results['data']

params = results['params'].item()

modulation_freqs = params['modulation_freqs']
mod_depth = params['mod_depth']

freqs = np.linspace(params['start_freq'], params['stop_freq'], params['n_sweep'])
detuning = freqs - center_freq

max_slopes = []
max_slope_freqs = []
#
# # EXPIREMENTAL
#
# def cw_curve(x, b0, b1, A1, A2, x1, x2, gamma1, gamma2):
#     return b0 + b1 * x - A1 / (1 + ((x-x1)/gamma1) ** 2) - A2 / (1 + ((x-x2)/ gamma2) ** 2)
#
# def curve(x, b0, b1, A1, A2, x1, x2, gamma1, gamma2):
#     mod_depth = 3e6
#
#     high = cw_curve(x + mod_depth / 2, b0, b1, A1, A2, x1, x2, gamma1, gamma2)
#     low = cw_curve(x - mod_depth / 2, b0, b1, A1, A2, x1, x2, gamma1, gamma2)
#
#     return (high - low) / (high + low)
#
# def fit_curve(freqs, fluorescence, dip=True):
#     b0_guess = 1
#     b1_guess = 0.0
#
#     A1_guess = np.max(fluorescence) - np.min(fluorescence)
#     A2_guess = A1_guess
#
#     # Guess that the dips are symmetric around the center
#     x_index_guess = np.argmin(fluorescence) if dip else np.argmax(fluorescence)
#     x_midpoint = len(freqs) // 2
#
#     if x_index_guess <= x_midpoint:
#         x1_guess = freqs[x_index_guess]
#         x2_guess = freqs[x_midpoint + (x_midpoint - x_index_guess)]
#     else:
#         x1_guess = freqs[x_midpoint - (x_index_guess - x_midpoint)]
#         x2_guess = freqs[x_index_guess]
#
#     # Estimate linewidth of 1/50 of the range
#     gamma1_guess = (freqs.max() - freqs.min()) / 50
#     gamma2_guess = gamma1_guess
#
#     initial_guess = [b0_guess, b1_guess, A1_guess, A2_guess, x1_guess, x2_guess, gamma1_guess, gamma2_guess]
#
#     params, _ = curve_fit(curve, freqs, fluorescence, p0=initial_guess)
#
#     return curve, params
#
# data = fm_counts_per_modulation_freq[6]
#
# qcurve, params = fit_curve(freqs, data, dip=True)
#
# fit = qcurve(freqs, *params)
#
# plt.plot(data, 'x')
# plt.plot(fit)
# plt.show()

for i, (modulation_freq, fm_counts) in enumerate(list(zip(modulation_freqs, fm_counts_per_modulation_freq))):
    curve, params = fit_curve(detuning, fm_counts, mod_depth, dip=True)
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