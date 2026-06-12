import numpy as np
import matplotlib.pyplot as plt

from util.double_lorentzian_diff_fit import fit_curve
import util.plot_style

center_freq = 2.8e9

results = np.load('../persist/cw_fm_mod_sweep/2026-06-03T17.19.41.042999.npz', allow_pickle=True)
fm_counts_per_modulation_freq = results['data']

params = results['params'].item()

modulation_freqs = params['modulation_freqs']
mod_depth = params['mod_depth']

freqs = np.linspace(params['start_freq'], params['stop_freq'], params['n_sweep'])
detuning = freqs - 2.87e9

max_slopes = []
max_slope_freqs = []

fig, ax = plt.subplots(2, 3, sharex=True, sharey=True, layout='constrained')

fig.set_figwidth(7.5)
fig.set_figheight(5)

modulation_freqs = modulation_freqs[[0, 2, 4, 5, 7, 9]]
fm_counts_per_modulation_freq = fm_counts_per_modulation_freq[[0, 2, 4, 5, 7, 9]]

for i, (modulation_freq, fm_counts) in enumerate(list(zip(modulation_freqs, fm_counts_per_modulation_freq))):

    ax[i // 3, i % 3].set_title(f'{modulation_freq:.0f} Hz modulation')
    ax[i // 3, i % 3].plot(detuning / 1e6, fm_counts, marker="o", ls="none", ms=3.2,
                mfc="none", mec="0.33", mew=0.8, label=('Data' if i == 0 else None))

    if i // 3 == 1:
        ax[i // 3, i % 3].set_xlabel('Detuning [MHz]')

    if i % 3 == 0:
        ax[i // 3, i % 3].set_ylabel('FM signal [a.u]')

    ax[i // 3, i % 3].grid(True)

    try:
        curve, params = fit_curve(detuning, -fm_counts, mod_depth, dip=False)

        fit = -curve(detuning, *params)
        max_peak = np.max(fit)
        max_peak_detuning = detuning[np.argmax(fit)]

        slope = np.gradient(fit, detuning)
        max_slope = np.max(np.abs(slope))
        max_slope_detuning = detuning[np.argmax(np.abs(slope))]

        max_slopes.append(max_slope)
        max_slope_freqs.append(center_freq + max_slope_detuning)

        print(f"Max peak at {max_peak_detuning / 1e6} MHz ({(center_freq + max_peak_detuning) / 1e9} GHz)")
        print(f"Max slope at {max_slope_detuning / 1e6} MHz ({(center_freq + max_slope_detuning) / 1e9} GHz)")

        ax[i // 3, i % 3].plot(detuning / 1e6, fit, color="C3", lw=1.4,  label=('Lorentzian difference fit' if i == 0 else None))
        # ax[i].axvline(x=max_peak_detuning / 1e6, linestyle='--', label='Max Peak')
        # ax[i].axvline(x=max_slope_detuning / 1e6, linestyle='--', label='Max Slope')
    except:
        print(f"Fit failed for {modulation_freq} Hz")

print()
print(f"modulation_freqs = {list(modulation_freqs)}")
print(f"drive_freqs = {max_slope_freqs}")
print(f"slopes = {max_slopes}")

fig.suptitle(f"Frequency modulated ODMR spectra for different modulation frequencies\n")

fig.legend(loc="outside upper center", ncol=2, handlelength=1.6, borderaxespad=3, columnspacing=1.1)

fig.get_layout_engine().set(w_pad=8 / 72, h_pad=8 / 72, wspace=0.05, hspace=0.05)
plt.show()