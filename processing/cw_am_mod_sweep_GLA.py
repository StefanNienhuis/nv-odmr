import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from math import ceil
from util.double_lorentzian_fit import fit_curve

center_freq = 2.8e9   # NOTE: 2.8 GHz here (other scripts use 2.87); it cancels out of
                      # drive_freq, but shifts the detuning/peak columns and the x-axis.

results = np.load('../persist/cw_am_mod_sweep/2026-05-12T13.02.38.492524.npz',
                  allow_pickle=True)
counts = results['data']
meta = results['params'].item()              # the stored params dict (renamed to avoid
                                             # clashing with the fit params below)
modulation_freqs = meta['modulation_freqs']
freqs = np.linspace(meta['start_freq'], meta['stop_freq'], meta['n_sweep'])
detuning = freqs - center_freq

# adaptive subplot grid (was hardcoded 4x5, which broke for != 20 modulation freqs)
n = len(modulation_freqs)
ncols = min(5, n)
nrows = ceil(n / ncols)
plt.figure(figsize=(3 * ncols, 2.2 * nrows))

rows = []
for i, (modulation_freq, freq_counts) in enumerate(zip(modulation_freqs, counts)):
    active_counts = freq_counts[:, :, 0]
    inactive_counts = freq_counts[:, :, 1]
    mean_active = np.mean(active_counts, axis=1)
    mean_inactive = np.mean(inactive_counts, axis=1)
    am_counts = (mean_inactive - mean_active) / mean_inactive

    curve, fit_params = fit_curve(detuning, am_counts, dip=False)
    fit = curve(detuning, *fit_params)

    max_peak_detuning = detuning[np.argmax(fit)]
    slope = np.gradient(fit, detuning)
    max_slope = np.max(np.abs(slope))
    max_slope_detuning = detuning[np.argmax(np.abs(slope))]

    rows.append({
        "mod_freq_Hz":        modulation_freq,
        "drive_freq_GHz":     (center_freq + max_slope_detuning) / 1e9,
        "slope_detuning_MHz": max_slope_detuning / 1e6,
        "peak_detuning_MHz":  max_peak_detuning / 1e6,
        "max_abs_slope":      max_slope,
    })

    ax = plt.subplot(nrows, ncols, i + 1)
    ax.set_title(f"{modulation_freq} Hz", fontsize=8)
    ax.plot(detuning / 1e6, am_counts, "x", color="gray", ms=4, label="Data")
    ax.plot(detuning / 1e6, fit, color="red", lw=1.0, label="Fit")
    ax.axvline(max_peak_detuning / 1e6, ls="--", color="C0", lw=0.8)
    ax.axvline(max_slope_detuning / 1e6, ls=":", color="C1", lw=0.8)
    ax.tick_params("x", labelbottom=(i >= (nrows - 1) * ncols))   # labels on bottom row only

plt.tight_layout()

# ----------------------------------------------------------------------
# Summary table: one row per modulation frequency
# ----------------------------------------------------------------------
df = pd.DataFrame(rows)
# df = df.sort_values("max_abs_slope", ascending=False)  # uncomment for best-slope first

print(df.to_string(index=False))                 # clean console table
df.to_csv("am_mod_sweep_summary.csv", index=False)

# ---- LaTeX table for Overleaf -------------------------------------------------
# Preamble needs: \usepackage{booktabs} and \usepackage{makecell}
# Drive freq. and slope detuning are redundant (one is the other shifted by the
# centre), so we drop slope detuning; the slope is scaled to keep its column narrow.
SLOPE_SCALE = 1e8        # report slope in units of 1e-8 Hz^-1

tex = df.drop(columns=["slope_detuning_MHz"]).copy()
tex["max_abs_slope"] *= SLOPE_SCALE
tex = tex.rename(columns={
    "mod_freq_Hz":       r"\thead{Mod. freq.\\(Hz)}",
    "drive_freq_GHz":    r"\thead{Drive freq.\\(GHz)}",
    "peak_detuning_MHz": r"\thead{Peak det.\\(MHz)}",
    "max_abs_slope":     r"\thead{Max $|$slope$|$\\($10^{-8}\,$Hz$^{-1}$)}",
})
print("\n" + tex.to_latex(index=False, escape=False, column_format="rrrr",
                          formatters=[
                              "{:.0f}".format,   # mod freq
                              "{:.4f}".format,   # drive freq
                              "{:.1f}".format,   # peak det
                              "{:.3f}".format,   # slope
                          ]))

plt.show()