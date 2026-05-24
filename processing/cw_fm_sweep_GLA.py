import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from util.double_lorentzian_fit import fit_curve

# ----------------------------------------------------------------------
# Publication style (single-column journal figure)
# ----------------------------------------------------------------------
plt.rcParams.update({
    "figure.dpi":        150,     # on-screen
    "savefig.dpi":       600,     # PNG export
    "savefig.pad_inches": 0.02,
    "font.family":       "serif",
    "font.serif":        ["DejaVu Serif", "Times New Roman", "Times"],
    "mathtext.fontset":  "dejavuserif",
    "font.size":         9,
    "axes.labelsize":    9,
    "axes.titlesize":    9,
    "legend.fontsize":   7.5,
    "xtick.labelsize":   8,
    "ytick.labelsize":   8,
    "axes.linewidth":    0.8,
    "lines.linewidth":   1.3,
    "xtick.direction":   "in",
    "ytick.direction":   "in",
    "xtick.top":         True,
    "ytick.right":       True,
    "xtick.minor.visible": True,
    "ytick.minor.visible": True,
    "xtick.major.size":  3.5,
    "ytick.major.size":  3.5,
    "xtick.minor.size":  2.0,
    "ytick.minor.size":  2.0,
    "legend.frameon":    False,
    "axes.unicode_minus": True,
})

FIG_W, FIG_H = 3.4, 2.6          # inches; single column. Use ~7.0 for double.
SHOW = True                      # set False for unattended batch export

center_freq = 2.87e9             # NV zero-field splitting D
start_freq, stop_freq = 2.84e9, 2.90e9
freq_dev = 1.5e6                 # FM deviation (sideband offset)
TRIM_FRAC = 0.30                 # fraction of points cut from EACH edge; auto-scales with
                                 # sweep length (0.30 ~ the old 120/401). Set 0.0 to disable.


def strip_npy_suffix(name):
    while name.endswith((".npz", ".npy")):
        name = name.rsplit(".", 1)[0]
    return name


sweep_dir = Path("../data/cw_fm_sweep")
fig_dir = Path("../figures/fm_sweep")
fig_dir.mkdir(parents=True, exist_ok=True)

files = sorted(sweep_dir.glob("*.npy"))
print(f"Found {len(files)} file(s) in {sweep_dir}\n")

for path in files:
    print(f"=== {path.name} ===")
    try:
        counts = np.load(path)                  # expected shape (n_sweep, n_repeats, 2)
        print(f"  raw shape {counts.shape}")
        n_sweep = counts.shape[0]

        # .npy stores no frequency axis, so build it from the sweep length
        detuning = np.linspace(start_freq, stop_freq, n_sweep) - center_freq

        # auto-trim a fraction from each edge: scales with sweep length and
        # never empties the array (always leaves >= 5 points)
        trim = int(round(n_sweep * TRIM_FRAC))
        trim = min(trim, (n_sweep - 5) // 2)
        if trim > 0:
            detuning = detuning[trim:-trim]
            counts = counts[trim:-trim]
        print(f"  trimmed {trim} pts/side -> kept {len(detuning)} of {n_sweep}")

        mean_low = np.mean(counts[:, :, 0], axis=1)
        mean_high = np.mean(counts[:, :, 1], axis=1)
        fm_counts = (mean_high - mean_low) / (mean_high + mean_low)

        # Fit the two sidebands together, then reconstruct the dispersive signal
        low_detuning = detuning - freq_dev
        high_detuning = detuning + freq_dev
        curve, params = fit_curve(
            np.concatenate((low_detuning, high_detuning)),
            np.concatenate((mean_low, mean_high)),
        )
        low_fit = curve(low_detuning, *params)
        high_fit = curve(high_detuning, *params)
        fit = (high_fit - low_fit) / (high_fit + low_fit)

        slope = np.gradient(fit, detuning)
        max_slope_detuning = detuning[np.argmax(np.abs(slope))]
        slope_MHz = max_slope_detuning / 1e6
        print(f"  Max slope: {slope_MHz:.3f} MHz "
              f"({(center_freq + max_slope_detuning)/1e9:.6f} GHz)")

        # ----------------------------- figure -----------------------------
        fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))

        ax.plot(detuning / 1e6, fm_counts, marker="o", ls="none", ms=3.2,
                mfc="none", mec="0.35", mew=0.8, label="Data")
        ax.plot(detuning / 1e6, fit, color="C3", lw=1.4, label="Reconstructed fit")
        ax.axvline(slope_MHz, color="C1", ls=":", lw=1.0,
                   label=rf"Max slope ($\delta = {slope_MHz:.2f}$ MHz)")

        ax.set_xlabel(r"Detuning $\delta = f - D$ (MHz)")
        ax.set_ylabel(r"FM signal")
        ax.set_xlim((detuning / 1e6).min(), (detuning / 1e6).max())

        ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=2,
                  handlelength=1.6, columnspacing=1.1, borderaxespad=0.0)

        fig.tight_layout(rect=(0, 0, 1, 0.82))

        # ----------------------- export (PNG only) ------------------------
        base = strip_npy_suffix(path.name)
        fig.savefig(fig_dir / f"{base}.png", bbox_inches="tight")

        if SHOW:
            plt.show()
        else:
            plt.close(fig)

    except Exception as e:
        print(f"  SKIPPED: {type(e).__name__}: {e}")