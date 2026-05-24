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
start_freq, stop_freq = 2.84e9, 2.90e9   # fallback only (no axis in file)

DATA_KEYS = ("data", "counts", "y")
FREQ_KEYS = ("freqs", "frequencies", "frequency", "freq", "f", "x")


def pick(keys, candidates):
    for k in candidates:
        if k in keys:
            return k
    return None


def strip_suffix(name):
    while name.endswith((".npz", ".npy")):
        name = name.rsplit(".", 1)[0]
    return name


def load_data(path):
    """Return (array, freqs_or_None, npz_keys_or_None) for either .npy or .npz."""
    obj = np.load(path)
    if hasattr(obj, "files"):                 # .npz / .npy.npz archive
        dkey = pick(obj.files, DATA_KEYS)
        if dkey is None:
            if len(obj.files) == 1:
                dkey = obj.files[0]
            else:
                raise KeyError(f"no data array found; keys are {obj.files}")
        arr = np.asarray(obj[dkey])
        fkey = pick(obj.files, FREQ_KEYS)
        freqs = (np.asarray(obj[fkey]).squeeze()
                 if fkey is not None and len(obj[fkey]) == arr.shape[0] else None)
        return arr, freqs, obj.files
    return np.asarray(obj), None, None        # plain .npy (no stored axis)


sweep_dir = Path("../data/cw_sweep")
fig_dir = Path("../figures/cw_sweep")
fig_dir.mkdir(parents=True, exist_ok=True)

files = sorted(sweep_dir.glob("*.np*"))       # matches .npy, .npz and .npy.npz
print(f"Found {len(files)} file(s) in {sweep_dir}\n")

for path in files:
    print(f"=== {path.name} ===")
    try:
        arr, freqs, keys = load_data(path)
        counts = np.asarray(arr).squeeze()

        if freqs is None:
            freqs = np.linspace(start_freq, stop_freq, counts.shape[0])
            if keys is not None:              # was an npz but no usable axis
                print(f"  no usable freq axis (keys: {keys}); "
                      f"assuming {counts.shape[0]} pts over "
                      f"{start_freq/1e9}-{stop_freq/1e9} GHz")

        detuning = freqs - center_freq
        counts = counts / np.max(counts)

        curve, params = fit_curve(detuning, counts)
        fit = curve(detuning, *params)

        max_dip_detuning = detuning[np.argmin(np.abs(fit))]
        slope = np.gradient(fit, detuning)
        max_slope_detuning = detuning[np.argmax(np.abs(slope))]

        dip_MHz = max_dip_detuning / 1e6
        slope_MHz = max_slope_detuning / 1e6
        print(f"  Max dip at {dip_MHz:.3f} MHz")
        print(f"  Max slope at {slope_MHz:.3f} MHz")

        # ----------------------------- figure -----------------------------
        fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))

        ax.plot(detuning / 1e6, counts, marker="o", ls="none", ms=3.2,
                mfc="none", mec="0.35", mew=0.8, label="Data")
        ax.plot(detuning / 1e6, fit, color="C3", lw=1.4, label="Double-Lorentzian fit")
        ax.axvline(dip_MHz, color="C0", ls="--", lw=1.0,
                   label=rf"Max dip ($\delta = {dip_MHz:.2f}$ MHz)")
        ax.axvline(slope_MHz, color="C1", ls=":", lw=1.0,
                   label=rf"Max slope ($\delta = {slope_MHz:.2f}$ MHz)")

        ax.set_xlabel(r"Detuning $\delta = f - D$ (MHz)")
        ax.set_ylabel(r"Normalized PL")
        ax.set_xlim((detuning / 1e6).min(), (detuning / 1e6).max())

        ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=2,
                  handlelength=1.6, columnspacing=1.1, borderaxespad=0.0)

        fig.tight_layout(rect=(0, 0, 1, 0.82))

        # ----------------------- export (PNG only) ------------------------
        # fig.savefig(fig_dir / f"{strip_suffix(path.name)}.png", bbox_inches="tight")

        if SHOW:
            plt.show()
        else:
            plt.close(fig)

    except Exception as e:
        print(f"  SKIPPED: {type(e).__name__}: {e}")