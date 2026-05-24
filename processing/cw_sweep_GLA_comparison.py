
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
SHOW = True

DATA_MS = 1.8                    # data marker size (smaller = less overlap)
DATA_ALPHA = 0.5                 # data marker transparency (0-1)

center_freq = 2.87e9             # NV zero-field splitting D
start_freq, stop_freq = 2.84e9, 2.90e9   # fallback only (no axis in file)

DATA_KEYS = ("data", "counts", "y")
FREQ_KEYS = ("freqs", "frequencies", "frequency", "freq", "f", "x")

# === the files to compare: (path, legend label). Add more rows to compare >2. ===
FILES = [
    ("../persist/cw_sweep/2026-05-06T12.13.59.362988.npy", "Week 3"),
    ("../data/cw_sweep/2026-05-19T11.45.42.733044.npz", "Week 5"),
]
COLORS = ["C0", "C3", "C2", "C1", "C4"]      # cycled per dataset
OUT_PNG = Path("../figures/cw_sweep/comparison.png")


def pick(keys, candidates):
    for k in candidates:
        if k in keys:
            return k
    return None


def load_data(path):
    """Return (array, freqs_or_None, npz_keys_or_None) for either .npy or .npz."""
    obj = np.load(path)
    if hasattr(obj, "files"):
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
    return np.asarray(obj), None, None


def analyse(path):
    arr, freqs, keys = load_data(path)
    counts = np.asarray(arr).squeeze()
    if freqs is None:
        freqs = np.linspace(start_freq, stop_freq, counts.shape[0])
        if keys is not None:
            print(f"  no usable freq axis (keys: {keys}); "
                  f"assuming {counts.shape[0]} pts over "
                  f"{start_freq/1e9}-{stop_freq/1e9} GHz")
    detuning = freqs - center_freq
    counts = counts / np.max(counts)          # each trace normalised to its own max
    curve, params = fit_curve(detuning, counts)
    fit = curve(detuning, *params)
    dip = detuning[np.argmin(np.abs(fit))]
    slope = np.gradient(fit, detuning)
    max_slope = detuning[np.argmax(np.abs(slope))]
    return detuning, counts, fit, dip, max_slope


fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
results = []

for i, (path, label) in enumerate(FILES):
    color = COLORS[i % len(COLORS)]
    print(f"=== {label}: {Path(path).name} ===")
    try:
        detuning, counts, fit, dip, max_slope = analyse(path)
        print(f"  Max dip at {dip/1e6:.3f} MHz")
        print(f"  Max slope at {max_slope/1e6:.3f} MHz")
        results.append((label, dip, max_slope))

        ax.plot(detuning / 1e6, counts, marker="o", ls="none", ms=DATA_MS,
                mfc=color, mec="none", alpha=DATA_ALPHA, label=f"{label} (data)")
        ax.plot(detuning / 1e6, fit, color=color, lw=1.4, label=f"{label} (fit)")
        ax.axvline(dip / 1e6, color=color, ls="--", lw=0.8, alpha=0.6)  # dip guide (no legend)
    except Exception as e:
        print(f"  SKIPPED: {type(e).__name__}: {e}")

# numeric comparison (relative to the first dataset)
if len(results) >= 2:
    ref_label, ref_dip, ref_slope = results[0]
    print("\n--- comparison ---")
    for label, dip, max_slope in results[1:]:
        print(f"  {label} - {ref_label}:  "
              f"dip shift {(dip - ref_dip)/1e6:+.3f} MHz, "
              f"slope shift {(max_slope - ref_slope)/1e6:+.3f} MHz")

ax.set_xlabel(r"Detuning $\delta = f - D$ (MHz)")
ax.set_ylabel(r"Normalized PL")

ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=2,
          handlelength=1.6, columnspacing=1.1, borderaxespad=0.0)
fig.tight_layout(rect=(0, 0, 1, 0.82))

OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT_PNG, bbox_inches="tight")

if SHOW:
    plt.show()
else:
    plt.close(fig)