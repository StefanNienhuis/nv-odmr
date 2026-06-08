import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ----------------------------------------------------------------------
# Publication style (single-column journal figure)
# Copied verbatim from the CW-ODMR sweep script so the two figures match.
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

# ----------------------------------------------------------------------
# Laser-peak specifics
# ----------------------------------------------------------------------
PREFER_REAL_TIME = True  # If the file is an .npz carrying a measured 't' array,
                         # plot against that real (per-sample) time. Falls back to
                         # SAMPLE_DT_MS / sample index for plain .npy files.
SAMPLE_DT_MS = None     # Fallback only, used when no measured 't' is available.
                        # A float (e.g. 1.0) plots a NOMINAL ms axis from the sample
                        # index; None plots the sample index itself. Nominal ms is
                        # not measured time -- prefer None unless you know the loop
                        # interval is truly uniform.
NORMALIZE    = False    # True -> peak = 1 (matches the ODMR "Normalized PL" axis)

peak_dir = Path("../data/laser_peak")
fig_dir  = Path("../figures/laser_peak")
fig_dir.mkdir(parents=True, exist_ok=True)


def strip_suffix(name):
    while name.endswith((".npz", ".npy")):
        name = name.rsplit(".", 1)[0]
    return name


files = sorted(peak_dir.glob("*.np*"))       # matches .npy, .npz and .npy.npz
print(f"Found {len(files)} file(s) in {peak_dir}\n")

for path in files:
    print(f"=== {path.name} ===")
    try:
        # --- load: .npz carries measured time; .npy is rate-only ---
        loaded = np.load(path)
        t = None
        if isinstance(loaded, np.lib.npyio.NpzFile):
            if "rate" not in loaded.files:
                raise ValueError(f"npz has no 'rate' array; found {loaded.files}")
            y = np.asarray(loaded["rate"]).squeeze().astype(float)
            if PREFER_REAL_TIME and "t" in loaded.files:
                t = np.asarray(loaded["t"]).squeeze().astype(float)
        else:
            y = np.asarray(loaded).squeeze().astype(float)
        if y.ndim != 1:
            raise ValueError(f"expected a 1-D window, got shape {y.shape}")
        if t is not None and t.shape != y.shape:
            raise ValueError(f"t shape {t.shape} != rate shape {y.shape}")

        peak_i = int(np.argmax(y))           # the window is built around its maximum

        # --- x-axis: measured time if available, else nominal ms / sample index ---
        if t is not None:
            x = (t - t[peak_i]) * 1e3        # measured seconds -> ms, zeroed at peak
            xlabel = r"Time from peak $t$ (ms)"
        elif SAMPLE_DT_MS is not None:
            x = (np.arange(y.size) - peak_i) * SAMPLE_DT_MS
            xlabel = r"Time from peak $t$ (ms, nominal)"
        else:
            x = np.arange(y.size) - peak_i
            xlabel = r"Samples from peak"

        # --- y-axis: kcps, or normalized to the peak ---
        if NORMALIZE:
            yp = y / y.max()
            ylabel = r"Normalized count rate"
            unit_y = ""
        else:
            yp = y / 1e3
            ylabel = r"Count rate (kcps)"
            unit_y = " kcps"

        peak_val = yp[peak_i]
        print(f"  Peak at sample {peak_i} of {y.size}; height {peak_val:.3g}{unit_y}")

        # ----------------------------- figure -----------------------------
        fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))

        ax.plot(x, yp, color="0.6", lw=0.8, zorder=1, label="_nolegend_")  # connect dots
        ax.plot(x, yp, marker="o", ls="none", ms=3.2,
                mfc="none", mec="0.35", mew=0.8, zorder=2, label="Data")

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_xlim(x.min(), x.max())

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