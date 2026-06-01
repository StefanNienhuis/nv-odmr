import matplotlib
matplotlib.use("TkAgg")          # set backend BEFORE importing pyplot
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from TimeTagger import TimeTagStream, createTimeTaggerNetwork

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
TT_CLICK_CHANNEL = 1
TARGET_PEAK = 4         # freeze the window around this peak (1-indexed)
BEFORE_N    = 15         # samples to show before the peak
AFTER_N     = 30         # samples to show after the peak
THRESHOLD   = 10000        # counts/s above which we are "inside" a peak.
                         # >>> SET THIS from the live monitor (red dashed line) <<<
EXIT_FRAC   = 0.8        # hysteresis: a peak ends when rate < THRESHOLD * EXIT_FRAC
LIVE_HISTORY = 100       # samples shown in the rolling live monitor

FIG_DIR = Path("../figures/laser_peak")
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------------------
# Time Tagger initialization
# ----------------------------------------------------------------------
tt = createTimeTaggerNetwork('localhost:41101')
tt.setTriggerLevel(TT_CLICK_CHANNEL, 0.25)
stream = TimeTagStream(tt, 1e6, [TT_CLICK_CHANNEL])

# ----------------------------------------------------------------------
# Live monitor (rolls until the target peak is captured)
# ----------------------------------------------------------------------
plt.ion()
fig, ax = plt.subplots()
line, = ax.plot([], [], marker='o', ms=3, lw=1)
ax.axhline(THRESHOLD, color="red", ls="--", lw=0.8, label="peak threshold")
ax.set_xlabel("sample")
ax.set_ylabel("count rate (1/s)")
ax.legend(loc="upper right", fontsize=8)

y_data = []
peak_count = 0
in_peak = False
tracking_target = False
peak_max_val = 0.0
peak_idx = None

print(f"Monitoring... will freeze around peak #{TARGET_PEAK} "
      f"({BEFORE_N} before, {AFTER_N} after).")
print("Adjust THRESHOLD if the red line doesn't sit between baseline and peaks.")

while True:
    data = stream.getData()
    diff_t = (data.tGetData - data.tStart) / 1e12       # seconds
    if diff_t <= 0:
        plt.pause(0.001)
        continue
    rate = data.size / diff_t
    y_data.append(rate)
    i = len(y_data) - 1

    # --- live monitor refresh ---
    yt = y_data[-LIVE_HISTORY:]
    xt = list(range(len(y_data) - len(yt), len(y_data)))
    line.set_data(xt, yt)
    ax.set_xlim(max(0, len(y_data) - LIVE_HISTORY), len(y_data) + 1)
    ax.set_ylim(0, max(yt) * 1.1)
    plt.pause(0.001)

    # --- peak detection (rising edge over the threshold, with hysteresis) ---
    if not in_peak and rate > THRESHOLD:
        in_peak = True
        peak_count += 1
        if peak_count == TARGET_PEAK:
            tracking_target = True
            peak_max_val = rate
            peak_idx = i
    elif in_peak and rate < THRESHOLD * EXIT_FRAC:
        in_peak = False
        tracking_target = False          # target peak ended; peak_idx holds its maximum

    # track the maximum sample of the target peak
    if tracking_target and rate > peak_max_val:
        peak_max_val = rate
        peak_idx = i

    # stop once we have AFTER_N samples past the target peak's maximum
    if peak_idx is not None and not tracking_target and i >= peak_idx + AFTER_N:
        break

plt.ioff()
plt.close(fig)

# ----------------------------------------------------------------------
# Frozen window around the target peak
# ----------------------------------------------------------------------
y = np.asarray(y_data)
lo = max(0, peak_idx - BEFORE_N)
hi = min(len(y), peak_idx + AFTER_N + 1)
window = y[lo:hi]
offset = np.arange(lo, hi) - peak_idx      # sample index relative to the peak (0 at peak)

print(f"Peak #{TARGET_PEAK} at sample {peak_idx} "
      f"(rate {peak_max_val:.3g} 1/s); showing samples {lo}..{hi - 1}.")

# light publication styling for the static figure
plt.rcParams.update({
    "figure.dpi":        150,
    "savefig.dpi":       600,
    "font.family":       "serif",
    "font.serif":        ["DejaVu Serif", "Times New Roman", "Times"],
    "mathtext.fontset":  "dejavuserif",
    "font.size":         9,
    "axes.labelsize":    9,
    "xtick.labelsize":   8,
    "ytick.labelsize":   8,
    "axes.linewidth":    0.8,
    "xtick.direction":   "in",
    "ytick.direction":   "in",
    "xtick.top":         True,
    "ytick.right":       True,
    "xtick.minor.visible": True,
    "ytick.minor.visible": True,
    "axes.unicode_minus": True,
})

fig, ax = plt.subplots(figsize=(3.6, 2.6))
ax.plot(offset, window, marker='o', ms=3, color="C0", lw=1.2)
ax.set_xlabel("Sample (relative to peak)")
ax.set_ylabel("Count rate (1/s)")
ax.set_xlim(offset.min(), offset.max())
ax.set_ylim(0, window.max() * 1.1)
fig.tight_layout()
fig.savefig(FIG_DIR / f"peak_{TARGET_PEAK}_window.png", bbox_inches="tight")
plt.show()