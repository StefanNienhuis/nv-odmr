import time
import numpy as np
from pathlib import Path
from TimeTagger import TimeTagStream, createTimeTaggerNetwork

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
TT_CLICK_CHANNEL = 1
TARGET_PEAK = 4          # capture this peak (1-indexed)
BEFORE_N    = 15         # samples to keep before the peak
AFTER_N     = 30         # samples to keep after the peak
THRESHOLD   = 10000      # counts/s above which we are "inside" a peak
EXIT_FRAC   = 0.8        # hysteresis: a peak ends when rate < THRESHOLD * EXIT_FRAC

DATA_DIR = Path("../data/laser_peak")   # <<< set output folder here
DATA_DIR.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------------------
# Time Tagger initialization
# ----------------------------------------------------------------------
tt = createTimeTaggerNetwork('localhost:41101')
tt.setTriggerLevel(TT_CLICK_CHANNEL, 0.25)
stream = TimeTagStream(tt, 1e6, [TT_CLICK_CHANNEL])

# ----------------------------------------------------------------------
# Acquisition loop (headless: runs until the target peak is captured)
# ----------------------------------------------------------------------
y_data = []
dt_data = []                                        # NEW: real elapsed time per sample (s)
peak_count = 0
in_peak = False
tracking_target = False
peak_max_val = 0.0
peak_idx = None

print(f"Acquiring... will stop after peak #{TARGET_PEAK} "
      f"({BEFORE_N} before, {AFTER_N} after).")

while True:
    time.sleep(0.001)                               # pace the loop (~1 ms / sample)
    data = stream.getData()
    diff_t = (data.tGetData - data.tStart) / 1e12   # seconds
    if diff_t <= 0:
        continue
    rate = data.size / diff_t
    y_data.append(rate)
    dt_data.append(diff_t)                          # NEW: keep the true interval
    i = len(y_data) - 1

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

# ----------------------------------------------------------------------
# Extract and save the window around the target peak
# ----------------------------------------------------------------------
y  = np.asarray(y_data)
dt = np.asarray(dt_data)
lo = max(0, peak_idx - BEFORE_N)
hi = min(len(y), peak_idx + AFTER_N + 1)

window    = y[lo:hi]
dt_window = dt[lo:hi]                                 # NEW: matching intervals

# Build a real time axis for the window, measured from the peak sample.
# cumsum of the intervals gives elapsed time; subtract the peak's time to
# center t = 0 on the peak. (t[k] is the time at the END of sample k.)
peak_in_window = peak_idx - lo
t_abs    = np.cumsum(dt_window)                      # seconds, monotonically increasing
t_window = t_abs - t_abs[peak_in_window]             # seconds, zeroed at the peak

out_path = DATA_DIR / f"peak_{TARGET_PEAK}_window.npz"
np.savez(out_path, rate=window, t=t_window, dt=dt_window)

print(f"Peak #{TARGET_PEAK} at sample {peak_idx} (rate {peak_max_val:.3g} 1/s).")
print(f"Saved {window.size} samples to {out_path} "
      f"(peak at index {peak_in_window} within the window).")
print(f"Mean sample interval {dt_window.mean()*1e3:.3g} ms "
      f"(min {dt_window.min()*1e3:.3g}, max {dt_window.max()*1e3:.3g}).")