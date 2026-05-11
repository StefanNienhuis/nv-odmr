from datetime import datetime, timedelta
import numpy as np
import matplotlib.pyplot as plt

from util.cw_fm import (
    setup_awg,
    setup_time_tagger,
    run_fm_sweep,
    run_fm_singleshot,
    max_slope_per_pulse_length_fm,
    std_per_pulse_length_fm,
)

start_date = datetime.now()

# Frequency sweep parameters
start_freq      = 2.84e9
stop_freq       = 2.90e9
n_sweep         = 401
n_meas          = 5
meas_delay_ns   = 20e3
pulse_length_ns = 100e6     # 100 ms — operating point from pulse_length calibration

# Mod depth sweep parameters
start_mod_depth_hz = 0.5e6     # 500 kHz, well below typical δν
stop_mod_depth_hz  = 20e6      # 20 MHz, well above typical δν
n_mod_depths       = 7

# Std measurement parameters
n_std_meas = 100   # shots at each freqs_at_max for noise estimation

# Save
params = {
    "start_mod_depth_hz": start_mod_depth_hz,
    "stop_mod_depth_hz":  stop_mod_depth_hz,
    "n_mod_depths":       n_mod_depths,
    "pulse_length_ns":    pulse_length_ns,
    "meas_delay_ns":      meas_delay_ns,
    "start_freq":         start_freq,
    "stop_freq":          stop_freq,
    "n_sweep":            n_sweep,
    "n_meas":             n_meas,
    "n_std_meas":         n_std_meas,
}

# log-spaced mod depths in Hz (was: np.linspace(log10(...)) which gave log10(Hz), not Hz)
mod_depths_hz = np.logspace(
    np.log10(start_mod_depth_hz),
    np.log10(stop_mod_depth_hz),
    n_mod_depths,
)

t_sweep                 = 2 * n_sweep * n_meas * pulse_length_ns / 1e9
t_std                   = 2 *           n_std_meas * pulse_length_ns / 1e9
expected_sweep_duration = n_mod_depths * t_sweep
expected_std_duration   = n_mod_depths * t_std
expected_duration       = expected_sweep_duration + expected_std_duration
print(f"Mod depths (MHz): {mod_depths_hz / 1e6}")
print(f"Pulse length: {pulse_length_ns / 1e6:.1f} ms")
print(f"Per-sweep duration: {t_sweep:.1f}s")
print(f"Per-std   duration: {t_std:.1f}s")
print(f"Expected total duration: {expected_duration:.1f}s")
print(f"Finished at: {(datetime.now() + timedelta(seconds=expected_duration)).time()}")

freq = np.linspace(start_freq, stop_freq, n_sweep)

# Hardware setup (once — oscillators are reconfigured by the sequence per iteration)
awg_channel = setup_awg(start_freq=start_freq, mod_depth=mod_depths_hz[0])
tt          = setup_time_tagger()

# === Frequency sweep per mod depth ==========================================
all_counts = []

for md_idx, mod_depth in enumerate(mod_depths_hz):
    print(f"\n[sweep {md_idx+1}/{n_mod_depths}] "
          f"mod_depth = {mod_depth/1e6:.2f} MHz "
          f"(freq_dev = ±{mod_depth/2/1e6:.2f} MHz), "
          f"sweep duration ≈ {t_sweep:.1f}s")

    counts = run_fm_sweep(
        awg_channel, tt,
        pulse_length_ns=pulse_length_ns,
        start_freq=start_freq, stop_freq=stop_freq,
        mod_depth=mod_depth,
        n_sweep=n_sweep, n_meas=n_meas,
        meas_delay_ns=meas_delay_ns,
    )
    all_counts.append(counts)

max_slopes_fm, freqs_at_max_fm = max_slope_per_pulse_length_fm(all_counts, freq)

# === Std measurement at freqs_at_max_fm =====================================
# For each mod depth, park at the steepest point of that mod depth's sweep
# and run n_std_meas shots at fixed frequency for noise estimation.
all_std_counts = []

for md_idx, (mod_depth, f_at_max) in enumerate(zip(mod_depths_hz, freqs_at_max_fm)):
    print(f"\n[std {md_idx+1}/{n_mod_depths}] "
          f"mod_depth = {mod_depth/1e6:.2f} MHz, "
          f"freq = {f_at_max/1e9:.6f} GHz, "
          f"duration ≈ {t_std:.1f}s")

    counts = run_fm_singleshot(
        awg_channel, tt,
        pulse_length_ns=pulse_length_ns,
        freq=f_at_max,
        mod_depth=mod_depth,
        n_meas=n_std_meas,
        meas_delay_ns=meas_delay_ns,
    )
    all_std_counts.append(counts)

stds        = std_per_pulse_length_fm(all_std_counts)
T_shot = 2 * pulse_lengths_ns * 1e-9          # seconds, per pulse length
sensitivity = stds / max_slopes_fm * np.sqrt(T_shot)   

# === Save ===================================================================
np.savez(
    f"../persist/cw_fm_sweep_mod_depth_sweep/{start_date.isoformat().replace(':', '.')}.npz",
    max_slopes=max_slopes_fm,
    freqs_at_max=freqs_at_max_fm,
    stds=stds,
    sensitivity=sensitivity,
    mod_depths_hz=mod_depths_hz,
    params=params,
)

# === Plots ==================================================================
plt.plot(mod_depths_hz / 1e6, max_slopes_fm, 'o-')
plt.xlabel('mod depth [MHz]')
plt.ylabel(r'max $|d\,\mathrm{fm\_counts}/df|$')
plt.show()

plt.plot(mod_depths_hz / 1e6, freqs_at_max_fm, 'o-')
plt.xlabel('mod depth [MHz]')
plt.ylabel('frequency at max slope [Hz]')
plt.show()

plt.plot(mod_depths_hz / 1e6, sensitivity, 'o-')
plt.xlabel('mod depth [MHz]')
plt.ylabel(r'sensitivity $\sigma_\mathrm{FM}/\max|d\,\mathrm{fm}/df|$ [Hz]')
plt.show()