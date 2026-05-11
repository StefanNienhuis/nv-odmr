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
start_freq    = 2.84e9
stop_freq     = 2.90e9
mod_depth     = 3e6
n_sweep       = 401
n_meas        = 5
meas_delay_ns = 20e3

# Pulse length sweep parameters
start_pulse_length_ns = 5e6      # 5 ms   -> mod_freq = 100 Hz
stop_pulse_length_ns  = 200e6    # 200 ms -> mod_freq = 2.5 Hz
n_pulse_lengths       = 5

# Std measurement parameters
n_std_meas = 100   # shots at each freqs_at_max for noise estimation

# Save
params = {
    "start_pulse_length_ns": start_pulse_length_ns,
    "stop_pulse_length_ns":  stop_pulse_length_ns,
    "n_pulse_lengths":       n_pulse_lengths,
    "meas_delay_ns":         meas_delay_ns,
    "start_freq":            start_freq,
    "stop_freq":             stop_freq,
    "mod_depth":             mod_depth,
    "n_sweep":               n_sweep,
    "n_meas":                n_meas,
    "n_std_meas":            n_std_meas,
}

pulse_lengths_ns = np.logspace(
    np.log10(start_pulse_length_ns),
    np.log10(stop_pulse_length_ns),
    n_pulse_lengths,
)

expected_sweep_duration = sum(2 * n_sweep * n_meas * pl / 1e9 for pl in pulse_lengths_ns)
expected_std_duration   = sum(2 *           n_std_meas * pl / 1e9 for pl in pulse_lengths_ns)
expected_duration       = expected_sweep_duration + expected_std_duration
print(f"Pulse lengths (ms): {pulse_lengths_ns / 1e6}")
print(f"Modulation freq (Hz): {1e9 / (2 * pulse_lengths_ns)}")
print(f"Expected sweep duration: {expected_sweep_duration:.1f}s")
print(f"Expected std   duration: {expected_std_duration:.1f}s")
print(f"Expected total duration: {expected_duration:.1f}s")
print(f"Finished at: {(datetime.now() + timedelta(seconds=expected_duration)).time()}")

freq = np.linspace(start_freq, stop_freq, n_sweep)

# Hardware setup (once)
awg_channel = setup_awg(start_freq=start_freq, mod_depth=mod_depth)
tt          = setup_time_tagger()

# === Frequency sweep per pulse length =======================================
all_counts = []

for pl_idx, pl_ns in enumerate(pulse_lengths_ns):
    modulation_freq = 1e9 / (2 * pl_ns)
    t_sweep = 2 * n_sweep * n_meas * pl_ns / 1e9
    print(f"\n[sweep {pl_idx+1}/{n_pulse_lengths}] "
          f"pulse_length = {pl_ns/1e6:.2f} ms, "
          f"mod_freq = {modulation_freq:.2f} Hz, "
          f"sweep duration ≈ {t_sweep:.1f}s")

    counts = run_fm_sweep(
        awg_channel, tt,
        pulse_length_ns=pl_ns,
        start_freq=start_freq, stop_freq=stop_freq,
        mod_depth=mod_depth,
        n_sweep=n_sweep, n_meas=n_meas,
        meas_delay_ns=meas_delay_ns,
    )
    all_counts.append(counts)

max_slopes_fm, freqs_at_max_fm = max_slope_per_pulse_length_fm(all_counts, freq)

# === Std measurement at freqs_at_max_fm =====================================
# For each pulse length park at the steepest point and run n_std_meas shots
# of the FM measurement to estimate per-shot noise.
all_std_counts = []

for pl_idx, (pl_ns, f_at_max) in enumerate(zip(pulse_lengths_ns, freqs_at_max_fm)):
    t_std = 2 * n_std_meas * pl_ns / 1e9
    print(f"\n[std {pl_idx+1}/{n_pulse_lengths}] "
          f"pulse_length = {pl_ns/1e6:.2f} ms, "
          f"freq = {f_at_max/1e9:.6f} GHz, "
          f"duration ≈ {t_std:.1f}s")

    counts = run_fm_singleshot(
        awg_channel, tt,
        pulse_length_ns=pl_ns,
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
    f"../persist/cw_fm_sweep_mod_sweep/{start_date.isoformat().replace(':', '.')}.npz",
    max_slopes=max_slopes_fm,
    freqs_at_max=freqs_at_max_fm,
    stds=stds,
    sensitivity=sensitivity,
    pulse_lengths_ns=pulse_lengths_ns,
    params=params,
)

# === Plots ==================================================================
plt.plot(pulse_lengths_ns / 1e6, max_slopes_fm, 'o-')
plt.xlabel('pulse length [ms]')
plt.ylabel(r'max $|d\,\mathrm{fm\_counts}/df|$')
plt.show()

plt.plot(pulse_lengths_ns / 1e6, freqs_at_max_fm, 'o-')
plt.xlabel('pulse length [ms]')
plt.ylabel('frequency at max slope [Hz]')
plt.show()

plt.plot(pulse_lengths_ns / 1e6, sensitivity, 'o-')
plt.xlabel('pulse length [ms]')
plt.ylabel(r'sensitivity $\sigma_\mathrm{FM}/\max|d\,\mathrm{fm}/df|$ [Hz]')
plt.show()