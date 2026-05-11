from datetime import datetime, timedelta
import numpy as np
import matplotlib.pyplot as plt

from util.cw_am import (
    CENTER_FREQ,
    ns_to_samples,
    init_awg,
    init_time_tagger,
    configure_sweep,
    run_sweep,
    max_slope_per_pulse_length,
    std_per_pulse_length
)

start_date = datetime.now()

# Frequency sweep parameters
start_freq      = 2.84e9
stop_freq       = 2.90e9
n_sweep         = 401
n_meas          = 5
osc             = 0
meas_delay_ns   = 20e3

# Pulse length sweep parameters
start_pulse_length_ns = 5e6
stop_pulse_length_ns  = 200e6
n_pulse_lengths       = 5

# Std measurement parameters
n_std_meas = 100 

params = {
    "start_pulse_length_ns": start_pulse_length_ns,
    "stop_pulse_length_ns":stop_pulse_length_ns,
    "n_pulse_lengths":n_pulse_lengths,
    "meas_delay_ns": meas_delay_ns,
    "start_freq": start_freq,
    "stop_freq": stop_freq,
    "n_sweep": n_sweep,
    "n_meas": n_meas,
    "n_std_meas": n_std_meas,
}

pulse_lengths_ns = np.logspace(
    np.log10(start_pulse_length_ns),
    np.log10(stop_pulse_length_ns),
    n_pulse_lengths,
)

expected_sweep_duration = sum(2 * n_sweep * n_meas * pl / 1e9 for pl in pulse_lengths_ns)
expected_std_duration   = sum(2 * 1       * n_std_meas * pl / 1e9 for pl in pulse_lengths_ns)
expected_duration = expected_sweep_duration + expected_std_duration
print(f"Pulse lengths (ms): {pulse_lengths_ns/1e6}")
print(f"Modulation freq (Hz): {1e9/(2*pulse_lengths_ns)}")
print(f"Expected sweep duration: {expected_sweep_duration:.1f}s")
print(f"Expected std   duration: {expected_std_duration:.1f}s")
print(f"Expected total duration: {expected_duration:.1f}s")
print(f"Finished at: {(datetime.now() + timedelta(seconds=expected_duration)).time()}")


meas_delay = ns_to_samples(meas_delay_ns)

relative_start_freq = start_freq - CENTER_FREQ
freq = np.linspace(start_freq, stop_freq, n_sweep)
freq_incr = (stop_freq - start_freq) / (n_sweep - 1)

# Hardware init (once)
awg_channel = init_awg(relative_start_freq, osc=osc)
tt = init_time_tagger()

# Storage: raw counts only
all_counts = []

for pl_idx, pl_ns in enumerate(pulse_lengths_ns):
    pulse_length = ns_to_samples(pl_ns)
    modulation_freq = 1e9 / (2 * pl_ns)
    t_sweep = 2 * n_sweep * n_meas * pl_ns / 1e9
    print(f"\n[{pl_idx+1}/{n_pulse_lengths}] "
          f"pulse_length = {pl_ns/1e6:.2f} ms, "
          f"mod_freq = {modulation_freq:.2f} Hz, "
          f"sweep duration ≈ {t_sweep:.1f}s")

    configure_sweep(
        awg_channel, pulse_length, meas_delay,
        relative_start_freq, freq_incr,
        n_sweep, n_meas, osc=osc,
    )
    counts = run_sweep(awg_channel, tt, n_sweep, n_meas, timeout=t_sweep * 1.5)
    all_counts.append(counts)


max_slopes, freqs_at_max = max_slope_per_pulse_length(all_counts, freq)

all_std_counts = []
 
for pl_idx, (pl_ns, f_at_max) in enumerate(zip(pulse_lengths_ns, freqs_at_max)):
    pulse_length = ns_to_samples(pl_ns)
    rel_f        = f_at_max - CENTER_FREQ
    t_std        = 2 * 1 * n_std_meas * pl_ns / 1e9
    print(f"\n[std {pl_idx+1}/{n_pulse_lengths}] "
          f"pulse_length = {pl_ns/1e6:.2f} ms, "
          f"freq = {f_at_max/1e9:.6f} GHz, "
          f"duration ≈ {t_std:.1f}s")
 
    configure_sweep(
        awg_channel, pulse_length, meas_delay,
        rel_f, 0.0,
        n_sweep=1, n_meas=n_std_meas, osc=osc,
    )
    counts = run_sweep(awg_channel, tt, n_sweep=1, n_meas=n_std_meas, timeout=t_std * 1.5)
    all_std_counts.append(counts)
 
stds = std_per_pulse_length(all_std_counts)
T_shot = 2 * pulse_lengths_ns * 1e-9          # seconds, per pulse length
sensitivity = stds / max_slopes_fm * np.sqrt(T_shot)   
 
# === Save ====================================================================
np.savez(
    f"../persist/cw_am_sweep_mod_sweep/{start_date.isoformat().replace(':', '.')}.npz",
    max_slopes=max_slopes,
    freqs_at_max=freqs_at_max,
    stds=stds,
    sensitivity=sensitivity,
    pulse_lengths_ns=pulse_lengths_ns,
    params=params,
)
 
# === Plots ===================================================================
plt.plot(pulse_lengths_ns / 1e6, max_slopes, 'o-')
plt.xlabel('pulse length [ms]')
plt.ylabel(r'max $|d\,\mathrm{am\_counts}/df|$')
plt.show()
 
plt.plot(pulse_lengths_ns / 1e6, freqs_at_max, 'o-')
plt.xlabel('pulse length [ms]')
plt.ylabel('frequency at max slope [Hz]')
plt.show()
 
plt.plot(pulse_lengths_ns / 1e6, sensitivity, 'o-')
plt.xlabel('pulse length [ms]')
plt.ylabel(r'sensitivity $\sigma_\mathrm{AM}/\max|d\,\mathrm{am}/df|$ [Hz]')
plt.show()
