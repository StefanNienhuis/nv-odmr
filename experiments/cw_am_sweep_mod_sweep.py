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

params = {
    "start_pulse_length_ns": start_pulse_length_ns,
    "stop_pulse_length_ns":stop_pulse_length_ns,
    "n_pulse_lengths":n_pulse_lengths,
    "meas_delay_ns": meas_delay_ns,
    "start_freq": start_freq,
    "stop_freq": stop_freq,
    "n_sweep": n_sweep,
    "n_meas": n_meas,
}

pulse_lengths_ns = np.logspace(
    np.log10(start_pulse_length_ns),
    np.log10(stop_pulse_length_ns),
    n_pulse_lengths,
)

expected_duration = sum(2 * n_sweep * n_meas * pl / 1e9 for pl in pulse_lengths_ns)
print(f"Pulse lengths (ms): {pulse_lengths_ns/1e6}")
print(f"Modulation freq (Hz): {1e9/(2*pulse_lengths_ns)}")
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

np.savez(f'../persist/cw_am_sweep_mod_sweep/{start_date.isoformat().replace(':', '.')}.npy', data=counts, params=params)

# Aggregated view: heatmap of normalized AM signal vs (frequency, pulse length)
all_counts_arr = np.array(all_counts)
mean_active   = all_counts_arr[:, :, :, 0].mean(axis=2)
mean_inactive = all_counts_arr[:, :, :, 1].mean(axis=2)
am_signal = mean_inactive - mean_active  # (n_pulse_lengths, n_sweep)

# Per-row normalization so line shapes are comparable across pulse lengths
am_norm = am_signal / np.max(np.abs(am_signal), axis=1, keepdims=True)

fig, ax = plt.subplots(figsize=(8, 5))
mesh = ax.pcolormesh(
    freq / 1e9, pulse_lengths_ns / 1e6, am_norm,
    shading='nearest', cmap='RdBu_r', vmin=-1, vmax=1,
)
ax.set_yscale('log')  # pulse_lengths_ns is logspaced
ax.set_xlabel('MW frequency (GHz)')
ax.set_ylabel('Pulse length (ms)')
ax.set_title('Normalized AM signal (inactive − active)')
fig.colorbar(mesh, ax=ax, label='Normalized')
plt.tight_layout()
plt.show()