from datetime import datetime, timedelta
import numpy as np
import matplotlib.pyplot as plt

from util.cw_fm import setup_awg, setup_time_tagger, run_fm_sweep

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

# Save
params = {
    "start_pulse_length_ns":start_pulse_length_ns,
    "stop_pulse_length_ns":stop_pulse_length_ns,
    "n_pulse_lengths":n_pulse_lengths,
    "meas_delay_ns":   meas_delay_ns,
    "start_freq":      start_freq,
    "stop_freq":       stop_freq,
    "mod_depth":       mod_depth,
    "n_sweep":         n_sweep,
    "n_meas":          n_meas,
}

pulse_lengths_ns = np.logspace(
    np.log10(start_pulse_length_ns),
    np.log10(stop_pulse_length_ns),
    n_pulse_lengths,
)

expected_duration = sum(2 * n_sweep * n_meas * pl / 1e9 for pl in pulse_lengths_ns)
print(f"Pulse lengths (ms): {pulse_lengths_ns / 1e6}")
print(f"Modulation freq (Hz): {1e9 / (2 * pulse_lengths_ns)}")
print(f"Expected total duration: {expected_duration:.1f}s")
print(f"Finished at: {(datetime.now() + timedelta(seconds=expected_duration)).time()}")

freq = np.linspace(start_freq, stop_freq, n_sweep)

# Hardware setup (once)
awg_channel = setup_awg(start_freq=start_freq, mod_depth=mod_depth)
tt          = setup_time_tagger()

# Storage: raw counts only
all_counts = []

for pl_idx, pl_ns in enumerate(pulse_lengths_ns):
    modulation_freq = 1e9 / (2 * pl_ns)
    t_sweep = 2 * n_sweep * n_meas * pl_ns / 1e9
    print(f"\n[{pl_idx+1}/{n_pulse_lengths}] "
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

np.savez(
    f"../persist/cw_fm_sweep_mod_sweep/{start_date.isoformat().replace(':', '.')}.npz",
    data=np.array(all_counts),
    params=params,
)

# Aggregated view: heatmap of normalized FM signal vs (frequency, pulse length)
all_counts_arr = np.array(all_counts)
mean_low  = all_counts_arr[:, :, :, 0].mean(axis=2)
mean_high = all_counts_arr[:, :, :, 1].mean(axis=2)
fm_signal = (mean_high - mean_low) / (mean_high + mean_low)  # (n_pulse_lengths, n_sweep)
# Per-row normalization so line shapes are comparable across pulse lengths
fm_norm = fm_signal / np.max(np.abs(fm_signal), axis=1, keepdims=True)
fig, ax = plt.subplots(figsize=(8, 5))
mesh = ax.pcolormesh(
    freq / 1e9, pulse_lengths_ns / 1e6, fm_norm,
    shading='nearest', cmap='RdBu_r', vmin=-1, vmax=1,
)
ax.set_yscale('log')  # pulse_lengths_ns is logspaced
ax.set_xlabel('MW frequency (GHz)')
ax.set_ylabel('Pulse length (ms)')
ax.set_title('Normalized FM signal (H − L) / (H + L)')
fig.colorbar(mesh, ax=ax, label='Normalized')
plt.tight_layout()
plt.show()