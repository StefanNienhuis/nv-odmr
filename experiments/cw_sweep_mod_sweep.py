from datetime import datetime, timedelta
import numpy as np
import matplotlib.pyplot as plt

from util.cw import (
    CENTER_FREQ,
    init_awg,
    init_time_tagger,
    make_count_between_markers,
    ns_to_samples,
    run_acquisition,
    upload_command_table,
)
from util.load_sequence import load_sequence


start_date = datetime.now()

# Frequency sweep parameters
start_freq      = 2.84e9
stop_freq       = 2.90e9
n_sweep         = 401
n_meas          = 5
osc             = 0
meas_delay_ns   = 50e3

# Pulse length sweep parameters
start_pulse_length_ns = 5e6      # 5 ms
stop_pulse_length_ns  = 200e6    # 200 ms
n_pulse_lengths       = 5


pulse_lengths_ns = np.logspace(
    np.log10(start_pulse_length_ns),
    np.log10(stop_pulse_length_ns),
    n_pulse_lengths,
)

# Save
params = {
    "start_pulse_length_ns":start_pulse_length_ns,
    "stop_pulse_length_ns":stop_pulse_length_ns,
    "n_pulse_lengths":n_pulse_lengths,
    "meas_delay_ns": meas_delay_ns,
    "start_freq": start_freq,
    "stop_freq": stop_freq,
    "n_sweep": n_sweep,
    "n_meas": n_meas,
}

expected_duration = sum(n_sweep * n_meas * pl / 1e9 for pl in pulse_lengths_ns)
print(f"Pulse lengths (ms): {pulse_lengths_ns/1e6}")
print(f"Expected total duration: {expected_duration:.1f}s")
print(f"Finished at: {(datetime.now() + timedelta(seconds=expected_duration)).time()}")

meas_delay = ns_to_samples(meas_delay_ns)

relative_start_freq = start_freq - CENTER_FREQ
freq = np.linspace(start_freq, stop_freq, n_sweep)
freq_incr = (stop_freq - start_freq) / (n_sweep - 1)

# Hardware setup (once)
awg_channel = init_awg(osc_frequency=relative_start_freq, osc=osc)
tt = init_time_tagger()

# Storage: raw counts only
all_counts = []

for pl_idx, pl_ns in enumerate(pulse_lengths_ns):
    pulse_length = ns_to_samples(pl_ns)
    t_sweep = n_sweep * n_meas * pl_ns / 1e9
    print(f"\n[{pl_idx+1}/{n_pulse_lengths}] pulse_length = {pl_ns/1e6:.2f} ms "
          f"(sweep duration ≈ {t_sweep:.1f}s)")

    sequence = load_sequence("../awg_sequences/cw_sweep.c")
    sequence.constants = {
        'PULSE_LENGTH': pulse_length,
        'MEAS_DELAY': meas_delay,
        'OSC': osc,
        'START_FREQ': relative_start_freq,
        'FREQ_INCR': freq_incr,
        'N_SWEEP': n_sweep,
        'N_MEAS': n_meas,
    }
    awg_channel.awg.load_sequencer_program(sequence)
    awg_channel.awg.wait_done()

    upload_command_table(awg_channel, pulse_length, meas_delay)

    cbm = make_count_between_markers(tt, n_sweep)
    counts = run_acquisition(awg_channel, tt, cbm, timeout=t_sweep*1.5)
    all_counts.append(counts)

np.savez(
    f"../persist/cw_mod_sweep/{start_date.isoformat().replace(':', '.')}.npz",
    data=np.array(all_counts),
    params=params,
)

# Sanity-check plot: one curve per pulse_length, normalized
all_counts_arr = np.array(all_counts)  # (n_pulse_lengths, n_sweep)

# Dip signal = baseline (per-row max) − counts.
# Per-row normalized so line shapes are comparable across pulse lengths.
dip_signal = all_counts_arr.max(axis=1, keepdims=True) - all_counts_arr
dip_norm = dip_signal / np.max(np.abs(dip_signal), axis=1, keepdims=True)

fig, ax = plt.subplots(figsize=(8, 5))
mesh = ax.pcolormesh(
    freq / 1e9,
    pulse_lengths_ns / 1e6,
    dip_norm,
    shading='nearest',
    cmap='viridis',
    vmin=0,
    vmax=1,
)
ax.set_yscale('log')  # pulse_lengths_ns is logspaced
ax.set_xlabel('MW frequency (GHz)')
ax.set_ylabel('Pulse length (ms)')
ax.set_title('Normalized CW dip signal (baseline − counts)')
fig.colorbar(mesh, ax=ax, label='Normalized')
plt.tight_layout()
plt.show()