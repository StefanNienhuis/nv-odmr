from datetime import datetime, timedelta
import numpy as np
import matplotlib.pyplot as plt
from util import cw_am, set_mw

start_date = datetime.now()

# Parameters
meas_delay_ns   = 1e3      # Delay before measuring (ns)
osc             = 0        # Oscillator being swept
start_freq      = 2.855e9   # Sweep start frequency (Hz)
stop_freq       = 2.885e9   # Sweep stop frequency (Hz)
n_sweep         = 201      # Number of sweep steps
meas_time       = 1        # Time to measure for at each frequency

modulation_freqs = np.round(np.logspace(0, 5, 10))
#modulation_freqs = np.round(np.logspace(1, 2, 2)).astype(int)

# Parameters stored in output file
params = {
    "modulation_freqs": modulation_freqs,
    "meas_delay_ns": meas_delay_ns,
    "start_freq": start_freq,
    "stop_freq": stop_freq,
    "n_sweep": n_sweep,
    "meas_time": meas_time,
}

expected_duration = n_sweep * meas_time * len(modulation_freqs)
print(f"Expected duration: {expected_duration}s")
print(f"Finished at: {(datetime.now() + timedelta(seconds=expected_duration)).time()}")

am_counts_per_modulation_freq = []

print()

for i, modulation_freq in enumerate(modulation_freqs):
    print(f'[{i + 1}/{len(modulation_freqs)}] Sweeping at {modulation_freq} Hz modulation...')
    
    n_meas = int(round(modulation_freq * meas_time))
    if n_meas != round(n_meas):
        print(f"Warning: number of measurements is rounded: {n_meas} instead of {modulation_freq * meas_time}")
    
    freq, sweep_counts = cw_am.perform_sweep(modulation_freq, meas_delay_ns, osc, start_freq, stop_freq, n_sweep, n_meas)
    
    active_counts = sweep_counts[:, :, 0]
    inactive_counts = sweep_counts[:, :, 1]

    mean_active_counts = np.mean(active_counts, axis=1)
    mean_inactive_counts = np.mean(inactive_counts, axis=1)

    am_counts = (mean_inactive_counts - mean_active_counts) / mean_inactive_counts
    
    am_counts_per_modulation_freq.append(am_counts)
    
    plt.plot(freq, am_counts, label=f'{modulation_freq} Hz')

set_mw.set_steady()

am_counts_per_modulation_freq = np.array(am_counts_per_modulation_freq)
np.savez(f'../data/cw_am_mod_sweep/{start_date.isoformat().replace(":", ".")}.npz', data=am_counts_per_modulation_freq, params=params)

plt.legend()
plt.show()