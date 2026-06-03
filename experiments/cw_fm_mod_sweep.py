from datetime import datetime, timedelta
import numpy as np
import matplotlib.pyplot as plt
from util import cw_fm

start_date = datetime.now()

# Parameters
meas_delay_ns   = 1e3      # Delay before measuring (ns)
osc1            = 0        # First oscillator being swept
osc2            = 1        # Second oscillator being swept
start_freq      = 2.85e9   # Sweep start frequency (Hz)
stop_freq       = 2.89e9   # Sweep stop frequency (Hz)
mod_depth       = 3e6      # FM modulation depth (Hz)
n_sweep         = 101      # Number of sweep steps
meas_time       = 1        # Time to measure for at each frequency

modulation_freqs = np.round(np.logspace(0, 5, 20))
#modulation_freqs = np.round(np.logspace(1, 2, 2)).astype(int)

freq_dev = mod_depth / 2

# Parameters stored in output file
params = {
    "modulation_freqs": modulation_freqs,
    "meas_delay_ns": meas_delay_ns,
    "start_freq": start_freq,
    "stop_freq": stop_freq,
    "mod_depth": mod_depth,
    "n_sweep": n_sweep,
    "meas_time": meas_time,
}

expected_duration = n_sweep * meas_time * len(modulation_freqs)
print(f"Expected duration: {expected_duration}s")
print(f"Finished at: {(datetime.now() + timedelta(seconds=expected_duration)).time()}")

fm_counts_per_modulation_freq = []

print()

for i, modulation_freq in enumerate(modulation_freqs):
    print(f'[{i + 1}/{len(modulation_freqs)}] Sweeping at {modulation_freq} Hz modulation...')
    
    n_meas = int(round(modulation_freq * meas_time))
    if n_meas != round(n_meas):
        print(f"Warning: number of measurements is rounded: {n_meas} instead of {modulation_freq * meas_time}")
    
    freq, sweep_counts = cw_fm.perform_sweep(modulation_freq, freq_dev, osc1, osc2, start_freq, stop_freq, n_sweep, n_meas)
    
    low_counts = sweep_counts[:, :, 0]
    high_counts = sweep_counts[:, :, 1]

    mean_low_counts = np.mean(low_counts, axis=1)
    mean_high_counts = np.mean(high_counts, axis=1)

    fm_counts = (mean_high_counts - mean_low_counts) / (mean_high_counts + mean_low_counts)

    fm_counts_per_modulation_freq.append(fm_counts)
    
    plt.plot(freq, fm_counts, label=f'{modulation_freq} Hz')

fm_counts_per_modulation_freq = np.array(fm_counts_per_modulation_freq)
np.savez(f'../data/cw_fm_mod_sweep/{start_date.isoformat().replace(":", ".")}.npz', data=fm_counts_per_modulation_freq, params=params)

plt.legend()
plt.show()
