from datetime import datetime, timedelta
import numpy as np
import matplotlib.pyplot as plt
from util import cw_am_bin, set_mw

start_date = datetime.now()

# Parameters
osc             = 0        # Oscillator being swept
start_freq      = 2.85e9   # Sweep start frequency (Hz)
stop_freq       = 2.89e9   # Sweep stop frequency (Hz)
n_sweep         = 101      # Number of sweep steps
meas_time       = 1        # Time to measure for at each frequency

modulation_freqs = np.logspace(0, 5, 20)
modulation_freqs = cw_am_bin.round_frequency(modulation_freqs)

# Parameters stored in output file
params = {
    "modulation_freqs": modulation_freqs,
    "start_freq": start_freq,
    "stop_freq": stop_freq,
    "n_sweep": n_sweep,
    "meas_time": meas_time,
}

expected_duration = n_sweep * meas_time * len(modulation_freqs)
print(f"Expected duration: {expected_duration}s")
print(f"Finished at: {(datetime.now() + timedelta(seconds=expected_duration)).time()}")

counts_per_modulation_freq = []

print()

for i, modulation_freq in enumerate(modulation_freqs):
    print(f'[{i + 1}/{len(modulation_freqs)}] Sweeping at {modulation_freq} Hz modulation...')
    
    n_meas = int(round(modulation_freq * meas_time))
    if n_meas != round(n_meas):
        print(f"Warning: number of measurements is rounded: {n_meas} instead of {modulation_freq * meas_time}")
    
    freq, am_counts = cw_am_bin.perform_sweep(modulation_freq, osc, start_freq, stop_freq, n_sweep, n_meas)
    counts_per_modulation_freq.append(am_counts)
    
    plt.plot(freq, am_counts, label=f'{modulation_freq} Hz')

set_mw.set_steady()

save_array = np.empty(len(counts_per_modulation_freq), object)
save_array[:] = counts_per_modulation_freq
np.savez(f'../data/cw_am_bin_mod_sweep/{start_date.isoformat().replace(":", ".")}.npz', data=save_array, params=params)

plt.legend()
plt.show()
