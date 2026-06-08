from datetime import datetime, timedelta
import numpy as np
import matplotlib.pyplot as plt
from util import cw_fm_bin, set_mw

start_date = datetime.now()

# Parameters
osc1            = 0        # First oscillator being swept
osc2            = 1        # Second oscillator being swept
start_freq      = 2.855e9   # Sweep start frequency (Hz)
stop_freq       = 2.885e9   # Sweep stop frequency (Hz)
mod_depth       = 3e6      # FM modulation depth (Hz)
n_sweep         = 201      # Number of sweep steps
meas_time       = 1        # Time to measure for at each frequency
lock_in_mode    = 'corr'

n_chunks = 2

freq_dev = mod_depth / 2

modulation_freqs = np.logspace(0, 5, 10)
modulation_freqs = cw_fm_bin.round_frequency(modulation_freqs)

# Parameters stored in output file
params = {
    "modulation_freqs": modulation_freqs,
    "start_freq": start_freq,
    "stop_freq": stop_freq,
    "mod_depth": mod_depth,
    "n_sweep": n_sweep,
    "meas_time": meas_time,
    "lock_in_mode": lock_in_mode
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
    
    # Split run so doesn't memory limit
    f = np.linspace(start_freq, stop_freq, n_sweep)
    f_chunked = np.array_split(f, n_chunks)

    freq = np.array([])
    fm_signals = np.array([])
    
    for i, f_chunk in enumerate(f_chunked):
        print(f"[Chunk {i+1}/{n_chunks}] {f_chunk[0]/1e9:.4f} - {f_chunk[-1]/1e9:.4f} GHz")
        freq_chunk, fm_signals_chunk = cw_fm_bin.perform_sweep(modulation_freq, freq_dev, osc1, osc2, f_chunk[0], f_chunk[-1], len(f_chunk), n_meas, lock_in_mode=lock_in_mode)
        
        freq = np.concatenate([freq, freq_chunk])
        fm_signals = np.concatenate([fm_signals, fm_signals_chunk])
    
    counts_per_modulation_freq.append(fm_signals)
    
    plt.plot(freq, fm_signals, label=f'{modulation_freq} Hz')

set_mw.set_steady()

save_array = np.empty(len(counts_per_modulation_freq), object)
save_array[:] = counts_per_modulation_freq
np.savez(f'../data/cw_fm_bin_mod_sweep/{start_date.isoformat().replace(":", ".")}.npz', data=save_array, params=params)

plt.legend()
plt.show()
