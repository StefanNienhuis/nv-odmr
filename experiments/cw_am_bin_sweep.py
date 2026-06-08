from datetime import datetime, timedelta
import numpy as np
import matplotlib.pyplot as plt
from util import cw_am_bin, set_mw

start_date = datetime.now()

# Parameters
modulation_freq = 10
modulation_freq = cw_am_bin.round_frequency(modulation_freq)

osc             = 0        # Oscillator being swept
start_freq      = 2.855e9   # Sweep start frequency (Hz)
stop_freq       = 2.885e9   # Sweep stop frequency (Hz)
n_sweep         = 201      # Number of sweep steps
meas_time       = 1        # Time to measure for at each frequency
lock_in_mode    = 'corr'

n_chunks        = 2         # Split run into n chunks, to avoid memory issues

n_meas = int(round(meas_time * modulation_freq))

# Parameters stored in output file
params = {
    "modulation_freq": modulation_freq,
    "start_freq": start_freq,
    "stop_freq": stop_freq,
    "n_sweep": n_sweep,
    "meas_time": meas_time,
    'lock_in_mode': lock_in_mode
}

# Calculate pulse length from modulation frequency
period_ns = 1e9 / modulation_freq
pulse_length_ns = period_ns / 2

expected_duration = n_sweep * meas_time
print(f"Expected duration: {expected_duration}s")
print(f"Finished at: {(datetime.now() + timedelta(seconds=expected_duration)).time()}")

# Split run so doesn't memory limit
f = np.linspace(start_freq, stop_freq, n_sweep)
f_chunked = np.array_split(f, n_chunks)

freq = np.array([])
am_signals = np.array([])

for i, f_chunk in enumerate(f_chunked):
    print(f"[{i+1}/{n_chunks}] Sweeping {f_chunk[0]/1e9:.4f} - {f_chunk[-1]/1e9:.4f} GHz")
    freq_chunk, am_signals_chunk = cw_am_bin.perform_sweep(modulation_freq, osc, f_chunk[0], f_chunk[-1], len(f_chunk), n_meas, lock_in_mode=lock_in_mode)
    
    freq = np.concatenate([freq, freq_chunk])
    am_signals = np.concatenate([am_signals, am_signals_chunk])

# summed = summed - np.mean(summed)
# summed = summed / np.max(np.abs(summed))

# plt.plot(summed)
# plt.plot(ref)
# plt.show()

set_mw.set_steady()

np.savez(f'../data/cw_am_bin_sweep/{start_date.isoformat().replace(":", ".")}.npz', data=am_signals, params=params)
    
plt.plot(freq, am_signals)
plt.show()