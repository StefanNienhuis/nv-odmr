from datetime import datetime, timedelta
import numpy as np
import matplotlib.pyplot as plt
from util import cw_am_bin

start_date = datetime.now()

# Parameters
modulation_freq = 10e3
modulation_freq = cw_am_bin.round_frequency(modulation_freq)

osc             = 0        # Oscillator being swept
start_freq      = 2.855e9   # Sweep start frequency (Hz)
stop_freq       = 2.885e9   # Sweep stop frequency (Hz)
n_sweep         = 101      # Number of sweep steps
meas_time       = 1        # Time to measure for at each frequency

n_meas = int(round(meas_time * modulation_freq))

# Parameters stored in output file
params = {
    "modulation_freq": modulation_freq,
    "start_freq": start_freq,
    "stop_freq": stop_freq,
    "n_sweep": n_sweep,
    "meas_time": meas_time,
}

# Calculate pulse length from modulation frequency
period_ns = 1e9 / modulation_freq
pulse_length_ns = period_ns / 2

expected_duration = n_sweep * meas_time
print(f"Expected duration: {expected_duration}s")
print(f"Finished at: {(datetime.now() + timedelta(seconds=expected_duration)).time()}")

freq, am_signals = cw_am_bin.perform_sweep(modulation_freq, osc, start_freq, stop_freq, n_sweep, n_meas)

np.savez(f'../data/cw_am_bin_sweep/{start_date.isoformat().replace(":", ".")}.npz', data=am_signals, params=params)
    
plt.plot(freq, am_signals)
plt.show()