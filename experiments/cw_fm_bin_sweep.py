from datetime import datetime, timedelta
import numpy as np
import matplotlib.pyplot as plt
from util import cw_fm_bin, set_mw

start_date = datetime.now()

# Parameters
modulation_freq = 100
modulation_freq = cw_fm_bin.round_frequency(modulation_freq)

osc1            = 0        # First oscillator being swept
osc2            = 1        # Second oscillator being swept
start_freq      = 2.855e9   # Sweep start frequency (Hz)
stop_freq       = 2.885e9   # Sweep stop frequency (Hz)
mod_depth       = 3e6      # FM modulation depth (Hz)
n_sweep         = 201      # Number of sweep steps
meas_time       = 1        # Time to measure for at each frequency

freq_dev = mod_depth / 2
n_meas = int(round(meas_time * modulation_freq))

# Parameters stored in output file
params = {
    "modulation_freq": modulation_freq,
    "start_freq": start_freq,
    "stop_freq": stop_freq,
    "mod_depth": mod_depth,
    "n_sweep": n_sweep,
    "meas_time": meas_time,
}

# Calculate pulse length from modulation frequency
period_ns = 1e9 / modulation_freq
pulse_length_ns = period_ns / 2

expected_duration = n_sweep * meas_time
print(f"Expected duration: {expected_duration}s")
print(f"Finished at: {(datetime.now() + timedelta(seconds=expected_duration)).time()}")

# Split run so doesn't memory limit
f = np.linspace(start_freq, stop_freq, n_sweep)
f_a = f[0:len(f)//2]
f_b = f[len(f)//2:]

freq_a, fm_signals_a = cw_fm_bin.perform_sweep(modulation_freq, freq_dev, osc1, osc2, f_a[0], f_a[-1], len(f_a), n_meas)
freq_b, fm_signals_b = cw_fm_bin.perform_sweep(modulation_freq, freq_dev, osc1, osc2, f_b[0], f_b[-1], len(f_b), n_meas)

freq = np.concatenate([freq_a, freq_b])
fm_signals = np.concatenate([fm_signals_a, fm_signals_b])

set_mw.set_steady()

np.savez(f'../data/cw_fm_bin_sweep/{start_date.isoformat().replace(":", ".")}.npz', data=fm_signals, params=params)
    
plt.plot(freq, fm_signals)
plt.show()