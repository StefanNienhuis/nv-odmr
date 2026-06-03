import time
from datetime import datetime, timedelta
import numpy as np
import matplotlib.pyplot as plt
from zhinst.toolkit import Session, CommandTable
from TimeTagger import createTimeTaggerNetwork, CountBetweenMarkers
import pycobolt
from util.load_sequence import load_sequence
from util import cw_fm, set_mw

start_date = datetime.now()

# Device parameters
AWG_SERVER_HOST = 'localhost'
AWG_SERVER_PORT = 8004
AWG_DEVICE = 'DEV12120'
AWG_CHANNEL = 2
AWG_SAMPLE_RATE = 2e9

TT_CLICK_CHANNEL = 1
TT_MARKER_CHANNEL = 2

LASER_SN = '31977'
LASER_CURRENT = 57

# Parameters
modulation_freq = 5        # FM modulation frequency
meas_delay_ns   = 20e3     # Delay before measuring (ns)
osc1            = 0        # First oscillator being swept
osc2            = 1        # Second oscillator being swept
start_freq      = 2.84e9   # Sweep start frequency (Hz)
stop_freq       = 2.90e9   # Sweep stop frequency (Hz)
mod_depth       = 3e6      # FM modulation depth (Hz)
n_sweep         = 401      # Number of sweep steps
meas_time       = 1        # Time to measure for at each frequency

n_meas = meas_time * modulation_freq

# Parameters stored in output file
params = {
    "modulation_freq": modulation_freq,
    "meas_delay_ns": meas_delay_ns,
    "start_freq": start_freq,
    "stop_freq": stop_freq,
    "mod_depth": mod_depth,
    "n_sweep": n_sweep,
    "meas_time": meas_time,
}

# Calculate the +- frequency deviation from modulation depth
freq_dev = mod_depth / 2

# Calculate pulse length from modulation frequency
period_ns = 1e9 / modulation_freq
pulse_length_ns = period_ns / 2

expected_duration = n_sweep * n_meas * period_ns / 1e9
print(f"Expected duration: {expected_duration}s")
print(f"Finished at: {(datetime.now() + timedelta(seconds=expected_duration)).time()}")

freq, counts = cw_fm.perform_sweep(modulation_freq, freq_dev, meas_delay_ns, osc1, osc2, start_freq, stop_freq, n_sweep, n_meas)

set_mw.set_steady()

np.savez(f'../data/cw_fm_sweep/{start_date.isoformat().replace(":", ".")}.npz', data=counts, params=params)

low_counts = counts[:,:,0]
high_counts = counts[:,:,1]

mean_low_counts = np.mean(low_counts, axis=1)
mean_high_counts = np.mean(high_counts, axis=1)

fm_counts = (mean_high_counts - mean_low_counts) / (mean_high_counts + mean_low_counts)

plt.plot(freq, mean_low_counts)
plt.plot(freq, mean_high_counts)
plt.show()

plt.plot(freq, fm_counts)
plt.show()