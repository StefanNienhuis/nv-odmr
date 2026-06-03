import time
from datetime import datetime, timedelta
import numpy as np
import matplotlib.pyplot as plt
from zhinst.toolkit import Session, CommandTable
from TimeTagger import CountBetweenMarkers, createTimeTaggerNetwork, CHANNEL_UNUSED
import pycobolt
from util.load_sequence import load_sequence
from util import cw, set_mw
from tqdm import tqdm

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
pulse_length_ns = 100e6    # Pulse duration (ns) - WARNING: overflow may occur - 1 second is too much
meas_delay_ns   = 50e3     # Delay before measuring (ns)
osc             = 0        # Oscillator being swept
start_freq      = 2.855e9   # Sweep start frequency (Hz)
stop_freq       = 2.885e9   # Sweep stop frequency (Hz)
n_sweep         = 201      # Number of sweep steps
n_meas          = 10        # Number of measurements at each frequency

# Parameters stored in output file
params = {
    "pulse_length_ns": pulse_length_ns,
    "meas_delay_ns": meas_delay_ns,
    "start_freq": start_freq,
    "stop_freq": stop_freq,
    "n_sweep": n_sweep,
    "n_meas": n_meas,
}

expected_duration = n_sweep * n_meas * pulse_length_ns / 1e9
print(f"Expected duration: {expected_duration}s")
print(f"Finished at: {(datetime.now() + timedelta(seconds=expected_duration)).time()}")

freq, counts = cw.perform_sweep(pulse_length_ns, meas_delay_ns, osc, start_freq, stop_freq, n_sweep, n_meas)
set_mw.set_steady()

np.savez(f'../data/cw_sweep/{start_date.isoformat().replace(":", ".")}.npz', data=counts, params=params)

counts_norm = counts / np.max(counts)

plt.plot(freq, counts_norm)
plt.show()