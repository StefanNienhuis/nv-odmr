import time
from datetime import datetime, timedelta
import numpy as np
import matplotlib.pyplot as plt
from zhinst.toolkit import Session, CommandTable
from TimeTagger import CountBetweenMarkers, createTimeTaggerNetwork, CHANNEL_UNUSED
import pycobolt
from util import cw, set_mw

start_date = datetime.now()

# Parameters - should match the ones used in sweep
pulse_length_ns = 100e6     # Pulse duration (ns) - WARNING: overflow may occur - 1 second is too much
meas_delay_ns   = 50e3      # Delay before measuring (ns)
osc             = 0         # Oscillator being used
n_meas          = 10
n_std           = 50

# Obtain from cw_sweep
drive_freq = 2867750000.0
slope = 4.43097649969663e-08
normalizer = 292518

# Parameters stored in output file
params = {
    "pulse_length_ns": pulse_length_ns,
    "meas_delay_ns": meas_delay_ns,
    "drive_freq": drive_freq,
    "n_meas": n_meas,
    "n_std": n_std,
    # Included for further processing
    "slope": slope,
    "normalizer": normalizer
}

expected_duration = n_std * pulse_length_ns / 1e9
print(f"Expected duration: {expected_duration}s")
print(f"Finished at: {(datetime.now() + timedelta(seconds=expected_duration)).time()}")

counts = []

for n in range(n_std):
    freq, sweep_counts = cw.perform_sweep(pulse_length_ns, meas_delay_ns, osc, drive_freq, drive_freq, 1, n_meas, show_progress=False)
    print(sweep_counts)
    mean_counts = np.mean(sweep_counts[0])
    mean_counts = mean_counts / normalizer

    counts.append(mean_counts)

set_mw.set_steady()

counts = np.array(counts)
np.savez(f'../data/cw_std/{start_date.isoformat().replace(":", ".")}.npz', data=counts, params=params)

print(f"Mean: {counts.mean()}")
print(f"Stddev: {counts.std()}")

plt.plot(counts)
plt.show()