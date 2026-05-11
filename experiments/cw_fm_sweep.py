from datetime import datetime, timedelta
import numpy as np
import matplotlib.pyplot as plt

from util.cw_fm import setup_awg, setup_time_tagger, run_fm_sweep

start_date = datetime.now()

# Parameters
modulation_freq = 5         # FM modulation frequency (Hz)
meas_delay_ns   = 20e3      # Delay before measuring (ns)
start_freq      = 2.84e9    # Sweep start frequency (Hz)
stop_freq       = 2.90e9    # Sweep stop frequency (Hz)
mod_depth       = 3e6       # FM modulation depth (Hz)
n_sweep         = 401       # Number of sweep steps
n_meas          = 5         # Number of measurements at each frequency

# Parameters stored in output file
params = {
    "modulation_freq": modulation_freq,
    "meas_delay_ns":   meas_delay_ns,
    "start_freq":      start_freq,
    "stop_freq":       stop_freq,
    "mod_depth":       mod_depth,
    "n_sweep":         n_sweep,
    "n_meas":          n_meas,
}

# Calculate pulse length from modulation frequency
period_ns       = 1e9 / modulation_freq
pulse_length_ns = period_ns / 2

expected_duration = n_sweep * n_meas * period_ns / 1e9
print(f"Expected duration: {expected_duration}s")
print(f"Finished at: {(datetime.now() + timedelta(seconds=expected_duration)).time()}")

freq = np.linspace(start_freq, stop_freq, n_sweep)

# Hardware setup
awg_channel = setup_awg(start_freq=start_freq, mod_depth=mod_depth)
tt          = setup_time_tagger()

# Run sweep
counts = run_fm_sweep(
    awg_channel, tt,
    pulse_length_ns=pulse_length_ns,
    start_freq=start_freq, stop_freq=stop_freq,
    mod_depth=mod_depth,
    n_sweep=n_sweep, n_meas=n_meas,
    meas_delay_ns=meas_delay_ns,
)
np.savez(
    f"../persist/cw_fm_sweep/{start_date.isoformat().replace(':', '.')}.npy",
    data=counts,
    params=params,
)

low_counts  = counts[:, :, 0]
high_counts = counts[:, :, 1]

mean_low_counts  = np.mean(low_counts,  axis=1)
mean_high_counts = np.mean(high_counts, axis=1)

fm_counts = (mean_high_counts - mean_low_counts) / (mean_high_counts + mean_low_counts)

plt.plot(freq, mean_low_counts)
plt.plot(freq, mean_high_counts)
plt.show()

plt.plot(freq, fm_counts)
plt.show()