from datetime import datetime, timedelta
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

from util.cw_am import (
    CENTER_FREQ,
    ns_to_samples,
    init_awg,
    init_time_tagger,
    configure_sweep,
    run_sweep,
)

start_date = datetime.now()

# Parameters
modulation_freq = 5        # AM modulation frequency (Hz)
meas_delay_ns   = 20e3     # Delay before measuring (ns)
osc             = 0        # Oscillator being swept
start_freq      = 2.84e9   # Sweep start frequency (Hz)
stop_freq       = 2.90e9   # Sweep stop frequency (Hz)
n_sweep         = 401      # Number of sweep steps
n_meas          = 5        # Number of measurements at each frequency

period_ns       = 1e9 / modulation_freq
pulse_length_ns = period_ns / 2

params = {
    "modulation_freq": modulation_freq,
    "meas_delay_ns": meas_delay_ns,
    "start_freq": start_freq,
    "stop_freq": stop_freq,
    "n_sweep": n_sweep,
    "n_meas": n_meas,
}

t_sweep = n_sweep * n_meas * period_ns / 1e9
print(f"Expected duration: {t_sweep:.1f}s")
print(f"Finished at: {(datetime.now() + timedelta(seconds=t_sweep)).time()}")

pulse_length = ns_to_samples(pulse_length_ns)
meas_delay   = ns_to_samples(meas_delay_ns)

relative_start_freq = start_freq - CENTER_FREQ
freq = np.linspace(start_freq, stop_freq, n_sweep)
freq_incr = (stop_freq - start_freq) / (n_sweep - 1)

awg_channel = init_awg(relative_start_freq, osc=osc)
tt = init_time_tagger()

configure_sweep(
    awg_channel, pulse_length, meas_delay,
    relative_start_freq, freq_incr,
    n_sweep, n_meas, osc=osc,
)
counts = run_sweep(awg_channel, tt, n_sweep, n_meas, timeout=t_sweep * 1.5)

mean_active   = counts[:, :, 0].mean(axis=1)   # MW ON
mean_inactive = counts[:, :, 1].mean(axis=1)   # MW OFF
am_counts = (mean_inactive - mean_active) / mean_inactive


np.savez(f'../persist/cw_am_sweep/{start_date.isoformat().replace(':', '.')}.npy', data=counts, params=params)

plt.plot(freq, mean_active_counts, label='on')
plt.plot(freq, mean_inactive_counts, label='off')
plt.legend()
plt.show()

plt.plot(freq, am_counts, label='on')
plt.show()