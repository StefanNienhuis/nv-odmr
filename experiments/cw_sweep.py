from datetime import datetime, timedelta

import numpy as np
import matplotlib.pyplot as plt

from util.experiment import (
    CENTER_FREQ,
    init_awg,
    init_time_tagger,
    make_count_between_markers,
    ns_to_samples,
    run_acquisition,
    upload_command_table,
)
from util.load_sequence import load_sequence


start_date = datetime.now()

# Parameters
pulse_length_ns = 100e6     # Pulse duration (ns)
meas_delay_ns   = 50e3      # Delay before measuring (ns)
osc             = 0         # Oscillator being swept
start_freq      = 2.84e9    # Sweep start frequency (Hz)
stop_freq       = 2.90e9    # Sweep stop frequency (Hz)
n_sweep         = 401       # Number of sweep steps
n_meas          = 5         # Number of measurements at each frequency

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

pulse_length = ns_to_samples(pulse_length_ns)
meas_delay = ns_to_samples(meas_delay_ns)

relative_start_freq = start_freq - CENTER_FREQ
freq = np.linspace(start_freq, stop_freq, n_sweep)
freq_incr = (stop_freq - start_freq) / (n_sweep - 1)

# Hardware setup
awg_channel = init_awg(osc_frequency=relative_start_freq, osc=osc)
tt = init_time_tagger()
cbm = make_count_between_markers(tt, n_sweep)

# Load AWG sequence
sequence = load_sequence("../awg_sequences/cw_sweep.c")
sequence.constants = {
    'PULSE_LENGTH': pulse_length,
    'MEAS_DELAY': meas_delay,
    'OSC': osc,
    'START_FREQ': relative_start_freq,
    'FREQ_INCR': freq_incr,
    'N_SWEEP': n_sweep,
    'N_MEAS': n_meas,
}
awg_channel.awg.load_sequencer_program(sequence)
awg_channel.awg.wait_done()

upload_command_table(awg_channel, pulse_length, meas_delay)

counts = run_acquisition(awg_channel, tt, cbm, timeout=expected_duration*1.5)

np.savez(f'../data/cw_sweep/{start_date.isoformat().replace(":", ".")}.npy', data=counts, params=params)

print(counts)
print(cbm.getBinWidths())

counts_norm = counts / np.max(counts)
plt.plot(freq, counts_norm)
plt.show()