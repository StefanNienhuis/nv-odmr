from datetime import datetime, timedelta
import numpy as np
import matplotlib.pyplot as plt

from util.cw import (
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

# Parameters - should match the ones used in sweep
pulse_length_ns = 100e6     # Pulse duration (ns)
meas_delay_ns   = 50e3      # Delay before measuring (ns)
osc             = 0         # Oscillator being used
freq            = 2.85e9    # The frequency to look at
n_meas          = 1000      # Number of measurements at this frequency

# Parameters stored in output file
params = {
    "pulse_length_ns": pulse_length_ns,
    "meas_delay_ns": meas_delay_ns,
    "freq": freq,
    "n_meas": n_meas,
}

expected_duration = n_meas * pulse_length_ns / 1e9
print(f"Expected duration: {expected_duration}s")
print(f"Finished at: {(datetime.now() + timedelta(seconds=expected_duration)).time()}")

pulse_length = ns_to_samples(pulse_length_ns)
meas_delay = ns_to_samples(meas_delay_ns)

relative_freq = freq - CENTER_FREQ

# Hardware setup
awg_channel = init_awg(osc_frequency=relative_freq, osc=osc)
tt = init_time_tagger()
cbm = make_count_between_markers(tt, n_meas)

# Load AWG sequence
sequence = load_sequence("../awg_sequences/cw_sweep.c")
sequence.constants = {
    'PULSE_LENGTH': pulse_length,
    'MEAS_DELAY': meas_delay,
    'OSC': osc,
    'N_MEAS': n_meas,
}
awg_channel.awg.load_sequencer_program(sequence)
awg_channel.awg.wait_done()

upload_command_table(awg_channel, pulse_length, meas_delay)

counts = run_acquisition(awg_channel, tt, cbm, timeout=expected_duration*1.5)

np.savez(f'../persist/cw_single/{start_date.isoformat().replace(":", ".")}.npy', data=counts, params=params)

print(f"Mean: {counts.mean()}")
print(f"Stddev: {counts.std()}")

plt.plot(counts)
plt.show()