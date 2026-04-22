# TODO:
#  - AWG server parameters correct?
#  - AWG channel number and power
#  - Time tagger channel numbers, trigger level
#  - Appropriate pulse length

import time
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from zhinst.toolkit import Session, Sequence
from TimeTagger import createTimeTagger, CountBetweenMarkers

from experiments.load_sequence import load_sequence

start_date = datetime.now()

# Device parameters
AWG_SERVER_HOST = 'localhost'
AWG_SERVER_PORT = 8004
AWG_DEVICE = 'DEV12120'
AWG_CHANNEL = 0

TT_CLICK_CHANNEL = 0
TT_MARKER_CHANNEL = 0

# Parameters
pulse_length = 3000     # Pulse duration (ps)
meas_delay   = 1500     # Delay before measuring (ps)
osc          = 0        # Oscillator being swept
start_freq   = 2.84e9   # Sweep start frequency (Hz)
stop_freq    = 2.90e9   # Sweep stop frequency (Hz)
n_sweep      = 401      # Number of sweep steps
n_meas       = 1        # Number of measurements at each frequency

center_freq = 2.87e9
relative_start_freq = start_freq - center_freq
relative_stop_freq = stop_freq - center_freq

freq = np.linspace(start_freq, stop_freq, n_sweep)
freq_incr = (stop_freq - start_freq) / (n_sweep - 1)

# Arbitrary Waveform Generator initialization
awg_session = Session(AWG_SERVER_HOST, AWG_SERVER_PORT)
awg_device = awg_session.connect_device(AWG_DEVICE)

awg_device.check_compatibility()

awg_channel = awg_device.sgchannels[AWG_CHANNEL]

awg_channel.configure_channel(
    enable=True,
    output_range=0,
    center_frequency=center_freq,
    rf_path=True
)

# Time Tagger initialization
tt = createTimeTagger()

tt.setTriggerLevel(TT_CLICK_CHANNEL, 0.5)
tt.setTriggerLevel(TT_MARKER_CHANNEL, 0.5)

cmb = CountBetweenMarkers(tt, TT_CLICK_CHANNEL, TT_MARKER_CHANNEL, -TT_MARKER_CHANNEL, n_sweep * n_meas)

# Load AWG sequence
sequence = load_sequence("../awg_sequences/cw_sweep.c")
sequence.constants = {
    'PULSE_LENGTH': pulse_length,
    'MEAS_DELAY': meas_delay,
    'OSC': osc,
    'START_FREQ': start_freq,
    'FREQ_INCR': freq_incr,
    'N_SWEEP': n_sweep,
    'N_MEAS': n_meas
}

awg_channel.awg.load_sequencer_program(sequence)
awg_channel.awg.wait_done()

# Start time tagger and AWG sequence
cmb.start()
tt.sync()

awg_channel.awg.enable_sequencer(single=True)
awg_channel.awg.wait_done()

while not cbm.ready():
    time.sleep(0.2)

counts = cbm.getData()
counts = np.array(counts)

counts = counts.reshape((n_sweep, n_meas))

np.save(f'../data/cw_sweep/{start_date.isoformat()}.npy', counts)

mean_counts = counts.mean(axis=1)
mean_counts_norm = mean_counts / np.max(mean_counts)

plt.plot(freq, mean_counts_norm)
plt.show()