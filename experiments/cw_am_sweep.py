# TODO:
#  - AWG server parameters correct?
#  - AWG channel number and power
#  - Time tagger channel numbers, trigger level

import time
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from zhinst.toolkit import Session, CommandTable
from TimeTagger import createTimeTagger, CountBetweenMarkers
from util.load_sequence import load_sequence

start_date = datetime.now()

# Device parameters
AWG_SERVER_HOST = 'localhost'
AWG_SERVER_PORT = 8004
AWG_DEVICE = 'DEV12120'
AWG_CHANNEL = 0
AWG_SAMPLE_RATE = 2e9

TT_CLICK_CHANNEL = 0
TT_MARKER_CHANNEL = 1

# Parameters
modulation_freq = 5e3      # AM modulation frequency (Hz)
meas_delay_ns   = 1e3      # Delay before measuring (ns)
osc             = 0        # Oscillator being swept
start_freq      = 2.84e9   # Sweep start frequency (Hz)
stop_freq       = 2.90e9   # Sweep stop frequency (Hz)
n_sweep         = 401      # Number of sweep steps
n_meas          = 50       # Number of measurements at each frequency

# Calculate pulse length from modulation frequency
period_ns = 1e9 / modulation_freq
pulse_length_ns = period_ns / 2

# Convert ns -> samples
pulse_length = pulse_length_ns * AWG_SAMPLE_RATE / 1e9
meas_delay = meas_delay_ns * AWG_SAMPLE_RATE / 1e9

# Round counts to 16 - AWG zero pads otherwise
pulse_length = int(round(pulse_length / 16) * 16)
meas_delay = int(round(meas_delay / 16) * 16)

center_freq = 2.87e9
relative_start_freq = start_freq - center_freq

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

awg_channel.configure_sine_generation(
    enable=True,
    osc_index=osc,
    osc_frequency=relative_start_freq,
    gains=(0.0, 1.0, 1.0, 0.0),
    phase=0
)

# Time Tagger initialization
tt = createTimeTagger()

tt.setTriggerLevel(TT_CLICK_CHANNEL, 0.5)
tt.setTriggerLevel(TT_MARKER_CHANNEL, 0.5)

# Twice the number of samples since we get one with pulse and one without pulse (square AM modulation)
cbm = CountBetweenMarkers(tt, TT_CLICK_CHANNEL, TT_MARKER_CHANNEL, -TT_MARKER_CHANNEL, 2 * n_sweep * n_meas)

# Load AWG sequence
sequence = load_sequence("../awg_sequences/cw_am_sweep.c")
sequence.constants = {
    'PULSE_LENGTH': pulse_length,
    'MEAS_DELAY': meas_delay,
    'OSC': osc,
    'START_FREQ': relative_start_freq,
    'FREQ_INCR': freq_incr,
    'N_SWEEP': n_sweep,
    'N_MEAS': n_meas
}

awg_channel.awg.load_sequencer_program(sequence)
awg_channel.awg.wait_done()

# Load command table
# Command table used since it's more efficient than playWave
# https://docs.zhinst.com/shfsg_user_manual/tutorials/tutorial_command_table.html#introduction-to-the-command-table
ct_schema = awg_channel.awg.commandtable.load_validation_schema()
ct = CommandTable(ct_schema)

# Entry 0: play waveform 0
ct.table[0].waveform.index = 0

# Entry 1: play waveform 1
ct.table[1].waveform.index = 1

# Entry 2: play waveform 2
ct.table[2].waveform.index = 2

# Entry 3: play waveform 3
ct.table[3].waveform.index = 3

awg_channel.awg.commandtable.upload_to_device(ct)

# Start time tagger and AWG sequence
cbm.start()
tt.sync()

awg_channel.awg.enable_sequencer(single=True)
awg_channel.awg.wait_done()

while not cbm.ready():
    time.sleep(0.2)

counts = cbm.getData()
counts = np.array(counts)
counts = counts.reshape((n_sweep, n_meas, 2))
np.save(f'../data/cw_am_sweep/{start_date.isoformat()}.npy', counts)

active_counts = counts[:,:,0]
inactive_counts = counts[:,:,1]

mean_active_counts = np.mean(active_counts, axis=1)
mean_inactive_counts = np.mean(inactive_counts, axis=1)

am_counts = (inactive_counts - active_counts) / inactive_counts

plt.plot(freq, am_counts)
plt.show()