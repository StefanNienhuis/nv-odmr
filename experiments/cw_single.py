import time
from datetime import datetime, timedelta
import numpy as np
import matplotlib.pyplot as plt
from zhinst.toolkit import Session, CommandTable
from TimeTagger import CountBetweenMarkers, createTimeTaggerNetwork, CHANNEL_UNUSED
from util.load_sequence import load_sequence

start_date = datetime.now()

# Device parameters
AWG_SERVER_HOST = 'localhost'
AWG_SERVER_PORT = 8004
AWG_DEVICE = 'DEV12120'
AWG_CHANNEL = 2
AWG_SAMPLE_RATE = 2e9

TT_CLICK_CHANNEL = 1
TT_MARKER_CHANNEL = 2

# Parameters - should match the ones used in sweep
pulse_length_ns = 100e6     # Pulse duration (ns)
meas_delay_ns   = 50e3      # Delay before measuring (ns)
osc             = 0         # Oscillator being used
freq            = 2.85e9    # The frequency to look at
n_meas          = 1000       # Number of measurements at each frequency

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

# Convert ns -> samples
pulse_length = pulse_length_ns * AWG_SAMPLE_RATE / 1e9
meas_delay = meas_delay_ns * AWG_SAMPLE_RATE / 1e9

# Round counts to 16 - AWG zero pads otherwise
pulse_length = int(round(pulse_length / 16) * 16)
meas_delay = int(round(meas_delay / 16) * 16)

center_freq = 2.8e9
relative_freq = freq - center_freq

# Arbitrary Waveform Generator initialization
awg_session = Session(AWG_SERVER_HOST, AWG_SERVER_PORT)
awg_device = awg_session.connect_device(AWG_DEVICE)

awg_device.check_compatibility()

awg_channel = awg_device.sgchannels[AWG_CHANNEL]

awg_channel.configure_channel(
    enable=True,
    output_range=10,
    center_frequency=center_freq,
    rf_path=True
)

awg_channel.configure_sine_generation(
    enable=False,
    osc_index=osc,
    osc_frequency=relative_freq,
    phase=0
)

awg_channel.configure_pulse_modulation(
    enable=True,
    osc_index=osc,
    osc_frequency=relative_freq,
    phase=0
)

awg_channel.awg.configure_marker_and_trigger(
    trigger_in_source='trigin0',
    trigger_in_slope='rising_edge',
    marker_out_source='output0_marker0'
)

# Time Tagger initialization
tt = createTimeTaggerNetwork('localhost:41101')

tt.setTriggerLevel(TT_CLICK_CHANNEL, 0.25)
tt.setTriggerLevel(TT_MARKER_CHANNEL, 0.5)

# Marker channel is inverted
cbm = CountBetweenMarkers(tt, TT_CLICK_CHANNEL, -TT_MARKER_CHANNEL, TT_MARKER_CHANNEL, n_meas)

# Load AWG sequence
sequence = load_sequence("../awg_sequences/cw_sweep.c")
sequence.constants = {
    'PULSE_LENGTH': pulse_length,
    'MEAS_DELAY': meas_delay,
    'OSC': osc,
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

# Entry 2: hold for remaining time
ct.table[2].waveform.playHold = True
ct.table[2].waveform.length = pulse_length - meas_delay - 1024

awg_channel.awg.commandtable.upload_to_device(ct)

# Start time tagger and AWG sequence
cbm.start()
tt.sync()

awg_channel.awg.enable_sequencer(single=True)
awg_channel.awg.wait_done(timeout=expected_duration*1.5)

while not cbm.ready():
    time.sleep(0.2)

counts = cbm.getData()
counts = np.array(counts)
np.savez(f'../data/cw_single/{start_date.isoformat().replace(':', '.')}.npy', data=counts, params=params)

print(f"Mean: {counts.mean()}")
print(f"Stddev: {counts.std()}")

plt.plot(counts)
plt.show()