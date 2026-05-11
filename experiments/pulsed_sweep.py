import time
from datetime import datetime, timedelta
import numpy as np
import matplotlib.pyplot as plt
from zhinst.toolkit import Session, CommandTable
from TimeTagger import CountBetweenMarkers, createTimeTaggerNetwork
from util.load_sequence import load_sequence

start_date = datetime.now()

# Device parameters
AWG_SERVER_HOST = 'localhost'
AWG_SERVER_PORT = 8004
AWG_DEVICE = 'DEV12120'
AWG_MW_CHANNEL = 2
AWG_LASER_CHANNEL = 4
AWG_SAMPLE_RATE = 2e9

TT_CLICK_CHANNEL = 1
TT_MARKER_CHANNEL = 2

# Pi pulse length from rabi_sweep
mw_length_ns = 5e3

init_length_ns = 3e3
dark_length_ns = 250
meas_length_ns = 1e3
osc = 0

start_freq = 2.84e9
stop_freq = 2.90e9
n_sweep = 401
n_meas = 100

# Synchronization is done by sending internal trigger periodically. Pulse period is used, with some margin for safety.
max_period_ns = init_length_ns + dark_length_ns + mw_length_ns + meas_length_ns
sync_overhead = 1.05

# Parameters stored in output file
params = {
    "init_length_ns": init_length_ns,
    "dark_length_ns": dark_length_ns,
    "mw_length_ns": mw_length_ns,
    "meas_length_ns": meas_length_ns,
    "start_freq": start_freq,
    "stop_freq": stop_freq,
    "n_sweep": n_sweep,
    "n_meas": n_meas,
}

expected_duration = n_sweep * n_meas * max_period_ns * sync_overhead / 1e9
print(f"Expected duration: {expected_duration}s")
print(f"Finished at: {(datetime.now() + timedelta(seconds=expected_duration)).time()}")

init_length = init_length_ns * AWG_SAMPLE_RATE / 1e9
dark_length = dark_length_ns * AWG_SAMPLE_RATE / 1e9
mw_length = mw_length_ns * AWG_SAMPLE_RATE / 1e9
meas_length = meas_length_ns * AWG_SAMPLE_RATE / 1e9

center_freq = 2.8e9
relative_start_freq = start_freq - center_freq

freq = np.linspace(start_freq, stop_freq, n_sweep)
freq_incr = (stop_freq - start_freq) / (n_sweep - 1)

center_freq = 2.8e9

# Arbitrary Waveform Generator initialization
awg_session = Session(AWG_SERVER_HOST, AWG_SERVER_PORT)
awg_device = awg_session.connect_device(AWG_DEVICE)

awg_device.check_compatibility()

# Configure internal trigger for synchronization
awg_device.system.internaltrigger.enable(1)
awg_device.system.internaltrigger.holdoff(max_period_ns * sync_overhead / 1e9)
awg_device.system.internaltrigger.repetitions(n_sweep * n_meas)
awg_device.system.internaltrigger.synchronization.enable(1)

awg_mw = awg_device.sgchannels[AWG_MW_CHANNEL]
awg_laser = awg_device.sgchannels[AWG_LASER_CHANNEL]

awg_mw.configure_channel(
    enable=True,
    output_range=10,
    center_frequency=center_freq,
    rf_path=True
)

awg_mw.synchronization.enable(1)

awg_mw.configure_sine_generation(
    enable=False,
    osc_index=osc,
    osc_frequency=relative_start_freq,
    phase=0
)

awg_mw.configure_pulse_modulation(
    enable=True,
    osc_index=osc,
    osc_frequency=relative_start_freq,
    phase=0
)

awg_mw.awg.configure_marker_and_trigger(
    trigger_in_source='internal_trigger',
    trigger_in_slope='rising_edge',
    marker_out_source='output0_marker0'
)

awg_laser.configure_channel(
    enable=True,
    output_range=0,
    center_frequency=center_freq,
    rf_path=True
)

awg_laser.synchronization.enable(1)

awg_laser.configure_sine_generation(
    enable=False
)

awg_laser.configure_pulse_modulation(
    enable=False
)

awg_laser.awg.configure_marker_and_trigger(
    trigger_in_source='internal_trigger',
    trigger_in_slope='rising_edge',
    marker_out_source='output0_marker0'
)

# Time Tagger initialization
tt = createTimeTaggerNetwork('localhost:41101')

tt.setTriggerLevel(TT_CLICK_CHANNEL, 0.25)
tt.setTriggerLevel(TT_MARKER_CHANNEL, 0.5)

cbm = CountBetweenMarkers(tt, TT_CLICK_CHANNEL, TT_MARKER_CHANNEL, -TT_MARKER_CHANNEL, 2 * n_sweep * n_meas)

# - Microwave sequence and CT
mw_sequence = load_sequence("../awg_sequences/rabi_sweep/mw.c")
mw_sequence.constants = {
    'INIT_LENGTH': init_length,
    'DARK_LENGTH': dark_length,
    'MW_LENGTH': mw_length,
    'MEAS_LENGTH': meas_length,
    'OSC': osc,
    'START_FREQ': start_freq,
    'FREQ_INCR': freq_incr,
    'N_SWEEP': n_sweep,
    'N_MEAS': n_meas
}

awg_mw.awg.load_sequencer_program(mw_sequence)
awg_mw.awg.wait_done()

# Load command table
mw_ct_schema = awg_mw.awg.commandtable.load_validation_schema()
mw_ct = CommandTable(mw_ct_schema)

# Entry 0: play waveform 0
mw_ct.table[0].waveform.index = 0

# Entry 1: play waveform 1
mw_ct.table[1].waveform.index = 1

# Entry 2: play waveform 2
mw_ct.table[2].waveform.index = 2

awg_mw.awg.commandtable.upload_to_device(mw_ct)

# - Laser sequence and CT
laser_sequence = load_sequence("../awg_sequences/rabi_sweep/laser.c")
laser_sequence.constants = {
    'INIT_LENGTH': init_length,
    'DARK_LENGTH': dark_length,
    'MW_LENGTH': mw_length,
    'MEAS_LENGTH': meas_length,
    'N_SWEEP': n_sweep,
    'N_MEAS': n_meas
}

awg_laser.awg.load_sequencer_program(laser_sequence)
awg_laser.awg.wait_done()

# Load command table
laser_ct_schema = awg_laser.awg.commandtable.load_validation_schema()
laser_ct = CommandTable(laser_ct_schema)

# Entry 0: play waveform 0
laser_ct.table[0].waveform.index = 0

# Entry 1: play waveform 1
laser_ct.table[1].waveform.index = 1

# Entry 2: play waveform 2
laser_ct.table[2].waveform.index = 2

awg_laser.awg.commandtable.upload_to_device(laser_ct)

# Start time tagger and AWG sequence
cbm.start()
tt.sync()

awg_mw.awg.enable_sequencer(single=True)
awg_laser.awg.enable_sequencer(single=True)
awg_mw.awg.wait_done(timeout=expected_duration*1.5)

while not cbm.ready():
    time.sleep(0.2)

counts = cbm.getData()
counts = np.array(counts)
counts = counts.reshape((n_sweep, n_meas, 2))
np.savez(f'../data/pulsed_sweep/{start_date.isoformat().replace(':', '.')}.npy', data=counts, params=params)

ref_counts = counts[:,:,0]
meas_counts = counts[:,:,1]

mean_ref_counts = np.mean(ref_counts, axis=1)
mean_meas_counts = np.mean(meas_counts, axis=1)

mean_counts_norm = (mean_ref_counts - mean_meas_counts) / mean_ref_counts

plt.plot(freq, mean_counts_norm)
plt.show()