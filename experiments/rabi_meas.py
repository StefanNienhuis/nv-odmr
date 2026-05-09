import time
from datetime import datetime, timedelta
import numpy as np
import matplotlib.pyplot as plt
from zhinst.toolkit import Session, CommandTable
from TimeTagger import createTimeTaggerNetwork, CountBetweenMarkers
from util.load_sequence import load_sequence

start_date = datetime.now()

# Device parameters
AWG_SERVER_HOST = 'localhost'
AWG_SERVER_PORT = 8004
AWG_DEVICE = 'DEV12120'
AWG_CHANNEL_MW = 2
AWG_CHANNEL_LASER = 3
AWG_SAMPLE_RATE = 2e9

TT_CLICK_CHANNEL = 1
TT_MARKER_CHANNEL = 2

# Parameters
init_laser_ns    = 5         #Time initial laser is on (ns)
readout_ns          = 5         #Time measurements is on per period (ns)
drive_freq          = 3.6e9     # drive frequency (Hz)
mw_delay_ns         = 5e9       #Delay time after laser to start mw (ns)
meas_delay_ns       = 20e3      # Delay before measuring (ns)
osc                 = 0         # Oscillator being swept
shortest_pulse_ns   = 2.84e9    # Shortest mw pulse duration
longest_pulse_ns    = 2.90e9    # Longest mw pulse duration
n_sweep             = 401       # Number of sweep steps
n_meas              = 5         # Number of measurements at each time delay
shortest_period_ns  = init_laser_ns + mw_delay_ns + shortest_pulse_ns + meas_delay_ns + meas_on_ns #Shortest total period length (ns)
# Parameters stored in output file


params = {
    "laser_on_ns": laser_on_ns,
    "drive_freq": drive_freq,
    "meas_on_ns": meas_on_ns,
    "mw_delay_ns": mw_delay_ns,
    "meas_delay_ns":meas_delay_ns,
    "shortest_pulse_ns": shortest_pulse_ns,
    "longest_pulse_ns": longest_pulse_ns,
    "n_sweep": n_sweep,
    "n_meas": n_meas,
}

# Calculate pulse length from modulation frequency
avg_pulse = (shortest_pulse_ns+longest_pulse_ns)*0.5
avg_period_ns = 1e9 / (init_laser_ns+mw_delay_ns+meas_delay_ns+avg_pulse+meas_on_ns)

expected_duration = n_sweep * n_meas * avg_period_ns / 1e9
print(f"Expected duration: {expected_duration}s")
print(f"Finished at: {(datetime.now() + timedelta(seconds=expected_duration)).time()}")

# Convert ns -> samples
init_laser = init_laser_ns * AWG_SAMPLE_RATE / 1e9
readout = readout_ns * AWG_SAMPLE_RATE / 1e9
mw_delay = mw_delay_ns * AWG_SAMPLE_RATE / 1e9
meas_delay = meas_delay_ns * AWG_SAMPLE_RATE / 1e9
shortest_pulse = shortest_pulse_ns * AWG_SAMPLE_RATE / 1e9
longest_pulse = longest_pulse_ns * AWG_SAMPLE_RATE / 1e9
shortest_period = shortest_period_ns * AWG_SAMPLE_RATE / 1e9

# Round counts to 16 - AWG zero pads otherwise
init_laser_on = int(round(init_laser / 16) * 16)
readout = int(round(readout / 16) * 16)
mw_delay = int(round(mw_delay / 16) * 16)
meas_delay = int(round(meas_delay / 16) * 16)
shortest_pulse = int(round(shortest_pulse / 16) * 16)
longest_pulse = int(round(longest_pulse / 16) * 16)
shortest_period = int(round(shortest_period / 16) * 16)

pulse = np.linspace(shortest_pulse_ns, longest_pulse_ns, n_sweep)
pulse_incr = (shortest_pulse - longest_pulse) / (n_sweep - 1)
pulse_incr = int(round(pulse_incr / 16) * 16)

# Arbitrary Waveform Generator initialization
awg_session = Session(AWG_SERVER_HOST, AWG_SERVER_PORT)
awg_device = awg_session.connect_device(AWG_DEVICE)

awg_device.check_compatibility()

awg_channel_mw = awg_device.sgchannels[AWG_CHANNEL_MW]

awg_channel_mw.configure_channel(
    enable=True,
    output_range=10,
    center_frequency=drive_freq,
    rf_path=True
)

awg_channel_mw.configure_sine_generation(
    enable=False,
    osc_index=osc,
    osc_frequency=0,
    phase=0
)

awg_channel_mw.configure_pulse_modulation(
    enable=True,
    osc_index=osc,
    osc_frequency=0,
    phase=0
)

awg_channel_mw.awg.configure_marker_and_trigger(
    trigger_in_source='trigin3',
    trigger_in_slope='rising_edge',
    marker_out_source='output0_marker0'
)



awg_channel_laser = awg_device.sgchannels[AWG_CHANNEL_LASER]

awg_channel_laser.configure_channel(
    enable=True,
    output_range=10,
    center_frequency=drive_freq,
    rf_path=True
)

awg_channel_laser.configure_sine_generation(
    enable=False,
    osc_index=osc,
    osc_frequency=0,
    phase=0
)

awg_channel_laser.configure_pulse_modulation(
    enable=True,
    osc_index=osc,
    osc_frequency=drive_freq,
    phase=0
)

awg_channel_laser.awg.configure_marker_and_trigger(
    trigger_in_source='trigin2',
    trigger_in_slope='falling_edge',
    marker_out_source='output0_marker0'
)


# Time Tagger initialization
tt = createTimeTaggerNetwork('localhost:41101')

tt.setTriggerLevel(TT_CLICK_CHANNEL, 0.5)
tt.setTriggerLevel(TT_MARKER_CHANNEL, 0.5)

# Twice the number of samples since we get one with pulse and one without pulse (square AM modulation)
cbm = CountBetweenMarkers(tt, TT_CLICK_CHANNEL, TT_MARKER_CHANNEL, -TT_MARKER_CHANNEL,n_sweep*n_meas)

# Load AWG sequence
mw_sequence = load_sequence("../awg_sequences/rabi_meas_mw.c")
mw_sequence.constants = {
    'INIT_LASER': init_laser,
    'SHORTEST_PULSE': shortest_pulse,
    'PULSE_INCR': pulse_incr,
    'READOUT': readout,
    'N_MEAS': n_meas,
}

laser_sequence = load_sequence("../awg_sequences/rabi_meas_laser.c")
laser_sequence.constants = {
    'INIT_LASER': init_laser,
    'MW_DELAY':mw_delay,
    'READOUT': readout,
    'N_SWEEP': n_sweep,
    'N_MEAS': n_meas,
}

awg_channel_mw.awg.load_sequencer_program(mw_sequence)
awg_channel_mw.awg.wait_done()
awg_channel_laser.awg.load_sequencer_program(laser_sequence)
awg_channel_laser.awg.wait_done()

# Load command table
# Command table used since it's more efficient than playWave
# https://docs.zhinst.com/shfsg_user_manual/tutorials/tutorial_command_table.html#introduction-to-the-command-table
ct_schema_mw = awg_channel_mw.awg.commandtable.load_validation_schema()
ct_mw = CommandTable(ct_schema_mw)
ct_schema_laser = awg_channel_laser.awg.commandtable.load_validation_schema()
ct_laser = CommandTable(ct_schema_laser)

# Entry 0: inital laser
ct_laser.table[0].waveform.index = 0

# Entry 1: Readout
ct_laser.table[1].waveform.index = 1


# Entry 0: initial laser 
ct_mw.table[0].waveform.index = 0

# Entry 1: mw on, playhold inside sequencer
ct_mw.table[1].waveform.index = 1

# Entry 2: readout
ct_mw.table[2].waveform.index = 2




awg_channel_mw.awg.commandtable.upload_to_device(ct_mw)
awg_channel_laser.awg.commandtable.upload_to_device(ct_laser)

# Start time tagger and AWG sequence
cbm.start()
tt.sync()

awg_channel_mw.awg.enable_sequencer(single=True)
awg_channel_laser.awg.enable_sequencer(single=True)
awg_channel_mw.awg.wait_done(timeout=expected_duration*1.5)
awg_channel_laser.awg.wait_done(timeout=expected_duration*1.5)

while not cbm.ready():
    time.sleep(0.2)

counts = cbm.getData()
counts = np.array(counts)
counts = counts.reshape((n_sweep, n_meas))
np.savez(f'../data/cw_am_sweep/{start_date.isoformat().replace(':', '.')}.npy', data=counts, params=params)

mean_counts = np.mean(counts, axis=1)
print(mean_counts)
print(cbm.getBinWidths())


plt.plot(pulse, mean_counts, label='Mean counts')
plt.legend()
plt.show()
