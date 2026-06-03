import time
import math
from datetime import datetime, timedelta
import numpy as np
import matplotlib.pyplot as plt
from zhinst.toolkit import Session, CommandTable
from TimeTagger import CountBetweenMarkers, createTimeTaggerNetwork
import pycobolt
from tqdm import tqdm

from util import set_mw
from util.load_sequence import load_sequence

start_date = datetime.now()

# Device parameters
AWG_SERVER_HOST = 'localhost'
AWG_SERVER_PORT = 8004
AWG_DEVICE = 'DEV12120'
AWG_MW_CHANNEL = 2
AWG_LASER_CHANNEL = 0
AWG_SAMPLE_RATE = 2e9

TT_CLICK_CHANNEL = 1
TT_MARKER_CHANNEL = 2

LASER_SN = '31977'
LASER_CURRENT = 106

# Largely based on: https://iopscience.iop.org/article/10.1088/1367-2630/ad20b0
init_length_ns = 2e3
dark_length_ns = 400
readout_length_ns = 2e3 + 8 # Must be different from init length - AWG doesn't work otherwise
meas_length_ns = 250
ref_length_ns = 500
drive_freq = 2.8748e9
osc = 0

start_tau_ns = 8
stop_tau_ns = 1024
n_sweep = 128
n_meas = int(1e6)

assert readout_length_ns < init_length_ns + dark_length_ns + stop_tau_ns

# Synchronization is done by sending internal trigger periodically. Max period is used, with some margin for safety.
max_period_ns = init_length_ns + dark_length_ns + stop_tau_ns + readout_length_ns
sync_overhead = 1.1

# Parameters stored in output file
params = {
    "init_length_ns": init_length_ns,
    "dark_length_ns": dark_length_ns,
    "readout_length_ns": readout_length_ns,
    "meas_length_ns": meas_length_ns,
    "ref_length_ns": ref_length_ns,
    "drive_freq": drive_freq,
    "start_tau_ns": start_tau_ns,
    "stop_tau_ns": stop_tau_ns,
    "n_sweep": n_sweep,
    "n_meas": n_meas,
}

expected_duration = math.ceil(n_sweep * n_meas * max_period_ns * sync_overhead / 1e9)
print(f"Expected duration: {expected_duration}s")
print(f"Finished at: {(datetime.now() + timedelta(seconds=expected_duration)).time()}")

init_length = init_length_ns * AWG_SAMPLE_RATE / 1e9
dark_length = dark_length_ns * AWG_SAMPLE_RATE / 1e9
readout_length = readout_length_ns * AWG_SAMPLE_RATE / 1e9
meas_length = meas_length_ns * AWG_SAMPLE_RATE / 1e9
ref_length = ref_length_ns * AWG_SAMPLE_RATE / 1e9

tau_incr_ns = (stop_tau_ns - start_tau_ns) / (n_sweep - 1)

start_tau = start_tau_ns * AWG_SAMPLE_RATE / 1e9
tau_incr = tau_incr_ns * AWG_SAMPLE_RATE / 1e9

if tau_incr != round(tau_incr):
    print(f"Tau sweep increment should be an integer! Currently: {tau_incr}")
    exit()

tau_ns = np.linspace(start_tau_ns, stop_tau_ns, n_sweep)

center_freq = 2.8e9

# Arbitrary Waveform Generator initialization
awg_session = Session(AWG_SERVER_HOST, AWG_SERVER_PORT)
awg_device = awg_session.connect_device(AWG_DEVICE)

awg_device.check_compatibility()

# Configure internal trigger for synchronization
awg_device.system.internaltrigger.enable(0)
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
    osc_frequency=drive_freq - center_freq,
    phase=0
)

awg_mw.configure_pulse_modulation(
    enable=True,
    osc_index=osc,
    osc_frequency=drive_freq - center_freq,
    global_amp=1,
    phase=0
)

awg_mw.awg.configure_marker_and_trigger(
    trigger_in_source='internal_trigger',
    trigger_in_slope='rising_edge',
    marker_out_source='output0_marker0'
)

awg_laser.configure_channel(
    enable=True,
    output_range=10,
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
    'READOUT_LENGTH': readout_length,
    'MEAS_LENGTH': meas_length,
    'REF_LENGTH': ref_length,
    'START_TAU': start_tau,
    'TAU_INCR': tau_incr,
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
    'READOUT_LENGTH': readout_length,
    'START_TAU': start_tau,
    'TAU_INCR': tau_incr,
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

# Laser setup
laser = pycobolt.CoboltLaser(serialnumber=LASER_SN)
laser.current_modulation_mode()
laser.set_modulation_current(LASER_CURRENT)
print(f"Laser mode: {laser.get_mode()}")

# Start time tagger and AWG sequence
cbm.start()
tt.sync()

awg_device.system.internaltrigger.enable(1)
awg_mw.awg.enable_sequencer(single=True)
awg_laser.awg.enable_sequencer(single=True)

with tqdm(total=100) as pbar:
    progress = 0
    while progress < 100:
        time.sleep(1)
        progress = awg_device.system.internaltrigger.progress()
        progress = round(progress * 100, 1)
        pbar.update(progress - pbar.n)
        

awg_mw.awg.wait_done(timeout=50)

while not cbm.ready():
    time.sleep(0.2)

set_mw.set_steady()

counts = cbm.getData()
counts = np.array(counts)
counts = counts.reshape((n_sweep, n_meas, 2))
np.savez(f'../data/rabi_sweep/{start_date.isoformat().replace(":", ".")}.npz', data=counts, params=params)

print(counts)

print(np.sum(counts, axis=1))

meas_counts = counts[:,:,0]
ref_counts = counts[:,:,1]

total_meas_counts = np.sum(meas_counts, axis=1)
total_ref_counts = np.sum(ref_counts, axis=1)

total_counts_norm = (total_ref_counts - total_meas_counts) / total_ref_counts

plt.plot(tau_ns / 1e3, total_ref_counts / (ref_length_ns / meas_length_ns), label='Ref')
plt.plot(tau_ns / 1e3, total_meas_counts, label='Meas')
plt.xlabel('tau (us)')
plt.legend()
plt.show()

plt.plot(tau_ns / 1e3, total_counts_norm)
plt.xlabel('tau (us)')
plt.show()