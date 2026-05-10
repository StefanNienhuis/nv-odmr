import time
from datetime import datetime, timedelta
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
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

# Parameters
pulse_length_ns = 100e6      # Pulse duration (ns)
meas_delay_ns   = 50e3      # Delay before measuring (ns)
osc             = 0        # Oscillator being swept
start_freq      = 2.84e9   # Sweep start frequency (Hz)
stop_freq       = 2.90e9   # Sweep stop frequency (Hz)
n_sweep         = 401      # Number of sweep steps
n_meas          = 5        # Number of measurements at each frequency

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

# Convert meas_delay ns -> samples, round to multiple of 16
meas_delay = int(round(meas_delay_ns * AWG_SAMPLE_RATE / 1e9 / 16) * 16)

center_freq = 2.8e9
relative_start_freq = start_freq - center_freq
freq = np.linspace(start_freq, stop_freq, n_sweep)
freq_incr = (stop_freq - start_freq) / (n_sweep - 1)

# AWG initialization (once)
awg_session = Session(AWG_SERVER_HOST, AWG_SERVER_PORT)
awg_device = awg_session.connect_device(AWG_DEVICE)
awg_device.check_compatibility()

awg_channel = awg_device.sgchannels[AWG_CHANNEL]
awg_channel.configure_channel(
    enable=True, output_range=10,
    center_frequency=center_freq, rf_path=True
)
awg_channel.configure_sine_generation(
    enable=False, osc_index=osc,
    osc_frequency=relative_start_freq, phase=0
)
awg_channel.configure_pulse_modulation(
    enable=True, osc_index=osc,
    osc_frequency=relative_start_freq, phase=0
)
awg_channel.awg.configure_marker_and_trigger(
    trigger_in_source='trigin0',
    trigger_in_slope='rising_edge',
    marker_out_source='output0_marker0'
)

# Time Tagger initialization (once)
tt = createTimeTaggerNetwork('localhost:41101')
tt.setTriggerLevel(TT_CLICK_CHANNEL, 0.25)
tt.setTriggerLevel(TT_MARKER_CHANNEL, 0.5)

# Lorentzian dip
def lorentzian(f, f0, gamma, amplitude, offset):
    return offset - amplitude * (gamma/2)**2 / ((f - f0)**2 + (gamma/2)**2)

# Storage
linewidth_per_pl  = np.full(n_pulse_lengths, np.nan)
contrast_per_pl   = np.full(n_pulse_lengths, np.nan)
count_rate_per_pl = np.full(n_pulse_lengths, np.nan)
f0_per_pl         = np.full(n_pulse_lengths, np.nan)
eta_B_per_pl      = np.full(n_pulse_lengths, np.nan)
sigma_B_per_pl    = np.full(n_pulse_lengths, np.nan)
all_counts        = []

# Loop over pulse_length
for pl_idx, pl_ns in enumerate(pulse_lengths_ns):
    pulse_length = int(round(pl_ns * AWG_SAMPLE_RATE / 1e9 / 16) * 16)
    sweep_duration = n_sweep * n_meas * pl_ns / 1e9
    t_sweep = sweep_duration
    print(f"\n[{pl_idx+1}/{n_pulse_lengths}] pulse_length = {pl_ns/1e6:.2f} ms "
          f"(sweep duration ≈ {sweep_duration:.1f}s)")

    # Reload AWG sequence
    sequence = load_sequence("../awg_sequences/cw_sweep.c")
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
np.savez(f'../data/cw_sweep/{start_date.isoformat().replace(':', '.')}.npy', data=counts, params=params)

print(counts)
print(cbm.getBinWidths())

counts_norm = counts / np.max(counts)

plt.plot(freq, counts_norm)
plt.show()