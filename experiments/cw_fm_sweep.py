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
AWG_CHANNEL = 2
AWG_SAMPLE_RATE = 2e9

TT_CLICK_CHANNEL = 1
TT_MARKER_CHANNEL = 2

# Parameters
modulation_freq = 5      # FM modulation frequency
meas_delay_ns   = 20e3      # Delay before measuring (ns)
osc1            = 0        # First oscillator being swept
osc2            = 1        # Second oscillator being swept
start_freq      = 2.84e9   # Sweep start frequency (Hz)
stop_freq       = 2.90e9   # Sweep stop frequency (Hz)
mod_depth       = 3e6      # FM modulation depth (Hz)
n_sweep         = 401      # Number of sweep steps
n_meas          = 5        # Number of measurements at each frequency

freq_dev        = mod_depth / 2

# Calculate pulse length from modulation frequency
period_ns = 1e9 / modulation_freq
pulse_length_ns = period_ns / 2

expected_duration = n_sweep * n_meas * period_ns / 1e9
print(f"Expected duration: {expected_duration}s")
print(f"Finished at: {(datetime.now() + timedelta(seconds=expected_duration)).time()}")

# Convert ns -> samples
pulse_length = pulse_length_ns * AWG_SAMPLE_RATE / 1e9
meas_delay = meas_delay_ns * AWG_SAMPLE_RATE / 1e9

# Round counts to 16 - AWG zero pads otherwise
pulse_length = int(round(pulse_length / 16) * 16)
meas_delay = int(round(meas_delay / 16) * 16)

center_freq = 2.8e9
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
    enable=False,
    osc_index=osc1,
    osc_frequency=relative_start_freq - freq_dev,
    phase=0
)

awg_channel.configure_pulse_modulation(
    enable=True,
    osc_index=osc1,
    osc_frequency=relative_start_freq - freq_dev,
    phase=0
)

awg_channel.configure_sine_generation(
    enable=False,
    osc_index=osc2,
    osc_frequency=relative_start_freq + freq_dev,
    phase=0
)

awg_channel.configure_pulse_modulation(
    enable=True,
    osc_index=osc2,
    osc_frequency=relative_start_freq + freq_dev,
    phase=0
)

awg_channel.awg.configure_marker_and_trigger(
    trigger_in_source='trigin0',
    trigger_in_slope='rising_edge',
    marker_out_source='output0_marker0'
)

# Time Tagger initialization
tt = createTimeTaggerNetwork('localhost:41101')

tt.setTriggerLevel(TT_CLICK_CHANNEL, 0.5)
tt.setTriggerLevel(TT_MARKER_CHANNEL, 0.5)

# Twice the number of samples since we get two pulses at different frequencies (square FM modulation)
cbm = CountBetweenMarkers(tt, TT_CLICK_CHANNEL, -TT_MARKER_CHANNEL, TT_MARKER_CHANNEL, 2 * n_sweep * n_meas)

# Load AWG sequence
sequence = load_sequence("../awg_sequences/cw_fm_sweep.c")
sequence.constants = {
    'PULSE_LENGTH': pulse_length,
    'MEAS_DELAY': meas_delay,
    'OSC1': osc1,
    'OSC2': osc2,
    'START_FREQ': relative_start_freq,
    'FREQ_DEV': freq_dev,
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

# Entry 0: play waveform 0, osc1
ct.table[0].waveform.index = 0
ct.table[0].oscillatorSelect.value = osc1

# Entry 1: play waveform 1, osc1
ct.table[1].waveform.index = 1
ct.table[1].oscillatorSelect.value = osc1

# Entry 2: play waveform 0, osc2
ct.table[2].waveform.index = 0
ct.table[2].oscillatorSelect.value = osc2

# Entry 3: play waveform 1, osc2
ct.table[3].waveform.index = 1
ct.table[3].oscillatorSelect.value = osc2

ct.table[4].waveform.playHold = True
ct.table[4].waveform.length = pulse_length - meas_delay - 1024

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
counts = counts.reshape((n_sweep, n_meas, 2))
np.save(f'../data/cw_fm_sweep/{start_date.isoformat().replace(':', '.')}.npy', counts)

low_counts = counts[:,:,0]
high_counts = counts[:,:,1]

mean_low_counts = np.mean(low_counts, axis=1)
mean_high_counts = np.mean(high_counts, axis=1)

fm_counts = (mean_high_counts - mean_low_counts) / (mean_high_counts + mean_low_counts)

plt.plot(freq, mean_low_counts)
plt.plot(freq, mean_high_counts)
plt.show()

last_valid_idx = np.where(valid)[0][-1]
fm_sig = all_fm_signals[last_valid_idx]
f0_show = f0_per_pl[last_valid_idx]
slope_show = slope_per_pl[last_valid_idx]
fig, ax = plt.subplots()
ax.plot(freq/1e9, fm_sig, 'o-', markersize=4, label='FM signal')
ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
ax.axvline(f0_show/1e9, color='red', linestyle='--', alpha=0.5,
           label=f'zero crossing at {f0_show/1e9:.5f} GHz')
f_line = np.linspace(f0_show - mod_depth, f0_show + mod_depth, 50)
s_line = slope_show * (f_line - f0_show)
ax.plot(f_line/1e9, s_line, 'r-', alpha=0.7, label='linear fit (slope)')
ax.set_xlabel('MW center frequency (GHz)')
ax.set_ylabel('FM signal (H − L)/(H + L)')
ax.set_title(f'FM demodulated signal at pulse_length = '
             f'{pulse_lengths_ns[last_valid_idx]/1e6:.0f} ms')
ax.legend()
ax.grid(True, alpha=0.5)
plt.tight_layout()
plt.show()