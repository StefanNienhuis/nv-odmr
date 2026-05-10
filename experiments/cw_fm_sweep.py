import time
from datetime import datetime, timedelta
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
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
    enable=True, output_range=0,
    center_frequency=center_freq, rf_path=True
)
awg_channel.configure_sine_generation(
    enable=False, osc_index=osc1,
    osc_frequency=relative_start_freq - freq_dev, phase=0
)
awg_channel.configure_pulse_modulation(
    enable=True, osc_index=osc1,
    osc_frequency=relative_start_freq - freq_dev, phase=0
)
awg_channel.configure_sine_generation(
    enable=False, osc_index=osc2,
    osc_frequency=relative_start_freq + freq_dev, phase=0
)
awg_channel.configure_pulse_modulation(
    enable=True, osc_index=osc2,
    osc_frequency=relative_start_freq + freq_dev, phase=0
)
awg_channel.awg.configure_marker_and_trigger(
    trigger_in_source='trigin0',
    trigger_in_slope='rising_edge',
    marker_out_source='output0_marker0'
)

# Time Tagger initialization (once)
tt = createTimeTaggerNetwork('localhost:41101')
tt.setTriggerLevel(TT_CLICK_CHANNEL, 0.5)
tt.setTriggerLevel(TT_MARKER_CHANNEL, 0.5)

# Lorentzian (diagnostic only)
def lorentzian(f, f0, gamma, amplitude, offset):
    return offset - amplitude * (gamma/2)**2 / ((f - f0)**2 + (gamma/2)**2)

def find_slope_at_zero(freq, fm_signal, search_window_hz=10e6):
    """Find the strongest zero crossing of the FM signal and fit a line locally."""
    sign_changes = np.where(np.diff(np.sign(fm_signal)))[0]
    if len(sign_changes) == 0:
        raise ValueError("no zero crossing in FM signal")
    deltas = np.abs(np.diff(fm_signal)[sign_changes])
    best_zc = sign_changes[np.argmax(deltas)]
    f1, f2 = freq[best_zc], freq[best_zc + 1]
    s1, s2 = fm_signal[best_zc], fm_signal[best_zc + 1]
    f_zero = f1 - s1 * (f2 - f1) / (s2 - s1)
    mask = np.abs(freq - f_zero) < search_window_hz
    if mask.sum() < 3:
        start = max(0, best_zc - 2)
        stop  = min(len(freq), best_zc + 3)
        mask = np.zeros_like(freq, dtype=bool)
        mask[start:stop] = True
    slope, _ = np.polyfit(freq[mask], fm_signal[mask], 1)
    return f_zero, slope, mask

# Storage
f0_per_pl          = np.full(n_pulse_lengths, np.nan)
slope_per_pl       = np.full(n_pulse_lengths, np.nan)
linewidth_per_pl   = np.full(n_pulse_lengths, np.nan)
contrast_per_pl    = np.full(n_pulse_lengths, np.nan)
count_rate_per_pl  = np.full(n_pulse_lengths, np.nan)
eta_B_per_pl       = np.full(n_pulse_lengths, np.nan)
sigma_B_per_pl     = np.full(n_pulse_lengths, np.nan)
all_counts         = []
all_fm_signals     = []

for pl_idx, pl_ns in enumerate(pulse_lengths_ns):
    pulse_length = int(round(pl_ns * AWG_SAMPLE_RATE / 1e9 / 16) * 16)
    modulation_freq = 1e9 / (2 * pl_ns)
    t_sweep = 2 * n_sweep * n_meas * pl_ns / 1e9
    print(f"\n[{pl_idx+1}/{n_pulse_lengths}] "
          f"pulse_length = {pl_ns/1e6:.2f} ms, mod_freq = {modulation_freq:.2f} Hz, "
          f"sweep duration ≈ {t_sweep:.1f}s")

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

    ct_schema = awg_channel.awg.commandtable.load_validation_schema()
    ct = CommandTable(ct_schema)
    ct.table[0].waveform.index = 0
    ct.table[0].oscillatorSelect.value = osc1
    ct.table[1].waveform.index = 1
    ct.table[1].oscillatorSelect.value = osc1
    ct.table[2].waveform.index = 0
    ct.table[2].oscillatorSelect.value = osc2
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