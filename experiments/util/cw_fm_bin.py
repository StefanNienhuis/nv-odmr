import time
import numpy as np
import matplotlib.pyplot as plt
from zhinst.toolkit import Session, CommandTable
from TimeTagger import createTimeTaggerNetwork, CountBetweenMarkers, CHANNEL_UNUSED
import pycobolt
from util.load_sequence import load_sequence
from tqdm import tqdm

bin_length = 1024

# Device parameters
AWG_SERVER_HOST = 'localhost'
AWG_SERVER_PORT = 8004
AWG_DEVICE = 'DEV12120'
AWG_CHANNEL = 2
AWG_SAMPLE_RATE = 2e9

TT_CLICK_CHANNEL = 1
TT_MARKER_CHANNEL = 2

LASER_SN = '31977'
LASER_CURRENT = 57

# Arbitrary Waveform Generator initialization
awg_session = Session(AWG_SERVER_HOST, AWG_SERVER_PORT)
awg_device = awg_session.connect_device(AWG_DEVICE)

awg_device.check_compatibility()

awg_channel = awg_device.sgchannels[AWG_CHANNEL]

awg_channel.synchronization.enable(0)

center_freq = 2.8e9
    
awg_channel.configure_channel(
    enable=True,
    output_range=10,
    center_frequency=center_freq,
    rf_path=True
)

# Time Tagger initialization
tt = createTimeTaggerNetwork('localhost:41101')

tt.setTriggerLevel(TT_CLICK_CHANNEL, 0.5)
tt.setTriggerLevel(TT_MARKER_CHANNEL, 0.5)

# Laser setup
laser = pycobolt.CoboltLaser(serialnumber=LASER_SN)
laser.constant_current()
laser.set_current(LASER_CURRENT)
print(f"Laser mode: {laser.get_mode()}")

def round_frequency(frequency):
    period_ns = 1e9 / frequency
    pulse_ns = period_ns / 2
    pulse_length = pulse_ns * AWG_SAMPLE_RATE / 1e9

    rounded_pulse_length = int(np.round(pulse_length / bin_length) * bin_length)

    rounded_pulse_ns = rounded_pulse_length * 1e9 / AWG_SAMPLE_RATE
    rounded_period_ns = rounded_pulse_ns * 2
    rounded_frequency = 1e9 / rounded_period_ns

    return rounded_frequency

def lockin(sig, ref):
    assert len(sig) == len(ref)

    theta = np.linspace(0, 2 * np.pi, len(sig))

    # Normalize signals
    s = sig - np.mean(sig)
    r = ref - np.mean(ref)

    # Invert signal since MW high means PL low
    s = -s

    Xs = np.sum(s * np.cos(theta))
    Ys = np.sum(s * np.sin(theta))

    phi_s = np.arctan2(Ys, Xs)

    Xr = np.sum(r * np.cos(theta))
    Yr = np.sum(r * np.sin(theta))

    phi_r = np.arctan2(Yr, Xr)

    d_phi = phi_s - phi_r
    d_phi = d_phi % (2 * np.pi)
    
    print(phi_s, phi_r)

    R = np.sqrt(Xs**2 + Ys**2)
    shift = d_phi / (2 * np.pi) * len(sig)

    return R, shift

def perform_sweep(modulation_freq, freq_dev, osc1, osc2, start_freq, stop_freq, n_sweep, n_meas):
    # Calculate pulse length from modulation frequency
    period_ns = 1e9 / modulation_freq
    pulse_length_ns = period_ns / 2
    expected_duration = n_sweep * n_meas * period_ns / 1e9

    # Convert ns -> samples
    pulse_length = pulse_length_ns * AWG_SAMPLE_RATE / 1e9

    n_bins = pulse_length / bin_length
    assert round(n_bins) == n_bins
    n_bins = int(n_bins)

    relative_start_freq = start_freq - center_freq

    freq = np.linspace(start_freq, stop_freq, n_sweep)
    freq_incr = (stop_freq - start_freq) / max(1, n_sweep - 1)
    
    # AWG channel configuration
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

    # Load AWG sequence
    sequence = load_sequence("../awg_sequences/cw_fm_bin_sweep.c")
    sequence.constants = {
        'BIN_LENGTH': bin_length,
        'N_BINS': n_bins,
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

    # Entry 0: play waveform 0
    ct.table[0].waveform.index = 0
    ct.table[0].oscillatorSelect.value = osc1

    # Entry 1: play waveform 1
    ct.table[1].waveform.index = 0
    ct.table[1].oscillatorSelect.value = osc2

    awg_channel.awg.commandtable.upload_to_device(ct)

    cbm = CountBetweenMarkers(tt, TT_CLICK_CHANNEL, TT_MARKER_CHANNEL, CHANNEL_UNUSED, n_sweep * n_meas * 2 * n_bins)

    # Start time tagger and AWG sequence
    cbm.start()
    tt.sync()

    awg_channel.awg.enable_sequencer(single=True)
    
    steps = n_sweep if n_sweep > 1 else n_meas
    for _ in tqdm(range(steps)):
        time.sleep(expected_duration / steps)
    
    awg_channel.awg.wait_done(timeout=expected_duration*1.5)

    while not cbm.ready():
        time.sleep(0.2)

    counts = cbm.getData()
    counts = np.array(counts)
    counts = counts.reshape((n_sweep, n_meas, 2 * n_bins))
    
    ref = np.zeros(2 * n_bins)
    ref[0:n_bins//2] = 1
    
    fm_signals = np.zeros(n_sweep)
    
    for i in range(n_sweep):
        summed_period = np.sum(counts[i], axis=0)
        R, shift = lockin(summed_period, ref)
        
        sig_unshifted = np.roll(counts[i], -round(shift))

        low_counts = np.sum(sig_unshifted[:, :n_bins], axis=1)
        high_counts = np.sum(sig_unshifted[:, n_bins:], axis=1)

        mean_low_counts = np.mean(low_counts)
        mean_high_counts = np.mean(high_counts)

        fm_signal = (mean_high_counts - mean_low_counts) / (mean_high_counts + mean_low_counts)
        fm_signals[i] = fm_signal
    
    return freq, fm_signals

