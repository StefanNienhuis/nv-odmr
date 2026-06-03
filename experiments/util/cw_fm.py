import time
import numpy as np
import matplotlib.pyplot as plt
from zhinst.toolkit import Session, CommandTable
from TimeTagger import createTimeTaggerNetwork, CountBetweenMarkers
import pycobolt
from util.load_sequence import load_sequence
from tqdm import tqdm

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

def perform_sweep(modulation_freq, freq_dev, meas_delay_ns, osc1, osc2, start_freq, stop_freq, n_sweep, n_meas, show_progress=True):
    # Calculate pulse length from modulation frequency
    period_ns = 1e9 / modulation_freq
    pulse_length_ns = period_ns / 2
    expected_duration = n_sweep * n_meas * period_ns / 1e9

    # Convert ns -> samples
    pulse_length = pulse_length_ns * AWG_SAMPLE_RATE / 1e9
    meas_delay = meas_delay_ns * AWG_SAMPLE_RATE / 1e9

    # Round counts to 16 - AWG zero pads otherwise
    pulse_length = int(round(pulse_length / 16) * 16)
    meas_delay = int(round(meas_delay / 16) * 16)

    relative_start_freq = start_freq - center_freq

    freq = np.linspace(start_freq, stop_freq, n_sweep)
    freq_incr = (stop_freq - start_freq) / max(1, n_sweep - 1)

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

    if show_progress:
        steps = n_sweep if n_sweep > 1 else n_meas
        for _ in tqdm(range(steps)):
            time.sleep(expected_duration / steps)

    awg_channel.awg.wait_done(timeout=expected_duration * 1.5)

    while not cbm.ready():
        time.sleep(0.2)

    counts = cbm.getData()
    counts = np.array(counts)
    counts = counts.reshape((n_sweep, n_meas, 2))

    return freq, counts