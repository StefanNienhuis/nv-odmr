import time
import numpy as np
from zhinst.toolkit import Session, CommandTable
from TimeTagger import createTimeTaggerNetwork, CountBetweenMarkers

from util.load_sequence import load_sequence

# Default device parameters
AWG_SERVER_HOST  = 'localhost'
AWG_SERVER_PORT  = 8004
AWG_DEVICE       = 'DEV12120'
AWG_CHANNEL      = 2
AWG_SAMPLE_RATE  = 2e9

TT_HOST          = 'localhost:41101'
TT_CLICK_CHANNEL = 1
TT_MARKER_CHANNEL = 2

CENTER_FREQ      = 2.8e9
SEQUENCE_PATH    = "../awg_sequences/cw_fm_sweep.c"


def _ns_to_samples(ns, sample_rate=AWG_SAMPLE_RATE):
    """Convert ns to AWG samples, rounded to a multiple of 16
    (the AWG zero-pads otherwise)."""
    return int(round(ns * sample_rate / 1e9 / 16) * 16)


def setup_awg(start_freq, mod_depth,
              osc1=0, osc2=1,
              host=AWG_SERVER_HOST, port=AWG_SERVER_PORT,
              device=AWG_DEVICE, channel=AWG_CHANNEL,
              center_freq=CENTER_FREQ):
    """Connect to the AWG and configure the channel + both oscillators.

    Returns the configured ``sgchannel`` handle.
    """
    freq_dev = mod_depth / 2
    relative_start_freq = start_freq - center_freq

    awg_session = Session(host, port)
    awg_device = awg_session.connect_device(device)
    awg_device.check_compatibility()

    awg_channel = awg_device.sgchannels[channel]
    awg_channel.configure_channel(
        enable=True,
        output_range=0,
        center_frequency=center_freq,
        rf_path=True,
    )
    awg_channel.configure_sine_generation(
        enable=False, osc_index=osc1,
        osc_frequency=relative_start_freq - freq_dev, phase=0,
    )
    awg_channel.configure_pulse_modulation(
        enable=True, osc_index=osc1,
        osc_frequency=relative_start_freq - freq_dev, phase=0,
    )
    awg_channel.configure_sine_generation(
        enable=False, osc_index=osc2,
        osc_frequency=relative_start_freq + freq_dev, phase=0,
    )
    awg_channel.configure_pulse_modulation(
        enable=True, osc_index=osc2,
        osc_frequency=relative_start_freq + freq_dev, phase=0,
    )
    awg_channel.awg.configure_marker_and_trigger(
        trigger_in_source='trigin0',
        trigger_in_slope='rising_edge',
        marker_out_source='output0_marker0',
    )
    return awg_channel


def setup_time_tagger(host=TT_HOST,
                      click_channel=TT_CLICK_CHANNEL,
                      marker_channel=TT_MARKER_CHANNEL,
                      trigger_level=0.5):
    """Connect to the Time Tagger and set trigger levels. Returns the handle."""
    tt = createTimeTaggerNetwork(host)
    tt.setTriggerLevel(click_channel, trigger_level)
    tt.setTriggerLevel(marker_channel, trigger_level)
    return tt


def run_fm_sweep(awg_channel, tt,
                 pulse_length_ns,
                 start_freq, stop_freq, mod_depth,
                 n_sweep, n_meas, meas_delay_ns,
                 osc1=0, osc2=1,
                 sample_rate=AWG_SAMPLE_RATE,
                 center_freq=CENTER_FREQ,
                 click_channel=TT_CLICK_CHANNEL,
                 marker_channel=TT_MARKER_CHANNEL,
                 sequence_path=SEQUENCE_PATH,
                 timeout_factor=1.5):
    """Run one square-FM CW sweep.

    Returns ``counts`` with shape ``(n_sweep, n_meas, 2)``; the last axis
    is ``(low_freq_counts, high_freq_counts)``.
    """
    freq_dev = mod_depth / 2
    relative_start_freq = start_freq - center_freq
    freq_incr = (stop_freq - start_freq) / (n_sweep - 1)

    pulse_length = _ns_to_samples(pulse_length_ns, sample_rate)
    meas_delay   = _ns_to_samples(meas_delay_ns, sample_rate)

    expected_duration = 2 * n_sweep * n_meas * pulse_length_ns / 1e9

    # Load AWG sequence
    sequence = load_sequence(sequence_path)
    sequence.constants = {
        'PULSE_LENGTH': pulse_length,
        'MEAS_DELAY':   meas_delay,
        'OSC1':         osc1,
        'OSC2':         osc2,
        'START_FREQ':   relative_start_freq,
        'FREQ_DEV':     freq_dev,
        'FREQ_INCR':    freq_incr,
        'N_SWEEP':      n_sweep,
        'N_MEAS':       n_meas,
    }
    awg_channel.awg.load_sequencer_program(sequence)
    awg_channel.awg.wait_done()

    # Load command table (more efficient than playWave - see
    # https://docs.zhinst.com/shfsg_user_manual/tutorials/tutorial_command_table.html)
    ct_schema = awg_channel.awg.commandtable.load_validation_schema()
    ct = CommandTable(ct_schema)

    # Entries 0/1: waveform 0/1 on osc1; 2/3: waveform 0/1 on osc2
    ct.table[0].waveform.index = 0
    ct.table[0].oscillatorSelect.value = osc1
    ct.table[1].waveform.index = 1
    ct.table[1].oscillatorSelect.value = osc1
    ct.table[2].waveform.index = 0
    ct.table[2].oscillatorSelect.value = osc2
    ct.table[3].waveform.index = 1
    ct.table[3].oscillatorSelect.value = osc2

    # Entry 4: hold to pad out the rest of the pulse
    ct.table[4].waveform.playHold = True
    ct.table[4].waveform.length = pulse_length - meas_delay - 1024
    awg_channel.awg.commandtable.upload_to_device(ct)

    # 2x the number of samples since we get two pulses per step (square FM)
    cbm = CountBetweenMarkers(
        tt, click_channel,
        -marker_channel, marker_channel,
        2 * n_sweep * n_meas,
    )
    cbm.start()
    tt.sync()

    awg_channel.awg.enable_sequencer(single=True)
    awg_channel.awg.wait_done(timeout=expected_duration * timeout_factor)

    while not cbm.ready():
        time.sleep(0.2)

    return np.array(cbm.getData()).reshape((n_sweep, n_meas, 2))