import time
import numpy as np
from zhinst.toolkit import Session, CommandTable
from TimeTagger import createTimeTaggerNetwork, CountBetweenMarkers
from util.load_sequence import load_sequence

# --- Device parameters -------------------------------------------------------
AWG_SERVER_HOST = 'localhost'
AWG_SERVER_PORT = 8004
AWG_DEVICE = 'DEV12120'
AWG_CHANNEL = 2
AWG_SAMPLE_RATE = 2e9

TT_CLICK_CHANNEL = 1
TT_MARKER_CHANNEL = 2
TT_TRIGGER_LEVEL = 0.5
TT_NETWORK_ADDRESS = 'localhost:41101'

CENTER_FREQ = 2.8e9
SEQUENCE_PATH = "../../awg_sequences/cw_am_sweep.c"


def ns_to_samples(ns):
    """ns -> AWG samples, rounded to a multiple of 16.

    The AWG zero-pads waveforms whose length is not a multiple of 16, so we
    always quantize before handing a length to the sequencer.
    """
    return int(round(ns * AWG_SAMPLE_RATE / 1e9 / 16) * 16)


def init_awg(relative_start_freq, osc=0, center_freq=CENTER_FREQ):
    """Connect to the AWG and configure the SG channel for AM sweeps."""
    session = Session(AWG_SERVER_HOST, AWG_SERVER_PORT)
    device = session.connect_device(AWG_DEVICE)
    device.check_compatibility()

    channel = device.sgchannels[AWG_CHANNEL]
    channel.configure_channel(
        enable=True,
        output_range=10,
        center_frequency=center_freq,
        rf_path=True,
    )
    channel.configure_sine_generation(
        enable=False,
        osc_index=osc,
        osc_frequency=relative_start_freq,
        phase=0,
    )
    channel.configure_pulse_modulation(
        enable=True,
        osc_index=osc,
        osc_frequency=relative_start_freq,
        phase=0,
    )
    channel.awg.configure_marker_and_trigger(
        trigger_in_source='trigin0',
        trigger_in_slope='rising_edge',
        marker_out_source='output0_marker0',
    )
    return channel


def init_time_tagger():
    """Connect to the network Time Tagger and set trigger levels."""
    tt = createTimeTaggerNetwork(TT_NETWORK_ADDRESS)
    tt.setTriggerLevel(TT_CLICK_CHANNEL, TT_TRIGGER_LEVEL)
    tt.setTriggerLevel(TT_MARKER_CHANNEL, TT_TRIGGER_LEVEL)
    return tt


def configure_sweep(awg_channel, pulse_length, meas_delay,
                    relative_start_freq, freq_incr,
                    n_sweep, n_meas, osc=0):
    """Load the cw_am_sweep sequence + command table for one frequency sweep.

    `pulse_length` and `meas_delay` are in AWG samples (use `ns_to_samples`).
    """
    sequence = load_sequence(SEQUENCE_PATH)
    sequence.constants = {
        'PULSE_LENGTH': pulse_length,
        'MEAS_DELAY': meas_delay,
        'OSC': osc,
        'START_FREQ': relative_start_freq,
        'FREQ_INCR': freq_incr,
        'N_SWEEP': n_sweep,
        'N_MEAS': n_meas,
    }
    awg_channel.awg.load_sequencer_program(sequence)
    awg_channel.awg.wait_done()

    ct_schema = awg_channel.awg.commandtable.load_validation_schema()
    ct = CommandTable(ct_schema)
    ct.table[0].waveform.index = 0
    ct.table[1].waveform.index = 1
    ct.table[2].waveform.index = 2
    ct.table[3].waveform.playZero = True
    ct.table[3].waveform.length = 16
    ct.table[4].waveform.playHold = True
    ct.table[4].waveform.length = pulse_length - meas_delay - 1024
    awg_channel.awg.commandtable.upload_to_device(ct)


def run_sweep(awg_channel, tt, n_sweep, n_meas, timeout):
    """Run one frequency sweep and return counts of shape (n_sweep, n_meas, 2).

    Trailing axis is (MW-active, MW-inactive) for each AM half-period.
    """
    cbm = CountBetweenMarkers(
        tt, TT_CLICK_CHANNEL,
        -TT_MARKER_CHANNEL, TT_MARKER_CHANNEL,
        2 * n_sweep * n_meas,
    )
    cbm.start()
    tt.sync()
    awg_channel.awg.enable_sequencer(single=True)
    awg_channel.awg.wait_done(timeout=timeout)
    while not cbm.ready():
        time.sleep(0.2)
    return np.array(cbm.getData()).reshape((n_sweep, n_meas, 2))