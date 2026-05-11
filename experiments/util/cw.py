import time

import numpy as np
from zhinst.toolkit import Session, CommandTable
from TimeTagger import CountBetweenMarkers, createTimeTaggerNetwork


# Device parameters
AWG_SERVER_HOST = 'localhost'
AWG_SERVER_PORT = 8004
AWG_DEVICE = 'DEV12120'
AWG_CHANNEL = 2
AWG_SAMPLE_RATE = 2e9

TT_NETWORK_ADDRESS = 'localhost:41101'
TT_CLICK_CHANNEL = 1
TT_MARKER_CHANNEL = 2

CENTER_FREQ = 2.8e9


def ns_to_samples(ns):
    """Convert nanoseconds to AWG samples, rounded to a multiple of 16
    (the AWG zero-pads otherwise)."""
    return int(round(ns * AWG_SAMPLE_RATE / 1e9 / 16) * 16)


def init_awg(osc_frequency, osc=0):
    """Connect to the AWG and configure the signal generator channel:
    channel output, sine generation (disabled), pulse modulation, and
    marker/trigger. Returns the configured awg_channel."""
    session = Session(AWG_SERVER_HOST, AWG_SERVER_PORT)
    device = session.connect_device(AWG_DEVICE)
    device.check_compatibility()

    channel = device.sgchannels[AWG_CHANNEL]
    channel.configure_channel(
        enable=True,
        output_range=10,
        center_frequency=CENTER_FREQ,
        rf_path=True,
    )
    channel.configure_sine_generation(
        enable=False,
        osc_index=osc,
        osc_frequency=osc_frequency,
        phase=0,
    )
    channel.configure_pulse_modulation(
        enable=True,
        osc_index=osc,
        osc_frequency=osc_frequency,
        phase=0,
    )
    channel.awg.configure_marker_and_trigger(
        trigger_in_source='trigin0',
        trigger_in_slope='rising_edge',
        marker_out_source='output0_marker0',
    )
    return channel


def init_time_tagger():
    """Connect to the time tagger and set trigger levels."""
    tt = createTimeTaggerNetwork(TT_NETWORK_ADDRESS)
    tt.setTriggerLevel(TT_CLICK_CHANNEL, 0.25)
    tt.setTriggerLevel(TT_MARKER_CHANNEL, 0.5)
    return tt


def make_count_between_markers(tt, n_values):
    """Create a CountBetweenMarkers with the standard channel configuration.
    The marker channel is inverted."""
    return CountBetweenMarkers(
        tt,
        TT_CLICK_CHANNEL,
        -TT_MARKER_CHANNEL,
        TT_MARKER_CHANNEL,
        n_values,
    )


def upload_command_table(awg_channel, pulse_length, meas_delay):
    """Upload the standard 3-entry command table:
        Entry 0: play waveform 0
        Entry 1: play waveform 1
        Entry 2: hold for the remaining pulse_length - meas_delay - 1024 samples

    A command table is used since it's more efficient than playWave:
    https://docs.zhinst.com/shfsg_user_manual/tutorials/tutorial_command_table.html#introduction-to-the-command-table
    """
    schema = awg_channel.awg.commandtable.load_validation_schema()
    ct = CommandTable(schema)
    ct.table[0].waveform.index = 0
    ct.table[1].waveform.index = 1
    ct.table[2].waveform.playHold = True
    ct.table[2].waveform.length = pulse_length - meas_delay - 1024
    awg_channel.awg.commandtable.upload_to_device(ct)


def run_acquisition(awg_channel, tt, cbm, timeout):
    """Start the count-between-markers measurement, sync the time tagger,
    run the AWG sequencer once, wait for it to finish, then return the
    collected counts as a numpy array."""
    cbm.start()
    tt.sync()
    awg_channel.awg.enable_sequencer(single=True)
    awg_channel.awg.wait_done(timeout=timeout)
    while not cbm.ready():
        time.sleep(0.2)
    return np.array(cbm.getData())