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

def max_slope_per_pulse_length(all_counts, freq):
    """
    For each pulse length, build the normalized AM signal
        am_counts = (mean_inactive - mean_active) / mean_inactive
    and return:
      - the maximum |d(am_counts)/d(freq)|
      - the frequency at which that maximum occurs

    Parameters
    ----------
    all_counts : list of np.ndarray
        One entry per pulse length, each of shape (n_sweep, n_meas, 2).
        The trailing axis is (MW-active, MW-inactive).
    freq : np.ndarray, shape (n_sweep,)
        Frequency axis used in the sweeps (same for every pulse length).

    Returns
    -------
    max_slopes : np.ndarray, shape (n_pulse_lengths,)
        Maximum absolute slope of am_counts vs freq.
    freqs_at_max : np.ndarray, shape (n_pulse_lengths,)
        Frequency at which the maximum slope is reached.
    """
    n_pl = len(all_counts)
    max_slopes   = np.empty(n_pl)
    freqs_at_max = np.empty(n_pl)

    for i, counts in enumerate(all_counts):
        # average over the n_meas repetitions at each frequency step
        mean_active   = counts[:, :, 0].mean(axis=1)   # MW ON
        mean_inactive = counts[:, :, 1].mean(axis=1)   # MW OFF
        am_counts = (mean_inactive - mean_active) / mean_inactive

        # slope d(am_counts)/d(freq) via central differences
        slope = np.gradient(am_counts, freq)

        idx = int(np.argmax(np.abs(slope)))
        max_slopes[i]   = np.abs(slope[idx])
        freqs_at_max[i] = freq[idx]

    return max_slopes, freqs_at_max

def std_per_pulse_length(all_std_counts):
    """
    For each pulse length, compute the per-shot normalized AM signal
 
        am_per_shot = (inactive - active) / inactive
 
    at a single (fixed) frequency, then return the standard deviation of
    that quantity across the n_std_meas repetitions. The input shape is
    expected to be (1, n_std_meas, 2) -- i.e. produced by `run_sweep` with
    `n_sweep=1` and `n_meas=n_std_meas`.
 
    Parameters
    ----------
    all_std_counts : list of np.ndarray
        One entry per pulse length, each of shape (1, n_std_meas, 2).
        The trailing axis is (MW-active, MW-inactive).
 
    Returns
    -------
    stds : np.ndarray, shape (n_pulse_lengths,)
        Std of am_per_shot across the n_std_meas repetitions.
    """
    n_pl = len(all_std_counts)
    stds = np.empty(n_pl)
 
    for i, counts in enumerate(all_std_counts):
        active   = counts[0, :, 0]   # MW ON
        inactive = counts[0, :, 1]   # MW OFF
        am_per_shot = (inactive - active) / inactive
        stds[i] = am_per_shot.std()
 
    return stds
