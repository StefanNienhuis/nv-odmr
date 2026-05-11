# TODO:
#  - AWG server parameters correct?
#  - AWG channel number and power
#  - Time tagger channel numbers, trigger level

import os
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

# Frequency sweep parameters
start_freq      = 2.84e9
stop_freq       = 2.90e9
n_sweep         = 401
n_meas          = 5
osc             = 0
meas_delay_ns   = 50e3

# Pulse length sweep parameters
start_pulse_length_ns = 5e6      # 5 ms
stop_pulse_length_ns  = 200e6    # 200 ms
n_pulse_lengths       = 5

pulse_lengths_ns = np.logspace(
    np.log10(start_pulse_length_ns),
    np.log10(stop_pulse_length_ns),
    n_pulse_lengths
)

# Gyromagnetic conversion (h / g μ_B)
H_OVER_G_MU_B = 36e-6  # T per MHz

expected_duration = sum(n_sweep * n_meas * pl / 1e9 for pl in pulse_lengths_ns)
print(f"Pulse lengths (ms): {pulse_lengths_ns/1e6}")
print(f"Expected total duration: {expected_duration:.1f}s")
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

    # Update command table
    ct_schema = awg_channel.awg.commandtable.load_validation_schema()
    ct = CommandTable(ct_schema)
    ct.table[0].waveform.index = 0
    ct.table[1].waveform.index = 1
    ct.table[2].waveform.playHold = True
    ct.table[2].waveform.length = pulse_length - meas_delay - 1024
    awg_channel.awg.commandtable.upload_to_device(ct)

    # Single sweep
    cbm = CountBetweenMarkers(
        tt, TT_CLICK_CHANNEL,
        -TT_MARKER_CHANNEL, TT_MARKER_CHANNEL, n_sweep
    )
    cbm.start()
    tt.sync()
    awg_channel.awg.enable_sequencer(single=True)
    awg_channel.awg.wait_done(timeout=sweep_duration*1.5)
    while not cbm.ready():
        time.sleep(0.2)
    counts = np.array(cbm.getData())
    all_counts.append(counts)

    # Fit Lorentzian
    try:
        offset_guess    = np.median(counts)
        amplitude_guess = max(offset_guess - np.min(counts), 1.0)
        f0_guess        = freq[np.argmin(counts)]
        gamma_guess     = 5e6
        popt, _ = curve_fit(
            lorentzian, freq, counts,
            p0=[f0_guess, gamma_guess, amplitude_guess, offset_guess],
            bounds=([freq[0], 100e3, 0, 0],
                    [freq[-1], 50e6, np.inf, np.inf]),
            maxfev=5000
        )
        f0, gamma, amplitude, offset = popt
        R = offset / (pl_ns / 1e9)
        C = amplitude / offset

        f0_per_pl[pl_idx]         = f0
        linewidth_per_pl[pl_idx]  = gamma
        contrast_per_pl[pl_idx]   = C
        count_rate_per_pl[pl_idx] = R

        eta_B = H_OVER_G_MU_B * (gamma / 1e6) / (C * np.sqrt(R))
        eta_B_per_pl[pl_idx] = eta_B

        sigma_B = eta_B / np.sqrt(t_sweep)
        sigma_B_per_pl[pl_idx] = sigma_B

        print(f"  δν = {gamma/1e6:.2f} MHz, C = {C:.3f}, R = {R/1e3:.0f} kHz")
        print(f"  η_B  = {eta_B*1e6:.2f} μT/√Hz")
        print(f"  σ_B  = {sigma_B*1e9:.1f} nT (per sweep, {t_sweep:.2f} s)")
    except Exception as e:
        print(f"  fit failed: {e}")

# Save
os.makedirs("../data/pulse_length_calib", exist_ok=True)
np.savez(
    f"../data/pulse_length_calib/{start_date.isoformat().replace(':', '.')}.npz",
    pulse_lengths_ns=pulse_lengths_ns,
    linewidth_per_pl=linewidth_per_pl,
    contrast_per_pl=contrast_per_pl,
    count_rate_per_pl=count_rate_per_pl,
    f0_per_pl=f0_per_pl,
    eta_B_per_pl=eta_B_per_pl,
    sigma_B_per_pl=sigma_B_per_pl,
    n_sweep=n_sweep,
    n_meas=n_meas,
    freq=freq,
    raw_counts=np.array(all_counts),
)

valid = ~np.isnan(eta_B_per_pl)
if not valid.any():
    print("No valid fits — aborting before plot.")
    raise SystemExit

# Plot: dual view of sensitivity
fig, axes = plt.subplots(1, 2, figsize=(11, 4))

axes[0].loglog(pulse_lengths_ns[valid]/1e6, sigma_B_per_pl[valid]*1e6,
               'o-', markersize=8)
axes[0].set_xlabel('Pulse length (ms)')
axes[0].set_ylabel(r'$\sigma_B$ per sweep ($\mu$T)')
axes[0].set_title(r'CW per-sweep field precision: scales as $1/\sqrt{t}$')
axes[0].grid(True, which='both', alpha=0.5)

axes[1].semilogx(pulse_lengths_ns[valid]/1e6, eta_B_per_pl[valid]*1e6,
                 'o-', markersize=8)
axes[1].set_xlabel('Pulse length (ms)')
axes[1].set_ylabel(r'$\eta_B$ ($\mu$T/$\sqrt{\mathrm{Hz}}$)')
axes[1].set_title(r'CW bandwidth-normalized sensitivity: flat ideally')
axes[1].grid(True, which='both', alpha=0.5)

plt.tight_layout()
plt.show()

# Diagnostic: fit parameters separately
fig, axes = plt.subplots(1, 3, figsize=(13, 4))
axes[0].semilogx(pulse_lengths_ns[valid]/1e6, linewidth_per_pl[valid]/1e6, 'o-')
axes[0].set_xlabel('Pulse length (ms)')
axes[0].set_ylabel('δν (MHz)')
axes[0].set_title('Linewidth')
axes[0].grid(True, which='both', alpha=0.5)

axes[1].semilogx(pulse_lengths_ns[valid]/1e6, contrast_per_pl[valid], 'o-')
axes[1].set_xlabel('Pulse length (ms)')
axes[1].set_ylabel('C')
axes[1].set_title('Contrast')
axes[1].grid(True, which='both', alpha=0.5)

axes[2].semilogx(pulse_lengths_ns[valid]/1e6, count_rate_per_pl[valid]/1e3, 'o-')
axes[2].set_xlabel('Pulse length (ms)')
axes[2].set_ylabel('R (kHz)')
axes[2].set_title('Off-resonance count rate')
axes[2].grid(True, which='both', alpha=0.5)

plt.tight_layout()
plt.show()