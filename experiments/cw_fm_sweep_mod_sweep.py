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

# Frequency sweep parameters
start_freq      = 2.84e9
stop_freq       = 2.90e9
mod_depth       = 3e6
n_sweep         = 401
n_meas          = 5
osc1            = 0
osc2            = 1
meas_delay_ns   = 20e3

freq_dev = mod_depth / 2

# Pulse length sweep parameters
start_pulse_length_ns = 5e6      # 5 ms → mod_freq = 100 Hz
stop_pulse_length_ns  = 200e6    # 200 ms → mod_freq = 2.5 Hz
n_pulse_lengths       = 5

pulse_lengths_ns = np.logspace(
    np.log10(start_pulse_length_ns),
    np.log10(stop_pulse_length_ns),
    n_pulse_lengths
)

# Gyromagnetic conversion: 36 μT/MHz = 36e-12 T/Hz
H_OVER_G_MU_B_T_PER_HZ = 36e-12

expected_duration = sum(2 * n_sweep * n_meas * pl / 1e9 for pl in pulse_lengths_ns)
print(f"Pulse lengths (ms): {pulse_lengths_ns/1e6}")
print(f"Modulation freq (Hz): {1e9/(2*pulse_lengths_ns)}")
print(f"Expected total duration: {expected_duration:.1f}s")
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

    cbm = CountBetweenMarkers(
        tt, TT_CLICK_CHANNEL,
        -TT_MARKER_CHANNEL, TT_MARKER_CHANNEL,
        2 * n_sweep * n_meas
    )
    cbm.start()
    tt.sync()
    awg_channel.awg.enable_sequencer(single=True)
    awg_channel.awg.wait_done(timeout=t_sweep*1.5)
    while not cbm.ready():
        time.sleep(0.2)
    counts = np.array(cbm.getData())
    counts = counts.reshape((n_sweep, n_meas, 2))
    all_counts.append(counts)

    mean_low  = counts[:, :, 0].mean(axis=1)
    mean_high = counts[:, :, 1].mean(axis=1)

    fm_signal = (mean_high - mean_low) / (mean_high + mean_low)
    all_fm_signals.append(fm_signal)

    try:
        # Primary metric: slope at zero crossing
        f0, slope, _ = find_slope_at_zero(freq, fm_signal, search_window_hz=mod_depth)

        # Off-resonance rate: top/bottom 10% of frequencies, excluding region near f0
        far_from_f0 = np.abs(freq - f0) > 2 * mod_depth
        edge_mask = ((freq < freq[0] + 0.1*(freq[-1]-freq[0])) |
                     (freq > freq[-1] - 0.1*(freq[-1]-freq[0])))
        off_mask = far_from_f0 & edge_mask
        if off_mask.sum() < 5:
            off_mask = edge_mask
        off_counts_per_pulse = (mean_low[off_mask].mean() + mean_high[off_mask].mean()) / 2
        R = off_counts_per_pulse / (pl_ns / 1e9)

        # Shot-noise on FM signal: per frequency point
        t_per_freq = 2 * n_meas * pl_ns / 1e9
        sigma_S = 1.0 / np.sqrt(R * t_per_freq)

        sigma_f0 = sigma_S / np.abs(slope)
        sigma_B = sigma_f0 * H_OVER_G_MU_B_T_PER_HZ
        eta_B = sigma_B * np.sqrt(t_sweep)

        f0_per_pl[pl_idx]         = f0
        slope_per_pl[pl_idx]      = slope
        count_rate_per_pl[pl_idx] = R
        eta_B_per_pl[pl_idx]      = eta_B
        sigma_B_per_pl[pl_idx]    = sigma_B

        # Diagnostic: fit Lorentzian to (low+high)/2 for δν, C
        mean_avg = (mean_low + mean_high) / 2.0
        try:
            offset_guess    = np.median(mean_avg)
            amplitude_guess = max(offset_guess - np.min(mean_avg), 1.0)
            f0_guess_diag   = freq[np.argmin(mean_avg)]
            gamma_guess     = 5e6
            popt, _ = curve_fit(
                lorentzian, freq, mean_avg,
                p0=[f0_guess_diag, gamma_guess, amplitude_guess, offset_guess],
                bounds=([freq[0], 100e3, 0, 0],
                        [freq[-1], 50e6, np.inf, np.inf]),
                maxfev=5000
            )
            _, gamma_lz, amp_lz, off_lz = popt
            linewidth_per_pl[pl_idx] = gamma_lz
            contrast_per_pl[pl_idx]  = amp_lz / off_lz
        except Exception:
            pass

        print(f"  f0 = {f0/1e9:.5f} GHz, slope = {slope*1e6:.3e} /MHz, R = {R/1e3:.0f} kHz")
        if not np.isnan(linewidth_per_pl[pl_idx]):
            print(f"  diagnostic: δν = {linewidth_per_pl[pl_idx]/1e6:.2f} MHz, "
                  f"C = {contrast_per_pl[pl_idx]:.3f}")
        print(f"  σ_f0 = {sigma_f0/1e3:.1f} kHz")
        print(f"  η_B  = {eta_B*1e6:.2f} μT/√Hz")
        print(f"  σ_B  = {sigma_B*1e9:.1f} nT (per sweep, {t_sweep:.2f} s)")
    except Exception as e:
        print(f"  FM analysis failed: {e}")

os.makedirs("../data/pulse_length_calib_fm", exist_ok=True)
np.savez(
    f"../data/pulse_length_calib_fm/{start_date.isoformat().replace(':', '.')}.npz",
    pulse_lengths_ns=pulse_lengths_ns,
    modulation_freqs=1e9/(2*pulse_lengths_ns),
    f0_per_pl=f0_per_pl,
    slope_per_pl=slope_per_pl,
    linewidth_per_pl=linewidth_per_pl,
    contrast_per_pl=contrast_per_pl,
    count_rate_per_pl=count_rate_per_pl,
    eta_B_per_pl=eta_B_per_pl,
    sigma_B_per_pl=sigma_B_per_pl,
    n_sweep=n_sweep,
    n_meas=n_meas,
    mod_depth=mod_depth,
    freq=freq,
    raw_counts=np.array(all_counts),
    fm_signals=np.array(all_fm_signals),
)

valid = ~np.isnan(eta_B_per_pl)
if not valid.any():
    print("No valid fits — aborting before plot.")
    raise SystemExit

fig, axes = plt.subplots(1, 2, figsize=(11, 4))

axes[0].loglog(pulse_lengths_ns[valid]/1e6, sigma_B_per_pl[valid]*1e6,
               'o-', markersize=8)
axes[0].set_xlabel('Pulse length (ms)')
axes[0].set_ylabel(r'$\sigma_B$ per sweep ($\mu$T)')
axes[0].set_title(r'FM per-sweep field precision: scales as $1/\sqrt{t}$')
axes[0].grid(True, which='both', alpha=0.5)

axes[1].semilogx(pulse_lengths_ns[valid]/1e6, eta_B_per_pl[valid]*1e6,
                 'o-', markersize=8)
axes[1].set_xlabel('Pulse length (ms)')
axes[1].set_ylabel(r'$\eta_B$ ($\mu$T/$\sqrt{\mathrm{Hz}}$)')
axes[1].set_title(r'FM bandwidth-normalized sensitivity: flat ideally')
axes[1].grid(True, which='both', alpha=0.5)

plt.tight_layout()
plt.show()

fig, axes = plt.subplots(2, 2, figsize=(11, 8))

axes[0, 0].semilogx(pulse_lengths_ns[valid]/1e6,
                    np.abs(slope_per_pl[valid])*1e6, 'o-')
axes[0, 0].set_xlabel('Pulse length (ms)')
axes[0, 0].set_ylabel('|dS/df| (per MHz)')
axes[0, 0].set_title('FM signal slope at zero crossing')
axes[0, 0].grid(True, which='both', alpha=0.5)

axes[0, 1].semilogx(pulse_lengths_ns[valid]/1e6,
                    linewidth_per_pl[valid]/1e6, 'o-')
axes[0, 1].set_xlabel('Pulse length (ms)')
axes[0, 1].set_ylabel('δν (MHz)')
axes[0, 1].set_title('Linewidth (diagnostic, from (H+L)/2)')
axes[0, 1].grid(True, which='both', alpha=0.5)

axes[1, 0].semilogx(pulse_lengths_ns[valid]/1e6, contrast_per_pl[valid], 'o-')
axes[1, 0].set_xlabel('Pulse length (ms)')
axes[1, 0].set_ylabel('C')
axes[1, 0].set_title('Contrast (diagnostic)')
axes[1, 0].grid(True, which='both', alpha=0.5)

axes[1, 1].semilogx(pulse_lengths_ns[valid]/1e6,
                    count_rate_per_pl[valid]/1e3, 'o-')
axes[1, 1].set_xlabel('Pulse length (ms)')
axes[1, 1].set_ylabel('R (kHz)')
axes[1, 1].set_title('Off-resonance count rate')
axes[1, 1].grid(True, which='both', alpha=0.5)

plt.tight_layout()
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