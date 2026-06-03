from datetime import datetime, timedelta
import numpy as np
import matplotlib.pyplot as plt

from util import cw_fm

start_date = datetime.now()

# Parameters
meas_delay_ns   = 1e3     # Delay before measuring (ns)
osc1            = 0        # First oscillator being swept
osc2            = 1        # Second oscillator being swept
meas_time       = 1        # Time to measure for at each frequency
mod_depth       = 3e6      # FM modulation depth (Hz)
n_std           = 50

freq_dev = mod_depth / 2

# Obtain from cw_fm_mod_sweep - find maximum slope frequency for each modulation frequency
modulation_freqs = [1.0, 2.0, 3.0, 6.0, 11.0, 21.0, 38.0, 70.0, 127.0, 234.0, 428.0, 785.0, 1438.0, 2637.0, 4833.0, 8859.0, 16238.0, 29764.0, 54556.0, 100000.0]
drive_freqs = [2865600000.0, 2874400000.0, 2866000000.0, 2866000000.0, 2866000000.0, 2866000000.0, 2865600000.0, 2866000000.0, 2866000000.0, 2866000000.0, 2866000000.0, 2866000000.0, 2866000000.0, 2866000000.0, 2866000000.0, 2866400000.0, 2874000000.0, 2866000000.0, 2866400000.0, 2868000000.0]
slopes = [1.97335326469559e-08, 4.4853112745912805e-08, 2.3311749527669316e-08, 2.38000646019405e-08, 2.205877108814881e-08, 2.18016007880072e-08, 2.0799809894860057e-08, 2.2318638692230916e-08, 2.072805698337305e-08, 2.10708404571209e-08, 1.9010174853594503e-08, 1.808603212144027e-08, 1.654849468248466e-08, 1.5097324308013797e-08, 1.2452869870625879e-08, 1.0432453893594902e-08, 9.009540181040789e-09, 5.525078576855236e-09, 3.957393777901247e-09, 7.085585810659469e-09]

# Parameters stored in output file
params = {
    "modulation_freqs": modulation_freqs,
    "meas_delay_ns": meas_delay_ns,
    "drive_freqs": drive_freqs,
    "mod_depth": mod_depth,
    "meas_time": meas_time,
    "slopes": slopes # included for further processing
}

expected_duration = n_std * meas_time * len(modulation_freqs)
print(f"Expected duration: {expected_duration}s")
print(f"Finished at: {(datetime.now() + timedelta(seconds=expected_duration)).time()}")

fm_counts_per_modulation_freq = []

print()

for i, (modulation_freq, drive_freq) in enumerate(zip(modulation_freqs, drive_freqs)):
    print(f'[{i + 1}/{len(modulation_freqs)}] Sweeping at {modulation_freq} Hz modulation...')
    
    n_meas = int(round(modulation_freq * meas_time))
    if n_meas != round(n_meas):
        print(f"Warning: number of measurements is rounded: {n_meas} instead of {modulation_freq * meas_time}")

    fm_counts = []
    
    for n in range(n_std):
        freq, sweep_counts = cw_fm.perform_sweep(modulation_freq, freq_dev, meas_delay_ns, osc1, osc2, drive_freq, drive_freq, 1, n_meas)

        # Taking sweep index 0 as only one frequency is used
        mean_low_counts = np.mean(sweep_counts[0, :, 0])
        mean_high_counts = np.mean(sweep_counts[0, :, 1])

        fm_counts.append((mean_high_counts - mean_low_counts) / (mean_high_counts + mean_low_counts))

    fm_counts_per_modulation_freq.append(np.array(fm_counts))

fm_counts_per_modulation_freq = np.array(fm_counts_per_modulation_freq)
np.savez(f'../data/cw_fm_mod_sweep_std/{start_date.isoformat().replace(":", ".")}.npz', data=fm_counts_per_modulation_freq, params=params)

std_per_modulation_freq = [np.std(counts, ddof=1) for counts in fm_counts_per_modulation_freq]

plt.semilogx(modulation_freqs, std_per_modulation_freq)
plt.legend()
plt.show()
