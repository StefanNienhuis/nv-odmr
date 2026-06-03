from datetime import datetime, timedelta
import numpy as np
import matplotlib.pyplot as plt

from util import cw_fm, set_mw

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
modulation_freqs = [np.float64(1.0), np.float64(4.0), np.float64(13.0), np.float64(46.0), np.float64(167.0), np.float64(599.0), np.float64(2154.0), np.float64(7743.0), np.float64(27826.0), np.float64(100000.0)]
drive_freqs = [np.float64(2873200000.0), np.float64(2873200000.0), np.float64(2873200000.0), np.float64(2873200000.0), np.float64(2873200000.0), np.float64(2873200000.0), np.float64(2873200000.0), np.float64(2873200000.0), np.float64(2873200000.0), np.float64(2873200000.0)]
slopes = [np.float64(2.3123299369765924e-08), np.float64(2.246584035770707e-08), np.float64(2.227749635803615e-08), np.float64(2.2072198178773187e-08), np.float64(2.146769842659301e-08), np.float64(1.9966719908730045e-08), np.float64(1.7825020202729907e-08), np.float64(1.477820425801757e-08), np.float64(1.0294056471115221e-08), np.float64(5.95219033621928e-09)]

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
        freq, sweep_counts = cw_fm.perform_sweep(modulation_freq, freq_dev, meas_delay_ns, osc1, osc2, drive_freq, drive_freq, 1, n_meas, show_progress=False)

        # Taking sweep index 0 as only one frequency is used
        mean_low_counts = np.mean(sweep_counts[0, :, 0])
        mean_high_counts = np.mean(sweep_counts[0, :, 1])

        fm_counts.append((mean_high_counts - mean_low_counts) / (mean_high_counts + mean_low_counts))

    fm_counts_per_modulation_freq.append(np.array(fm_counts))

set_mw.set_steady()

fm_counts_per_modulation_freq = np.array(fm_counts_per_modulation_freq)
np.savez(f'../data/cw_fm_mod_sweep_std/{start_date.isoformat().replace(":", ".")}.npz', data=fm_counts_per_modulation_freq, params=params)

std_per_modulation_freq = [np.std(counts, ddof=1) for counts in fm_counts_per_modulation_freq]

plt.semilogx(modulation_freqs, std_per_modulation_freq)
plt.legend()
plt.show()
