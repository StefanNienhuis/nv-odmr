from datetime import datetime, timedelta
import numpy as np
import matplotlib.pyplot as plt

from util import cw_fm_bin, set_mw

start_date = datetime.now()

# Parameters
osc1            = 0        # First oscillator being swept
osc2            = 1        # Second oscillator being swept
mod_depth       = 3e6      # FM modulation depth (Hz)
meas_time       = 1        # Time to measure for at each frequency
n_std           = 50
lock_in_mode    = 'corr'

freq_dev = mod_depth / 2

# Obtain from cw_fm_bin_mod_sweep - find maximum slope frequency for each modulation frequency
modulation_freqs = [1.0000005120002622, 3.5938178512810324, 12.915443315875788, 46.41677361091307, 166.81969593440382, 599.4858809085329, 2155.7671081677704, 7750.496031746032, 27901.785714285714, 97656.25]
drive_freqs = [2873150000.0, 2873000000.0, 2873150000.0, 2873300000.0, 2873300000.0, 2873300000.0, 2873450000.0, 2873300000.0, 2869400000.0, 2868500000.0]
slopes = [2.339312503263235e-08, 2.371435476923821e-08, 2.257971618605307e-08, 2.1740709942055653e-08, 2.097017327611756e-08, 2.0265262465755565e-08, 1.7124748880540454e-08, 1.1509786489582848e-08, 2.287104748403852e-09, 2.6228844054999794e-09]
# Parameters stored in output file
params = {
    "modulation_freqs": modulation_freqs,
    "drive_freqs": drive_freqs,
    "mod_depth": mod_depth,
    "meas_time": meas_time,
    "slopes": slopes, # included for further processing
    "lock_in_mode": lock_in_mode
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
        freq, sweep_fm_counts = cw_fm_bin.perform_sweep(modulation_freq, freq_dev, osc1, osc2, drive_freq, drive_freq, 1, n_meas, show_progress=False, lock_in_mode=lock_in_mode)

        fm_counts.append(sweep_fm_counts[0])

    fm_counts_per_modulation_freq.append(np.array(fm_counts))

set_mw.set_steady()

np.savez(f'../data/cw_fm_bin_mod_sweep_std/{start_date.isoformat().replace(":", ".")}.npz', data=fm_counts_per_modulation_freq, params=params)

std_per_modulation_freq = [np.std(counts, ddof=1) for counts in fm_counts_per_modulation_freq]

plt.semilogx(modulation_freqs, std_per_modulation_freq)
plt.legend()
plt.show()
