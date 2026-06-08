from datetime import datetime, timedelta
import numpy as np
import matplotlib.pyplot as plt

from util import cw_am_bin, set_mw

start_date = datetime.now()

# Parameters
osc             = 0        # Oscillator being swept
meas_time       = 1        # Time to measure for at each frequency
n_std           = 50
lock_in_mode    = 'corr'

# Obtain from cw_am_bin_mod_sweep - find maximum slope frequency for each modulation frequency
modulation_freqs = [1.0000005120002622, 3.5938178512810324, 12.915443315875788, 46.41677361091307, 166.81969593440382, 599.4858809085329, 2155.7671081677704, 7750.496031746032, 27901.785714285714, 97656.25]
drive_freqs = [2867450000.0, 2867300000.0, 2867300000.0, 2874200000.0, 2874500000.0, 2874650000.0, 2874650000.0, 2867600000.0, 2867600000.0, 2867750000.0]
slopes = [4.021967680045762e-08, 3.6766592210885816e-08, 3.6220753811134534e-08, 1.947143100611379e-08, 3.2508174854627886e-08, 3.1279098163388185e-08, 2.702608255070423e-08, 2.1066131393939318e-08, 1.315323917244926e-08, 4.099103631904031e-09]
params = {
    "modulation_freqs": modulation_freqs,
    "drive_freqs": drive_freqs,
    "meas_time": meas_time,
    "slopes": slopes, # included for further processing
    'lock_in_mode': lock_in_mode
}

expected_duration = n_std * meas_time * len(modulation_freqs)
print(f"Expected duration: {expected_duration}s")
print(f"Finished at: {(datetime.now() + timedelta(seconds=expected_duration)).time()}")

am_counts_per_modulation_freq = []

print()

for i, (modulation_freq, drive_freq) in enumerate(zip(modulation_freqs, drive_freqs)):
    print(f'[{i + 1}/{len(modulation_freqs)}] Sweeping at {modulation_freq} Hz modulation...')
    
    n_meas = int(round(modulation_freq * meas_time))
    if n_meas != round(n_meas):
        print(f"Warning: number of measurements is rounded: {n_meas} instead of {modulation_freq * meas_time}")

    am_counts = []
    
    for n in range(n_std):
        freq, sweep_am_counts = cw_am_bin.perform_sweep(modulation_freq, osc, drive_freq, drive_freq, 1, n_meas, show_progress=False, lock_in_mode=lock_in_mode)

        am_counts.append(sweep_am_counts[0])

    am_counts_per_modulation_freq.append(np.array(am_counts))

set_mw.set_steady()

np.savez(f'../data/cw_am_bin_mod_sweep_std/{start_date.isoformat().replace(":", ".")}.npz', data=am_counts_per_modulation_freq, params=params)

std_per_modulation_freq = [np.std(counts, ddof=1) for counts in am_counts_per_modulation_freq]

plt.semilogx(modulation_freqs, std_per_modulation_freq)
plt.legend()
plt.show()
