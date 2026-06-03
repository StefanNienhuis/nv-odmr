from datetime import datetime, timedelta
import numpy as np
import matplotlib.pyplot as plt

from util import cw_am

start_date = datetime.now()

# Parameters
meas_delay_ns   = 1e3     # Delay before measuring (ns)
osc             = 0        # Oscillator being swept
meas_time       = 1        # Time to measure for at each frequency
n_std           = 50

# Obtain from cw_am_mod_sweep - find maximum slope frequency for each modulation frequency
modulation_freqs = [1.0, 4.0, 13.0, 46.0, 167.0, 599.0, 2154.0, 7743.0, 27826.0, 100000.0]
drive_freqs = [2867000000.0, 2874650000.0, 2874650000.0, 2867300000.0, 2867150000.0, 2867150000.0, 2874500000.0, 2874350000.0, 2874500000.0, 2867000000.0]
slopes = [2.4364741645768695e-08, 2.722195171357124e-08, 2.787380734796216e-08, 2.7732434312461632e-08, 2.7310118180031892e-08, 2.61876394397798e-08, 2.1949315644410604e-08, 1.973484106278914e-08, 1.0690769519668385e-08, 4.89900293656966e-09]

# Parameters stored in output file
params = {
    "modulation_freqs": modulation_freqs,
    "meas_delay_ns": meas_delay_ns,
    "drive_freqs": drive_freqs,
    "meas_time": meas_time,
    "slopes": slopes # included for further processing
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
        freq, sweep_counts = cw_am.perform_sweep(modulation_freq, meas_delay_ns, osc, drive_freq, drive_freq, 1, n_meas, show_progress=False)

        # Taking sweep index 0 as only one frequency is used
        mean_active_counts = np.mean(sweep_counts[0, :, 0])
        mean_inactive_counts = np.mean(sweep_counts[0, :, 1])

        am_counts.append((mean_inactive_counts - mean_active_counts) / mean_inactive_counts)

    am_counts_per_modulation_freq.append(np.array(am_counts))

am_counts_per_modulation_freq = np.array(am_counts_per_modulation_freq)
np.savez(f'../data/cw_am_mod_sweep_std/{start_date.isoformat().replace(":", ".")}.npz', data=am_counts_per_modulation_freq, params=params)

std_per_modulation_freq = [np.std(counts, ddof=1) for counts in am_counts_per_modulation_freq]

plt.semilogx(modulation_freqs, std_per_modulation_freq)
plt.legend()
plt.show()
