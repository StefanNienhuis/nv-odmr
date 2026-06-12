import numpy as np
import matplotlib.pyplot as plt
from util.sensitivity import calculate_sensitivity

results = np.load('../persist/cw_std/2026-06-03T16.00.42.831349.npz', allow_pickle=True)
counts = results['data']
print(counts)
params = results['params'].item()

slope = params['slope']
meas_time = 1#params['pulse_length_ns'] * params['n_meas'] / 1e9

gamma_nv = 28.0e9

sensitivity = calculate_sensitivity(counts, slope, meas_time)
print(f"{sensitivity*1e6} uT/sqrt(Hz)")

print(np.std(counts))
print(slope)
plt.plot(counts - np.mean(counts))
plt.show()