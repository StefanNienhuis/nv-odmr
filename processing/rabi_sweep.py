import numpy as np
import matplotlib.pyplot as plt

from util.rabi_oscillation_fit import fit_curve

results = np.load('../persist/rabi_sweep/2026-05-11T16.55.35.007772.npz', allow_pickle=True)
counts = results['data']
params = results['params'].item()

tau_ns = np.linspace(params['start_tau_ns'], params['stop_tau_ns'], params['n_sweep'])

meas_counts = counts[:,:,0]
ref_counts = counts[:,:,1]

total_ref_counts = np.sum(ref_counts, axis=1)
total_meas_counts = np.sum(meas_counts, axis=1)

total_counts_norm = (total_ref_counts - total_meas_counts) / total_ref_counts

curve, params = fit_curve(tau_ns, total_counts_norm)
fit = curve(tau_ns, *params)

print(*params)

plt.plot(tau_ns, mean_counts_norm, 'x', color='gray', label='Data')
plt.plot(tau_ns, fit, color='red', label='Fit')
plt.xlabel('$\\tau$ (ns)')
plt.legend()
plt.show()