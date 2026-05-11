import numpy as np
import matplotlib.pyplot as plt

from util.rabi_oscillation_fit import fit_curve

results = np.load('../persist/rabi_sweep/')
counts = results['data']
params = results['params']

tau_ns = np.linspace(params['start_tau_ns'], params['stop_tau_ns'], params['n_sweep'])

ref_counts = counts[:,:,0]
meas_counts = counts[:,:,1]

mean_ref_counts = np.mean(ref_counts, axis=1)
mean_meas_counts = np.mean(meas_counts, axis=1)

mean_counts_norm = (mean_ref_counts - mean_meas_counts) / mean_ref_counts

curve, params = fit_curve(tau_ns, mean_counts_norm)
fit = curve(tau_ns, *params)

plt.plot(tau_ns * 1e9, mean_counts_norm, 'x', color='gray', label='Data')
plt.plot(tau_ns * 1e9, fit, color='red', label='Fit')
plt.xlabel('$\\tau$ (ns)')
plt.legend()
plt.show()