import numpy as np
import matplotlib.pyplot as plt

from util.rabi_oscillation_fit import fit_curve
from util import plot_style

results = np.load('../data/rabi_sweep/2026-05-26T15.48.16.526758.npz', allow_pickle=True)
counts = results['data']
params = results['params'].item()

tau_ns = np.linspace(params['start_tau_ns'], params['stop_tau_ns'], params['n_sweep'])

counts = counts[:, :, :]
tau_ns = tau_ns[:]

meas_counts = counts[:,:,0]
ref_counts = counts[:,:,1]

total_ref_counts = np.sum(ref_counts, axis=1)
total_meas_counts = np.sum(meas_counts, axis=1)

total_counts_norm = (total_ref_counts - total_meas_counts) / total_ref_counts

curve, params = fit_curve(tau_ns, total_counts_norm)
fit = curve(tau_ns, *params)

print(*params)

plt.suptitle('Normalized Rabi signal')
plt.plot(tau_ns, total_counts_norm, color="C3", lw=1.4, label="Data")
# plt.plot(tau_ns, total_ref_counts / 2, label='Ref')
# plt.plot(tau_ns, total_meas_counts, label='Meas')
# plt.plot(tau_ns, fit, color='red', label='Fit')
plt.xlabel('$\\tau$ (ns)')
plt.ylabel('Normalized signal (a.u)')
plt.grid(True)

plt.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=1,
          handlelength=1.6, columnspacing=1.1, borderaxespad=0.0)

plt.tight_layout(rect=(0, 0, 1, 1.025))
plt.show()