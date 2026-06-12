import numpy as np
import matplotlib.pyplot as plt
from cw import simulate_cw
from processing.util import double_lorentzian_diff_fit, plot_style

# Simulation parameters
Bz = 150e-6             # magnetic field along NV axis (T)
Amw = 6e5               # microwave field amplitude (Hz)

f_min = 2.84e9
f_max = 2.90e9
n_points = 401
freqs = np.linspace(f_min, f_max, n_points)

mod_depth = 3e6
delta_f = mod_depth / 2

a_results = [simulate_cw(Bz, Amw, f_mw - delta_f) for f_mw in freqs]
a_results = np.array(a_results)
a_results = a_results / np.max(a_results)

b_results = [simulate_cw(Bz, Amw, f_mw + delta_f) for f_mw in freqs]
b_results = np.array(b_results)
b_results = b_results / np.max(b_results)

background_noise = np.random.randn(len(a_results)) * 0.01

a_results += background_noise
b_results += background_noise

fm_results = b_results - a_results

detunings = freqs - 2.87e9

curve, params = double_lorentzian_diff_fit.fit_curve(detunings, -fm_results, mod_depth=mod_depth, dip=False)
fit = curve(detunings, *params)

if __name__ == '__main__':
    # Plot
    plt.figure(figsize=(5, 3.5))
    plt.plot(detunings / 1e6, fm_results, marker="o", ls="none", ms=3.2,
                mfc="none", mec="0.35", mew=0.8, label="Data")
    plt.plot(detunings / 1e6, -fit, color="C3", lw=1.4, label="Lorentzian difference fit")
    plt.xlabel("Detuning $\\delta = f - D$ (MHz)")
    plt.ylabel("FM signal (a.u.)")
    plt.suptitle(f"Simulated FM CW-ODMR frequency sweep (B = {Bz*1e6} $\\mu$T)")

    plt.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=2,
              handlelength=1.6, columnspacing=1.1, borderaxespad=0.0)

    plt.tight_layout(rect=(0, 0, 1, 1.025))

    plt.grid(True)
    plt.show()
