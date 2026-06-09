import numpy as np
import matplotlib.pyplot as plt
from cw import simulate_cw
from processing.util import plot_style, double_lorentzian_fit

# Simulation parameters
Bz = 150e-6             # magnetic field along NV axis (T)
Amw = 10e5               # microwave field amplitude (Hz)

f_min = 2.84e9
f_max = 2.90e9
n_points = 401
freqs = np.linspace(f_min, f_max, n_points)

fluorescence_results = [simulate_cw(Bz, Amw, f_mw) for f_mw in freqs]
fluorescence_results = np.array(fluorescence_results)
fluorescence_results = fluorescence_results / np.max(fluorescence_results)

background_noise = np.random.randn(len(fluorescence_results)) * 0.01

fluorescence_results += background_noise

detunings = freqs - 2.87e9

curve, params = double_lorentzian_fit.fit_curve(detunings, fluorescence_results)
fit = curve(detunings, *params)

if __name__ == '__main__':
    # Plot
    plt.figure()
    plt.plot(detunings / 1e6, fluorescence_results, marker="o", ls="none", ms=3.2,
                mfc="none", mec="0.35", mew=0.8, label="Data")
    plt.plot(detunings / 1e6, fit, color="C3", lw=1.4, label="Lorentzian fit")
    plt.xlabel("Detuning $\\delta = f - D$ (MHz)")
    plt.ylabel("Normalized fluorescence")
    plt.suptitle(f"Simulated CW-ODMR spectrum  (B = {Bz*1e6} $\\mu$T)")
    plt.grid(True)

    plt.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=2,
              handlelength=1.6, columnspacing=1.1, borderaxespad=0.0)

    plt.tight_layout(rect=(0, 0, 1, 1.025))

    plt.show()

    print(f"Dip frequency: {freqs[np.argmin(fluorescence_results)]}")