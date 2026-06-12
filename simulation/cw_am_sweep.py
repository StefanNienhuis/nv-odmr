import numpy as np
import matplotlib.pyplot as plt
from matplotlib.transforms import ScaledTranslation

from cw import simulate_cw
from processing.util import double_lorentzian_fit, plot_style

# Simulation parameters
Bz = 150e-6             # magnetic field along NV axis (T)
Amw = 6e5               # microwave field amplitude (Hz)

f_min = 2.84e9
f_max = 2.90e9
n_points = 401
freqs = np.linspace(f_min, f_max, n_points)

fluorescence_results = [simulate_cw(Bz, Amw, f_mw) for f_mw in freqs]
fluorescence_results = np.array(fluorescence_results)
fluorescence_results = fluorescence_results / np.max(fluorescence_results)

background_noise = np.random.randn(len(fluorescence_results)) * 0.01

fluorescence_results += background_noise

on_results = fluorescence_results
off_results = 1 + background_noise

am_results = off_results - on_results

detunings = freqs - 2.87e9

curve, params = double_lorentzian_fit.fit_curve(detunings, am_results, dip=False)
fit = curve(detunings, *params)

if __name__ == '__main__':
    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle(f"Simulated AM CW-ODMR frequency sweep (B = {Bz * 1e6} $\\mu$T)")
    fig.set_figwidth(9)
    fig.set_figheight(3.5)
    # plt.figure(figsize=(10.8, 4.8))

    ax1.plot(freqs / 1e9, on_results, label="MW on")
    ax1.plot(freqs / 1e9, off_results, label="MW off")
    ax1.set_xlabel("Detuning $\\delta = f - D$ (MHz)")
    ax1.set_ylabel("Fluorescence (a.u)")
    ax1.legend()
    ax1.grid(True)

    ax2.plot(detunings / 1e6, am_results, marker="o", ls="none", ms=3.2,
                mfc="none", mec="0.35", mew=0.8, label="Data")
    ax2.plot(detunings / 1e6, fit, color="C3", lw=1.4, label="Lorentzian fit")
    ax2.set_xlabel("Detuning $\\delta = f - D$ (MHz)")
    ax2.set_ylabel("AM signal (a.u)")
    ax2.legend()
    ax2.grid(True)

    ax1.set_title('a)', loc='left')
    ax2.set_title('b)', loc='left')

    ax1.legend(loc="lower center", bbox_to_anchor=(0.5, 1.005), ncol=2,
              handlelength=1.6, columnspacing=1.1, borderaxespad=0.0)
    ax2.legend(loc="lower center", bbox_to_anchor=(0.5, 1.005), ncol=2,
              handlelength=1.6, columnspacing=1.1, borderaxespad=0.0)

    plt.tight_layout(rect=(0, 0, 1, 1.025))

    fig.show()

    print(f"Peak frequency: {freqs[np.argmax(am_results)]}")