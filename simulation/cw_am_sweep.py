import numpy as np
import matplotlib.pyplot as plt
from cw import simulate_cw

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

if __name__ == '__main__':
    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle(f"Simulated AM CW-ODMR frequency sweep (B = {Bz * 1e6} $\\mu$T)")
    fig.set_figwidth(10.8)
    fig.set_figheight(4.8)
    # plt.figure(figsize=(10.8, 4.8))

    ax1.plot(freqs / 1e9, on_results, label="MW on")
    ax1.plot(freqs / 1e9, off_results, label="MW off")
    ax1.set_xlabel("Microwave frequency (GHz)")
    ax1.set_ylabel("Fluorescence (a.u)")
    ax1.legend()
    ax1.grid(True)

    ax2.plot(freqs / 1e9, am_results, label="AM signal")
    ax2.set_xlabel("Microwave frequency (GHz)")
    ax2.set_ylabel("Relative fluorescence")
    ax2.legend()
    ax2.grid(True)

    fig.tight_layout()
    fig.show()

    print(f"Peak frequency: {freqs[np.argmax(am_results)]}")