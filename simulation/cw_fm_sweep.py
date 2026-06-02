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

delta_f = 3e6

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

if __name__ == '__main__':
    # Plot
    plt.figure()
    plt.plot(freqs / 1e9, fm_results)
    plt.xlabel("Microwave frequency (GHz)")
    plt.ylabel("Normalized fluorescence")
    plt.title(f"Simulated FM CW-ODMR frequency sweep (B = {Bz*1e6} $\\mu$T)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
