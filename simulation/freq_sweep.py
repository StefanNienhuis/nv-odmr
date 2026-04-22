import numpy as np
import matplotlib.pyplot as plt
from cw import simulate_cw

# Simulation parameters
Bz = 250e-6             # magnetic field along NV axis (T)
Amw = 1e6               # microwave field amplitude (Hz)

f_min = 2.84e9
f_max = 2.90e9
n_points = 401
freqs = np.linspace(f_min, f_max, n_points)

fluorescence_results = [simulate_cw(Bz, Amw, f_mw) for f_mw in freqs]
fluorescence_results = np.array(fluorescence_results)
fluorescence_results = fluorescence_results / np.max(fluorescence_results)

if __name__ == '__main__':
    # Plot
    plt.figure()
    plt.plot(freqs / 1e9, fluorescence_results)
    plt.xlabel("Microwave frequency (GHz)")
    plt.ylabel("Normalized fluorescence")
    plt.title(f"ODMR frequency sweep  (B = {Bz*1e6} uT)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    print(f"Dip frequency: {freqs[np.argmin(fluorescence_results)]}")