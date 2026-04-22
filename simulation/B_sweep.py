import numpy as np
import matplotlib.pyplot as plt
from cw import simulate_cw

# Simulation parameters
Amw = 1e6               # microwave field amplitude (Hz)
f_mw = 2876950000.0           # microwave frequency

Bmin = 0
Bmax = 1e-3
Bs = np.linspace(Bmin, Bmax, 401)

fluorescence_results = [simulate_cw(B, Amw, f_mw) for B in Bs]
fluorescence_results = np.array(fluorescence_results)
fluorescence_results = fluorescence_results / np.max(fluorescence_results)

if __name__ == '__main__':
    # Plot
    plt.figure()
    plt.plot(Bs * 1e6, fluorescence_results)
    plt.xlabel("Magnetic field (uT)")
    plt.ylabel("Normalized fluorescence")
    plt.title(f"ODMR magnetic field sweep ($f_{{mw}}$ = {f_mw/1e9} GHz)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    print(f"Dip magnetic field: {Bs[np.argmin(fluorescence_results)]*1e6} uT")