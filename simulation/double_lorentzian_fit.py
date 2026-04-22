# Fit Lorentzian to output of cw_sweep

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from freq_sweep import freqs, fluorescence_results

def double_lorentzian(x, b0, b1, A1, A2, x1, x2, gamma1, gamma2):
    return b0 + b1 * x - A1 / (1 + ((x-x1)/gamma1) ** 2) - A2 / (1 + ((x-x2)/ gamma2) ** 2)

b0_guess = 1
b1_guess = 0.0

A1_guess = np.max(fluorescence_results) - np.min(fluorescence_results)
A2_guess = A1_guess

# Guess that the dips are symmetric around the center
x_index_guess = np.argmin(fluorescence_results)
x_midpoint = len(freqs) // 2
if x_index_guess <= x_midpoint:
    x1_guess = freqs[x_index_guess]
    x2_guess = freqs[x_midpoint + (x_midpoint - x_index_guess)]
else:
    x1_guess = freqs[x_midpoint - (x_index_guess - x_midpoint)]
    x2_guess = freqs[x_index_guess]

# Estimate linewidth of 1/50 of the range
gamma1_guess = (freqs.max() - freqs.min()) / 50
gamma2_guess = gamma1_guess

initial_guess = [b0_guess, b1_guess, A1_guess, A2_guess, x1_guess, x2_guess, gamma1_guess, gamma2_guess]

params, _ = curve_fit(double_lorentzian, freqs, fluorescence_results, p0=initial_guess)

fit_fluorescence = double_lorentzian(freqs, *params)

print(f"""
b0 = {b0_guess}
b1 = {b1_guess}
A1 = {A1_guess}
A2 = {A2_guess}
x1 = {x1_guess}
x2 = {x2_guess}
gamma1 = {gamma1_guess}
gamma2 = {gamma2_guess}
""")

if __name__ == '__main__':
    # Plot
    plt.figure()
    plt.plot(freqs / 1e9, fluorescence_results, label='Results')
    plt.plot(freqs / 1e9, fit_fluorescence, '--', label='Fit')
    plt.xlabel("Microwave frequency (GHz)")
    plt.ylabel("Normalized fluorescence")
    plt.title("Basic NV ODMR frequency sweep")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()