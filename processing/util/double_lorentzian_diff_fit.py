import numpy as np
from scipy.optimize import curve_fit

def cw_curve(x, b0, b1, A1, A2, x1, x2, gamma1, gamma2):
    return b0 + b1 * x - A1 / (1 + ((x-x1)/gamma1) ** 2) - A2 / (1 + ((x-x2)/ gamma2) ** 2)

def fit_curve(freqs, fluorescence, mod_depth, dip=True):
    b0_guess = 1
    b1_guess = 0.0

    A1_guess = np.max(fluorescence) - np.min(fluorescence)
    A2_guess = A1_guess

    # Guess that the dips are symmetric around the center
    x_index_guess = np.argmin(fluorescence) if dip else np.argmax(fluorescence)
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

    def model(x, b0, b1, A1, A2, x1, x2, gamma1, gamma2):
        high = cw_curve(x + mod_depth / 2, b0, b1, A1, A2, x1, x2, gamma1, gamma2)
        low = cw_curve(x - mod_depth / 2, b0, b1, A1, A2, x1, x2, gamma1, gamma2)

        return -(high - low) / (high + low)

    params, _ = curve_fit(model, freqs, fluorescence, p0=initial_guess, maxfev=18000)

    return model, params