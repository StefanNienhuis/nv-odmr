import numpy as np
from scipy.optimize import curve_fit

def curve(x, y0, a, b, c, d):
    return y0 + a * (1 - np.exp(-x / b) * np.cos(c * x + d))

def fit_curve(x, y):
    y0_guess = np.min(y)
    a_guess = np.ptp(y) / 2
    b_guess = 3e3 # 3 us
    c_guess = 2 * np.pi / 100
    d_guess = 0

    initial_guess = [y0_guess, a_guess, b_guess, c_guess, d_guess]

    params, _ = curve_fit(curve, x, y, p0=initial_guess)

    return curve, params