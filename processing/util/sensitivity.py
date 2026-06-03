import numpy as np

gamma_nv = 28.0e9

def calculate_sensitivity(S, slope, meas_time):
    std_S = np.std(S, ddof=1)
    
    std_f = std_S / slope
    std_B = std_f / gamma_nv

    sensitivity = std_B * np.sqrt(meas_time)
    return sensitivity