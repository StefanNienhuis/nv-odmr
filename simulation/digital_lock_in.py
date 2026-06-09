import numpy as np
from scipy.signal import square
import matplotlib.pyplot as plt

T = 500
N = 3

t = np.arange(T * N)

shift = int(np.round(np.random.random() * T/2))

ref = np.zeros(T)
ref[:T//2] = 1

# Simulate PL: 1 - ref * 0.3 + (noise)
sig_base = np.tile(ref, N)
sig = 1 - (np.roll(sig_base, shift) * 0.15) + (np.random.randn(len(sig_base)) * 0.1)

sig = np.reshape(sig, (N, T))

def lockin(sig, ref):
    assert len(sig) == len(ref)

    theta = np.linspace(0, 2 * np.pi, len(sig))

    # Normalize signals
    s = sig - np.mean(sig)
    r = ref - np.mean(ref)

    # Invert signal since MW high means PL low
    s = -s

    Xs = np.sum(s * np.cos(theta))
    Ys = np.sum(s * np.sin(theta))

    phi_s = np.arctan2(Ys, Xs)

    Xr = np.sum(r * np.cos(theta))
    Yr = np.sum(r * np.sin(theta))

    phi_r = np.arctan2(Yr, Xr)

    d_phi = phi_s - phi_r
    d_phi = d_phi % (2 * np.pi)

    R = np.sqrt(Xs**2 + Ys**2)
    shift = d_phi / (2 * np.pi) * len(sig)

    return R, shift

sig_summed = np.sum(sig, axis=0)
R, det_shift = lockin(sig_summed, ref)
print(R, shift, det_shift)

sig_unshifted = np.roll(sig, -round(det_shift), axis=1)

active_counts = np.sum(sig_unshifted[:, :T//2], axis=1)
inactive_counts = np.sum(sig_unshifted[:, T//2:], axis=1)

am_signal = (active_counts - inactive_counts) / inactive_counts
mean_am_signal = np.mean(am_signal)

print(mean_am_signal)

plt.step(np.arange(T), sig_unshifted[0])
plt.step(np.arange(T), ref)
plt.show()