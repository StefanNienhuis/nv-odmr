import numpy as np
from scipy.signal import square
import matplotlib.pyplot as plt

T = 500
N = 3

t = np.arange(T * N)

shift = int(np.round((np.random.random() * T/4) + T/8))

ref = np.zeros(T)
ref[:T//2] = 1

# Simulate PL: 1 - ref * 0.3 + (noise)
sig_base = np.tile(ref, N)
sig = 1 - (np.roll(sig_base, shift) * 0.15) + (np.random.randn(len(sig_base)) * 0.1)

sig = np.reshape(sig, (N, T))

def corr_lockin(sig, ref):

    sig_padded = np.pad(-sig, (0,len(ref)-1))
    ref_padded = np.pad(ref, (0, len(sig)-1))

    sig_padded = sig_padded - np.mean(sig_padded)
    ref_padded = ref_padded - np.mean(ref_padded)

    S = np.fft.fft(sig_padded)
    R = np.fft.fft(ref_padded)

    R_f = S * R.conj()
    Rsigref = np.real(np.fft.ifft(R_f))


    # plt.plot(np.arange(len(Rsigref)),Rsigref)
    # plt.xlim(0, len(sig)//2)
    # plt.show()

    shift = np.argmax(Rsigref[0:len(sig)//2])

    return shift

sig_summed = np.sum(sig, axis=0)
det_shift = corr_lockin(sig_summed, ref)
print(shift, det_shift)

sig_unshifted = np.roll(sig, -round(shift), axis=1)

active_counts = np.sum(sig_unshifted[:, :T//2], axis=1)
inactive_counts = np.sum(sig_unshifted[:, T//2:], axis=1)

am_signal = (active_counts - inactive_counts) / inactive_counts
mean_am_signal = np.mean(am_signal)

print(mean_am_signal)

plt.step(np.arange(T), sig_unshifted[1])
plt.step(np.arange(T), ref)
plt.show()