import numpy as np

def lock_in(sig, ref):
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

    shift = d_phi / (2 * np.pi) * len(sig)
    print(shift, shift % (len(sig) // 2))
    shift = shift % (len(sig) // 2)

    return shift


def lock_in_corr(sig, ref, am=False, plot=False):
    sig_norm = sig - sig.mean()
    sig_norm = sig_norm / np.max(np.abs(sig_norm))
    
    ref_norm = ref - ref.mean()
    ref_norm = ref_norm / np.max(np.abs(ref_norm))
    
    # AM signal should be flipped, as MW high means PL low
    if am:
        sig_norm = -sig_norm
    
    sig_padded = np.pad(-sig_norm, (0,len(ref)-1))
    ref_padded = np.pad(ref_norm, (0, len(sig)-1))

    S = np.fft.fft(sig_padded)
    R = np.fft.fft(ref_padded)

    R_f = S * R.conj()
    Rsigref = np.real(np.fft.ifft(R_f))

    if am:
        shift = np.argmax(Rsigref[0:len(sig)])
    else: 
        shift = np.argmax(np.abs(Rsigref[0:len(sig)]))
        
    if plot:
        import matplotlib.pyplot as plt
        plt.plot(Rsigref)
        plt.show()
        
        plt.plot(-sig_norm)
        plt.plot(ref_norm)
        plt.show()

    return shift