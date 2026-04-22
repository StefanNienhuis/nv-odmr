# Show effect of lock-in detection on Lorentzian data

# Values from double_lorentzian_fit
b0 = 1
b1 = 0.0
A1 = 0.5084616467365286
A2 = 0.5084616467365286
x1 = 2865000000.0
x2 = 2875000000.0
gamma1 = 2000000.0
gamma2 = 2000000.0

params = [b0, b1, A1, A2, x1, x2, gamma1, gamma2]

def double_lorentzian(x, b0, b1, A1, A2, x1, x2, gamma1, gamma2):
    return b0 + b1 * x - A1 / (1 + ((x-x1)/gamma1) ** 2) - A2 / (1 + ((x-x2)/ gamma2) ** 2)

