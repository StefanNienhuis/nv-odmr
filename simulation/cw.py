# CW ODMR simulation using Qutip - simplified Hamiltonian, RWA

import numpy as np
import qutip as qt

def simulate_cw(Bz, Amw, f_mw):
    # Basis states: |+1>, |0>, |-1>
    m_plus = qt.basis(3, 0)
    m_zero = qt.basis(3, 1)
    m_minus = qt.basis(3, 2)

    # Spin operators
    Sx = qt.jmat(1, 'x')

    # Physical constants
    D = 2.87e9  # zero-field splitting (Hz)
    gamma_e = 28e9  # electron gyromagnetic ratio (Hz/T)

    # Relaxation / optical pumping rates
    gamma_pump = 1e6  # pumps |±1> -> |0>
    gamma_dephase = 0.5e6  # dephasing rate

    # - Collapse operators
    c_ops = []

    # Optical pumping / relaxation from |+1> and |-1> into |0>
    c_ops.append(np.sqrt(gamma_pump) * (m_zero * m_plus.dag()))
    c_ops.append(np.sqrt(gamma_pump) * (m_zero * m_minus.dag()))

    # Pure dephasing on |+1> and |-1>
    c_ops.append(np.sqrt(gamma_dephase) * (m_plus * m_plus.dag()))
    c_ops.append(np.sqrt(gamma_dephase) * (m_minus * m_minus.dag()))

    # - Expectation operators

    # Population of |0> - fluorescence proportional to this value
    p_zero = m_zero * m_zero.dag()

    # - Simulation
    # Initialize in |0>
    psi0 = m_zero

    tlist = np.linspace(0, 10e-6, 100)

    # Ground state Hamiltonian: D Sz^2 + gamma B @ S
    # Assuming B in z-direction ->  D Sz^2 + gamma Bz Sz
    # For |1>:  E = D + gamma Bz
    # For |0>:  E = 0
    # For |-1>: E = D - gamma Bz

    # Energies in rotating frame at f_mw
    E_plus = (D + gamma_e * Bz) - f_mw
    E_minus = (D - gamma_e * Bz) - f_mw

    H0 = (
            E_plus * (m_plus * m_plus.dag()) +
            0 * m_zero * m_zero.dag() +
            E_minus * (m_minus * m_minus.dag())
    )

    # Microwave drive Hamiltonian in rotating frame at f_mw
    H_drive = Sx

    # Total Hamiltonian
    H = [H0, [H_drive, Amw]]

    solution = qt.mesolve(H, psi0, tlist, c_ops, e_ops=[p_zero])
    return solution.expect[0][-1]
