from common_imports import *

def rho0_from_init(init, n_qubits):
    d = 2**n_qubits
    psi = np.zeros(d, dtype=complex)

    if isinstance(init, str):
        # init[0] targets qubit 0 -> reverse before int()
        idx = int(init[::-1], 2)
        psi[idx] = 1.0
    else:
        # handle other types if you have them
        raise ValueError("init must be a bitstring")

    return np.outer(psi, psi.conj())