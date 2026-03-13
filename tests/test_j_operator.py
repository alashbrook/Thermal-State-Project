import numpy as np

from get_schwinger_hamiltonian import get_schwinger_hamiltonian
from L_n import L_n
from O_n import O_n
from build_J_operator import build_J_operator


def test_j_operator_blocks():
    L = 4
    a_lat = 1.0
    m_mass = 0.5
    e_coup = 0.71
    T = 5.0

    H = get_schwinger_hamiltonian(L, m_mass, e_coup, a_lat)
    L_ops = [L_n(n=n, H=H, L=L, a=a_lat, T=T) for n in range(L)]

    J = build_J_operator(L_ops=L_ops, n_sys=H.num_qubits)

    m = len(L_ops)
    n_sys = H.num_qubits
    d_sys = 2 ** n_sys

    Jm = J.to_matrix()
    Lmats = [op.to_matrix() for op in L_ops]

    def block(a, b):
        r0, r1 = a * d_sys, (a + 1) * d_sys
        c0, c1 = b * d_sys, (b + 1) * d_sys
        return Jm[r0:r1, c0:c1]

    for j in range(1, m + 1):
        np.testing.assert_allclose(block(j, 0), Lmats[j - 1], atol=1e-10, rtol=1e-10)

    for j in range(1, m + 1):
        np.testing.assert_allclose(
            block(0, j),
            Lmats[j - 1].conj().T,
            atol=1e-10,
            rtol=1e-10,
        )