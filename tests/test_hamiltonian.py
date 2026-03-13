import numpy as np
from get_schwinger_hamiltonian import get_schwinger_hamiltonian

def test_hamiltonian_reference_eigenvalues():
    L = 6
    a = 1.0
    m = 0.5
    e = 1.0

    H = get_schwinger_hamiltonian(L, m, e, a)
    H_dense = H.to_matrix()

    vals_all = np.linalg.eigvals(H_dense)
    vals_all = np.real_if_close(vals_all, tol=1e5)
    vals_all = np.asarray(vals_all, dtype=np.complex128)

    given = np.asarray([
        -2.17935655, -0.45117005, -0.36291496, -0.22088833, -0.06897418,
        -0.02267936,  0.64717163,  0.64799083,  0.80051069,  1.40791377,
         1.52841255,  1.69044329,  1.79807205,  1.86945912,  1.92068802,
         2.12033848,  2.6735862 ,  2.71617821,  3.87675283,  4.60846576
    ], dtype=np.complex128)

    remaining = list(range(len(vals_all)))
    ordered = []

    for lam in given:
        rem_vals = vals_all[remaining]
        j = int(np.argmin(np.abs(rem_vals - lam)))
        idx = remaining.pop(j)
        ordered.append(vals_all[idx])

    ordered = np.array(ordered, dtype=np.complex128)

    np.testing.assert_allclose(ordered, given, atol=1e-7, rtol=1e-7)