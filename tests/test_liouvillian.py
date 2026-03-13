import numpy as np

from get_schwinger_hamiltonian import get_schwinger_hamiltonian
from L_n import L_n
from O_n import O_n
from D_matrix import D_matrix
from build_liouvillian import build_liouvillian


def test_liouvillian_reference_eigenvalues():
    L = 4
    a_lat = 1.0
    T = 5.0
    m = 0.5
    e = 0.71

    H = get_schwinger_hamiltonian(L, m, e, a_lat)
    L_ops = [L_n(n=n, H=H, L=L, a=a_lat, T=T) for n in range(L)]
    D = D_matrix(L, a_lat, kind="const", D0=1.0)

    Lsuper = build_liouvillian(H, L_ops, D, a_lat)
    vals_all = np.linalg.eigvals(Lsuper.toarray())
    vals_all = np.asarray(vals_all, dtype=np.complex128)

    given = np.asarray([
    -3.27165339e-17+1.84374375e-17j, -9.24271504e-17+1.23084462e-16j,
    -5.49683823e-02-4.47083132e-01j, -5.49683823e-02+4.47083132e-01j,
    -6.06370298e-02+1.28174296e-02j, -6.06370298e-02-1.28174296e-02j,
    -1.07540816e-01+4.64566856e-01j, -1.07540816e-01-4.64566856e-01j,
    -1.08535026e-01-9.99650752e-17j, -1.65308941e-01+8.96133803e-17j,
    -3.13915019e-01-6.79522482e-17j, -4.17773467e-01-6.25506542e-16j,
    -4.60634662e-01+3.60048712e-02j, -4.60634662e-01-3.60048712e-02j,
    -4.77113769e-01-3.74875107e-01j, -4.77113769e-01+3.74875107e-01j,
    -1.05896504e+00-1.24389324e-15j, -1.47677327e+00-1.94707262e+00j,
    -1.47677327e+00+1.94707262e+00j, -1.48778673e+00+1.45821468e-17j,
    -1.53058783e+00+1.65481866e+00j, -1.53058783e+00-1.65481866e+00j,
    -1.65265405e+00+1.89292675e+00j, -1.65265405e+00-1.89292675e+00j,
    -1.68118664e+00-1.40596425e+00j, -1.68118664e+00+1.40596425e+00j,
    -1.68583228e+00+1.16965604e+00j, -1.68583228e+00-1.16965604e+00j,
    -1.74045706e+00-1.46933933e+00j, -1.74045706e+00+1.46933933e+00j,
    -1.75757329e+00-1.51180342e+00j, -1.75757329e+00+1.51180342e+00j,
    -1.87056050e+00-1.36547776e+00j, -1.87056050e+00+1.36547776e+00j,
    -7.75733831e+00+2.55247828e+00j, -7.75733831e+00-2.55247828e+00j
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