from common_imports import *

def lindblad_evolve_direct(
    Lsuper: sp.csr_matrix,
    rho0: np.ndarray,
    times: np.ndarray,
    *,
    enforce_physical: bool = True,
) -> list[np.ndarray]:
    """
    Direct evolution: vec(rho(t)) = exp(Lsuper * t) vec(rho0)
    """
    rho0 = np.asarray(rho0, dtype=complex)
    d = rho0.shape[0]
    v0 = vec_col_major(rho0)

    rhos = []
    for t in times:
        vt = spla.expm_multiply(Lsuper * t, v0) # double-check this line
        rho_t = unvec_col_major(vt, d)

        if enforce_physical:
            rho_t = 0.5 * (rho_t + rho_t.conj().T)
            tr = np.trace(rho_t)
            if tr != 0:
                rho_t = rho_t / tr
        rhos.append(rho_t)
    return rhos