from common_imports import *

def vec_col_major(rho: np.ndarray) -> np.ndarray:
    """
    vec(rho) using column-stacking; matches kron structure in build_liouvillian
    """
    return np.asarray(rho, dtype=complex).reshape(-1, order="F")