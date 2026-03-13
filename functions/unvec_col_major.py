from common_imports import *

def unvec_col_major(v: np.ndarray, d: int) -> np.ndarray:
    """
    inverse of vec_col_major
    """
    return np.asarray(v, dtype=complex).reshape((d, d), order="F")