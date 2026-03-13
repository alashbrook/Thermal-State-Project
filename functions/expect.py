from common_imports import *

def expect(op: np.ndarray, rho: np.ndarray) -> float:
    return float(np.real_if_close(np.trace(rho @ op)))