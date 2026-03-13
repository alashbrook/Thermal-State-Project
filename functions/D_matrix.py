from common_imports import *

def D_matrix(L: int, a: float, kind: str, D0: float, sigma: float | None = None):
    D = np.zeros((L, L), dtype = float)
    for n1 in range(L):
        for n2 in range(L):
            x = (n1 - n2) * a
            if kind == "delta":
                D[n1, n2] = D0 if n1 == n2 else 0.0
            elif kind == "gaussian":
                assert sigma is not None
                D[n1, n2] = D0 * np.exp(-(x*x) / (2.0*sigma*sigma))
            elif kind == "const":
                D[n1, n2] = D0
            else:
                raise ValueError("kind must be delta/gaussian/const")
    return D