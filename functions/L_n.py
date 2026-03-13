from common_imports import *
from O_n import O_n

def L_n(n: int, H: SparsePauliOp, L: int, a: float, T: float) -> SparsePauliOp:
    O = O_n(n, L, a)
    comm = H @ O - O @ H

    return (O - (1.0/(4.0*T)) * comm).simplify()