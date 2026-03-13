from common_imports import *

def ketbra_1q(ket: int, bra: int) -> SparsePauliOp:
    """
    |0><0| = 0.5(I + Z)
    |1><1| = 0.5(I - Z)
    |0><1| = 0.5(X + iY)
    |1><0| = 0.5(X - iY)
    """
    if ket == 0 and bra == 0:
        return SparsePauliOp.from_list([("I", 0.5), ("Z", 0.5)])
    if ket == 1 and bra == 1:
        return SparsePauliOp.from_list([("I", 0.5), ("Z", -0.5)])
    if ket == 0 and bra == 1:
        return SparsePauliOp.from_list([("X", 0.5), ("Y", 0.5j)])
    if ket == 1 and bra == 0:
        return SparsePauliOp.from_list([("X", 0.5), ("Y", -0.5j)])
    raise ValueError("ket and bra must be 0 or 1")