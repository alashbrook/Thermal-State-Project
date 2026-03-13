from common_imports import *

def O_n(n: int, L: int, a: float) -> SparsePauliOp:
    """
    n: site
    L: number of sites
    """
    s = (-1)**n

    I_label = "I" * L

    chars = ["I"] * L
    chars[-1 - n] = "Z" # rightmost is qubit 0
    Z_label = "".join(chars)

    O = SparsePauliOp.from_list([
        (I_label, s/(2.0*a)),
        (Z_label, s/(2.0*a)),
    ])

    return O.simplify()