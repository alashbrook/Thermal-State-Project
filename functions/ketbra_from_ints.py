from common_imports import *
from ketbra_from_bits import ketbra_from_bits
from int_to_bits import int_to_bits

def ketbra_from_ints(ket: int, bra: int, nbits: int) -> SparsePauliOp:
    """
    Straight from an integer to a tensor product.
    """
    return ketbra_from_bits(int_to_bits(ket, nbits), int_to_bits(bra, nbits))