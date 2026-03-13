from common_imports import *

def int_to_bitstring(idx: int, nbits: int) -> str:
    """
    Converts an integer to binary as a string.
    """
    bits = int_to_bits(idx, nbits)
    return "".join(str(bits[q]) for q in reversed(range(nbits)))