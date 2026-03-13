from common_imports import *

def int_to_bits(idx: int, nbits: int) -> list[int]:
    """
    Extracts an integer's binary digits as a list of 0/1 values.
    """
    if idx < 0 or idx >= (1 << nbits):
        raise ValueError(f"idx = {idx} out of range for nbits = {nbits}")
    return [(idx >> q) & 1 for q in range(nbits)] # q = 0 is qubit 0