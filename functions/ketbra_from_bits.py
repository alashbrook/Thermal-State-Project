from common_imports import *
from ketbra_1q import ketbra_1q
from int_to_bits import int_to_bits

def ketbra_from_bits(ket_bits: list[int], bra_bits: list[int]) -> SparsePauliOp:
    """
    Returns a SparsePauliOp for |ket><bra| on n qubits.
    ket_bits[q] corresponds to qubit q

    |ket><bra| = ⊗_{q=0}^{n-1} (|k_q><b_q|)
    """
    if len(ket_bits) != len(bra_bits):
        raise ValueError("ket_bits and bra_bits must have same length")
    n = len(ket_bits)
    if n == 0:
        raise ValueError("Need at least 1 qubit")

    op = ketbra_1q(ket_bits[n-1], bra_bits[n-1])
    for q in reversed(range(n-1)): # q = n-2 ... 0
        op = op.tensor(ketbra_1q(ket_bits[q], bra_bits[q]))
    return op.simplify()